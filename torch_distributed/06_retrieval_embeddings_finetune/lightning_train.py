import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
import lightning as L
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from lightning.pytorch.strategies import DDPStrategy

from typing import Dict, List, Any, Tuple
from functools import partial

from model import MyModel


class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.data = []
        with open(path, "r") as f:
            for line in tqdm(f, desc="Loading data"):
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(tokenizer, examples: List[Dict[str, Any]]):
    # Extract queries, positive and negative docs
    queries = [ex["query"] for ex in examples]
    pos_titles = [ex["pos_doc"][0] for ex in examples]
    pos_texts = [ex["pos_doc"][1] for ex in examples]
    
    # Tokenize queries
    query_encodings = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Tokenize positive docs with title+text pairs
    pos_doc_encodings = tokenizer(
        pos_titles,
        text_pair=pos_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Process negative docs
    neg_encodings_list = []
    for ex in examples:
        neg_titles = [neg[0] for neg in ex["neg_doc"]]
        neg_texts = [neg[1] for neg in ex["neg_doc"]]
        
        neg_encodings = tokenizer(
            neg_titles,
            text_pair=neg_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        neg_encodings_list.append(neg_encodings)

    return {
        "query": query_encodings,
        "pos_doc": pos_doc_encodings,
        "neg_docs": neg_encodings_list
    }


class MyLightningModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = MyModel(args.model_name_or_path)
        self.model.train()

    def training_step(self, batch, batch_idx):
        # Get query embeddings
        query_embeds = self.model.encode(batch["query"])
        
        # Combine positive and negative docs into one batch
        all_docs = [batch["pos_doc"]]
        for neg_docs in batch["neg_docs"]:
            all_docs.append(neg_docs)
        
        # Get all doc embeddings in one forward pass
        all_doc_embeds = []
        for docs in all_docs:
            doc_embeds = self.model.encode(docs)
            all_doc_embeds.append(doc_embeds)
        key_embeds = torch.cat(all_doc_embeds, dim=0)

        # Compute scores and labels
        scores, labels = self.full_contrastive_scores_and_labels(
            query=query_embeds,
            key=key_embeds,
            use_all_pairs=True
        )
        
        # Scale scores by temperature
        scores = scores / self.args.temperature
        
        # Compute loss
        loss = F.cross_entropy(scores, labels)
        
        self.log("train_loss", loss)
        return loss

    def full_contrastive_scores_and_labels(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            use_all_pairs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

        train_n_passages = key.shape[0] // query.shape[0]
        labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)
        labels = labels * train_n_passages

        # batch_size x (batch_size x n_psg)
        qk = torch.mm(query, key.t())

        if not use_all_pairs:
            return qk, labels

        # batch_size x dim
        sliced_key = key.index_select(dim=0, index=labels)
        assert query.shape[0] == sliced_key.shape[0]

        # batch_size x batch_size
        kq = torch.mm(sliced_key, query.t())
        kq.fill_diagonal_(float('-inf'))

        qq = torch.mm(query, query.t())
        qq.fill_diagonal_(float('-inf'))

        kk = torch.mm(sliced_key, sliced_key.t())
        kk.fill_diagonal_(float('-inf'))

        scores = torch.cat([qk, kq, qq, kk], dim=-1)

        return scores, labels

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser(description='Contrastive Learning')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = JsonlDataset(args.train_path)
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer),
        num_workers=4,
        pin_memory=True
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        strategy=DDPStrategy(find_unused_parameters=True), # otherwise lightning complains about unused params
        num_nodes=args.num_nodes,
        devices=args.devices
    )

    module = MyLightningModule(args)
    trainer.fit(module, train_dataloaders=train_loader)

    if dist.get_rank() == 0:
        print("Done training, saving model!")
        module.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
