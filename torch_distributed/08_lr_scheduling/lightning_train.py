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
from transformers import AutoTokenizer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lib import RestorableSampler
from typing import Dict, List, Any, Tuple, Optional
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import random
from model import MyModel


def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)

    all_tensors[dist.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


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


def collate_fn(tokenizer, args, examples: List[Dict[str, Any]]):
    # Extract queries and all docs (positive first, then negatives)
    queries = [ex["query"] for ex in examples]
    
    # Combine positive and negative docs for each example
    all_docs = []
    for ex in examples:
        pos_doc = random.choice(ex["pos_docs"])  # Take random positive doc
        neg_docs = random.sample(ex["neg_docs"], args.train_n_passages - 1)  # Take n-1 random negative docs
        docs = [pos_doc] + neg_docs  # Positive doc first, then negatives
        all_docs.extend([(doc[0], doc[1]) for doc in docs])  # (title, text) pairs
    
    # Tokenize all queries in one batch
    query_encodings = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=args.q_max_len,
        return_tensors="pt"
    )

    # Tokenize all docs in one batch
    doc_encodings = tokenizer(
        [doc[0] for doc in all_docs],  # titles
        text_pair=[doc[1] for doc in all_docs],  # texts
        padding=True,
        truncation=True,
        max_length=args.p_max_len,
        return_tensors="pt"
    )

    return {
        "query": query_encodings,
        "docs": doc_encodings
    }


class MyLightningModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = MyModel(args.model_name_or_path)
        self.model.train()
        self.save_hyperparameters()
        self.consumed_samples = 0
        self.sampler = None
        self.trainer = None

    def state_dict(self):
        state = super().state_dict()
        state['consumed_samples'] = self.consumed_samples
        return state

    def load_state_dict(self, state_dict):
        if 'consumed_samples' in state_dict:
            self.consumed_samples = state_dict.pop('consumed_samples')
        super().load_state_dict(state_dict)

    def common_step(self, batch):
        # Get query embeddings for all queries in one forward pass
        query_embeds = self.model.encode(batch["query"])
        
        # Get all doc embeddings in one forward pass
        doc_embeds = self.model.encode(batch["docs"])
        
        # Gather embeddings from all devices
        all_query_embeds = dist_gather_tensor(query_embeds)
        all_doc_embeds = dist_gather_tensor(doc_embeds)

        # Compute scores and labels
        scores, labels = self.full_contrastive_scores_and_labels(
            query=all_query_embeds,
            key=all_doc_embeds,
            use_all_pairs=True
        )
        
        # Scale scores by temperature
        scores = scores / self.args.temperature
        
        # Compute loss
        loss = F.cross_entropy(scores, labels)
        loss *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        # assume that last batch is always dropped
        self.consumed_samples += len(batch["query"]["input_ids"]) * self.args.world_size
        self.log("train_loss", loss)
        self.log("consumed_samples", self.consumed_samples)
        # print("Rank: ", dist.get_rank(), "Consumed samples: ", self.consumed_samples)
        if self.sampler is not None:
            self.sampler.consumed_samples = self.consumed_samples
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("val_loss", loss)
        return loss

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.parameters() if p.grad is not None]))
        self.log("grad_norm", grad_norm)


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
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.warmup_steps is None:
            scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)
            return [optimizer], [scheduler]
        
        def lr_lambda(current_step: int):
            if current_step < self.args.warmup_steps:
                # Linear warmup from lr/2 to lr
                return 0.5 + (current_step * 0.5 / self.args.warmup_steps)
            else:
                # Linear decay from lr to 0
                return max(0.0, (self.args.max_steps - current_step) / (self.args.max_steps - self.args.warmup_steps))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]



def main():
    parser = argparse.ArgumentParser(description='Contrastive Learning')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.02)
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--val_check_interval', type=int, default=5000)
    parser.add_argument('--q_max_len', type=int, default=32)
    parser.add_argument('--p_max_len', type=int, default=144)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument("--use_restorable_sampler", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument("--loss_scale", type=float, default=1.0, help="negative to scale by world size")
    parser.add_argument("--train_n_passages", type=int, default=16, help="Number of passages to train on. 1 positive, n-1 negatives")
    args = parser.parse_args()
    args.world_size = args.num_nodes * args.devices

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = JsonlDataset(args.train_path)
    val_dataset = JsonlDataset(args.val_path)
    args.steps_per_epoch = len(train_dataset) // args.batch_size // args.world_size
    if args.max_steps == -1:
        args.max_steps = args.steps_per_epoch * args.epochs
    
    # Load from checkpoint if specified, otherwise from model_name_or_path
    if args.resume_from_checkpoint:
        print("Restoring from checkpoint: ", args.resume_from_checkpoint)
        module = MyLightningModule.load_from_checkpoint(args.resume_from_checkpoint, args=args)
        print("Restored consumed samples: ", module.consumed_samples)
    else:
        module = MyLightningModule(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RestorableSampler(train_dataset, module.consumed_samples) if args.use_restorable_sampler else None,
        shuffle=False if args.use_restorable_sampler else True,
        collate_fn=partial(collate_fn, tokenizer, args),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    module.sampler = train_loader.sampler

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer, args),
        num_workers=4,
        pin_memory=True
    )

    wandb_logger = WandbLogger(project="your_project_name")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        strategy=DDPStrategy(find_unused_parameters=True), # otherwise lightning complains about unused params
        num_nodes=args.num_nodes,
        devices=args.devices,
        val_check_interval=args.val_check_interval,
        callbacks=[ModelCheckpoint(
            dirpath=args.output_dir,
            save_top_k=-1,
            every_n_train_steps=args.val_check_interval,
            save_weights_only=False
        )],
        logger=wandb_logger,
        gradient_clip_val=args.max_grad_norm,
    )

    module.trainer = trainer

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if dist.get_rank() == 0:
        print("Done training, saving model!")
        module.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
