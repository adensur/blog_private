In the previous chapter, we've explored how we can train a simple MNIST digit classifier using PyTorch Lightning.  

Over the course of the next few chapters, I wanted to go through main capabilities of Lightning, such as mixed precision, gradient accumulation, scaling and clipping, LR scheduling etc. To do that, I want to go through a real-world example of finetuning [simlm](https://huggingface.co/intfloat/simlm-base-msmarco-finetuned) or [bge-m3](https://huggingface.co/BAAI/bge-m3) embedding model for text retrieval on MSMarco dataset. bge-m3 model is close to SOTA on this task, and is used in many startups and enterprises, mine included. So, these guides will not only be useful from theoretical standpoint, but might also serve as a step-by-step guide for a practical task. Refer to my [video](https://youtu.be/UlSCPHtSVzk) for an in-depth explanation of retrieval embeddings.   

When attempting something like this, it is much easier to have a working example of training, evaluation scripts and metrics. I refer to [simlm](https://github.com/microsoft/unilm/tree/master/simlm) repo with finetuning examples and evaluation scripts. That one uses HF Trainer and `deepspeed`.  

First, I use a small wrapper around hf models to compute embeddings:  
```python
class MyModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, batch_dict):
        preds = self.model(**batch_dict)
        hidden_state = preds.last_hidden_state
        embeds = hidden_state[:, 0]
        embeds = F.normalize(embeds, dim=-1)
        embeds = embeds.contiguous()
        return embeds

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
```
It contains specific logic about getting an embedding out of the model (some model might use different logic, like avg pooling).   

Next, we want to evaluate some sort of metric on a dev set of MSMarco. Metric is evaluated in "retrieval" setting, i.e., for every input query, full corpus is sorted by embedding dot product, and relevance of top-k results is evaluated. I use [./encode_passages.py](./encode_passages.py) to build corpus embeddings, and [./search.py](./search.py) to evaluate the metric. Both are copied and adapted from simlm repo.   
This is how to call them:
```bash
python encode_passages.py --data-dir msmarco_bm25_official --encode_save_dir encoded_passages --encode_batch_size 128 --encode_shard_size 1000000 --p_max_len 512 --model_name_or_path intfloat/simlm-base-msmarco-finetuned --dataloader_num_workers 16
python search.py --data-dir msmarco_bm25_official --encode_save_dir encoded_passages --q_max_len 512 --model_name_or_path intfloat/simlm-base-msmarco-finetuned --dataloader_num_workers 16 --search_split dev --search_out_dir search_out
```
As `--model_name_or_path`, we can pass either an existing model from HF, or a local path, to a finetuned model, for example.   

This is how results look like for `intfloat/simlm-base-msmarco-finetuned` - one of the models provided by simlm:  
```
{
    "NDCG@10": 0.47579,
    "NDCG@50": 0.51799,
    "NDCG@100": 0.525,
    "NDCG@200": 0.52965,
    "NDCG@1000": 0.53388,
    "MAP@10": 0.40329,
    "MAP@50": 0.41333,
    "MAP@100": 0.414,
    "MAP@200": 0.41425,
    "MAP@1000": 0.41437,
    "Recall@10": 0.69687,
    "Recall@50": 0.87782,
    "Recall@100": 0.91984,
    "Recall@200": 0.95207,
    "Recall@1000": 0.9868,
    "mrr": 40.9831
}
```
We will only refer to NDCG@10 in the future.  
Here is another measurement for a model finetuned by me, using simlm code, on the simplest bm25-hard-negative msmarco version:  
```
"NDCG@10": 0.44149,
```
This is the metric we will target in our future finetunes.   
## Dataset
I wanted a simple dataset logic, so I wrote a [converter script](msmarco_converter.py) to convert msmarco dataset (consisting of several files) into a single jsonl file.   
To get the data, refer to [simlm](https://github.com/microsoft/unilm/blob/master/simlm/scripts/download_msmarco_data.sh).  
To convert:
```bash
python msmarco_convert.py --data-dir data/msmarco_bm25_official --output-dir data
```
Each json line now contains data like this:   
```
{
    "query": "what are some tissues found in the skin",
    "pos_docs":
    [
        [
            "Areolar Tissue",
            "Areolar Tissue. holds organs and tissue fluid in place, ..."
        ],
    ],
    "neg_docs":
    [
        [
            "Vaginal Prolapse",
            "Vaginal Prolapse Overview. The network of muscles, ..."
        ],
        [
            "All About Minerals",
            "Why silica is good for you. Silica is a trace mineral, ..."
        ],
    ]
}
```
Each row has a query and some positive and negative docs. Each doc contains title and text, as separate fields. Some models, like `intfloat/simlm-base-msmarco-finetuned`, expects them separately, and will generate different tokentype embeddings for them. Some, like `bge-m3`, will just use single concatenated field.  

We tokenize in the data collator (to benefit from HF tokenizers' batching logic):   
```python
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
```
We also select a single positive randomly, and n-1 random negatives, where n-1 (`train_n_passages`) is a hyperparam. Our further code relies on the shape of each tensor to be the same on every batch, so we will cut off some negatives if needed. Random sampling makes sure that we will use different positives and negatives in each epoch, which helps a bit.
## Model inference and loss
Here is how our lightning module code looks like:  
```python
class MyLightningModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = MyModel(args.model_name_or_path)
        self.model.train()

    def training_step(self, batch, batch_idx):
        query_embeds = self.model.encode(batch["query"])
        doc_embeds = self.model.encode(batch["docs"])

        scores, labels = self.full_contrastive_scores_and_labels(
            query=query_embeds,
            key=doc_embeds,
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
```
`full_contrastive_scores_and_labels` is copied from `simlm` as well. Basic logic is that we assign query,pos pair score as 1.0, and query,neg pair scores as 0.0. In addition, we add extra exampes:   
- "in batch negatives" - positives and negatives for one query are added as negatives for other queries
- doc-to-doc negatives - one query's positive multiplied by another query's positive is a negative (one positive acts as a "query" in this case)
- query-to-query negatives - one query multiplied by another query is a negative

As a loss, we use cross entropy, though the specific logic of positives and negatives combined with it is usually called "contrastive loss".   
We scale loss by temperature, which is a constant hyperparam for now, though we will setup warmup/scheduling later.  
## Training
```python
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
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--q_max_len', type=int, default=32)
    parser.add_argument('--p_max_len', type=int, default=144)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = JsonlDataset(args.train_path)
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer, args),
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

```
Training code is almost unchanged. I had to add `find_unused_parameters=True` to `DDPStrategy`, otherwise lightning complains about unused params.   

This is how to run it, and then do evaluation:
```bash
export MY_RUN_NAME=train_v2
python lightning_train.py --model_name_or_path intfloat/simlm-base-msmarco --train_path data/train.jsonl --output_dir runs/$MY_RUN_NAME --epochs 3 --batch-size 16 --p_max_len 144 --q_max_len 32 --temperature 0.02
python encode_passages.py --data-dir msmarco_bm25_official --encode_save_dir encode_runs/$MY_RUN_NAME --encode_batch_size 128 --encode_shard_size 1000000 --p_max_len 512 --model_name_or_path runs/$MY_RUN_NAME --dataloader_num_workers 16
python search.py --data-dir msmarco_bm25_official --encode_save_dir encode_runs/$MY_RUN_NAME --q_max_len 512 --model_name_or_path runs/$MY_RUN_NAME --dataloader_num_workers 16 --search_split dev --search_out_dir search_runs/$MY_RUN_NAME
```
In my case, this configuration yielded `"NDCG@10": 0.42248`, a bit lower than simlm version, though amount of data and main params are the same.   

In the next few chapters, we will add missing functionality to the training script and see how that impacts the results.