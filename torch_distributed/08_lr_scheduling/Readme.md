Now that the basic hygiene is set up, let's add some missing functionality. In this post we will talk about:
- In-batch negatives
- LR scheduling
- Gradient clipping
## In-batch negatives
Our current code actually has these:
```python
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
# ...
 def common_step(self, batch):
    query_embeds = self.model.encode(batch["query"])
    
    # Get all doc embeddings in one forward pass
    doc_embeds = self.model.encode(batch["docs"])

    # Compute scores and labels
    scores, labels = self.full_contrastive_scores_and_labels(
        query=query_embeds,
        key=doc_embeds,
        use_all_pairs=True
    )
    
    # Scale scores by temperature
    scores = scores / self.args.temperature
    
    # Compute loss
    loss = F.cross_entropy(scores, labels)
```
Dataloader yields 1 positive and 15 mined hard negatives per example. The `full_constrastive_scores_and_labels` function call generates extra "negative" examples from neighbouring queries in the batch. Since dataset is quite big, and queries are random, a document that is positive for one query ("How old is Trump?") will most likely be negative to another query ("Best running shoes").   
The benefit of in-batch negative mining is two-fold. First, we want to provide extra negative examples for our training to better distinguish between different documents. The process of embedding training can be visualized as placing embedding vectors on the other side of the query projection on a hypersphere (check out this [cool paper](https://arxiv.org/pdf/2012.09740) about it). The more examples you have in a batch, the more you can uniformly "scatter them out" accross embedding space.   
Second benefit is that, as it turns out, these in-batch-negatives are almost free from the memory/compute consumption standpoint. For the forward pass, you only need to add a bunch of dot product computations over the final embeddings of the documents - this is a minor overhead compared to computing of embedding themselves. Same for backward pass: we propagate gradients from loss to document scores to these embeddings; memory/compute footprint for backwards pass through the model will be the same.   
Due to these 2 reasons, it turns out that the more in-batch negatives you have, the better. That is why it is probably beneficial to also share the negatives between different per-device microbatches, i.e., exchange them accross devices!   
For that, we will use some `torch.distributed` communication functions:
```python
def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)

    all_tensors[dist.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors
```
This is a convenience function that basically wraps `torch.distributed.all_gather` - a function to exchange different versions of the same tensor across devices.   

And we rewrite our training step like this:
```python
def common_step(self, batch):
    # Get query embeddings for all queries in one forward pass
    query_embeds = self.model.encode(batch["query"])
    
    # Get all doc embeddings in one forward pass
    doc_embeds = self.model.encode(batch["docs"])
    
    # Gather embeddings from all devices
    # shape before: micro_batch_size, emb_dim
    all_query_embeds = dist_gather_tensor(query_embeds)
    # shape after: micro_batch_size * world_size, emb_dim

    # shape before: micro_batch_size * n_passages, emb_dim
    all_doc_embeds = dist_gather_tensor(doc_embeds)
    # shape after: micro_batch_size * n_passages * world_size, emb_dim

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
```
### What about the gradients?
When we write PyTorch code, we don't even think about what will happen during backprop - it is so seamless. In this case, this had me a bit puzzled. Yes, we exchanged the values of a tensor across devices and used it to compute final scores and loss. But how will that work during backwards pass? Will the gradients through that operation be correctly computed?   
[Here](https://pytorch.org/docs/stable/rpc/distributed_autograd.html) is the explanation from official documentation about it. Long story short: the gradients will be correctly computed, we don't have to worry about that. Under the hood, operations like `dist.all_gather` keep track of network communications the same way as all the other computation, and use this information to correctly propagate gradients during backwards pass across different devices
## Learning Rate scheduling
It was shown that learning rate scheduling, usually with gradual reduction of learning rate to zero over the course of training, is beneficial to stochastic gradient descent convergence (see, for example, [this](https://arxiv.org/pdf/1803.02865) paper). The process of warmup with further decay of LR is similar to metal annealing in blacksmithing: first we heat the material up to give the atoms enough energy to escape their local minima. Then we cool it down to let them settle in the optimal places in the crystallic lattice.   
Here is how we can implement that in PyTorch Lightning:
```python
class MyLightningModule(L.LightningModule):
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
```
We don't have direct access to the training loop, but we can provide lr scheduling mechanism by overriding `configure_optimizers` method.  
Depending on our arguments, we turn "lr warmup" on or off. The `LambdaLR` mechanism allows writing an arbitrary function for learning rate based on current step.   

To see/debug problems with lr scheduling, it would also be a good idea to log current `lr` for each step: 
```python
    def training_step(self, batch, batch_idx):
        # ...
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr)
        return loss
```
## Gradient clipping
Gradient clipping is more of a safety technique. It is unlikely to improve accuracy in a small training example like ours, but it will save you from "exploding gradients" that might come up every once in a while if you are training, say, LLMs for weeks at a time. Better safe than sorry!
```python
# good idea to monitor gradients as well
class MyLightningModule(L.LightningModule):
    # ...
    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.parameters() if p.grad is not None]))
        self.log("grad_norm", grad_norm)
#
parser.add_argument('--max_grad_norm', type=float, default=None)
# 
trainer = L.Trainer(
    # ...
    gradient_clip_val=args.max_grad_norm,
)
```
We provide Trainer with `gradient_clip_val` argument, which will cause it to clip gradients using [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_). What this does is multiply all the gradients by the same coefficient if overall norm is above 1. 