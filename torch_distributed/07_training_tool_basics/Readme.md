We have a lot of functionality to add to our training, but before we do that, I want to setup basic hygyene tools - checkpoints, validation dataset and wandb.
## Validation dataset
In our current lightning module, we've defined `training_step` that performs inference and computes loss. We can similarly define `validation_step`, refactoring away common code:
```python
def common_step(self, batch):
    # compute scores and loss

def training_step(self, batch, batch_idx):
    loss = self.common_step(batch)
    self.log("train_loss", loss)
    return loss

def validation_step(self, batch, batch_idx):
    loss = self.common_step(batch)
    self.log("val_loss", loss)
    return loss
```
Lightning calls `model.eval()` automatically under the hood, stopping some model intrinsics like accumulation of BatchNorm buffer.
## Wandb logging
This is the easiest part. We simply init the logger (in Lightning format) and pass it to trainer. Make sure you do it with Lightning and not just with `wandb.init()`, because Lightining creates many processes under the hood. If setup is incorrect, each process will spawn its own wandb run - not what we want!   
```python
from lightning.pytorch.loggers import WandbLogger
# ...
wandb_logger = WandbLogger(project="your_project_name")
#...
trainer = L.Trainer(
        #...
        logger=wandb_logger
    )
```
Make sure you also export WANDB_API_KEY in env vars.   

Next launch creates a wandb run and prints a link to it in stdout. It plots some metrics, such as train/val loss, that we added ourselves, as well as some general metrics:  
- epochs
- global step - number of batches processed
- process gpu power usage %. great to benchmark. If usage is much lower than 100%, some improvements can be made!
- process gpu memory usage. Allows determining batch size. For training purposes, usually the bigger batch size, the better. GPU memory is a primary bottleneck, rule of thumb is to have gpu memory usage above 50%  
## Checkpoints
When training takes months, it is important to save intermediate checkpoints, so that training can resume exactly from the same place. For example, LLama-3 developers reported that they had a team of engineers keep an eye on the model, and it was restarted more than 150 times over the course of training. In order for it to work, we need to save checkpoints with everything needed to restart training mid-epoch, which includes weights, optimizer state, dataloader state etc.  

This area is rather underexplored. [lightning docs](https://www.restack.io/p/pytorch-lightning-answer-modelcheckpoint-callback-cat-ai) rather vaguely say simply that they save "model weights, optimizer states, etc". So let's go through it step by step.  

### Model weights
Minimal info that needs to be saved in order to use the model is model weights. Those will be enough for inference, if we need to export the model to something like triton/onnx. To get them for a hf model, we can do:   
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("intfloat/simlm-base-msmarco")
for k, v in model.state_dict().items():
    print(k, v.shape)
```
```
embeddings.word_embeddings.weight torch.Size([30522, 768])
embeddings.position_embeddings.weight torch.Size([512, 768])
embeddings.token_type_embeddings.weight torch.Size([2, 768])
embeddings.LayerNorm.weight torch.Size([768])
embeddings.LayerNorm.bias torch.Size([768])
encoder.layer.0.attention.self.query.weight torch.Size([768, 768])
encoder.layer.0.attention.self.query.bias torch.Size([768])
encoder.layer.0.attention.self.key.weight torch.Size([768, 768])
encoder.layer.0.attention.self.key.bias torch.Size([768])
encoder.layer.0.attention.self.value.weight torch.Size([768, 768])
encoder.layer.0.attention.self.value.bias torch.Size([768])
encoder.layer.0.attention.output.dense.weight torch.Size([768, 768])
encoder.layer.0.attention.output.dense.bias torch.Size([768])
encoder.layer.0.attention.output.LayerNorm.weight torch.Size([768])
encoder.layer.0.attention.output.LayerNorm.bias torch.Size([768])
encoder.layer.0.intermediate.dense.weight torch.Size([3072, 768])
encoder.layer.0.intermediate.dense.bias torch.Size([3072])
encoder.layer.0.output.dense.weight torch.Size([768, 3072])
encoder.layer.0.output.dense.bias torch.Size([768])
encoder.layer.0.output.LayerNorm.weight torch.Size([768])
encoder.layer.0.output.LayerNorm.bias torch.Size([768])
...
pooler.dense.weight torch.Size([768, 768])
pooler.dense.bias torch.Size([768])
```
We can use them with bare `torch.save` (this function is a low-level backend of saving anything torch-related to disk, and is used under the hood in all other checkpointing methods).   
So, state dict contains only weights - of embeddings, dense layers, LayerNorms etc. Our model doesn't really have BatchNorm, so we can't check if it is saved or not.   
### Save pretrained
Next thing we can do:   
```python
model.save_pretrained("test_checkpoints/save_pretrained")
```
This is the "official" best way from HF to save a model. Inside the folder, it will save a `config.json` and `model.safetensors`. Safetensors is just a "safer" way to save everything inside a Torch module, avoiding pickling to prevent some security issues. This is the content of `config.json`:  
```json
{
  "_name_or_path": "intfloat/simlm-base-msmarco",
  "architectures": [
    "BertModel"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "torch_dtype": "float32",
  "transformers_version": "4.47.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```
This is just some high-level info about the model, which allows HF to understand its architecture completely. Config plus weights is enough to load the model, even if you don't have access to the original code that was used to create the model.   
### Lightning checkpoints
This is how we can add checkpoint saving to our training:   
```python
from lightning.pytorch.callbacks import ModelCheckpoint
# ...
parser.add_argument('--val_check_interval', type=int, default=5000)
# ...
trainer = L.Trainer(
        # ...
        val_check_interval=args.val_check_interval,
        callbacks=[ModelCheckpoint(
            dirpath=args.output_dir,
            save_top_k=-1,
            every_n_train_steps=args.val_check_interval,
            save_weights_only=False
        )]
    )
```
I use same interval for computing val metrics and saving checkpoints. Logic behind the length of the interval should be simple: we don't want to spend too much time on val evaluation/saving, but we don't want to do that too rarely as well.   
After launching the training, we can see that some checkpoints were saved to `runs/epoch=0-step=500.ckpt`. We can load the checkpoint and check out what's inside:  
```python
cp = torch.load("runs/train_v1/epoch=0-step=500.ckpt")
list(cp.keys())
```
```
['epoch',
 'global_step',
 'pytorch-lightning_version',
 'state_dict',
 'loops',
 'callbacks',
 'optimizer_states',
 'lr_schedulers',
 'hparams_name',
 'hyper_parameters'
]
```
`state_dict` is the same state of the model that we've seen before.   
`optimizer_states` holds everything that optimizer needs to resume training from the same place. For Adam, that would be buffers and current running L1 and L2 momentums.   
`lr_schedulers` holds scheduler state. More complex schedulers might start with lr warmup, than follow some sort of schedule like cosine.   

Note that there is no "dataloader" state here. We use native torch dataloader with `shuffle=True`, which uses random number generator internally with no way to return to a midway point. So we either agree to only save checkpoints between epochs (not going to work for petabyte datasets and month-long epochs), or we need to write our custom sampler with restorable state.   

There are couple of challenges with this. First of all, we need to save sampler state. If we just picked indices during sampling at random, this is not something we can restore from an arbitrary position. Common solution is to precompute sampled indices and save them.  
Pytorch's [Sampler](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py#L198) does something similar. Main problem with this is memory consumption. For 10 billion dataset, assuming 32 bits per integer, that is already 40GB of cpu memory, multiplied by 8 if you spawn 8 processes per node and don't shard the sampler.   

[NeMo](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/data/language_modeling/megatron/data_samplers.py#L200) does something similar, albeit correctly sharding the sampler, so that the same index is only stored once.   

Second problem is how to actually save-restore sampler state during training. Torch Lightning doesn't really provide API for this. The problem is that Lightning preloads a bunch of samplers from the dataloader, and there is no tracking of which of those were already fed into the model, and which weren't.   

What we can do is to store `consumed_samples` - number of total samples that were actually used by the model during training_step. And then learn to deterministically restore sampler state from that single number. Here is my code for RestorableSampler:  
```python
class RestorableSampler(Sampler):
    def __init__(self, data_source, consumed_samples=0, batch_size=32):
        self.data_source = data_source
        self.consumed_samples = consumed_samples
        self.batch_size = batch_size

    def get_data_size(self):
        return len(self.data_source) - (len(self.data_source) % (self.batch_size * dist.get_world_size()))

    def __iter__(self):
        data_size = self.get_data_size()
        # Calculate current epoch and offset within epoch
        epoch = self.consumed_samples // data_size
        offset = self.consumed_samples % data_size
        print(f"RestorableSampler consumed_samples: {self.consumed_samples}, epoch: {epoch}, offset: {offset}")

        # Set random seed based on epoch for consistent shuffling
        rng = np.random.RandomState(epoch)
        indices = list(range(data_size))
        rng.shuffle(indices)

        # Yield remaining indices in current epoch starting from offset
        for idx in indices[offset:]:
            yield idx

        print(f"RestorableSampler epoch {epoch} done. Consumed samples: {self.consumed_samples}")
```
Since `training_step` will be done in parallel on many devices, we have no way of tracking (unless we want explicit communication step on every step) how many samples were consumed, unless we KNOW that every device processed the same number of samples. To ensure that, we will always drop last batch from the sampler. To make sure that state restoration is deterministic, our shuffler is seeded with epoch number.   
This is how we track `consumed_samples`:
```python
def training_step(self, batch, batch_idx):
    loss = self.common_step(batch)
    # assume that last batch is always dropped
    self.consumed_samples += len(batch["query"]["input_ids"]) * dist.get_world_size()
    if self.sampler is not None:
        self.sampler.consumed_samples = self.consumed_samples
    return loss
```
We had to connect our `module` and `sampler` to make sure that states are properly updated.   

Finally, the code to restore from checkpoint and correctly connect everything:   
```python
parser.add_argument('--resume_from_checkpoint', type=str, default=None,
    help='Path to checkpoint to resume training from')
# ...
if args.resume_from_checkpoint:
    print("Restoring from checkpoint: ", args.resume_from_checkpoint)
    module = MyLightningModule.load_from_checkpoint(args.resume_from_checkpoint, args=args)
    print("Restored consumed samples: ", module.consumed_samples)
else:
    module = MyLightningModule(args)
# ...
train_loader = DataLoader(
        # ...
        sampler=RestorableSampler(train_dataset, module.consumed_samples),
    )
```
As you might've noticed, this thing is quite brittle and relies on the fact that:  
- Batch size is not changed between checkpoint save and load
- We drop last batch
- World size does not change
- Torch Lightning doesn't introduce something unexpected in the way they use dataloaders, like pre-loading samples several epochs ahead.   

For our small-scale problem, it might be a better idea to roll back to end-of-epoch checkpointing instead.  
## Outro
Minimal hygiene is done, now we have some introspection into what is happening to the model, and checkpointing so that we don't lose progress. In the next posts, we will do some modifications to our training so that we can actually reach proper benchmark accuracy. 