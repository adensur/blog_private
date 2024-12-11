This blog series is about distributed inference and training in PyTorch. It will go over basic concepts of distributed training, such as data parallel and model parallel, multi-gpu and multi-node training. It will then explore main distributed primitives of PyTorch, and main libraries that build on top of them for distributed training - [Huggingface Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) and [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html). I also wanted to go over some often-overlooked concepts like distributed optimizeres, automatic-mixed precision and gradient checkpointing.Finally, we will explore some advanced topics for enterprise-level distributed training and model parallelism such as [Megatron](https://github.com/NVIDIA/Megatron-LM) and [NeMo](https://github.com/NVIDIA/NeMo). As a final "project" for the series, I will build a distributed, production-ready training pipeline for information retrieval model [bge-m3](https://huggingface.co/BAAI/bge-m3) using Pytorch Lightning and Slurm.  

Outline:  
- Parallelism methods. Model and Data parallelism
- torch.nn.DataParallel
- torch.dist module. Primitives of distributed training
- Pytorch Lighting

The series assumes that you are familiar with PyTorch and deep learning fundamentals. Most of the examples will be regarding NLP, so knowledge of modern transformers and text processing models is helpful.