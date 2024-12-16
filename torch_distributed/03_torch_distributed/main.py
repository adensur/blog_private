"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist

def run(rank, size):
    tensor = torch.zeros(1)
    torch.distributed.breakpoint(0)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        print("My rank: ", dist.get_rank())
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    if "MASTER_ADDR" not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if "MASTER_PORT" not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    print("MASTER_ADDR: ", os.environ['MASTER_ADDR'])
    print("MASTER_PORT: ", os.environ['MASTER_PORT'])
    print("Starting init process group. Current rank: ", rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("Finished init process group. Current rank: ", rank)
    fn(rank, size)


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    current_rank = int(os.environ["RANK"])
    init_process(current_rank, world_size, run)
