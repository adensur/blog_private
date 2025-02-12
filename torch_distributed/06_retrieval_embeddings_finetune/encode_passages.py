import argparse
import os
import tqdm
import torch
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple, Mapping, Dict

from lib import move_to_cuda
from model import MyModel

from contextlib import nullcontext
from torch.utils.data import DataLoader
from functools import partial
from datasets import load_dataset
from typing import Dict, List
from transformers.file_utils import PaddingStrategy
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    HfArgumentParser,
    BatchEncoding
)



def _psg_transform_func(tokenizer: PreTrainedTokenizerFast, args,
                        examples: Dict[str, List]) -> BatchEncoding:
    batch_dict = tokenizer(examples['title'],
                           text_pair=examples['contents'],
                           max_length=args.p_max_len,
                           padding=PaddingStrategy.DO_NOT_PAD,
                           truncation=True)
    # for co-Condenser reproduction purpose only
    if args.model_name_or_path.startswith('Luyu/'):
        del batch_dict['token_type_ids']

    return batch_dict


@torch.no_grad()
def _worker_encode_passages(gpu_idx: int, args):
    def _get_out_path(shard_idx: int = 0) -> str:
        path = '{}/shard_{}_{}'.format(args.encode_save_dir, gpu_idx, shard_idx)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    if os.path.exists(_get_out_path(0)):
        print('{} already exists, will skip encoding'.format(_get_out_path(0)))
        return

    dataset = load_dataset('json', data_files=os.path.join(args.data_dir, 'passages.jsonl.gz'))['train']
    if args.dry_run:
        dataset = dataset.select(range(4096))
    dataset = dataset.shard(num_shards=torch.cuda.device_count(),
                            index=gpu_idx,
                            contiguous=True)

    print('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = MyModel(args.model_name_or_path)
    model.eval()
    model.cuda()

    dataset.set_transform(partial(_psg_transform_func, tokenizer, args))

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    data_loader = DataLoader(
        dataset,
        batch_size=args.encode_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True)

    num_encoded_docs, encoded_embeds, cur_shard_idx = 0, [], 0
    for batch_dict in tqdm.tqdm(data_loader, desc='passage encoding', mininterval=8):
        batch_dict = move_to_cuda(batch_dict)

        with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
            embeds = model.encode(batch_dict)
            
        encoded_embeds.append(embeds.cpu())
        num_encoded_docs += embeds.shape[0]

        if num_encoded_docs >= args.encode_shard_size:
            out_path = _get_out_path(cur_shard_idx)
            concat_embeds = torch.cat(encoded_embeds, dim=0)
            print('GPU {} save {} embeds to {}'.format(gpu_idx, concat_embeds.shape[0], out_path))
            torch.save(concat_embeds, out_path)

            cur_shard_idx += 1
            num_encoded_docs = 0
            encoded_embeds.clear()

    if num_encoded_docs > 0:
        out_path = _get_out_path(cur_shard_idx)
        concat_embeds = torch.cat(encoded_embeds, dim=0)
        print('GPU {} save {} embeds to {}'.format(gpu_idx, concat_embeds.shape[0], out_path))
        torch.save(concat_embeds, out_path)

    print('Done computing score for worker {}'.format(gpu_idx))


def _batch_encode_passages(args: argparse.Namespace):
    print('Args={}'.format(str(args)))
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print('No gpu available')
        return

    os.makedirs(args.encode_save_dir, exist_ok=True)

    print('Use {} gpus'.format(gpu_count))
    if not args.dry_run:
        torch.multiprocessing.spawn(_worker_encode_passages, args=(args,), nprocs=gpu_count)
    else:
        _worker_encode_passages(0, args)
    print('Done batch encode passages')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data-dir', type=str, default="msmarco_bm25_official")
    args.add_argument('--encode_save_dir', type=str, required=True)
    args.add_argument('--encode_batch_size', type=int, default=128)
    args.add_argument('--encode_shard_size', type=int, default=1000000)
    args.add_argument('--p_max_len', type=int, default=512)
    args.add_argument('--model_name_or_path', type=str, default='Luyu/co-Condenser-marco')
    args.add_argument('--fp16', action='store_true')
    args.add_argument('--dataloader_num_workers', type=int, default=16)
    args.add_argument('--dry_run', action='store_true')
    args = args.parse_args()
    _batch_encode_passages(args)
