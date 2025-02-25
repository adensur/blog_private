from typing import List
import argparse
import os
import json
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
import multiprocessing as mp
from functools import partial

def _slice_with_mod(elements: List, offset: int, cnt: int) -> List:
    return [elements[(offset + idx) % len(elements)] for idx in range(cnt)]


def convert_example(example: dict, corpus: Dataset) -> dict:
    result = {}
    result["query"] = example["query"]
    pos_docids = example["positives"]["doc_id"]
    neg_docids = example["negatives"]["doc_id"]
    # fix out of bonds error for some docids
    pos_docids = [docid for docid in pos_docids if int(docid) < len(corpus)]
    neg_docids = [docid for docid in neg_docids if int(docid) < len(corpus)]
    pos_contents = [corpus[int(docid)]["contents"] for docid in pos_docids]
    pos_titles = [corpus[int(docid)]["title"] for docid in pos_docids]
    result["pos_docs"] = list(zip(pos_titles, pos_contents))
    neg_contents = [corpus[int(docid)]["contents"] for docid in neg_docids]
    neg_titles = [corpus[int(docid)]["title"] for docid in neg_docids]
    result["neg_docs"] = list(zip(neg_titles, neg_contents))
    return result

def process_chunk(chunk: list, corpus: Dataset, output_file: str):
    with open(output_file, "a") as f:
        for i, example in enumerate(chunk):
            try:
                converted_example = convert_example(example, corpus)
                f.write(json.dumps(converted_example) + "\n")
            except Exception as e:
                print(f"Error processing example: {e}")
                print(f"Example {i} of {len(chunk)}")
                print(example)
                raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_jsonl = os.path.join(args.data_dir, "train.jsonl")
    corpus_path = os.path.join(args.data_dir, 'passages.jsonl.gz')
    corpus: Dataset = load_dataset('json', data_files=corpus_path)
    data_files = {}
    data_files["train"] = args.data_dir + "/train.jsonl"
    data_files["validation"] = args.data_dir + "/dev.jsonl"
    raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)

    for split in ["train", "validation"]:
        output_file = os.path.join(args.output_dir, f"{split}.jsonl")
        # Clear the output file if it exists
        open(output_file, "w").close()
        
        # Split data into chunks for parallel processing
        examples = list(raw_datasets[split])
        chunk_size = max(1, len(examples) // args.num_workers)
        chunks = [examples[i:i + chunk_size] for i in range(0, len(examples), chunk_size)]
        
        # Create process pool and process chunks in parallel
        with mp.Pool(args.num_workers) as pool:
            process_fn = partial(process_chunk, corpus=corpus["train"], output_file=output_file)
            list(tqdm(
                pool.imap(process_fn, chunks),
                total=len(chunks),
                desc=f"Converting {split} examples"
            ))

if __name__ == "__main__":
    main()
