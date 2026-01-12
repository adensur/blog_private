"""Generate synthetic parquet data for benchmarking parquet crates."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

# A small vocabulary to build pseudo-natural text.
WORD_BANK: Sequence[str] = [
    "rust",
    "python",
    "storage",
    "analytics",
    "columnar",
    "vector",
    "index",
    "parquet",
    "polars",
    "arrow",
    "spark",
    "delta",
    "lake",
    "iceberg",
    "schema",
    "field",
    "partition",
    "compression",
    "zstd",
    "snappy",
    "lz4",
    "dataset",
    "batch",
    "row",
    "group",
    "merge",
    "shuffle",
    "sort",
    "query",
    "filter",
    "projection",
    "scan",
    "async",
    "stream",
    "blocking",
    "engine",
    "tokio",
    "thread",
    "buffer",
    "io",
    "read",
    "write",
    "throughput",
    "latency",
    "benchmark",
    "fixture",
    "random",
    "sample",
    "distribution",
    "gaussian",
    "semantic",
    "embedding",
    "distance",
    "cosine",
    "dot",
    "ann",
    "hnsw",
    "knn",
    "payload",
    "document",
    "token",
    "chunk",
    "vectorize",
    "normalize",
]


def generate_text_column(rows: int, words_per_row: int, rng: random.Random) -> list[str]:
    return [" ".join(rng.choices(WORD_BANK, k=words_per_row)) for _ in range(rows)]


def generate_embeddings(rows: int, dim: int, seed: int | None) -> list[list[float]]:
    np_rng = np.random.default_rng(seed)
    embeddings = np_rng.standard_normal(size=(rows, dim), dtype=np.float32)
    return embeddings.tolist()


def build_dataframe(
    rows: int, text_words: int, query_words: int, embedding_dim: int, seed: int | None
) -> pl.DataFrame:
    py_rng = random.Random(seed)
    text_col = generate_text_column(rows, text_words, py_rng)
    query_col = generate_text_column(rows, query_words, py_rng)
    embedding_col = generate_embeddings(rows, embedding_dim, seed)

    return pl.DataFrame(
        {
            "text": text_col,
            "query": query_col,
            "embedding": embedding_col,
        }
    )


def resolve_row_group_size(rows: int, row_group_size: int | None, row_groups: int | None) -> int | None:
    if row_group_size is not None and row_groups is not None:
        raise ValueError("Use either --row-group-size or --row-groups, not both.")
    if row_groups is not None and row_groups > 0:
        return math.ceil(rows / row_groups)
    return row_group_size


def write_parquet(
    df: pl.DataFrame, output: str, compression: str, row_group_size: int | None
) -> None:
    # Polars can write directly to S3 URIs; only create parent dirs for local paths.
    if output.startswith("s3://"):
        target = output
    else:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path

    df.write_parquet(
        target,
        compression=None if compression.lower() == "none" else compression,
        row_group_size=row_group_size,
        statistics=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic parquet data using polars.")
    parser.add_argument("--rows", type=int, required=True, help="Number of rows to generate.")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path or S3 URI.")
    parser.add_argument(
        "--row-group-size",
        type=int,
        help="Number of rows per row group. Mutually exclusive with --row-groups.",
    )
    parser.add_argument(
        "--row-groups",
        type=int,
        help="Number of row groups to target. Mutually exclusive with --row-group-size.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Compression codec (zstd, lz4, snappy, uncompressed/none). Default: zstd.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--text-words", type=int, default=500, help="Words per row in the text column.")
    parser.add_argument("--query-words", type=int, default=10, help="Words per row in the query column.")
    parser.add_argument(
        "--embedding-dim", type=int, default=1024, help="Length of the embedding vector per row."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_group_size = resolve_row_group_size(args.rows, args.row_group_size, args.row_groups)

    print(
        f"Generating {args.rows} rows "
        f"(text={args.text_words} words, query={args.query_words} words, embedding_dim={args.embedding_dim})"
    )
    df = build_dataframe(args.rows, args.text_words, args.query_words, args.embedding_dim, args.seed)

    print(
        f"Writing parquet to {args.output} "
        f"(compression={args.compression}, row_group_size={row_group_size or 'auto'})"
    )
    write_parquet(df, args.output, args.compression, row_group_size)
    print("Done.")


if __name__ == "__main__":
    main()
