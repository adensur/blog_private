#!/usr/bin/env python3
"""
Generate test Parquet data for benchmarking different Rust Parquet crates.

This script generates synthetic data with various column types to test
different aspects of Parquet reading/writing performance.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
from faker import Faker


def is_s3_path(path: str) -> bool:
    """Check if path is an S3 URI."""
    return path.startswith("s3://")


def generate_text(fake: Faker, num_rows: int, num_words: int) -> List[str]:
    """Generate text strings with approximately the specified number of words."""
    return [" ".join(fake.words(num_words)) for _ in range(num_rows)]


def generate_features(num_rows: int, num_features: int) -> List[List[float]]:
    """Generate arrays of random float64 features."""
    return np.random.randn(num_rows, num_features).tolist()


def generate_parquet_data(
    num_rows: int,
    output_path: str,
    row_group_size: int | None = None,
    seed: int = 42,
) -> None:
    """
    Generate synthetic Parquet data with various column types.

    Args:
        num_rows: Number of rows to generate
        output_path: Path where the Parquet file will be written (local path or s3:// URI)
        row_group_size: Number of rows per row group (default: all rows in one group)
        seed: Random seed for reproducibility
    """
    print(f"Generating {num_rows:,} rows of test data...")

    # Initialize random generators
    np.random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    # Generate data for each column
    print("  - Generating text column (~500 words per row)...")
    text_data = generate_text(fake, num_rows, num_words=500)

    print("  - Generating query column (~10 words per row)...")
    query_data = generate_text(fake, num_rows, num_words=10)

    print("  - Generating features column (50 f64 values per row)...")
    features_data = generate_features(num_rows, num_features=50)

    print("  - Generating user_id column (uint64)...")
    user_id_data = np.random.randint(0, 2**32, size=num_rows, dtype=np.uint64)

    print("  - Generating email column...")
    email_data = [fake.email() for _ in range(num_rows)]

    # Create Polars DataFrame
    print("Creating DataFrame...")
    df = pl.DataFrame({
        "text": text_data,
        "query": query_data,
        "features": features_data,
        "user_id": user_id_data,
        "email": email_data,
    })

    # Display DataFrame info
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Memory usage: {df.estimated_size('mb'):.2f} MB")
    print(f"\nFirst few rows:")
    print(df.head(2))

    # Write to Parquet
    print(f"\nWriting to {output_path}...")

    # Prepare write options
    write_kwargs = {
        "compression": "snappy",
        "statistics": True,
    }

    if row_group_size is not None:
        write_kwargs["row_group_size"] = row_group_size
        num_row_groups = (num_rows + row_group_size - 1) // row_group_size
        print(f"  - Row group size: {row_group_size:,} rows")
        print(f"  - Number of row groups: {num_row_groups}")
    else:
        print(f"  - Using default row group size (single row group)")

    df.write_parquet(output_path, **write_kwargs)

    # Display file size (only for local paths)
    if is_s3_path(output_path):
        print(f"\n✓ Successfully written to {output_path}")
    else:
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\n✓ Successfully written {file_size_mb:.2f} MB to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test Parquet data for benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n", "--num-rows",
        type=int,
        required=True,
        help="Number of rows to generate",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output path for the Parquet file (local path or s3:// URI)",
    )

    parser.add_argument(
        "-r", "--row-group-size",
        type=int,
        default=None,
        help="Number of rows per row group (default: single row group)",
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist (only for local paths)
    if not is_s3_path(args.output):
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Generate data
    generate_parquet_data(
        num_rows=args.num_rows,
        output_path=args.output,
        row_group_size=args.row_group_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
