import polars as pl

data = {
    "id": list(range(5)),
}

df = pl.DataFrame(data)
df.write_parquet("data.parquet")