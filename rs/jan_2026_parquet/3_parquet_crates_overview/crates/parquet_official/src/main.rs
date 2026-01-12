use std::{fs::File, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use arrow_array::RecordBatch;
use clap::Parser;
use parquet::arrow::{
    arrow_reader::ParquetRecordBatchReaderBuilder, arrow_writer::ArrowWriter,
};

#[derive(Parser, Debug)]
#[command(name = "parquet_official_bench", about = "Benchmark reading Parquet with the official crate")]
struct Args {
    /// Input parquet file
    #[arg(long)]
    input: PathBuf,

    /// Output parquet file (written from the read batches)
    #[arg(long)]
    output: PathBuf,

    /// Record batch size used while reading
    #[arg(long, default_value_t = 8192)]
    batch_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input_size = std::fs::metadata(&args.input)
        .with_context(|| format!("reading metadata for {}", args.input.display()))?
        .len();

    let start = Instant::now();
    let file = File::open(&args.input)
        .with_context(|| format!("opening input {}", args.input.display()))?;

    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(args.batch_size)
        .build()?;

    let mut writer: Option<ArrowWriter<File>> = None;
    let mut total_rows: usize = 0;
    let mut first_batch_elapsed: Option<f64> = None;

    for batch in reader {
        let batch: RecordBatch = batch?;
        if writer.is_none() {
            // Initialize writer using the schema from the first batch.
            let out_file = File::create(&args.output)
                .with_context(|| format!("creating output {}", args.output.display()))?;
            writer = Some(ArrowWriter::try_new(out_file, batch.schema(), None)?);
            first_batch_elapsed = Some(start.elapsed().as_secs_f64());
        }

        let batch_rows = batch.num_rows();
        total_rows += batch_rows;

        if let Some(w) = writer.as_mut() {
            w.write(&batch)?;
        }
    }

    let writer = writer.context("no record batches were read from the input")?;
    writer.close()?;

    let total_elapsed = start.elapsed();
    let seconds = total_elapsed.as_secs_f64().max(f64::EPSILON);
    let rows_per_sec = total_rows as f64 / seconds;
    let mb_per_sec = (input_size as f64 / 1_048_576.0) / seconds;

    println!("Input:  {} ({} bytes)", args.input.display(), input_size);
    println!("Output: {}", args.output.display());
    println!(
        "Time to first batch: {:.3}s",
        first_batch_elapsed.unwrap_or(seconds)
    );
    println!("Total rows: {}", total_rows);
    println!("Total time: {:.3}s", seconds);
    println!("Rows/sec: {:.2}", rows_per_sec);
    println!("MB/sec: {:.2}", mb_per_sec);

    Ok(())
}
