use std::{fs::File, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use arrow_array::RecordBatch;
use clap::Parser;
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, arrow_writer::ArrowWriter},
    file::reader::{FileReader, SerializedFileReader},
    record::Row,
};

#[derive(Parser, Debug)]
#[command(
    name = "parquet_official_bench",
    about = "Benchmark reading Parquet rows with the official crate"
)]
struct Args {
    /// Input parquet file
    #[arg(long)]
    input: PathBuf,

    /// Output parquet file (written after a second pass)
    #[arg(long)]
    output: PathBuf,

    /// Record batch size used while copying to output
    #[arg(long, default_value_t = 8192)]
    batch_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input_size = std::fs::metadata(&args.input)
        .with_context(|| format!("reading metadata for {}", args.input.display()))?
        .len();

    // Row-wise pass for latency/throughput metrics.
    let row_reader = SerializedFileReader::new(
        File::open(&args.input).with_context(|| format!("opening input {}", args.input.display()))?,
    )?;
    let rows = row_reader.get_row_iter(None)?;

    let row_start = Instant::now();
    let mut first_row_elapsed: Option<f64> = None;
    let mut total_rows: usize = 0;

    for row in rows {
        let _row: Row = row?;
        total_rows += 1;
        if first_row_elapsed.is_none() {
            first_row_elapsed = Some(row_start.elapsed().as_secs_f64());
        }
    }

    let row_elapsed = row_start.elapsed();
    let row_seconds = row_elapsed.as_secs_f64().max(f64::EPSILON);
    let rows_per_sec = total_rows as f64 / row_seconds;
    let mb_per_sec = (input_size as f64 / 1_048_576.0) / row_seconds;

    println!("Row-wise read stats");
    println!("  Input:  {} ({} bytes)", args.input.display(), input_size);
    println!(
        "  Time to first row: {:.3}s",
        first_row_elapsed.unwrap_or(row_seconds)
    );
    println!("  Total rows: {}", total_rows);
    println!("  Total time: {:.3}s", row_seconds);
    println!("  Rows/sec: {:.2}", rows_per_sec);
    println!("  MB/sec: {:.2}", mb_per_sec);

    // Second pass to copy to output using the batch-oriented Arrow writer.
    let copy_start = Instant::now();
    let reader = ParquetRecordBatchReaderBuilder::try_new(
        File::open(&args.input)
            .with_context(|| format!("opening input {}", args.input.display()))?,
    )?
    .with_batch_size(args.batch_size)
    .build()?;

    let mut writer: Option<ArrowWriter<File>> = None;
    let mut first_batch_elapsed: Option<f64> = None;

    for batch in reader {
        let batch: RecordBatch = batch?;
        if writer.is_none() {
            let out_file = File::create(&args.output)
                .with_context(|| format!("creating output {}", args.output.display()))?;
            writer = Some(ArrowWriter::try_new(out_file, batch.schema(), None)?);
            first_batch_elapsed = Some(copy_start.elapsed().as_secs_f64());
        }

        if let Some(w) = writer.as_mut() {
            w.write(&batch)?;
        }
    }

    let writer = writer.context("no record batches were read from the input")?;
    writer.close()?;

    println!(
        "Copy to output (batch writer): {}",
        args.output.display()
    );
    if let Some(t) = first_batch_elapsed {
        println!("  Time to first batch in copy pass: {:.3}s", t);
    }
    println!(
        "  Copy pass total time: {:.3}s",
        copy_start.elapsed().as_secs_f64()
    );

    Ok(())
}
