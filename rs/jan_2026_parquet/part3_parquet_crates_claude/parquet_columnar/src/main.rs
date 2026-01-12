use anyhow::{Context, Result};
use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "parquet_columnar")]
#[command(about = "Read Parquet file, generate embeddings, and write results")]
struct Args {
    /// Input Parquet file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output Parquet file path
    #[arg(short, long)]
    output: PathBuf,

    /// Name of the text column to process
    #[arg(short, long)]
    text_column: String,

    /// Batch size for processing embeddings
    #[arg(short, long, default_value = "32")]
    batch_size: usize,

    /// Server concurrency limit (max concurrent requests)
    #[arg(short = 'c', long, default_value = "4")]
    server_concurrency: usize,

    /// Expected latency per request in milliseconds (can be fractional, e.g., 0.01 for 10 microseconds)
    #[arg(short = 'l', long, default_value = "50.0")]
    latency_ms: f64,
}

struct Metrics {
    time_to_first_byte: std::time::Duration,
    total_rows: usize,
    total_time: std::time::Duration,
    file_size_bytes: u64,
}

impl Metrics {
    fn report(&self) {
        println!("\n=== Benchmark Results ===");
        println!("Time to first byte: {:.3}s", self.time_to_first_byte.as_secs_f64());
        println!("Total rows processed: {}", self.total_rows);
        println!("Total time: {:.3}s", self.total_time.as_secs_f64());
        println!(
            "Throughput: {:.2} rows/sec",
            self.total_rows as f64 / self.total_time.as_secs_f64()
        );
        println!(
            "Throughput: {:.2} MB/sec",
            (self.file_size_bytes as f64 / 1_048_576.0) / self.total_time.as_secs_f64()
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let start_time = Instant::now();

    // Open input file and get metadata
    let input_file = File::open(&args.input)
        .with_context(|| format!("Failed to open input file: {:?}", args.input))?;

    let file_size_bytes = input_file.metadata()?.len();

    // Create Parquet reader
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file)
        .context("Failed to create Parquet reader")?;

    let schema = builder.schema().clone();
    let reader = builder.build().context("Failed to build Parquet reader")?;

    let mut time_to_first_byte = None;
    let mut all_texts = Vec::new();
    let mut all_batches = Vec::new();
    let mut total_rows = 0;

    // Read all batches and extract text column
    println!("Reading Parquet file...");
    for batch_result in reader {
        if time_to_first_byte.is_none() {
            time_to_first_byte = Some(start_time.elapsed());
            println!("First byte received after: {:.3}s", time_to_first_byte.unwrap().as_secs_f64());
        }

        let batch = batch_result.context("Failed to read record batch")?;
        total_rows += batch.num_rows();

        // Extract text column
        let text_column = batch
            .column_by_name(&args.text_column)
            .with_context(|| format!("Column '{}' not found", args.text_column))?;

        // Handle both String and LargeString types
        let texts: Vec<String> = if let Some(string_array) = text_column.as_any().downcast_ref::<arrow::array::StringArray>() {
            (0..string_array.len())
                .map(|i| {
                    if string_array.is_valid(i) {
                        string_array.value(i).to_string()
                    } else {
                        String::new()
                    }
                })
                .collect()
        } else if let Some(large_string_array) = text_column.as_any().downcast_ref::<arrow::array::LargeStringArray>() {
            (0..large_string_array.len())
                .map(|i| {
                    if large_string_array.is_valid(i) {
                        large_string_array.value(i).to_string()
                    } else {
                        String::new()
                    }
                })
                .collect()
        } else {
            anyhow::bail!("Column '{}' is not a string type", args.text_column);
        };

        all_texts.extend(texts);

        all_batches.push(batch);
    }

    println!("Read {} rows in {:.3}s", total_rows, start_time.elapsed().as_secs_f64());

    // Create embedding client
    let client = util::EmbeddingClient::new(args.latency_ms, args.server_concurrency);
    println!(
        "Embedding server: latency={:.3}ms, concurrency={}",
        args.latency_ms, args.server_concurrency
    );

    // Process in batches to generate embeddings (concurrently)
    let num_batches = (all_texts.len() + args.batch_size - 1) / args.batch_size;
    println!("Generating embeddings in {} batches of {} (processing concurrently)...", num_batches, args.batch_size);

    let client = Arc::new(client);
    let mut tasks = vec![];

    for (i, chunk) in all_texts.chunks(args.batch_size).enumerate() {
        let client = client.clone();
        let texts = chunk.to_vec();
        let task = tokio::spawn(async move {
            let embeddings = client.generate_embeddings(texts).await;
            (i, embeddings)
        });
        tasks.push(task);
    }

    // Wait for all tasks and collect results in order
    let mut results: Vec<(usize, Vec<Vec<f32>>)> = vec![];
    for task in tasks {
        results.push(task.await.unwrap());
    }
    results.sort_by_key(|(idx, _)| *idx);

    let mut all_embeddings = Vec::new();
    for (_, embeddings) in results {
        all_embeddings.extend(embeddings);
    }

    println!("Generated {} embeddings", all_embeddings.len());

    // Write output with embeddings
    println!("Writing output Parquet file...");
    write_parquet_with_embeddings(
        &args.output,
        &schema,
        &all_batches,
        all_embeddings,
    )?;

    let total_time = start_time.elapsed();

    // Report metrics
    let metrics = Metrics {
        time_to_first_byte: time_to_first_byte.unwrap(),
        total_rows,
        total_time,
        file_size_bytes,
    };
    metrics.report();

    Ok(())
}

fn write_parquet_with_embeddings(
    output_path: &PathBuf,
    original_schema: &Schema,
    batches: &[RecordBatch],
    embeddings: Vec<Vec<f32>>,
) -> Result<()> {
    // Create new schema with embedding field
    let mut fields: Vec<FieldRef> = original_schema.fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(
        "embedding",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            1024,
        ),
        false,
    )));
    let new_schema = Arc::new(Schema::new(fields));

    // Create output file
    let output_file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {:?}", output_path))?;

    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(output_file, new_schema.clone(), Some(props))
        .context("Failed to create Arrow writer")?;

    // Reconstruct batches with embeddings
    let mut embedding_offset = 0;
    for batch in batches {
        let num_rows = batch.num_rows();

        // Collect original columns
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();

        // Create embedding array for this batch
        let batch_embeddings = &embeddings[embedding_offset..embedding_offset + num_rows];
        embedding_offset += num_rows;

        // Flatten embeddings into a single array
        let mut flat_values = Vec::with_capacity(num_rows * 1024);
        for emb in batch_embeddings {
            flat_values.extend_from_slice(emb);
        }

        let values = Arc::new(Float32Array::from(flat_values)) as ArrayRef;
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let embedding_array = FixedSizeListArray::try_new(field, 1024, values, None)
            .context("Failed to create FixedSizeListArray")?;

        columns.push(Arc::new(embedding_array));

        // Create new batch
        let new_batch = RecordBatch::try_new(new_schema.clone(), columns)
            .context("Failed to create record batch with embeddings")?;

        writer.write(&new_batch).context("Failed to write batch")?;
    }

    writer.close().context("Failed to close writer")?;

    Ok(())
}
