use anyhow::{Context, Result};
use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use clap::Parser;
use futures::StreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::buffered::BufWriter;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::arrow::AsyncArrowWriter;
use std::sync::Arc;
use std::time::Instant;
use url::Url;

#[derive(Parser, Debug)]
#[command(name = "parquet_columnar_s3")]
#[command(about = "Read Parquet from S3, generate embeddings, and write results to S3")]
struct Args {
    /// Input S3 URI (s3://bucket/path/to/file.parquet)
    #[arg(short, long)]
    input: String,

    /// Output S3 URI (s3://bucket/path/to/output.parquet)
    #[arg(short, long)]
    output: String,

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
    reading_time: std::time::Duration,
    embedding_time: std::time::Duration,
    writing_time: std::time::Duration,
    total_rows: usize,
    total_time: std::time::Duration,
    file_size_bytes: u64,
}

impl Metrics {
    fn report(&self) {
        println!("\n=== Benchmark Results ===");
        println!(
            "Time to first byte: {:.3}s",
            self.time_to_first_byte.as_secs_f64()
        );
        println!("Total rows processed: {}", self.total_rows);
        println!();
        println!("=== Stage Breakdown ===");
        println!("Reading from S3:        {:.3}s ({:.1}%)",
            self.reading_time.as_secs_f64(),
            (self.reading_time.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0
        );
        println!("Computing embeddings:   {:.3}s ({:.1}%)",
            self.embedding_time.as_secs_f64(),
            (self.embedding_time.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0
        );
        println!("Writing to S3:          {:.3}s ({:.1}%)",
            self.writing_time.as_secs_f64(),
            (self.writing_time.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0
        );
        println!();
        println!("Total time:             {:.3}s", self.total_time.as_secs_f64());
        println!(
            "Throughput:             {:.2} rows/sec",
            self.total_rows as f64 / self.total_time.as_secs_f64()
        );
        println!(
            "Throughput:             {:.2} MB/sec",
            (self.file_size_bytes as f64 / 1_048_576.0) / self.total_time.as_secs_f64()
        );
    }
}

/// Parse S3 URI into bucket and key
fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let url = Url::parse(uri).context("Failed to parse S3 URI")?;

    if url.scheme() != "s3" {
        anyhow::bail!("URI must start with s3://");
    }

    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("No bucket specified in S3 URI"))?
        .to_string();

    let key = url.path().trim_start_matches('/').to_string();

    if key.is_empty() {
        anyhow::bail!("No key specified in S3 URI");
    }

    Ok((bucket, key))
}

/// Create an S3 object store
fn create_s3_store(bucket: &str) -> Result<Arc<dyn ObjectStore>> {
    let s3 = AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .build()
        .context("Failed to create S3 client")?;

    Ok(Arc::new(s3))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let start_time = Instant::now();

    // Parse input S3 URI
    let (input_bucket, input_key) = parse_s3_uri(&args.input)?;
    println!("Reading from s3://{}/{}", input_bucket, input_key);

    // Create S3 store for input
    let input_store = create_s3_store(&input_bucket)?;
    let input_path = ObjectPath::from(input_key.as_str());

    // Get file metadata to track size
    let input_meta = input_store
        .head(&input_path)
        .await
        .context("Failed to get input file metadata")?;
    let file_size_bytes = input_meta.size as u64;

    println!("Input file size: {:.2} MB", file_size_bytes as f64 / 1_048_576.0);

    // Create streaming Parquet reader from S3
    println!("Creating Parquet stream from S3...");
    let parquet_reader = ParquetObjectReader::new(input_store.clone(), input_path.clone());
    let builder = ParquetRecordBatchStreamBuilder::new(parquet_reader)
        .await
        .context("Failed to create Parquet stream builder")?;

    let schema = builder.schema().clone();
    let mut stream = builder.build().context("Failed to build Parquet stream")?;

    let mut time_to_first_byte = None;
    let mut all_texts = Vec::new();
    let mut all_batches = Vec::new();
    let mut total_rows = 0;

    // Read all batches from the stream
    println!("Reading Parquet file from S3 (streaming)...");
    while let Some(batch_result) = stream.next().await {
        if time_to_first_byte.is_none() {
            time_to_first_byte = Some(start_time.elapsed());
            println!(
                "First byte received after: {:.3}s",
                time_to_first_byte.unwrap().as_secs_f64()
            );
        }

        let batch = batch_result.context("Failed to read batch")?;

        total_rows += batch.num_rows();

        // Extract text column
        let text_column = batch
            .column_by_name(&args.text_column)
            .with_context(|| format!("Column '{}' not found", args.text_column))?;

        // Handle both String and LargeString types
        let texts: Vec<String> =
            if let Some(string_array) = text_column.as_any().downcast_ref::<arrow::array::StringArray>() {
                let len: usize = string_array.len();
                (0..len)
                    .map(|i| {
                        if string_array.is_valid(i) {
                            string_array.value(i).to_string()
                        } else {
                            String::new()
                        }
                    })
                    .collect()
            } else if let Some(large_string_array) =
                text_column.as_any().downcast_ref::<arrow::array::LargeStringArray>()
            {
                let len: usize = large_string_array.len();
                (0..len)
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

    println!(
        "Read {} rows in {:.3}s",
        total_rows,
        start_time.elapsed().as_secs_f64()
    );

    let reading_end_time = Instant::now();

    // Create embedding client
    let client = util::EmbeddingClient::new(args.latency_ms, args.server_concurrency);
    println!(
        "Embedding server: latency={:.3}ms, concurrency={}",
        args.latency_ms, args.server_concurrency
    );

    // Process in batches to generate embeddings (concurrently)
    let num_batches = (all_texts.len() + args.batch_size - 1) / args.batch_size;
    println!(
        "Generating embeddings in {} batches of {} (processing concurrently)...",
        num_batches, args.batch_size
    );

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

    let embedding_end_time = Instant::now();

    // Write output to S3
    println!("Writing output Parquet file to S3...");
    let (output_bucket, output_key) = parse_s3_uri(&args.output)?;
    println!("Writing to s3://{}/{}", output_bucket, output_key);

    write_parquet_to_s3(
        &output_bucket,
        &output_key,
        &schema,
        &all_batches,
        all_embeddings,
    )
    .await?;

    let writing_end_time = Instant::now();
    let total_time = start_time.elapsed();

    // Report metrics
    let metrics = Metrics {
        time_to_first_byte: time_to_first_byte.unwrap(),
        reading_time: reading_end_time - start_time,
        embedding_time: embedding_end_time - reading_end_time,
        writing_time: writing_end_time - embedding_end_time,
        total_rows,
        total_time,
        file_size_bytes,
    };
    metrics.report();

    Ok(())
}

async fn write_parquet_to_s3(
    bucket: &str,
    key: &str,
    original_schema: &Schema,
    batches: &[RecordBatch],
    embeddings: Vec<Vec<f32>>,
) -> Result<()> {
    // Create new schema with embedding field
    let mut fields: Vec<FieldRef> = original_schema.fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(
        "embedding",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 1024),
        false,
    )));
    let new_schema = Arc::new(Schema::new(fields));

    // Create S3 store for output
    let output_store = create_s3_store(bucket)?;
    let output_path = ObjectPath::from(key);

    // Use BufWriter for efficient multipart uploads
    let mut buf_writer = BufWriter::new(output_store.clone(), output_path.clone());

    // Create Arrow writer
    let mut arrow_writer = AsyncArrowWriter::try_new(&mut buf_writer, new_schema.clone(), None)
        .context("Failed to create Arrow writer")?;

    // Write batches with embeddings
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
        let embedding_array =
            FixedSizeListArray::try_new(field, 1024, values, None)
                .context("Failed to create FixedSizeListArray")?;

        columns.push(Arc::new(embedding_array));

        // Create new batch
        let new_batch = RecordBatch::try_new(new_schema.clone(), columns)
            .context("Failed to create record batch with embeddings")?;

        arrow_writer
            .write(&new_batch)
            .await
            .context("Failed to write batch")?;
    }

    // Close the Arrow writer (this flushes and shuts down the underlying writer)
    arrow_writer.close().await.context("Failed to close Arrow writer")?;

    println!("✓ Successfully written to s3://{}/{}", bucket, key);

    Ok(())
}
