use rand::Rng;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// Client for simulating an embedding API server with concurrency limits
pub struct EmbeddingClient {
    /// Latency for each embedding request (in microseconds precision)
    latency: Duration,
    /// Semaphore to limit concurrent requests (simulates server capacity)
    semaphore: Arc<Semaphore>,
}

impl EmbeddingClient {
    /// Create a new EmbeddingClient
    ///
    /// # Arguments
    /// * `latency_ms` - Expected latency in milliseconds (can be fractional, e.g., 0.01 for 10 microseconds)
    /// * `max_concurrency` - Maximum number of concurrent requests the server can handle
    pub fn new(latency_ms: f64, max_concurrency: usize) -> Self {
        let latency = Duration::from_micros((latency_ms * 1000.0) as u64);
        Self {
            latency,
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
        }
    }

    /// Generate embeddings for a batch of texts
    ///
    /// This simulates sending a request to an embedding API server.
    /// The method will block if the server is at capacity (all concurrent slots are taken).
    ///
    /// # Arguments
    /// * `texts` - Vector of strings to generate embeddings for
    ///
    /// # Returns
    /// A vector of embedding vectors, where each embedding is 1024 f32 values
    pub async fn generate_embeddings(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        // Acquire semaphore permit - this simulates waiting for server capacity
        let _permit = self.semaphore.acquire().await.unwrap();

        // Simulate API latency
        tokio::time::sleep(self.latency).await;

        // Generate random embeddings (1024 dimensions) for each text
        let num_texts = texts.len();
        let mut embeddings = Vec::with_capacity(num_texts);

        let mut rng = rand::thread_rng();
        for _ in 0..num_texts {
            let embedding: Vec<f32> = (0..1024).map(|_| rng.r#gen::<f32>()).collect();
            embeddings.push(embedding);
        }

        embeddings
        // Permit is automatically released when _permit goes out of scope
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding_client() {
        let client = EmbeddingClient::new(10.0, 2);
        let texts = vec!["Hello world".to_string(), "Test text".to_string()];

        let embeddings = client.generate_embeddings(texts.clone()).await;

        assert_eq!(embeddings.len(), texts.len());
        for embedding in embeddings {
            assert_eq!(embedding.len(), 1024);
        }
    }

    #[tokio::test]
    async fn test_concurrency_limit() {
        use std::time::Instant;

        // Create client with concurrency limit of 1
        let client = Arc::new(EmbeddingClient::new(50.0, 1));

        let start = Instant::now();

        // Launch 3 concurrent requests
        let mut handles = vec![];
        for i in 0..3 {
            let client = client.clone();
            let handle = tokio::spawn(async move {
                let texts = vec![format!("Text {}", i)];
                client.generate_embeddings(texts).await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let elapsed = start.elapsed();

        // With concurrency=1 and latency=50ms, 3 requests should take ~150ms
        // (they run sequentially)
        assert!(
            elapsed.as_millis() >= 140,
            "Expected sequential execution, got {:?}",
            elapsed
        );
    }
}
