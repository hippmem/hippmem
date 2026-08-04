//! acceptance test: API backend mock HTTP.
//!
//! Uses a local mock server to validate the API client parsing logic.

use hippmem_model::api::openai::OpenAiEmbedder;
use hippmem_model::traits::Embedder;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

/// Helper: start a mock HTTP server that returns a fixed JSON body.
/// Serves a single request, then the thread shuts down.
fn mock_server(response_body: &'static str, status_line: &'static str) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        if let Some(stream) = listener.incoming().flatten().next() {
            let mut stream = stream;
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status_line,
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    port
}

/// Restore `OPENAI_API_KEY` when dropped, so a test that clears the env var
/// cannot leak the mutation to parallel tests in the same binary.
struct EnvVarGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvVarGuard {
    fn remove(key: &'static str) -> Self {
        let prev = std::env::var(key).ok();
        std::env::remove_var(key);
        Self { key, prev }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

/// A real HTTP round-trip against the mock server is parsed into vectors.
#[test]
fn openai_mock_embedding_parses_correctly() {
    let body = r#"{
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1}
        ],
        "model": "text-embedding-3-small",
        "usage": {"total_tokens": 5}
    }"#;
    let port = mock_server(body, "HTTP/1.1 200 OK");
    let embedder = OpenAiEmbedder::new_with_base_url(
        "sk-test-key".into(),
        &format!("http://127.0.0.1:{port}/v1"),
        "text-embedding-3-small",
        1536,
    )
    .expect("explicit key, no env lookup");
    let vectors = embedder
        .embed_sync(&["hello".into(), "world".into()])
        .expect("mock response should parse");
    assert_eq!(vectors, vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]);
}

/// The OpenAiEmbedder type exists and compiles.
#[test]
fn openai_embedder_type_exists() {
    let e = OpenAiEmbedder::new("sk-test-key".into());
    assert_eq!(e.dim(), 1536);
    assert_eq!(e.backend_id(), "text-embedding-3-small");
}

/// An empty API key does not panic.
#[test]
fn empty_api_key_does_not_panic() {
    let _guard = EnvVarGuard::remove("OPENAI_API_KEY");
    let result = OpenAiEmbedder::new_with_base_url(
        String::new(),
        "https://api.openai.com/v1",
        "text-embedding-3-small",
        1536,
    );
    assert!(result.is_err(), "an empty key should return Err, not Ok");
}
