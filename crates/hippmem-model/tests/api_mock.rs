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

/// A real HTTP round-trip: the OpenAI-compatible extractor parses a
/// chat-completions response into structured understanding.
#[test]
fn openai_extract_mock_roundtrip_parses() {
    use hippmem_model::api::openai_extract::OpenAiExtractor;
    use hippmem_model::traits::Extractor;

    let body = r#"{
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "{\"entities\": [{\"text\": \"小明\", \"canonical\": \"小明\", \"entity_type\": \"person\"}], \"topics\": [{\"label\": \"住址\", \"confidence\": 0.8}], \"explicit_causals\": [], \"importance\": 0.7, \"language\": \"zh\", \"content_type\": \"user_statement\"}"
            }
        }],
        "usage": {"total_tokens": 42}
    }"#;
    let port = mock_server(body, "HTTP/1.1 200 OK");
    let extractor = OpenAiExtractor::new_with_base_url(
        "sk-test-key".into(),
        &format!("http://127.0.0.1:{port}/v1"),
        "test-model",
    )
    .expect("explicit key, no env lookup");

    let out = extractor
        .extract_immediate_sync(&hippmem_core::model::unit::MemoryContent {
            raw: "小明住在北京海淀区。".into(),
            summary: None,
            normalized: None,
            language: hippmem_core::model::unit::Language::Zh,
            content_type: hippmem_core::model::enums::ContentType::UserStatement,
        })
        .expect("mock response should parse");

    assert_eq!(out.entities.len(), 1);
    assert_eq!(out.entities[0].text, "小明");
    assert_eq!(
        out.entities[0].entity_type,
        hippmem_core::model::understanding::EntityType::Person
    );
    assert_eq!(out.topics[0].label, "住址");
    assert!((out.importance.value() - 0.7).abs() < 0.01);
    assert_eq!(out.language, hippmem_core::model::unit::Language::Zh);
    assert_eq!(
        out.content_type,
        Some(hippmem_core::model::enums::ContentType::UserStatement)
    );
}

/// The OpenAI-compatible extractor reports its backend id.
#[test]
fn openai_extract_backend_id() {
    use hippmem_model::api::openai_extract::OpenAiExtractor;
    use hippmem_model::traits::Extractor;

    let e = OpenAiExtractor::new("sk-test-key".into());
    assert_eq!(e.backend_id(), "openai-compatible");
}
