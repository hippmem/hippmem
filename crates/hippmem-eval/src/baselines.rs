//! Baseline comparison systems: five baselines (06 §3).

/// Baseline system identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Baseline {
    Bm25Only,
    EmbeddingOnly,
    HybridBm25Embedding,
    RagSummaryMemory,
    HippmemFull,
}

impl Baseline {
    pub fn name(&self) -> &str {
        match self {
            Baseline::Bm25Only => "BM25 Only",
            Baseline::EmbeddingOnly => "Embedding Only",
            Baseline::HybridBm25Embedding => "Hybrid BM25+Embedding",
            Baseline::RagSummaryMemory => "RAG Summary Memory",
            Baseline::HippmemFull => "HIPPMEM Full",
        }
    }

    pub fn all() -> Vec<Baseline> {
        vec![
            Baseline::Bm25Only,
            Baseline::EmbeddingOnly,
            Baseline::HybridBm25Embedding,
            Baseline::RagSummaryMemory,
            Baseline::HippmemFull,
        ]
    }
}
