//! contract test: validates trait contracts for the deterministic
//! fallback backend (08 §7).
//!
//! All offline, deterministic, and CI-passing.
//!
//! Locale-specific NLP test data is derived from `lang/` module's `LangData`
//! (no separate JSON fixture needed — Type A elimination per test-data-locale-architecture.md).

use hippmem_core::ids::MemoryId;
use hippmem_core::model::unit::{ContentType, Language, MemoryContent};
use hippmem_model::deterministic::embed::DeterministicEmbedder;
use hippmem_model::deterministic::extract::DeterministicExtractor;
use hippmem_model::deterministic::rerank::DeterministicReranker;
use hippmem_model::deterministic::summarize::DeterministicSummarizer;
use hippmem_model::lang::active_locales;
use hippmem_model::traits::{Embedder, Extractor, Reranker, SummarizeInput, Summarizer};

// ═══════════════════════════════════════════════════
// Embedder contract
// ═══════════════════════════════════════════════════

/// embed output length == input length.
#[test]
fn embedder_output_len_matches_input() {
    let e = DeterministicEmbedder::default();
    let texts: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
    let vecs = e.embed_sync(&texts).unwrap();
    assert_eq!(vecs.len(), texts.len());
}

/// Each vector length == dim().
#[test]
fn embedder_vector_len_equals_dim() {
    let e = DeterministicEmbedder::default();
    let vecs = e.embed_sync(&["hello world".into()]).unwrap();
    assert_eq!(vecs[0].len(), e.dim());
}

/// Deterministic: two calls with the same input yield the same output.
#[test]
fn embedder_deterministic() {
    let e = DeterministicEmbedder::default();
    let texts: Vec<String> = vec!["rust programming".into()];
    let v1 = e.embed_sync(&texts).unwrap();
    let v2 = e.embed_sync(&texts).unwrap();
    for (a, b) in v1[0].iter().zip(v2[0].iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

/// L2-normalized (for non-zero input).
#[test]
fn embedder_l2_normalized() {
    let e = DeterministicEmbedder::default();
    let vecs = e
        .embed_sync(&["hello world rust programming".into()])
        .unwrap();
    let norm: f32 = vecs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "L2 norm should be 1.0, got {norm}"
    );
}

// ═══════════════════════════════════════════════════
// Extractor contract
// ═══════════════════════════════════════════════════

/// Text with causal connectives must produce >= 1 explicit causal claim.
/// Test data is constructed from `LangData.causal_pairs` — no fixture needed.
/// All active locales are tested automatically.
#[test]
fn extractor_causal_connectives_produce_claims() {
    let ext = DeterministicExtractor;
    for lang in active_locales() {
        assert!(
            !lang.causal_pairs.is_empty(),
            "[{}] LangData.causal_pairs must not be empty",
            lang.locale
        );
        let language = locale_to_lang(lang.locale);
        for (cause, effect) in lang.causal_pairs {
            let text = format!("{cause} 某事件 {effect} 某结果");
            let content = MemoryContent {
                raw: text.clone(),
                summary: None,
                normalized: None,
                language,
                content_type: ContentType::UserStatement,
            };
            let r = ext.extract_sync_immediate(&content).unwrap();
            assert!(
                !r.explicit_causals.is_empty(),
                "[{}] text containing '{cause}...{effect}' should produce an explicit causal; text='{text}'",
                lang.locale
            );
            for claim in &r.explicit_causals {
                assert!(!claim.cause.is_empty());
                assert!(!claim.effect.is_empty());
                assert!(claim.confidence.value() >= 0.0 && claim.confidence.value() <= 1.0);
            }
        }
    }
}

/// Entity extraction hits known samples.
#[test]
fn extractor_entity_hits_known_samples() {
    let ext = DeterministicExtractor;
    let content = make_content_en("I used Rust and Tantivy today");
    let r = ext.extract_sync_immediate(&content).unwrap();
    let names: Vec<&str> = r.entities.iter().map(|e| e.text.as_str()).collect();
    assert!(names.contains(&"Rust"), "should extract Rust");
    assert!(names.contains(&"Tantivy"), "should extract Tantivy");
}

/// Output structure is valid: confidence within [0, 1].
#[test]
fn extractor_output_confidence_in_range() {
    let ext = DeterministicExtractor;
    let content = make_content_en("The weather is nice today");
    let imm = ext.extract_sync_immediate(&content).unwrap();
    let strong = ext.extract_sync_strong(&content).unwrap();

    for e in &imm.entities {
        let c = e.confidence.value();
        assert!((0.0..=1.0).contains(&c), "entity conf {c} out of range");
    }
    assert!(imm.importance.value() >= 0.0 && imm.importance.value() <= 1.0);

    for g in &strong.goals {
        let c = g.confidence.value();
        assert!((0.0..=1.0).contains(&c), "goal conf {c} out of range");
    }
    for p in &strong.preferences {
        let c = p.confidence.value();
        assert!((0.0..=1.0).contains(&c), "pref conf {c} out of range");
    }
    for e in &strong.emotions {
        let c = e.confidence.value();
        assert!((0.0..=1.0).contains(&c), "emotion conf {c} out of range");
    }
    assert!(strong.confidence.value() >= 0.0 && strong.confidence.value() <= 1.0);
}

/// The extractor does not panic on empty text.
#[test]
fn extractor_empty_text_does_not_panic() {
    let ext = DeterministicExtractor;
    let content = make_content_en("");
    let _ = ext.extract_sync_immediate(&content).unwrap();
    let _ = ext.extract_sync_strong(&content).unwrap();
}

// ═══════════════════════════════════════════════════
// Reranker contract
// ═══════════════════════════════════════════════════

/// Output length == number of candidates.
#[test]
fn reranker_output_len_matches_candidates() {
    let r = DeterministicReranker;
    let cands: Vec<String> = vec!["hello".into(), "world".into(), "rust".into()];
    let scores = r.rerank_sync("query", &cands).unwrap();
    assert_eq!(scores.len(), cands.len());
}

/// Scores are finite (not NaN, not inf).
#[test]
fn reranker_scores_are_finite() {
    let r = DeterministicReranker;
    let cands: Vec<String> = vec!["match".into(), "no match".into()];
    let scores = r.rerank_sync("match query", &cands).unwrap();
    for s in &scores {
        assert!(s.is_finite(), "score {} is not finite", s);
    }
}

/// Empty candidates return empty scores.
#[test]
fn reranker_empty_candidates() {
    let r = DeterministicReranker;
    let scores = r.rerank_sync("query", &[]).unwrap();
    assert!(scores.is_empty());
}

// ═══════════════════════════════════════════════════
// Summarizer contract
// ═══════════════════════════════════════════════════

/// covers == set of input ids.
#[test]
fn summarizer_covers_all_inputs() {
    let s = DeterministicSummarizer;
    let texts = [
        "Rust is very safe.",
        "Rust is very fast.",
        "Rust is very modern.",
    ];
    let sources: Vec<SummarizeInput> = texts
        .iter()
        .enumerate()
        .map(|(i, v)| SummarizeInput {
            id: MemoryId(i as u128 + 1),
            text: v.to_string(),
        })
        .collect();
    let out = s.summarize_sync(&sources).unwrap();
    let cover_ids: Vec<u128> = out.covers.iter().map(|id| id.as_u128()).collect();
    for src in &sources {
        assert!(
            cover_ids.contains(&src.id.as_u128()),
            "should cover id={}",
            src.id.as_u128()
        );
    }
}

/// Summary is non-empty.
#[test]
fn summarizer_output_non_empty() {
    let s = DeterministicSummarizer;
    let sources = vec![SummarizeInput {
        id: MemoryId(1),
        text: "Rust is a systems programming language. It guarantees memory safety. It performs excellently.".into(),
    }];
    let out = s.summarize_sync(&sources).unwrap();
    assert!(!out.summary.is_empty(), "summary should not be empty");
}

/// A single-sentence input still produces a summary.
#[test]
fn summarizer_single_sentence() {
    let s = DeterministicSummarizer;
    let sources = vec![SummarizeInput {
        id: MemoryId(42),
        text: "This is a short test sentence for summarization.".into(),
    }];
    let out = s.summarize_sync(&sources).unwrap();
    assert!(!out.summary.is_empty());
    assert_eq!(out.covers.len(), 1);
}

// ═══════════════════════════════════════════════════
// Provenance contract
// ═══════════════════════════════════════════════════

/// Each backend's backend_id() is stable and non-empty.
#[test]
fn backend_ids_stable_and_non_empty() {
    assert_eq!(
        DeterministicEmbedder::default().backend_id(),
        "deterministic-hash"
    );
    assert_eq!(DeterministicExtractor.backend_id(), "deterministic-rules");
    assert_eq!(
        DeterministicReranker.backend_id(),
        "deterministic-bm25-overlap"
    );
    assert_eq!(
        DeterministicSummarizer.backend_id(),
        "deterministic-extractive"
    );
}

// ═══════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════

fn locale_to_lang(locale: &str) -> Language {
    match locale {
        "zh" => Language::Zh,
        _ => Language::En,
    }
}

fn make_content_en(text: &str) -> MemoryContent {
    MemoryContent {
        raw: text.to_string(),
        summary: None,
        normalized: None,
        language: Language::En,
        content_type: ContentType::UserStatement,
    }
}
