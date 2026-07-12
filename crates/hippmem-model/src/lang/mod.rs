//! Multilingual NLP data for HIPPMEM's deterministic pipeline.
//!
//! Each supported language has a single file (zh.rs, en.rs, ...) exporting
//! a `LangData` constant containing ALL locale-specific patterns for that language.
//!
//! Adding a new language requires only:
//! 1. Create `lang/<locale>.rs` with the language's `LangData`
//! 2. Add the constant to `active_locales()` below
//!
//! No algorithm files need to be changed.

pub mod en;
pub mod zh;

use hippmem_core::model::understanding::EmotionKind;

/// All locale-specific NLP patterns for one language.
///
/// Fields are grouped by the algorithm that consumes them:
/// - Question type detection (retrieve_api.rs)
/// - Explanatory scoring (retrieve_api.rs)
/// - Content extraction (enrich.rs, extract.rs)
/// - Query pre-processing (retrieve_api.rs)
pub struct LangData {
    /// Locale identifier ("zh", "en", "ja", "ko", ...)
    pub locale: &'static str,

    // ── Question type detection ──
    /// Correction / change-of-mind signals
    pub q_correction: &'static [&'static str],
    /// Preference / liking signals
    pub q_preference: &'static [&'static str],
    /// Why / reason signals
    pub q_why: &'static [&'static str],
    /// How / method signals
    pub q_how: &'static [&'static str],
    /// What / fact signals
    pub q_what: &'static [&'static str],

    // ── Explanatory / causal scoring ──
    /// (keyword, boost_weight) pairs for explanatory-pattern detection.
    /// Used to boost documents that contain causal/reason language.
    pub explanatory: &'static [(&'static str, f32)],

    // ── Content marker extraction ──
    pub goal_markers: &'static [&'static str],
    pub event_markers: &'static [&'static str],
    pub decision_markers: &'static [&'static str],
    pub preference_pos: &'static [&'static str],
    pub preference_neg: &'static [&'static str],
    pub causal_pairs: &'static [(&'static str, &'static str)],
    pub emotion_keywords: &'static [(&'static str, EmotionKind)],

    // ── Query pre-processing ──
    /// Words filtered out during keyword extraction
    pub stop_words: &'static [&'static str],
    /// Delimiters that signal a what-is question (e.g., "是什么", "what is")
    pub what_delimiters: &'static [&'static str],
    /// Patterns for detecting definitional sentences (e.g., "{} 是", "{} is")
    pub definition_patterns: &'static [&'static str],
    /// Character used to split possessive constructions (e.g., '的' in Chinese)
    pub possessive_particle: Option<char>,
    /// Pair of keywords whose co-occurrence signals a change/correction
    /// (e.g., ("之前", "后来") in Chinese — "before ... later")
    pub change_pair: Option<(&'static str, &'static str)>,
}

/// Returns all active locales in priority order.
///
/// The first locale's patterns are tried first (higher priority for CJK queries).
pub fn active_locales() -> &'static [LangData] {
    &[zh::ZH, en::EN]
}
