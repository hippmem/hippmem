//! English (en) locale data for HIPPMEM's deterministic NLP pipeline.

use super::{EmotionKind, LangData};

pub const EN: LangData = LangData {
    locale: "en",

    // ── Question type detection ──
    q_correction: &[
        "corrected",
        "changed my mind",
        "no longer",
        "reconsidered",
        "updated",
        "replaced",
        "instead",
        "switched to",
        "used to",
    ],
    q_preference: &["prefer", "favorite", "like better", "rather", "enjoy"],
    q_why: &["why", "what for", "how come"],
    q_how: &["how", "in what way"],
    q_what: &["what", "which"],

    // ── Explanatory / causal scoring ──
    explanatory: &[
        ("because", 0.12),
        ("decided to", 0.10),
        ("chose", 0.10),
        ("reason", 0.15),
        ("due to", 0.15),
        ("therefore", 0.10),
        ("as a result", 0.10),
        ("led to", 0.12),
        ("caused by", 0.12),
        ("root cause", 0.15),
        ("in order to", 0.10),
    ],

    // ── Content marker extraction ──
    goal_markers: &["goal", "plan to", "aim to", "intend to"],
    event_markers: &["deploy", "release", "launch", "meet", "discuss"],
    decision_markers: &["decide", "choose", "select", "drop", "adopt"],
    preference_pos: &["prefer", "like", "enjoy", "favor"],
    preference_neg: &["avoid", "dislike", "hate"],
    causal_pairs: &[
        ("because", "so"),
        ("since", "therefore"),
        ("due to", "thus"),
    ],
    emotion_keywords: &[
        ("happy", EmotionKind::Joy),
        ("excited", EmotionKind::Joy),
        ("satisfied", EmotionKind::Joy),
        ("frustrated", EmotionKind::Frustration),
        ("disappointed", EmotionKind::Frustration),
        ("anxious", EmotionKind::Anxiety),
        ("angry", EmotionKind::Anger),
        ("afraid", EmotionKind::Fear),
    ],

    // ── Query pre-processing ──
    stop_words: &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can",
        "could", "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours", "he",
        "she", "it", "they", "them", "this", "that", "these", "those", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "about", "what", "which", "who", "whom", "how", "when",
        "where", "why",
    ],
    what_delimiters: &["what is", "what are", "what does", "define"],
    definition_patterns: &["is", "are", "refers to", "means", "defined as"],
    possessive_particle: None,
    change_pair: None,
};
