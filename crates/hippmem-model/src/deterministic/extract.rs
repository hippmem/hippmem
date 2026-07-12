//! Deterministic extractor: rule-based structured understanding (08 §4.2).
//!
//! Pure rules: dictionary + pattern matching + connective detection.
//! Deterministic, no network, no randomness.

use std::collections::BTreeSet;

use crate::error::ModelResult;
use crate::lang::active_locales;
use crate::traits::{Extractor, ImmediateExtraction, StrongExtraction};
use hippmem_core::hash::JIEBA;
use hippmem_core::model::understanding::{
    CausalClaim, CausalKind, DecisionFrame, EmotionFrame, EntityMention, EntityType, GoalFrame,
    GoalStatus, Polarity, PreferenceFrame, TopicTag,
};
use hippmem_core::model::unit::{ContentType, MemoryContent};
use hippmem_core::score::UnitScore;

/// Deterministic extractor: rule-based structured extraction.
#[derive(Default)]
pub struct DeterministicExtractor;

impl DeterministicExtractor {
    pub fn extract_sync_immediate(
        &self,
        content: &MemoryContent,
    ) -> ModelResult<ImmediateExtraction> {
        Ok(self.immediate_impl(content))
    }

    pub fn extract_sync_strong(&self, content: &MemoryContent) -> ModelResult<StrongExtraction> {
        Ok(self.strong_impl(content))
    }

    fn immediate_impl(&self, content: &MemoryContent) -> ImmediateExtraction {
        let text = &content.raw;
        ImmediateExtraction {
            entities: extract_entities(text),
            topics: extract_topics(text),
            explicit_causals: extract_explicit_causals(text),
            language: content.language,
            content_type: Some(content.content_type),
            importance: heuristic_importance(text, &content.content_type),
        }
    }

    fn strong_impl(&self, content: &MemoryContent) -> StrongExtraction {
        let text = &content.raw;
        StrongExtraction {
            goals: extract_goals(text),
            preferences: extract_preferences(text),
            emotions: extract_emotions(text),
            decisions: extract_decisions(text),
            implicit_causals: vec![],
            contradictions: vec![],
            confidence: UnitScore::new(0.35),
        }
    }
}

#[async_trait::async_trait]
impl Extractor for DeterministicExtractor {
    async fn extract_immediate(&self, c: &MemoryContent) -> ModelResult<ImmediateExtraction> {
        Ok(self.immediate_impl(c))
    }

    async fn extract_strong(&self, c: &MemoryContent) -> ModelResult<StrongExtraction> {
        Ok(self.strong_impl(c))
    }

    fn backend_id(&self) -> &str {
        "deterministic-rules"
    }
}

// ── Entities: ASCII capitalized proper names + jieba POS-tagged Chinese proper names ──

fn extract_entities(text: &str) -> Vec<EntityMention> {
    let mut out = Vec::new();
    let mut seen_canonical: BTreeSet<String> = BTreeSet::new();

    // Path 1: ASCII uppercase-led proper/library names (confidence=0.6)
    for word in text.split_whitespace() {
        let t = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
        if t.len() >= 2 && t.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            let canonical = t.to_lowercase();
            if seen_canonical.insert(canonical.clone()) {
                out.push(EntityMention {
                    text: t.to_string(),
                    canonical,
                    entity_type: EntityType::Other,
                    span: None,
                    confidence: UnitScore::new(0.6),
                });
            }
        }
    }

    // Path 2: jieba POS-tagged Chinese proper names (confidence=0.55)
    let chinese_entities = extract_chinese_entities(text);
    for e in chinese_entities {
        if seen_canonical.insert(e.canonical.clone()) {
            out.push(e);
        }
    }

    out
}

/// Extracts Chinese proper names via jieba POS tagging (nr/ns/nt/nz + x OOV words).
///
/// Uses the global JIEBA instance to avoid re-initializing the dictionary on each call.
fn extract_chinese_entities(text: &str) -> Vec<EntityMention> {
    // Only invoke jieba when the text contains CJK characters
    // (avoids pointless tokenization of pure-English text)
    if !text.contains(|c: char| c as u32 >= 0x4E00 && c as u32 <= 0x9FFF) {
        return vec![];
    }

    let tags = JIEBA.tag(text, true); // hmm=true enables new-word discovery

    tags.into_iter()
        .filter_map(|t| {
            let (entity_type, confidence) = match t.tag {
                "nr" => (EntityType::Person, UnitScore::new(0.55)),
                "ns" => (EntityType::Other, UnitScore::new(0.55)), // no Location variant yet
                "nt" => (EntityType::Org, UnitScore::new(0.55)),
                "nz" => (EntityType::Other, UnitScore::new(0.55)),
                "nrt" => (EntityType::Other, UnitScore::new(0.55)),
                "eng" => (EntityType::Other, UnitScore::new(0.50)),
                // x = non-morpheme / OOV word: multi-char CJK words may be proper names
                // (person names, brands, etc.) not covered by the dictionary
                "x" if is_potential_cjk_name(t.word) => (EntityType::Other, UnitScore::new(0.40)),
                _ => return None,
            };
            Some(EntityMention {
                text: t.word.to_string(),
                canonical: t.word.to_lowercase(),
                entity_type,
                span: None,
                confidence,
            })
        })
        .collect()
}

/// Decides whether a token tagged `x` (OOV) by jieba might be an unrecorded CJK proper name.
///
/// Condition: length >= 2 and composed entirely of CJK unified ideographs
/// (filters out punctuation / spaces / digits / latin letters).
fn is_potential_cjk_name(word: &str) -> bool {
    if word.len() < 2 {
        return false;
    }
    word.chars()
        .all(|c| matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF))
}

// ── Topics: first few non-short tokens ──

fn extract_topics(text: &str) -> Vec<TopicTag> {
    hippmem_core::hash::tokenize(text, "zh")
        .into_iter()
        .filter(|t| t.len() >= 2)
        .take(5)
        .map(|label| TopicTag {
            label,
            confidence: UnitScore::new(0.5),
        })
        .collect()
}

// ── Explicit causals ──

fn extract_explicit_causals(text: &str) -> Vec<CausalClaim> {
    for lang in active_locales() {
        for (cm, em) in lang.causal_pairs {
            if let (Some(cp), Some(ep)) = (text.find(cm), text.find(em)) {
                let cause = text[cp + cm.len()..ep].trim();
                let effect = text[ep + em.len()..].trim();
                if !cause.is_empty() && !effect.is_empty() {
                    return vec![CausalClaim {
                        cause: cause.chars().take(80).collect(),
                        effect: effect.chars().take(80).collect(),
                        kind: CausalKind::Explicit,
                        evidence_span: None,
                        confidence: UnitScore::new(0.7),
                    }];
                }
            }
        }
    }
    vec![]
}

// ── Preferences ──

fn extract_preferences(text: &str) -> Vec<PreferenceFrame> {
    for lang in active_locales() {
        for w in lang.preference_pos {
            if text.contains(w) {
                return vec![PreferenceFrame {
                    object: format!("detected marker: {w}"),
                    polarity: Polarity::Like,
                    strength: UnitScore::new(0.5),
                    still_valid: true,
                    confidence: UnitScore::new(0.5),
                }];
            }
        }
    }
    for lang in active_locales() {
        for w in lang.preference_neg {
            if text.contains(w) {
                return vec![PreferenceFrame {
                    object: format!("detected marker: {w}"),
                    polarity: Polarity::Dislike,
                    strength: UnitScore::new(0.5),
                    still_valid: true,
                    confidence: UnitScore::new(0.5),
                }];
            }
        }
    }
    vec![]
}

// ── Emotions ──

fn extract_emotions(text: &str) -> Vec<EmotionFrame> {
    for lang in active_locales() {
        for (w, ek) in lang.emotion_keywords {
            if text.contains(w) {
                return vec![EmotionFrame {
                    emotion: *ek,
                    intensity: UnitScore::new(0.6),
                    trigger: Some(format!("detected word: {w}")),
                    confidence: UnitScore::new(0.5),
                }];
            }
        }
    }
    vec![]
}

// ── Goals ──

fn extract_goals(text: &str) -> Vec<GoalFrame> {
    for lang in active_locales() {
        for m in lang.goal_markers {
            if text.contains(m) {
                return vec![GoalFrame {
                    description: format!("detected marker: {m}"),
                    status: GoalStatus::Active,
                    constraints: vec![],
                    confidence: UnitScore::new(0.4),
                }];
            }
        }
    }
    vec![]
}

// ── Decisions ──

fn extract_decisions(text: &str) -> Vec<DecisionFrame> {
    for lang in active_locales() {
        for m in lang.decision_markers {
            if text.contains(m) {
                return vec![DecisionFrame {
                    decision: format!("detected marker: {m}"),
                    rationale: Some(text.chars().take(100).collect()),
                    decided_at: None,
                    reverted: false,
                    confidence: UnitScore::new(0.45),
                }];
            }
        }
    }
    vec![]
}

// ── Importance ──

fn heuristic_importance(text: &str, ct: &ContentType) -> UnitScore {
    let base = match text.chars().count() {
        n if n > 200 => 0.7,
        n if n > 50 => 0.5,
        _ => 0.3,
    };
    let bonus = match ct {
        ContentType::Decision | ContentType::Preference => 0.2,
        _ => 0.0,
    };
    UnitScore::new(f32::min(base + bonus, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::model::unit::Language;

    #[test]
    fn detects_causal() {
        let content = MemoryContent {
            raw: "because memory was low, so the program crashed".into(),
            summary: None,
            normalized: None,
            language: Language::En,
            content_type: ContentType::UserStatement,
        };
        let r = DeterministicExtractor.immediate_impl(&content);
        assert!(!r.explicit_causals.is_empty());
    }
}
