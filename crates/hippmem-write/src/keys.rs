//! AssociationKeys generation: produce multi-dimensional recall keys from
//! understanding + context (03 §1).
//!
//! Deterministic: same input -> same output.

use hippmem_core::ids::{CausalKey, EntityKey, EventKey, GoalKey, TemporalKey, TopicKey};
use hippmem_core::model::links::{AssociationKeys, LexicalSignature, SemanticSignature};
use hippmem_core::model::understanding::MemoryUnderstanding;
use hippmem_core::model::unit::{MemoryContent, WriteContext};
use hippmem_core::time::Timestamp;
use xxhash_rust::xxh3::xxh3_64;

/// Generate the full AssociationKeys (deterministic).
pub fn generate_keys(
    content: &MemoryContent,
    understanding: &MemoryUnderstanding,
    context: &WriteContext,
    semantic: &SemanticSignature,
) -> Result<AssociationKeys, String> {
    Ok(AssociationKeys {
        entity_keys: entity_keys_from(understanding),
        temporal_keys: temporal_keys_from(context.local_time),
        lexical_signature: simhash_text(&content.raw),
        semantic_signature: semantic.clone(),
        topic_keys: topic_keys_from(understanding),
        emotion_keys: vec![],
        goal_keys: goal_keys_from(understanding),
        event_keys: event_keys_from(understanding),
        causal_keys: causal_keys_from(understanding),
    })
}

fn entity_keys_from(u: &MemoryUnderstanding) -> Vec<EntityKey> {
    u.entities
        .iter()
        .map(|e| xxh3_64(e.canonical.as_bytes()))
        .collect()
}

fn temporal_keys_from(ts: Timestamp) -> Vec<TemporalKey> {
    let ms = ts.0;
    vec![
        (ms / 3_600_000) as u32,   // hour bucket
        (ms / 86_400_000) as u32,  // day bucket
        (ms / 604_800_000) as u32, // week bucket
    ]
}

fn simhash_text(text: &str) -> LexicalSignature {
    let mut tokens = hippmem_core::hash::tokenize(text, "zh");
    tokens.extend(hippmem_core::hash::tokenize(text, "en"));
    let bits = 256usize;
    let mut acc = vec![0i64; bits];

    for token in &tokens {
        let h = xxh3_64(token.as_bytes());
        for (i, acc_val) in acc.iter_mut().enumerate().take(bits) {
            let seed = h.wrapping_add(i as u64) ^ (h >> 32);
            if (seed & 1) == 1 {
                *acc_val += 1;
            } else {
                *acc_val -= 1;
            }
        }
    }

    let mut simhash = [0u64; 4];
    for (chunk, word) in acc.chunks(64).zip(simhash.iter_mut()) {
        let mut w = 0u64;
        for (j, val) in chunk.iter().enumerate() {
            if *val > 0 {
                w |= 1u64 << j;
            }
        }
        *word = w;
    }
    LexicalSignature { simhash }
}

fn topic_keys_from(u: &MemoryUnderstanding) -> Vec<TopicKey> {
    u.topics
        .iter()
        .map(|t| xxh3_64(t.label.as_bytes()))
        .collect()
}

fn goal_keys_from(u: &MemoryUnderstanding) -> Vec<GoalKey> {
    u.goals
        .iter()
        .map(|g| xxh3_64(g.description.as_bytes()))
        .collect()
}

fn event_keys_from(u: &MemoryUnderstanding) -> Vec<EventKey> {
    u.events
        .iter()
        .map(|e| xxh3_64(e.action.as_bytes()))
        .collect()
}

fn causal_keys_from(u: &MemoryUnderstanding) -> Vec<CausalKey> {
    u.causal_claims
        .iter()
        .map(|c| {
            let mut input = c.cause.as_bytes().to_vec();
            input.extend_from_slice(b" -> ");
            input.extend_from_slice(c.effect.as_bytes());
            xxh3_64(&input)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simhash_deterministic() {
        let s1 = simhash_text("Rust programming language");
        let s2 = simhash_text("Rust programming language");
        assert_eq!(s1.simhash, s2.simhash);
    }
}
