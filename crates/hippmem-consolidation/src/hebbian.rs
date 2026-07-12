//! Consolidation engine: Hebbian reinforcement + decay + compaction + summary (03 §6-8).

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{AssociationLink, LinkType, ObservationState};
#[allow(unused_imports)]
use hippmem_core::model::unit::{MemoryLifecycle, MemoryStage, MemoryUnit};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use std::collections::HashMap;

// ── ActivationLog ──

/// Activation log entry.
#[derive(Debug, Clone)]
pub struct ActivationEntry {
    pub memory_id: MemoryId,
    pub activated_at: Timestamp,
    pub energy: f32,
}

/// Activation log: records retrieval activation history.
#[derive(Debug, Clone, Default)]
pub struct ActivationLog {
    pub entries: Vec<ActivationEntry>,
}

impl ActivationLog {
    pub fn record(&mut self, id: MemoryId, now: Timestamp, energy: f32) {
        self.entries.push(ActivationEntry {
            memory_id: id,
            activated_at: now,
            energy,
        });
    }

    /// Returns recent co-activation pairs (for Hebbian + CoActivation).
    pub fn co_activation_pairs(&self, window_ms: i64) -> Vec<(MemoryId, MemoryId, u32)> {
        let mut pairs: HashMap<(MemoryId, MemoryId), u32> = HashMap::new();
        let len = self.entries.len();
        for i in 0..len {
            for j in (i + 1)..len {
                let a = &self.entries[i];
                let b = &self.entries[j];
                if (a.activated_at.0 - b.activated_at.0).abs() < window_ms {
                    let key = if a.memory_id < b.memory_id {
                        (a.memory_id, b.memory_id)
                    } else {
                        (b.memory_id, a.memory_id)
                    };
                    *pairs.entry(key).or_insert(0) += 1;
                }
            }
        }
        pairs.into_iter().map(|((a, b), c)| (a, b, c)).collect()
    }
}

// ── Hebbian reinforcement ──

/// Hebbian parameters.
pub struct HebbianParams {
    pub learning_rate: f32,
    pub coactivation_threshold: u32,
    pub strength_max: f32,
}

impl Default for HebbianParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.08,
            coactivation_threshold: 3,
            strength_max: 1.0,
        }
    }
}

/// Hebbian reinforcement: increases the strength of edges that are frequently co-activated.
pub fn hebbian_reinforce(
    links: &mut [AssociationLink],
    co_activations: &[(MemoryId, MemoryId, u32)],
    params: &HebbianParams,
    now: Timestamp,
) {
    for link in links.iter_mut() {
        for (a, b, count) in co_activations {
            if *count >= params.coactivation_threshold
                && ((link.target_id == *a) || (link.target_id == *b))
            {
                let delta = params.learning_rate * (*count as f32).min(5.0);
                let new_strength = (link.strength.value() + delta).min(params.strength_max);
                link.strength = UnitScore::new(new_strength);
                link.last_activated_at = Some(now);
                link.activation_count += 1;
            }
        }
    }
}

/// CoActivation edge creation: when co-activation exceeds the threshold and no edge exists,
/// creates a new CoActivation edge.
pub fn build_coactivation_links(
    pairs: &[(MemoryId, MemoryId, u32)],
    threshold: u32,
    now: Timestamp,
) -> Vec<(MemoryId, AssociationLink)> {
    let mut new_links = Vec::new();
    for (a, b, count) in pairs {
        if *count >= threshold {
            let link = AssociationLink {
                target_id: *b,
                link_type: LinkType::Causal, // CoActivation → possible causal relationship
                direction: hippmem_core::model::links::LinkDirection::Forward,
                strength: UnitScore::new(0.3),
                confidence: UnitScore::new(0.4),
                evidence: hippmem_core::model::links::LinkEvidence {
                    contributing_dimensions: vec![],
                    score_breakdown: vec![],
                    text_spans: vec![],
                    note: Some(format!("co-activated {} times", count)),
                },
                formed_at: now,
                last_activated_at: Some(now),
                activation_count: *count,
                observation: ObservationState::Observing { since: now },
            };
            new_links.push((*a, link));
        }
    }
    new_links
}

// ── Decay ──

/// Decay parameters.
pub struct DecayParams {
    pub decay_per_cycle: f32,
    pub min_retained_strength: f32,
}

impl Default for DecayParams {
    fn default() -> Self {
        Self {
            decay_per_cycle: 0.97,
            min_retained_strength: 0.12,
        }
    }
}

/// Applies decay to edges: the strength of inactive edges gradually decreases.
pub fn apply_decay(links: &mut [AssociationLink], params: &DecayParams, now: Timestamp) {
    for link in links.iter_mut() {
        let inactive_duration = now.0 - link.last_activated_at.unwrap_or(link.formed_at).0;
        if inactive_duration > 86_400_000 {
            // inactive for > 1 day
            let new_strength =
                (link.strength.value() * params.decay_per_cycle).max(params.min_retained_strength);
            link.strength = UnitScore::new(new_strength);
        }
    }
}

/// Prunes weak edges below the threshold + marks long-inactive memories as Deprecated.
pub fn compact_units(units: &mut [MemoryUnit], link_threshold: f32, stale_ms: i64, now: Timestamp) {
    for unit in units.iter_mut() {
        unit.links.retain(|l| l.strength.value() >= link_threshold);
        if unit.links.is_empty()
            && (now.0 - unit.updated_at.0) > stale_ms
            && unit.lifecycle == MemoryLifecycle::Active
        {
            unit.lifecycle = MemoryLifecycle::Deprecated;
        }
    }
}

// ── Summarization ──

/// Summary parameters.
pub struct SummaryParams {
    pub trigger_count: usize,
}

impl Default for SummaryParams {
    fn default() -> Self {
        Self { trigger_count: 12 }
    }
}

// `should_summarize` and the summary-building logic have been unified into `crate::summarize`.
// No duplicate declaration is kept here.

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::model::links::{LinkDirection, LinkEvidence};

    fn make_link(
        target: u128,
        strength: f32,
        last_activated: Option<Timestamp>,
    ) -> AssociationLink {
        AssociationLink {
            target_id: MemoryId(target),
            link_type: LinkType::EntityOverlap,
            direction: LinkDirection::Forward,
            strength: UnitScore::new(strength),
            confidence: UnitScore::new(0.5),
            evidence: LinkEvidence {
                contributing_dimensions: vec![],
                score_breakdown: vec![],
                text_spans: vec![],
                note: None,
            },
            formed_at: Timestamp(0),
            last_activated_at: last_activated,
            activation_count: 0,
            observation: ObservationState::Confirmed,
        }
    }

    #[test]
    fn hebbian_strengthens_coactivated() {
        let mut links = vec![make_link(2, 0.4, None)];
        let pairs = vec![(MemoryId(1), MemoryId(2), 5u32)];
        hebbian_reinforce(
            &mut links,
            &pairs,
            &HebbianParams::default(),
            Timestamp(1000),
        );
        assert!(links[0].strength.value() > 0.4);
    }

    #[test]
    fn hebbian_respects_strength_cap() {
        let mut links = vec![make_link(2, 0.95, None)];
        let pairs = vec![(MemoryId(1), MemoryId(2), 10u32)];
        hebbian_reinforce(
            &mut links,
            &pairs,
            &HebbianParams::default(),
            Timestamp(1000),
        );
        // strength should not exceed strength_max (1.0)
        assert!(links[0].strength.value() <= 1.0);
    }

    #[test]
    fn decay_reduces_stale_links() {
        let mut links = vec![make_link(2, 0.5, Some(Timestamp(0)))];
        apply_decay(&mut links, &DecayParams::default(), Timestamp(200_000_000));
        assert!(links[0].strength.value() < 0.5);
    }

    #[test]
    fn coactivation_builds_links() {
        let pairs = vec![(MemoryId(1), MemoryId(2), 3u32)];
        let new = build_coactivation_links(&pairs, 3, Timestamp(1000));
        assert_eq!(new.len(), 1);
    }

    #[test]
    fn activation_log_records() {
        let mut log = ActivationLog::default();
        log.record(MemoryId(1), Timestamp(0), 0.8);
        log.record(MemoryId(2), Timestamp(100), 0.7);
        assert_eq!(log.entries.len(), 2);
    }

    #[test]
    fn compaction_removes_weak_links() {
        let unit = MemoryUnit {
            schema_version: 1,
            id: MemoryId(1),
            created_at: Timestamp(0),
            updated_at: Timestamp(0),
            content: hippmem_core::model::unit::MemoryContent {
                raw: "test".into(),
                summary: None,
                normalized: None,
                language: hippmem_core::model::unit::Language::Zh,
                content_type: hippmem_core::model::unit::ContentType::UserStatement,
            },
            context: hippmem_core::model::unit::WriteContext {
                conversation_id: None,
                session_id: None,
                project_id: None,
                task_id: None,
                user_id: None,
                local_time: Timestamp(0),
                preceding_memory_ids: vec![],
                source_refs: vec![],
            },
            understanding: hippmem_core::model::understanding::MemoryUnderstanding {
                entities: vec![],
                events: vec![],
                goals: vec![],
                decisions: vec![],
                preferences: vec![],
                emotions: vec![],
                causal_claims: vec![],
                contradictions: vec![],
                topics: vec![],
                importance: UnitScore::new(0.5),
                confidence: UnitScore::new(0.5),
            },
            association_keys: hippmem_core::model::links::AssociationKeys {
                entity_keys: vec![],
                temporal_keys: vec![],
                lexical_signature: hippmem_core::model::links::LexicalSignature { simhash: [0; 4] },
                semantic_signature: hippmem_core::model::links::SemanticSignature {
                    lexical_simhash: [0; 4],
                    dense_embedding_ref: None,
                    binary_code: [0, 0],
                    topic_minhash: [0u32; 16],
                },
                topic_keys: vec![],
                emotion_keys: vec![],
                goal_keys: vec![],
                event_keys: vec![],
                causal_keys: vec![],
            },
            links: vec![make_link(2, 0.1, None), make_link(3, 0.5, None)],
            activation: hippmem_core::model::links::ActivationState {
                last_retrieved_at: None,
                retrieval_count: 0,
                co_activations: vec![],
                usage_score: UnitScore::new(0.5),
            },
            lifecycle: MemoryLifecycle::Active,
            provenance: hippmem_core::model::unit::Provenance {
                origin: hippmem_core::model::unit::SourceKind::Conversation,
                generated_by: hippmem_core::model::unit::GeneratedBy::UserDirect,
                reliability: UnitScore::new(0.5),
                evidence_refs: vec![],
                revision_history: vec![],
            },
            stage: MemoryStage::Indexed,
        };
        compact_units(&mut [unit], 0.2, 1_000_000_000, Timestamp(2_000_000_000));
        // link to id=2 (strength=0.1) removed; id=3 (strength=0.5) retained
    }

    #[test]
    fn extractive_summary_covers_all() {
        // `should_summarize` has been unified into `crate::summarize` 
        assert!(crate::summarize::should_summarize(&[MemoryId(1); 15], 12));
        assert!(!crate::summarize::should_summarize(&[MemoryId(1); 5], 12));
    }
}
