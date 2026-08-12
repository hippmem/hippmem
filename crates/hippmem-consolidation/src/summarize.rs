//! Compaction and merging: low-level memories → summary memory + covers chain (03 §8).

use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{AssociationLink, LinkDirection, LinkType, ObservationState};
use hippmem_core::model::unit::{
    GeneratedBy, MemoryContent, MemoryLifecycle, MemoryStage, MemoryUnit,
};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use hippmem_model::deterministic::summarize::DeterministicSummarizer;
use hippmem_model::traits::SummarizeInput;
use std::collections::HashSet;

/// Token-set Jaccard similarity of two memory texts (deterministic, zh+en tokens).
///
/// 度量"词汇重叠"——即 §8 的"相似低层记忆"语义。不用 simhash：
/// 现有 simhash 签名熵低（近重复文本会落入完全不相关的签名阵营，相似度呈
/// 双峰 1.0/0.0），不适合做聚类度量。
fn text_jaccard(a_tokens: &HashSet<String>, b_tokens: &HashSet<String>) -> f32 {
    let union = a_tokens.union(b_tokens).count();
    if union == 0 {
        return 0.0;
    }
    let inter = a_tokens.intersection(b_tokens).count();
    inter as f32 / union as f32
}

fn token_set(text: &str) -> HashSet<String> {
    let mut toks = hippmem_core::hash::tokenize(text, "zh");
    toks.extend(hippmem_core::hash::tokenize(text, "en"));
    toks.into_iter()
        // 归一化：小写 + 去标点（zh 分词保留大小写并产出 ":"、"/"、"." 等标点 token，
        // 不归一化会把同词大小写变体与标点计入并集，严重稀释 Jaccard）
        .map(|t| t.to_lowercase())
        .filter(|t| t.chars().all(|c| c.is_alphanumeric()))
        .collect()
}

/// Plans summary clusters (03 §8): groups similar low-importance memories into
/// summary candidates. Deterministic (units are processed in id-ascending order).
///
/// A cluster must satisfy:
/// - token-set Jaccard similarity ≥ `similarity_threshold` (vs. the first member);
/// - member count ≥ `trigger_count` (boundary uses `>=`; the spec writes `>`, the
///   code and existing tests/tooling consistently use `>=` — kept for consistency);
/// - at least half of the members have importance < `low_importance_threshold`.
///
/// Excluded: units already compressed (`lifecycle == Compressed`), units already
/// covered by an existing summary (a unit with `generated_by == Consolidation`
/// lists its covered ids in `context.preceding_memory_ids`), and units with an
/// empty token set (no lexical signal).
///
/// Greedy: each unit is consumed into at most one cluster per cycle.
pub fn plan_summary_clusters(
    units: &[MemoryUnit],
    similarity_threshold: f32,
    trigger_count: u32,
    low_importance_threshold: f32,
) -> Vec<Vec<MemoryId>> {
    // 已被既有摘要覆盖的单元（covers 去重）
    let mut covered: HashSet<MemoryId> = HashSet::new();
    for u in units {
        if u.provenance.generated_by == GeneratedBy::Consolidation {
            covered.extend(u.context.preceding_memory_ids.iter().copied());
        }
    }

    // 候选：Active、未被覆盖、非摘要本身（E12: 摘要不得作为簇候选，
    // 否则会生成"摘要的摘要"，摘要链无限增长——covers 去重只挡已覆盖源）、
    // token 有信号（预计算 token 集，避免重复分词）
    let mut candidates: Vec<(&MemoryUnit, HashSet<String>)> = units
        .iter()
        .filter(|u| {
            !matches!(u.lifecycle, MemoryLifecycle::Compressed { .. })
                && !covered.contains(&u.id)
                && u.provenance.generated_by != GeneratedBy::Consolidation
        })
        .map(|u| (u, token_set(&u.content.raw)))
        .filter(|(_, toks)| !toks.is_empty())
        .collect();
    candidates.sort_by_key(|(u, _)| u.id);

    let mut clusters: Vec<Vec<MemoryId>> = Vec::new();
    while !candidates.is_empty() {
        let (pivot, pivot_tokens) = candidates.remove(0);
        let mut members: Vec<&MemoryUnit> = Vec::new();
        let mut remaining: Vec<(&MemoryUnit, HashSet<String>)> = Vec::new();
        for (u, toks) in candidates {
            if text_jaccard(&pivot_tokens, &toks) >= similarity_threshold {
                members.push(u);
            } else {
                remaining.push((u, toks));
            }
        }
        members.push(pivot);
        candidates = remaining;

        let low_importance = members
            .iter()
            .filter(|u| u.understanding.importance.value() < low_importance_threshold)
            .count();
        if (members.len() as u32) >= trigger_count && low_importance * 2 >= members.len() {
            let mut ids: Vec<MemoryId> = members.iter().map(|u| u.id).collect();
            ids.sort();
            clusters.push(ids);
        }
    }
    clusters
}

/// Builds a summary MemoryUnit: uses the Summarizer to generate summary text,
/// covering all original memories in `sources` (covers chain).
///
/// If the Summarizer returns confidence < 0.35, the caller should skip summary creation
/// (confidence gating, Constitution C7).
/// The returned MemoryUnit.understanding.confidence reflects the actual confidence.
pub fn build_summary_unit(
    sources: &[MemoryUnit],
    summarizer: &DeterministicSummarizer,
) -> MemoryUnit {
    // Use the Summarizer to generate the summary (the degraded backend does extractive summarization)
    let summarize_inputs: Vec<SummarizeInput> = sources
        .iter()
        .map(|u| SummarizeInput {
            id: u.id,
            text: u.content.raw.clone(),
        })
        .collect();

    let summary_output = match summarizer.summarize_sync(&summarize_inputs) {
        Ok(out) => out,
        Err(_) => {
            // Fallback summary when the Summarizer fails (simple concatenation)
            let fallback_text: String = sources
                .iter()
                .take(3)
                .map(|u| u.content.raw.chars().take(80).collect::<String>())
                .collect::<Vec<_>>()
                .join("; ");
            hippmem_model::traits::SummaryOutput {
                summary: fallback_text,
                covers: sources.iter().map(|u| u.id).collect(),
                confidence: UnitScore::new(0.1),
            }
        }
    };

    let summary_text = summary_output.summary;
    let summary_confidence = summary_output.confidence;
    let covers: Vec<MemoryId> = sources.iter().map(|u| u.id).collect();

    // Build an Elaboration edge for each original memory (summary → original)
    let links: Vec<AssociationLink> = covers
        .iter()
        .map(|target_id| AssociationLink {
            target_id: *target_id,
            link_type: LinkType::Elaboration,
            direction: LinkDirection::Forward,
            strength: UnitScore::new(0.5),
            confidence: UnitScore::new(0.6),
            evidence: hippmem_core::model::links::LinkEvidence {
                contributing_dimensions: vec![],
                score_breakdown: vec![],
                text_spans: vec![],
                note: Some("summary covers".into()),
            },
            formed_at: Timestamp(0),
            last_activated_at: None,
            activation_count: 0,
            observation: ObservationState::Confirmed,
        })
        .collect();

    MemoryUnit {
        schema_version: 1,
        id: MemoryId::generate(),
        created_at: Timestamp(0),
        updated_at: Timestamp(0),
        content: MemoryContent {
            raw: summary_text.clone(),
            summary: Some(summary_text),
            normalized: None,
            language: hippmem_core::model::unit::Language::Zh,
            content_type: hippmem_core::model::unit::ContentType::Reflection,
        },
        context: hippmem_core::model::unit::WriteContext {
            conversation_id: None,
            session_id: None,
            project_id: None,
            task_id: None,
            user_id: None,
            local_time: Timestamp(0),
            preceding_memory_ids: covers.clone(),
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
            confidence: summary_confidence,
        },
        association_keys: hippmem_core::model::links::AssociationKeys {
            entity_keys: vec![],
            temporal_keys: vec![],
            lexical_signature: hippmem_core::model::links::LexicalSignature { simhash: [0; 4] },
            semantic_signature: hippmem_core::model::links::SemanticSignature {
                lexical_simhash: [0; 4],
                dense_embedding_ref: None,
                binary_code: [0; 2],
                topic_minhash: [0u32; 16],
            },
            topic_keys: vec![],
            emotion_keys: vec![],
            goal_keys: vec![],
            event_keys: vec![],
            causal_keys: vec![],
        },
        links,
        activation: hippmem_core::model::links::ActivationState {
            last_retrieved_at: None,
            retrieval_count: 0,
            co_activations: vec![],
            usage_score: UnitScore::new(0.5),
        },
        lifecycle: hippmem_core::model::unit::MemoryLifecycle::Active,
        provenance: hippmem_core::model::unit::Provenance {
            origin: hippmem_core::model::unit::SourceKind::Conversation,
            generated_by: GeneratedBy::Consolidation,
            reliability: UnitScore::new(0.6),
            evidence_refs: vec![],
            revision_history: vec![],
        },
        stage: MemoryStage::Consolidated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippmem_core::model::links::{
        ActivationState, AssociationKeys, LexicalSignature, SemanticSignature,
    };
    use hippmem_core::model::understanding::MemoryUnderstanding;
    use hippmem_core::model::unit::{
        Language, MemoryContent, Provenance, SourceKind, WriteContext,
    };

    /// 构造簇规划测试用单元（仅填充规划器读取的字段）。
    fn unit(id: u128, text: &str, importance: f32) -> MemoryUnit {
        MemoryUnit {
            schema_version: 1,
            id: MemoryId(id),
            created_at: Timestamp(0),
            updated_at: Timestamp(0),
            content: MemoryContent {
                raw: text.into(),
                summary: None,
                normalized: None,
                language: Language::Zh,
                content_type: hippmem_core::model::enums::ContentType::UserStatement,
            },
            context: WriteContext {
                conversation_id: None,
                session_id: None,
                project_id: None,
                task_id: None,
                user_id: None,
                local_time: Timestamp(0),
                preceding_memory_ids: vec![],
                source_refs: vec![],
            },
            understanding: MemoryUnderstanding {
                entities: vec![],
                events: vec![],
                goals: vec![],
                decisions: vec![],
                preferences: vec![],
                emotions: vec![],
                causal_claims: vec![],
                contradictions: vec![],
                topics: vec![],
                importance: UnitScore::new(importance),
                confidence: UnitScore::new(0.5),
            },
            association_keys: AssociationKeys {
                entity_keys: vec![],
                temporal_keys: vec![],
                lexical_signature: LexicalSignature { simhash: [0; 4] },
                semantic_signature: SemanticSignature {
                    lexical_simhash: [0; 4],
                    dense_embedding_ref: None,
                    binary_code: [0; 2],
                    topic_minhash: [0u32; 16],
                },
                topic_keys: vec![],
                emotion_keys: vec![],
                goal_keys: vec![],
                event_keys: vec![],
                causal_keys: vec![],
            },
            links: vec![],
            activation: ActivationState {
                last_retrieved_at: None,
                retrieval_count: 0,
                co_activations: vec![],
                usage_score: UnitScore::new(0.5),
            },
            lifecycle: MemoryLifecycle::Active,
            provenance: Provenance {
                origin: SourceKind::Conversation,
                generated_by: GeneratedBy::UserDirect,
                reliability: UnitScore::new(0.5),
                evidence_refs: vec![],
                revision_history: vec![],
            },
            stage: MemoryStage::Indexed,
        }
    }

    /// 相似模板文本（仅 E 码变化、行号固定 → Jaccard 高）。
    fn make_cluster(n: u128) -> Vec<MemoryUnit> {
        (1..=n)
            .map(|i| {
                unit(
                    i,
                    &format!("Build output: warning E{i:04} fixed at src/main.rs line 42."),
                    0.3,
                )
            })
            .collect()
    }

    /// 互不相似的文本（每个文本词汇完全独立 → Jaccard 0）。
    fn make_dissimilar(n: u128) -> Vec<MemoryUnit> {
        (1..=n)
            .map(|i| {
                unit(
                    i,
                    &format!("memtopic{i} alpha{i} beta{i} gamma{i} delta{i}"),
                    0.3,
                )
            })
            .collect()
    }

    #[test]
    fn plan_cluster_forms_from_similar_units() {
        let units = make_cluster(15);
        let clusters = plan_summary_clusters(&units, 0.7, 12, 0.5);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 15);
    }

    #[test]
    fn plan_cluster_respects_threshold_and_trigger_count() {
        // 不足触发数 → 无簇
        assert!(plan_summary_clusters(&make_cluster(5), 0.7, 12, 0.5).is_empty());
        // 高 importance 占多数 → 无簇
        let high_imp: Vec<MemoryUnit> = (1..=13)
            .map(|i| {
                unit(
                    i,
                    &format!("Build output: warning E{i:04} fixed at src/main.rs line 42."),
                    0.8,
                )
            })
            .collect();
        assert!(plan_summary_clusters(&high_imp, 0.7, 12, 0.5).is_empty());
        // 词汇不重叠（Jaccard 低于阈值）→ 无簇
        assert!(plan_summary_clusters(&make_dissimilar(13), 0.7, 12, 0.5).is_empty());
    }

    #[test]
    fn plan_cluster_excludes_compressed_and_covered() {
        // 5 个已压缩 + 8 个 Active → 簇仅 8 人 → 无簇
        let mut units = make_cluster(13);
        for u in units.iter_mut().take(5) {
            u.lifecycle = MemoryLifecycle::Compressed {
                into: MemoryId(999),
            };
        }
        assert!(plan_summary_clusters(&units, 0.7, 12, 0.5).is_empty());

        // 未被压缩但被既有摘要覆盖 → 同样无簇
        let mut units = make_cluster(13);
        let mut summary = unit(999, "Build output: warning E0999 fixed at line 9990.", 0.5);
        summary.provenance.generated_by = GeneratedBy::Consolidation;
        summary.context.preceding_memory_ids = units.iter().map(|u| u.id).take(5).collect();
        units.push(summary);
        let clusters = plan_summary_clusters(&units, 0.7, 12, 0.5);
        assert!(clusters.is_empty(), "已被覆盖的单元不得再次进入簇");
    }

    #[test]
    fn plan_cluster_ignores_no_signal_text() {
        // 无 token 的文本（纯标点）→ 无簇
        let units: Vec<MemoryUnit> = (1..=15).map(|i| unit(i, "！？……", 0.3)).collect();
        assert!(plan_summary_clusters(&units, 0.7, 12, 0.5).is_empty());
    }

    #[test]
    fn plan_cluster_is_deterministic() {
        let units = make_cluster(15);
        let a = plan_summary_clusters(&units, 0.7, 12, 0.5);
        let b = plan_summary_clusters(&units, 0.7, 12, 0.5);
        assert_eq!(a, b);
    }

    /// E12: 摘要（generated_by = Consolidation）不得作为簇候选——
    /// 否则与新记忆相似时会把摘要本身卷入簇，生成"摘要的摘要"。
    #[test]
    fn plan_cluster_excludes_summaries_from_candidates() {
        let mut units = make_cluster(13);
        let mut summary = unit(
            999,
            "Build output: warning E0999 fixed at src/main.rs line 999.",
            0.5,
        );
        summary.provenance.generated_by = GeneratedBy::Consolidation;
        units.push(summary);

        let clusters = plan_summary_clusters(&units, 0.7, 12, 0.5);
        assert!(!clusters.is_empty(), "13 sources must still form a cluster");
        for c in &clusters {
            assert!(
                !c.contains(&MemoryId(999)),
                "summary must not become a cluster candidate (E12)"
            );
        }
    }
}
