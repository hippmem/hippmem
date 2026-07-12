//! acceptance tests: decay logic (protected set + per-cycle decay).

use hippmem_consolidation::decay::{apply_decay_with_protection, DecayParams};
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{AssociationLink, LinkDirection, LinkType, ObservationState};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;

fn make_link(
    target: u128,
    link_type: LinkType,
    strength: f32,
    observation: ObservationState,
) -> AssociationLink {
    AssociationLink {
        target_id: MemoryId(target),
        link_type,
        direction: LinkDirection::Undirected,
        strength: UnitScore::new(strength),
        confidence: UnitScore::new(0.5),
        evidence: hippmem_core::model::links::LinkEvidence {
            contributing_dimensions: vec![],
            score_breakdown: vec![],
            text_spans: vec![],
            note: None,
        },
        formed_at: Timestamp(0),
        last_activated_at: Some(Timestamp(0)),
        activation_count: 0,
        observation,
    }
}

#[test]
fn normal_edge_decays() {
    let mut links = vec![make_link(
        2,
        LinkType::EntityOverlap,
        0.5,
        ObservationState::Confirmed,
    )];
    let params = DecayParams::default();
    // Decay: 0.5 * 0.97 = 0.485
    apply_decay_with_protection(&mut links, &params, Timestamp(200_000_000));
    assert!(
        links[0].strength.value() < 0.5,
        "ordinary edge should decay"
    );
    assert!(links[0].strength.value() >= params.min_retained_strength);
}

#[test]
fn causal_edge_protected() {
    let mut links = vec![make_link(
        2,
        LinkType::Causal,
        0.5,
        ObservationState::Confirmed,
    )];
    let params = DecayParams::default();
    let original = links[0].strength.value();
    apply_decay_with_protection(&mut links, &params, Timestamp(200_000_000));
    assert_eq!(
        links[0].strength.value(),
        original,
        "Causal edge (protected set) should not decay"
    );
}

#[test]
fn correction_edge_protected() {
    let mut links = vec![make_link(
        2,
        LinkType::Correction,
        0.5,
        ObservationState::Confirmed,
    )];
    let params = DecayParams::default();
    let original = links[0].strength.value();
    apply_decay_with_protection(&mut links, &params, Timestamp(200_000_000));
    assert_eq!(
        links[0].strength.value(),
        original,
        "Correction edge should be protected"
    );
}

#[test]
fn supersedes_edge_protected() {
    let mut links = vec![make_link(
        2,
        LinkType::Supersedes,
        0.5,
        ObservationState::Confirmed,
    )];
    let params = DecayParams::default();
    let original = links[0].strength.value();
    apply_decay_with_protection(&mut links, &params, Timestamp(200_000_000));
    assert_eq!(
        links[0].strength.value(),
        original,
        "Supersedes edge should be protected"
    );
}

#[test]
fn observing_edge_not_protected() {
    // Observation-zone edges are not protected even when the type is Causal
    let mut links = vec![make_link(
        2,
        LinkType::Causal,
        0.5,
        ObservationState::Observing {
            since: Timestamp(0),
        },
    )];
    let params = DecayParams::default();
    apply_decay_with_protection(&mut links, &params, Timestamp(200_000_000));
    assert!(
        links[0].strength.value() < 0.5,
        "observation-zone edge (even Causal) should decay"
    );
}

#[test]
fn stale_observing_eliminated() {
    let mut links = vec![make_link(
        2,
        LinkType::EntityOverlap,
        0.11, // below min_retained (0.12)
        ObservationState::Observing {
            since: Timestamp(0),
        },
    )];
    let params = DecayParams::default();
    apply_decay_with_protection(&mut links, &params, Timestamp(200_000_000));
    assert!(
        links.is_empty(),
        "observation-zone candidates with strength below min_retained should be removed"
    );
}

#[test]
fn strength_floor_at_min_retained() {
    let mut links = vec![make_link(
        2,
        LinkType::SemanticSimilar,
        0.13,
        ObservationState::Confirmed,
    )];
    let params = DecayParams::default();
    apply_decay_with_protection(&mut links, &params, Timestamp(200_000_000));
    // 0.13 * 0.97 = 0.1261 < min_retained (0.12), so it should be 0.12
    assert!(
        links[0].strength.value() >= params.min_retained_strength,
        "should not go below min_retained_strength"
    );
}

// ── Compaction tests ──

use hippmem_consolidation::compaction::{compact_edges, CompactionParams};

fn make_c_link(target: u128, strength: f32) -> AssociationLink {
    make_link(
        target,
        LinkType::EntityOverlap,
        strength,
        ObservationState::Confirmed,
    )
}

#[test]
fn compaction_archives_weak_edges() {
    let links = vec![
        make_c_link(2, 0.05), // weak edge (below min_retained 0.12)
        make_c_link(3, 0.08), // weak edge
        make_c_link(4, 0.50), // normal edge
    ];
    let params = CompactionParams::default();
    let (kept, archived) = compact_edges(links, &params);
    assert_eq!(kept.len(), 1, "only the normal edge should be retained");
    assert_eq!(archived.len(), 2, "2 weak edges should be archived");
    assert_eq!(kept[0].target_id, MemoryId(4));
}

#[test]
fn compaction_protects_causal_edges() {
    let links = vec![
        make_link(2, LinkType::Causal, 0.05, ObservationState::Confirmed),
        make_c_link(3, 0.50),
    ];
    let params = CompactionParams::default();
    let (kept, _archived) = compact_edges(links, &params);
    // Causal edges are not subject to compaction
    assert_eq!(
        kept.len(),
        2,
        "Causal protected edge should not be archived"
    );
}

#[test]
fn compaction_respects_degree_limit() {
    let mut links = Vec::new();
    for i in 0..10 {
        links.push(make_c_link(i, 0.3 + i as f32 * 0.05));
    }
    let params = CompactionParams {
        degree_limit: 3,
        ..Default::default()
    };
    let (kept, _archived) = compact_edges(links, &params);
    assert_eq!(kept.len(), 3, "should retain only the strongest 3 edges");
}

// ═══════════════════════════════════════════════════════════════════
// complete the protected-set tests (Contradiction + TopicRelated)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn contradiction_edge_protected() {
    let mut links = vec![make_link(
        2,
        LinkType::Contradiction,
        0.5,
        ObservationState::Confirmed,
    )];
    let params = DecayParams::default();
    let original = links[0].strength.value();
    apply_decay_with_protection(&mut links, &params, Timestamp(200_000_000));
    assert_eq!(
        links[0].strength.value(),
        original,
        "Contradiction edge (protected set) should not decay"
    );
}

#[test]
fn topic_related_edge_decays() {
    let mut links = vec![make_link(
        2,
        LinkType::TopicRelated,
        0.5,
        ObservationState::Confirmed,
    )];
    let params = DecayParams::default();
    apply_decay_with_protection(&mut links, &params, Timestamp(200_000_000));
    assert!(
        links[0].strength.value() < 0.5,
        "TopicRelated edge (non-protected) should decay"
    );
}

#[test]
fn contradiction_protected_in_compaction() {
    let links = vec![
        make_link(
            2,
            LinkType::Contradiction,
            0.05,
            ObservationState::Confirmed,
        ),
        make_link(
            3,
            LinkType::EntityOverlap,
            0.50,
            ObservationState::Confirmed,
        ),
    ];
    let params = CompactionParams::default();
    let (kept, _archived) = compact_edges(links, &params);
    // Contradiction (0.05<min_retained but protected) + EntityOverlap (0.50>min_retained)
    assert_eq!(
        kept.len(),
        2,
        "Contradiction protected edge + normal edge should both be retained"
    );
    assert!(
        kept.iter().any(|l| l.link_type == LinkType::Contradiction),
        "Contradiction should be protected in compaction"
    );
}
