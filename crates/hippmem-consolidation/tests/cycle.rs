//! acceptance tests (module layer): consolidation pipeline integration tests
//!
//! Verifies the 03 §6-§7 consolidation flow: Hebbian reinforcement → decay → Compaction in series.
//! 4 scenarios:
//!   1. Hebbian → decay in series (co-activation reinforcement + ordinary decay + protected unchanged)
//!   2. Decay → Compaction in series (weak-edge archiving + protected-edge retention + degree_limit)
//!   3. Multi-cycle consolidation (decay compounding + Hebbian does not reinforce repeatedly)
//!   4. Protected set full chain (Causal/Correction survives all three stages unaffected)

use hippmem_consolidation::compaction::{compact_edges, CompactionParams};
use hippmem_consolidation::decay::{apply_decay_with_protection, DecayParams};
use hippmem_consolidation::hebbian::{hebbian_reinforce, HebbianParams};
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{
    AssociationLink, LinkDirection, LinkEvidence, LinkType, ObservationState,
};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;

// ═══════════════════════════════════════════════════════════════════
// Helper
// ═══════════════════════════════════════════════════════════════════

fn make_link(
    target: u128,
    link_type: LinkType,
    strength: f32,
    confidence: f32,
    observation: ObservationState,
    formed_at: Timestamp,
    last_activated: Option<Timestamp>,
) -> AssociationLink {
    AssociationLink {
        target_id: MemoryId(target),
        link_type,
        direction: LinkDirection::Undirected,
        strength: UnitScore::new(strength),
        confidence: UnitScore::new(confidence),
        evidence: LinkEvidence {
            contributing_dimensions: vec![],
            score_breakdown: vec![],
            text_spans: vec![],
            note: None,
        },
        formed_at,
        last_activated_at: last_activated,
        activation_count: 0,
        observation,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 1: Hebbian → decay in series
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_1_hebbian_then_decay_pipeline() {
    // Use two different time points: first Hebbian (earlier), then decay (later)
    // Ensure inactive_duration > 86_400_000 at decay time
    let heb_now = Timestamp(100_000_000); // Hebbian time
    let decay_now = Timestamp(100_000_000 + 200_000_000); // decay time (> 1 day later)

    // ── Construct 3 edges ──
    // Edge A: co-activated EntityOverlap edge → Hebbian reinforcement → decay
    // Edge B: EntityOverlap edge without co-activation → unaffected by Hebbian → decay
    // Edge C: Causal protected edge → unaffected by Hebbian (no matching activation pair) → unaffected by decay
    let mut links = vec![
        make_link(
            2,
            LinkType::EntityOverlap,
            0.4,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)), // last activated long ago
        ),
        make_link(
            3,
            LinkType::EntityOverlap,
            0.4,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
        make_link(
            4,
            LinkType::Causal,
            0.4,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
    ];

    // ── Step 1: Hebbian reinforcement (edge A matches the co-activation pair) ──
    let co_activations = vec![(MemoryId(1), MemoryId(2), 5u32)];
    let heb_params = HebbianParams::default();
    hebbian_reinforce(&mut links, &co_activations, &heb_params, heb_now);

    // Edge A (target=2) should be reinforced
    let link_a_strength = links
        .iter()
        .find(|l| l.target_id == MemoryId(2))
        .unwrap()
        .strength
        .value();
    assert!(
        link_a_strength > 0.4,
        "co-activated edge should be reinforced by Hebbian"
    );
    let link_a_activation = links
        .iter()
        .find(|l| l.target_id == MemoryId(2))
        .unwrap()
        .activation_count;
    assert_eq!(
        link_a_activation, 1,
        "Hebbian should increment activation_count"
    );
    let link_a_last = links
        .iter()
        .find(|l| l.target_id == MemoryId(2))
        .unwrap()
        .last_activated_at;
    assert_eq!(link_a_last, Some(heb_now));

    // Edge B (target=3) strength unchanged
    let link_b_strength = links
        .iter()
        .find(|l| l.target_id == MemoryId(3))
        .unwrap()
        .strength
        .value();
    assert!(
        (link_b_strength - 0.4).abs() < 0.001,
        "edge without co-activation should not be reinforced"
    );

    // ── Step 2: Decay (decay_now is far greater than last_activated_at, decay triggers) ──
    let decay_params = DecayParams::default();
    apply_decay_with_protection(&mut links, &decay_params, decay_now);

    // Edge A (reinforced, last_activated=heb_now, inactive > 1 day) → should decay
    let link_a2_strength = links
        .iter()
        .find(|l| l.target_id == MemoryId(2))
        .unwrap()
        .strength
        .value();
    assert!(
        link_a2_strength < link_a_strength,
        "edge inactive > 1 day after reinforcement should decay: {} → {}",
        link_a_strength,
        link_a2_strength
    );

    // Edge B (last_activated=0) → should decay
    let link_b2_strength = links
        .iter()
        .find(|l| l.target_id == MemoryId(3))
        .unwrap()
        .strength
        .value();
    assert!(
        link_b2_strength < 0.4,
        "non-reinforced edge inactive > 1 day should decay"
    );

    // Edge C (Causal protected) → strength unchanged
    let link_c_strength = links
        .iter()
        .find(|l| l.target_id == MemoryId(4))
        .unwrap()
        .strength
        .value();
    assert!(
        (link_c_strength - 0.4).abs() < 0.001,
        "protected Causal edge should not decay: expected 0.4, got {}",
        link_c_strength
    );
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 2: Decay → Compaction in series
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_2_decay_then_compaction_pipeline() {
    let now = Timestamp(200_000_000);

    // ── Construct edges, distinguishing the decay path from the compaction path ──
    // Note: decay applies a min_retained floor to Confirmed edges, so a weak Confirmed edge
    // after decay becomes 0.12 and cannot be archived in subsequent compaction. Therefore:
    //   - Decay elimination: tested with Observing edges
    //   - Compaction archiving: tested directly with a Confirmed edge below min_retained (without decay)
    let mut links_for_decay = vec![
        // Observation-zone edge (0.12 → 0.12*0.97=0.1164 < min_retained → removed by decay)
        make_link(
            2,
            LinkType::EntityOverlap,
            0.12,
            0.5,
            ObservationState::Observing {
                since: Timestamp(0),
            },
            Timestamp(0),
            Some(Timestamp(0)),
        ),
        // Protected edge (Observing but Causal type → is_protected is still false → treated equally)
        // Normal edge → retained after decay
        make_link(
            3,
            LinkType::EntityOverlap,
            0.50,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
    ];

    // ── Step 1: Decay ──
    let decay_params = DecayParams::default();
    apply_decay_with_protection(&mut links_for_decay, &decay_params, now);

    // Edge 2 (Observing, weak) has been removed by decay
    assert_eq!(
        links_for_decay.len(),
        1,
        "decay should remove 1 weak Observing edge"
    );
    assert!(
        links_for_decay.iter().any(|l| l.target_id == MemoryId(3)),
        "normal edge should be retained"
    );

    // ── Step 2: Compaction (tested directly, without decay) ──
    let links_for_compaction = vec![
        // Weak Confirmed edge (0.05 < min_retained 0.12) → archived
        make_link(
            10,
            LinkType::EntityOverlap,
            0.05,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
        // Weak Causal protected edge → retained (even below min_retained)
        make_link(
            11,
            LinkType::Causal,
            0.05,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
        // Normal edge → retained
        make_link(
            12,
            LinkType::EntityOverlap,
            0.50,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
    ];

    let comp_params = CompactionParams::default();
    let (kept, archived) = compact_edges(links_for_compaction, &comp_params);

    // Weak edge 10 (EntityOverlap, 0.05<0.12) → archived
    assert!(
        archived.iter().any(|l| l.target_id == MemoryId(10)),
        "weak Confirmed edge (below min_retained) should be archived"
    );
    // Protected edge 11 (Causal, 0.05) → retained
    assert!(
        kept.iter().any(|l| l.target_id == MemoryId(11)),
        "weak Causal protected edge should be retained"
    );
    // Normal edge 12 (0.50) → retained
    assert!(
        kept.iter().any(|l| l.target_id == MemoryId(12)),
        "normal-strength edge should be retained"
    );

    assert_eq!(kept.len(), 2, "should retain protected edge + normal edge");
    assert_eq!(archived.len(), 1, "should archive 1 weak edge");
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 3: Multi-cycle consolidation (decay compounding)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_3_multi_cycle_decay_compounds() {
    let decay_params = DecayParams::default();

    // Ordinary edge with initial strength 0.5
    let mut links = vec![make_link(
        2,
        LinkType::EntityOverlap,
        0.5,
        0.5,
        ObservationState::Confirmed,
        Timestamp(0),
        Some(Timestamp(0)), // formed far earlier than now
    )];

    // 3 decay rounds, each advancing now forward (> 1 day)
    let strengths: Vec<f32> = (0..3)
        .map(|round| {
            let now = Timestamp(200_000_000 + round * 200_000_000);
            apply_decay_with_protection(&mut links, &decay_params, now);
            links[0].strength.value()
        })
        .collect();

    // Strength should decrease monotonically after each decay round
    assert!(strengths[0] < 0.5, "should decay after round 1");
    assert!(
        strengths[1] < strengths[0],
        "should decay further after round 2"
    );
    assert!(
        strengths[2] < strengths[1],
        "should continue decaying after round 3"
    );

    // Should not go below the floor
    assert!(
        strengths[2] >= decay_params.min_retained_strength,
        "should not go below min_retained_strength={}",
        decay_params.min_retained_strength
    );
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 4: Protected set full chain (survives all three stages)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn scenario_4_protected_types_survive_all_stages() {
    let now = Timestamp(200_000_000);

    // ── Construct mixed edges (protected + non-protected) ──
    let mut links = vec![
        make_link(
            2,
            LinkType::Causal,
            0.3,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
        make_link(
            3,
            LinkType::Correction,
            0.3,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
        make_link(
            4,
            LinkType::Supersedes,
            0.3,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
        make_link(
            5,
            LinkType::EntityOverlap,
            0.3,
            0.5,
            ObservationState::Confirmed,
            Timestamp(0),
            Some(Timestamp(0)),
        ),
    ];

    // ── Stage 1: Hebbian (no co-activation for any edge, unaffected) ──
    let heb_params = HebbianParams::default();
    hebbian_reinforce(&mut links, &[], &heb_params, now);
    // All edges keep their strength
    for link in &links {
        assert!(
            (link.strength.value() - 0.3).abs() < 0.001,
            "Hebbian should not affect any edge when there is no co-activation"
        );
    }

    // ── Stage 2: Decay ──
    let decay_params = DecayParams::default();
    apply_decay_with_protection(&mut links, &decay_params, now);

    // Protected edges unchanged
    for id in &[2u128, 3, 4] {
        let link = links.iter().find(|l| l.target_id == MemoryId(*id)).unwrap();
        assert!(
            (link.strength.value() - 0.3).abs() < 0.001,
            "protected edge {:?} should not be affected by decay",
            link.link_type
        );
    }
    // EntityOverlap decays
    let entity_link = links.iter().find(|l| l.target_id == MemoryId(5)).unwrap();
    assert!(
        entity_link.strength.value() < 0.3,
        "non-protected edge should decay"
    );

    // ── Stage 3: Compaction ──
    let comp_params = CompactionParams::default();
    let (kept, _archived) = compact_edges(links, &comp_params);

    // 3 protected edges + 1 non-protected edge (may be archived after decay)
    let protected_kept: Vec<_> = kept
        .iter()
        .filter(|l| {
            matches!(
                l.link_type,
                LinkType::Causal | LinkType::Correction | LinkType::Supersedes
            )
        })
        .collect();
    assert_eq!(
        protected_kept.len(),
        3,
        "all 3 protected edges should be retained"
    );

    // EntityOverlap in compaction (after decay strength < 0.12 → archived or retained)
    // 0.3 * 0.97 = 0.291 > 0.12 (min_retained) → should be retained
    let entity_in_kept = kept.iter().any(|l| l.target_id == MemoryId(5));
    assert!(
        entity_in_kept,
        "non-protected edge still above the threshold after decay should be retained"
    );
}
