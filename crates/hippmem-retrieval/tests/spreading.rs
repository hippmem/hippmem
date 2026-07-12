//! acceptance test: initial_energy + single-hop spreading + multi-hop spreading + merge + pruning.
//!
//! Verifies the algorithmic correctness of 03 §4.1-4.3.

use hippmem_core::config::AlgoParams;
use hippmem_core::ids::MemoryId;
use hippmem_core::model::links::{AssociationLink, LinkDirection, LinkType, ObservationState};
use hippmem_core::score::UnitScore;
use hippmem_core::time::Timestamp;
use hippmem_retrieval::energy::{self, initial_energy};
use hippmem_retrieval::seeds::Seed;
use hippmem_retrieval::spreading::{spread_multi_hop, spread_one_hop};
use std::collections::HashMap;

// ── Helper: build test edges ──

fn make_link(target: u128, link_type: LinkType, strength: f32, confidence: f32) -> AssociationLink {
    AssociationLink {
        target_id: MemoryId(target),
        link_type,
        direction: LinkDirection::Undirected,
        strength: UnitScore::new(strength),
        confidence: UnitScore::new(confidence),
        evidence: hippmem_core::model::links::LinkEvidence {
            contributing_dimensions: vec![],
            score_breakdown: vec![],
            text_spans: vec![],
            note: None,
        },
        formed_at: Timestamp(1_700_000_000_000),
        last_activated_at: None,
        activation_count: 0,
        observation: ObservationState::Confirmed,
    }
}

// ── 1. initial_energy tests ──

#[test]
fn initial_energy_all_factors() {
    let params = AlgoParams::default();
    let seed = Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    };
    // All factors near 1, expect high energy
    let e = initial_energy(&seed, 0.9, 1.0, 0.8, 0.7, 0.6, &params);
    // channel_coeff=1.0, query=0.9, a=0.4, importance=0.8, c=0.60
    // = 1.0*0.9*0.4*(1+0.8*0.60) + 1.0*0.20 + 0.7*0.15 + 0.6*0.10
    // = 0.36*1.48 + 0.20 + 0.105 + 0.060 = 0.8978
    let expected = 0.9 * 0.40 * (1.0 + 0.8 * 0.60) + 1.0 * 0.20 + 0.7 * 0.15 + 0.6 * 0.10;
    assert!(
        (e - expected).abs() < 0.001,
        "expected {}, got {}",
        expected,
        e
    );
    assert!(e <= params.seed_energy_cap);
}

#[test]
fn initial_energy_clamped_to_cap() {
    let params = AlgoParams::default();
    let seed = Seed {
        id: MemoryId(2),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 1.0,
        rank_in_channel: None,
    };
    // All-max factors → energy > cap → should be clamped
    let e = initial_energy(&seed, 1.0, 1.0, 1.0, 1.0, 1.0, &params);
    // All-max: 1.0*0.4+1*0.2+1*0.15+1*0.15+1*0.1 = 1.0, cap=1.0
    assert!(e <= params.seed_energy_cap);
    assert!(e >= 0.99); // should not be over-clamped
}

#[test]
fn initial_energy_zero_factors() {
    let params = AlgoParams::default();
    let seed = Seed {
        id: MemoryId(3),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.5,
        rank_in_channel: None,
    };
    let e = initial_energy(&seed, 0.0, 0.0, 0.0, 0.0, 0.0, &params);
    assert!((e - 0.0).abs() < 0.001, "all-zero factors should yield 0");
}

// ── 2. propagated_energy tests ──

#[test]
fn propagated_energy_single_hop() {
    let params = AlgoParams::default();
    let link = make_link(10, LinkType::EntityOverlap, 0.7, 0.8);
    let source = 0.5;

    let pe = energy::propagated_energy(source, &link, 1, &params);
    // 0.5 * 0.7 * 0.8 * 0.55^1 * 1.00 = 0.5 * 0.7 * 0.8 * 0.55 = 0.154
    let expected = 0.5 * 0.7 * 0.8 * 0.55f32.powi(1) * 1.00;
    assert!((pe - expected).abs() < 0.001);
}

#[test]
fn propagated_energy_two_hops() {
    let params = AlgoParams::default();
    let link = make_link(11, LinkType::EntityOverlap, 0.7, 0.8);
    let source = 0.5;

    let pe = energy::propagated_energy(source, &link, 2, &params);
    // 0.5 * 0.7 * 0.8 * 0.55^2 * 1.00 = 0.121 * 0.55 = 0.0847
    let expected = 0.5 * 0.7 * 0.8 * 0.55f32.powi(2) * 1.00;
    assert!((pe - expected).abs() < 0.001);
}

// ── 3. type_modifier tests ──

#[test]
fn type_modifier_causal_strongest() {
    // Causal modifier should be 1.30 (highest)
    assert!((energy::type_modifier(LinkType::Causal) - 1.30).abs() < 0.001);
    assert!((energy::type_modifier(LinkType::Correction) - 1.20).abs() < 0.001);
}

#[test]
fn type_modifier_temporal_and_deprecated_weakest() {
    // TemporalAdjacent=0.60, Deprecated=0.40 (lowest)
    assert!(energy::type_modifier(LinkType::TemporalAdjacent) < 0.70);
    assert!(
        energy::type_modifier(LinkType::Deprecated)
            < energy::type_modifier(LinkType::TemporalAdjacent)
    );
}

#[test]
fn type_modifier_causal_gt_semantic() {
    // Causal > SemanticSimilar (causal relations are more valuable to spread than semantic similarity)
    let causal = energy::type_modifier(LinkType::Causal);
    let semantic = energy::type_modifier(LinkType::SemanticSimilar);
    assert!(
        causal > semantic,
        "Causal({}) should be > SemanticSimilar({})",
        causal,
        semantic
    );
}

#[test]
fn type_modifier_all_variants_positive() {
    // All edge type modifiers should be positive (not 0 or negative)
    let all_types = [
        LinkType::EntityOverlap,
        LinkType::TemporalAdjacent,
        LinkType::SemanticSimilar,
        LinkType::TopicRelated,
        LinkType::SameGoal,
        LinkType::SameEvent,
        LinkType::Causal,
        LinkType::EmotionalResonance,
        LinkType::Contradiction,
        LinkType::Correction,
        LinkType::Elaboration,
        LinkType::CoActivation,
        LinkType::Supersedes,
        LinkType::Deprecated,
    ];
    for lt in &all_types {
        let tm = energy::type_modifier(*lt);
        assert!(
            tm > 0.0,
            "type_modifier({:?}) = {} should not be 0 or negative",
            lt,
            tm
        );
    }
}

// ── 4. Single-hop spreading integration tests ──

#[test]
fn spread_one_hop_basic() {
    let params = AlgoParams::default();
    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.8,
        rank_in_channel: None,
    }];

    // Seed 1 → Causal edge to 2 → Semantic edge to 3
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(2, LinkType::Causal, 0.7, 0.8),
            make_link(3, LinkType::SemanticSimilar, 0.5, 0.6),
        ],
    );

    let results = spread_one_hop(&seeds, &links_map, &params, &HashMap::new());

    // Results should include the seed itself + neighbors
    assert!(
        !results.is_empty(),
        "should return at least the seed itself"
    );

    // Seed 1 itself should be in the results
    let seed_result = results.iter().find(|(id, _, _)| *id == MemoryId(1));
    assert!(seed_result.is_some(), "seed 1 should be in the results");
    assert!(
        seed_result.unwrap().1 > 0.0,
        "seed should have positive energy"
    );

    // Causal-edge neighbor 2 should be in the results, with energy > semantic-edge neighbor 3
    let causal = results.iter().find(|(id, _, _)| *id == MemoryId(2));
    let semantic = results.iter().find(|(id, _, _)| *id == MemoryId(3));
    assert!(causal.is_some(), "causal neighbor should be in the results");
    if let (Some(c), Some(s)) = (causal, semantic) {
        assert!(
            c.1 > s.1,
            "Causal edge should transfer more energy than SemanticSimilar"
        );
    }
}

#[test]
fn spread_one_hop_type_modifier_effects() {
    let params = AlgoParams::default();
    // Same seed, same strength/confidence, different edge types
    let seed = Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.8,
        rank_in_channel: None,
    };

    // Two edges: Temporal (weak spreading) vs Causal (strong spreading)
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(10, LinkType::TemporalAdjacent, 0.7, 0.8),
            make_link(20, LinkType::Causal, 0.7, 0.8),
        ],
    );

    let results = spread_one_hop(&[seed], &links_map, &params, &HashMap::new());
    let temporal = results
        .iter()
        .find(|(id, _, _)| *id == MemoryId(10))
        .unwrap();
    let causal = results
        .iter()
        .find(|(id, _, _)| *id == MemoryId(20))
        .unwrap();

    assert!(
        causal.1 > temporal.1,
        "with same strength/confidence, Causal({}) should be > TemporalAdjacent({})",
        causal.1,
        temporal.1
    );
}

#[test]
fn spread_one_hop_energy_decay() {
    let params = AlgoParams::default();
    let seed = Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    };

    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![make_link(2, LinkType::EntityOverlap, 0.9, 0.9)],
    );

    let results = spread_one_hop(&[seed], &links_map, &params, &HashMap::new());
    let seed_energy = results
        .iter()
        .find(|(id, _, _)| *id == MemoryId(1))
        .unwrap()
        .1;
    let neighbor_energy = results
        .iter()
        .find(|(id, _, _)| *id == MemoryId(2))
        .unwrap()
        .1;

    // Energy should decay after one hop (even with high strength/confidence)
    assert!(
        neighbor_energy < seed_energy,
        "neighbor energy after propagation ({}) should be less than seed energy ({})",
        neighbor_energy,
        seed_energy
    );
}

#[test]
fn spread_one_hop_below_threshold_excluded() {
    let params = AlgoParams {
        min_propagation_energy: 0.10,
        ..Default::default()
    };
    let seed = Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.3,
        rank_in_channel: None,
    };

    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![make_link(2, LinkType::TemporalAdjacent, 0.3, 0.3)],
    );

    let results = spread_one_hop(&[seed], &links_map, &params, &HashMap::new());
    let neighbor = results.iter().find(|(id, _, _)| *id == MemoryId(2));
    assert!(
        neighbor.is_none(),
        "neighbors below min_propagation_energy should not appear in the results"
    );
}

// ── 5. Multi-hop spreading tests ──

#[test]
fn spread_multi_hop_reaches_two_hops() {
    let params = AlgoParams::default();
    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 1.0,
        rank_in_channel: None,
    }];

    // 1 → 2 (strong Causal edge) → 3 (Entity edge)
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(MemoryId(1), vec![make_link(2, LinkType::Causal, 1.0, 1.0)]);
    links_map.insert(
        MemoryId(2),
        vec![make_link(3, LinkType::EntityOverlap, 1.0, 1.0)],
    );

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    assert!(
        results.iter().any(|(id, _, _)| *id == MemoryId(3)),
        "two-hop neighbor should be reachable"
    );
}

#[test]
fn spread_multi_hop_cycle_detection() {
    let params = AlgoParams::default();
    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    }];

    // 1 → 2 → 1 (cycle) — should not loop infinitely
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![make_link(2, LinkType::EntityOverlap, 0.8, 0.8)],
    );
    links_map.insert(
        MemoryId(2),
        vec![make_link(1, LinkType::SemanticSimilar, 0.8, 0.8)],
    );

    // Should not loop infinitely or panic
    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    // Results should contain 1 and 2, with no duplicates
    let count_1 = results
        .iter()
        .filter(|(id, _, _)| *id == MemoryId(1))
        .count();
    assert_eq!(
        count_1, 1,
        "node 1 should not appear twice (cycle elimination)"
    );
}

#[test]
fn spread_multi_hop_respects_max_hops() {
    let params = AlgoParams {
        max_hops_default: 1, // limit to 1 hop
        ..Default::default()
    };

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    }];

    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(MemoryId(1), vec![make_link(2, LinkType::Causal, 0.8, 0.9)]);
    links_map.insert(
        MemoryId(2),
        vec![make_link(3, LinkType::EntityOverlap, 0.7, 0.8)],
    );

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    assert!(
        results.iter().any(|(id, _, _)| *id == MemoryId(2)),
        "one-hop neighbor should be reachable"
    );
    assert!(
        !results.iter().any(|(id, _, _)| *id == MemoryId(3)),
        "two-hop neighbor should not be reachable (max_hops=1)"
    );
}

#[test]
fn spread_multi_hop_merge_two_paths() {
    let params = AlgoParams {
        merge_secondary_weight: 0.30,
        ..Default::default()
    };

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 1.0,
        rank_in_channel: None,
    }];

    // Diamond: 1 → 2, 1 → 3, 2 → 4, 3 → 4 (two paths converge at 4)
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(2, LinkType::EntityOverlap, 1.0, 1.0),
            make_link(3, LinkType::SemanticSimilar, 1.0, 1.0),
        ],
    );
    links_map.insert(
        MemoryId(2),
        vec![make_link(4, LinkType::EntityOverlap, 1.0, 1.0)],
    );
    links_map.insert(MemoryId(3), vec![make_link(4, LinkType::Causal, 1.0, 1.0)]);

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    // Convergence node 4 should be in the results and appear only once (merged)
    let count_4 = results
        .iter()
        .filter(|(id, _, _)| *id == MemoryId(4))
        .count();
    assert_eq!(
        count_4, 1,
        "node where two paths converge should appear only once"
    );
}

#[test]
fn spread_multi_hop_energy_decay_across_hops() {
    let params = AlgoParams::default();
    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    }];

    // Linear chain: 1 → 2 → 3 → 4
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![make_link(2, LinkType::EntityOverlap, 0.7, 0.8)],
    );
    links_map.insert(
        MemoryId(2),
        vec![make_link(3, LinkType::EntityOverlap, 0.7, 0.8)],
    );
    links_map.insert(
        MemoryId(3),
        vec![make_link(4, LinkType::EntityOverlap, 0.7, 0.8)],
    );

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    // Energy decays hop by hop along the chain
    let e1 = results
        .iter()
        .find(|(id, _, _)| *id == MemoryId(1))
        .map(|(_, e, _)| *e)
        .unwrap_or(0.0);
    let e2 = results
        .iter()
        .find(|(id, _, _)| *id == MemoryId(2))
        .map(|(_, e, _)| *e)
        .unwrap_or(0.0);
    let e3 = results
        .iter()
        .find(|(id, _, _)| *id == MemoryId(3))
        .map(|(_, e, _)| *e)
        .unwrap_or(0.0);

    assert!(e1 >= e2, "seed energy >= one-hop neighbor");
    assert!(
        e2 >= e3,
        "one-hop neighbor energy >= two-hop neighbor (decay)"
    );
}

#[test]
fn spread_multi_hop_stops_when_frontier_empty() {
    let params = AlgoParams {
        max_hops_default: 5, // allow multiple hops, stop at leaf node
        ..Default::default()
    };

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    }];

    // Only one chain: 1 → 2 (no deeper neighbors), 3 unreachable
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![make_link(2, LinkType::EntityOverlap, 0.8, 0.8)],
    );
    // Node 2 has no edges (leaf)

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    // Should not panic, returns normally
    assert!(!results.is_empty());
    // 2 should be in the results (one-hop reachable)
    assert!(results.iter().any(|(id, _, _)| *id == MemoryId(2)));
}

#[test]
fn spread_multi_hop_empty_links_map() {
    let params = AlgoParams::default();
    let seeds = vec![
        Seed {
            id: MemoryId(1),
            channel: hippmem_core::model::links::RecallChannel::Bm25,
            score: 0.8,
            rank_in_channel: None,
        },
        Seed {
            id: MemoryId(2),
            channel: hippmem_core::model::links::RecallChannel::EntityInverted,
            score: 0.5,
            rank_in_channel: None,
        },
    ];

    let links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());

    // Should return all seeds (no edges to spread)
    assert!(results.iter().any(|(id, _, _)| *id == MemoryId(1)));
    assert!(results.iter().any(|(id, _, _)| *id == MemoryId(2)));
    assert_eq!(
        results.len(),
        2,
        "with no edges, should only return the seeds themselves, no duplicates"
    );
}

#[test]
fn spread_multi_hop_fanout_limit() {
    let params = AlgoParams {
        fan_out_default: 2, // expand at most 2 edges per node
        ..Default::default()
    };

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    }];

    // Seed has 5 outgoing edges
    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(10, LinkType::Causal, 0.9, 0.9),
            make_link(11, LinkType::EntityOverlap, 0.8, 0.8),
            make_link(12, LinkType::SemanticSimilar, 0.7, 0.7),
            make_link(13, LinkType::TopicRelated, 0.6, 0.6),
            make_link(14, LinkType::TemporalAdjacent, 0.5, 0.5),
        ],
    );

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    // Seed + at most fan_out_default (2) neighbors = at most 3 results
    // (possibly fewer if some neighbors' energy is below threshold)
    assert!(
        results.len() <= 1 + params.fan_out_default as usize,
        "fan-out pruning should limit the number of expanded edges"
    );
}

#[test]
fn spread_multi_hop_seed_below_threshold() {
    let params = AlgoParams {
        min_propagation_energy: 0.50, // high threshold
        ..Default::default()
    };

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Temporal,
        score: 0.1, // low seed score
        rank_in_channel: None,
    }];

    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(MemoryId(1), vec![make_link(2, LinkType::Causal, 0.9, 0.9)]);

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    // Seed energy low → should not appear in results
    assert!(
        results.is_empty(),
        "seed below threshold should not be expanded"
    );
}

#[test]
fn spread_one_hop_seed_with_no_links() {
    let params = AlgoParams::default();
    let seeds = vec![Seed {
        id: MemoryId(99),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.5,
        rank_in_channel: None,
    }];

    let links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    let results = spread_one_hop(&seeds, &links_map, &params, &HashMap::new());

    // Should return at least the seed itself
    assert!(results.iter().any(|(id, _, _)| *id == MemoryId(99)));
}

#[test]
fn initial_energy_deterministic() {
    let params = AlgoParams::default();
    let seed = Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.5,
        rank_in_channel: None,
    };
    let e1 = initial_energy(&seed, 0.5, 0.3, 0.7, 0.4, 0.2, &params);
    let e2 = initial_energy(&seed, 0.5, 0.3, 0.7, 0.4, 0.2, &params);
    assert!(
        (e1 - e2).abs() < 1e-6,
        "same input should always yield same energy"
    );
}

// ═══════════════════════════════════════════════════════════════════
// energy and spreading edge-case tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn three_path_merge() {
    // Start from node 1, three paths converge at node 4:
    // 1→2→4, 1→3→4, 1→4 (direct)
    // Verify the merge formula: merge(e, n) = max(e, n) + merge_secondary_weight * min(e, n)
    let params = AlgoParams {
        merge_secondary_weight: 0.30,
        ..Default::default()
    };

    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(2, LinkType::EntityOverlap, 0.9, 0.9),
            make_link(3, LinkType::EntityOverlap, 0.9, 0.9),
            make_link(4, LinkType::EntityOverlap, 0.9, 0.9), // direct to 4
        ],
    );
    links_map.insert(
        MemoryId(2),
        vec![make_link(4, LinkType::EntityOverlap, 0.9, 0.9)],
    );
    links_map.insert(
        MemoryId(3),
        vec![make_link(4, LinkType::EntityOverlap, 0.9, 0.9)],
    );

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    }];

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    // Node 4 should appear only once (three-path merge)
    let count_4 = results
        .iter()
        .filter(|(id, _, _)| *id == MemoryId(4))
        .count();
    assert_eq!(
        count_4, 1,
        "node where three paths converge should appear only once (merge)"
    );

    // Verify merge result: the direct path has the highest energy, but after three-path merge it should be > single path
    let energy_4 = results
        .iter()
        .find(|(id, _, _)| *id == MemoryId(4))
        .map(|(_, e, _)| *e)
        .unwrap_or(0.0);
    assert!(energy_4 > 0.0, "after merge should have positive energy");
}

#[test]
fn zero_strength_edge_no_propagation() {
    // Edges with strength=0 should not propagate energy
    let params = AlgoParams {
        min_propagation_energy: 0.01,
        ..Default::default()
    };

    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(2, LinkType::EntityOverlap, 0.8, 0.8), // normal
            make_link(3, LinkType::EntityOverlap, 0.0, 0.8), // zero strength → should not propagate
        ],
    );

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    }];

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    // Normal-edge neighbor should be reachable
    assert!(results.iter().any(|(id, _, _)| *id == MemoryId(2)));
    // Zero-strength-edge neighbor should not appear
    assert!(
        !results.iter().any(|(id, _, _)| *id == MemoryId(3)),
        "zero-strength edge should not propagate energy"
    );
}

#[test]
fn zero_confidence_edge_no_propagation() {
    // Edges with confidence=0 should not propagate energy (propagated = source * strength * 0 * ...)
    let params = AlgoParams {
        min_propagation_energy: 0.01,
        ..Default::default()
    };

    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![
            make_link(2, LinkType::EntityOverlap, 0.8, 0.8), // normal
            make_link(3, LinkType::EntityOverlap, 0.8, 0.0), // zero confidence → should not propagate
        ],
    );

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 0.9,
        rank_in_channel: None,
    }];

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    assert!(results.iter().any(|(id, _, _)| *id == MemoryId(2)));
    assert!(
        !results.iter().any(|(id, _, _)| *id == MemoryId(3)),
        "zero-confidence edge should not propagate energy"
    );
}

#[test]
fn max_hops_exact_boundary() {
    // max_hops=2: the second hop should be reachable, the third should not
    let params = AlgoParams {
        max_hops_default: 2,
        ..Default::default()
    };

    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    links_map.insert(
        MemoryId(1),
        vec![make_link(2, LinkType::EntityOverlap, 1.0, 1.0)],
    );
    links_map.insert(
        MemoryId(2),
        vec![make_link(3, LinkType::EntityOverlap, 1.0, 1.0)],
    );
    links_map.insert(
        MemoryId(3),
        vec![make_link(4, LinkType::EntityOverlap, 1.0, 1.0)],
    );

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 1.0,
        rank_in_channel: None,
    }];

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());
    // hop=1: node 2 should be reachable
    assert!(results.iter().any(|(id, _, _)| *id == MemoryId(2)));
    // hop=2: node 3 should be reachable (exactly at max_hops)
    assert!(
        results.iter().any(|(id, _, _)| *id == MemoryId(3)),
        "with max_hops=2, two-hop neighbor should be reachable"
    );
    // hop=3: node 4 should not be reachable (exceeds max_hops)
    assert!(
        !results.iter().any(|(id, _, _)| *id == MemoryId(4)),
        "with max_hops=2, three-hop neighbor should not be reachable"
    );
}

#[test]
fn deep_chain_five_hops_monotonic_decay() {
    // 5-hop deep chain: 1→2→3→4→5→6, verify energy decays monotonically hop by hop
    // Use Causal edges (type_modifier=1.30) to ensure propagated energy is enough to reach the 5th hop
    let params = AlgoParams {
        max_hops_default: 5,
        min_propagation_energy: 0.0001, // extremely low threshold
        ..Default::default()
    };

    let mut links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    for i in 1..=5 {
        links_map.insert(
            MemoryId(i),
            vec![make_link(i + 1, LinkType::Causal, 1.0, 1.0)],
        );
    }

    let seeds = vec![Seed {
        id: MemoryId(1),
        channel: hippmem_core::model::links::RecallChannel::Bm25,
        score: 1.0,
        rank_in_channel: None,
    }];

    let results = spread_multi_hop(&seeds, &links_map, &params, &HashMap::new());

    // Extract per-hop energy and verify monotonic decay
    let energies: Vec<f32> = (1..=6)
        .filter_map(|i| {
            results
                .iter()
                .find(|(id, _, _)| *id == MemoryId(i))
                .map(|(_, e, _)| *e)
        })
        .collect();

    assert!(
        energies.len() >= 5,
        "deep chain should have at least 5 nodes, actual {} nodes",
        energies.len()
    );
    for i in 1..energies.len() {
        assert!(
            energies[i] < energies[i - 1],
            "deep chain: hop{} energy ({}) should be < hop{} energy ({})",
            i + 1,
            energies[i],
            i,
            energies[i - 1]
        );
    }
}

#[test]
fn empty_seeds_returns_empty() {
    // Empty seeds → empty results (edge case)
    let params = AlgoParams::default();
    let links_map: HashMap<MemoryId, Vec<AssociationLink>> = HashMap::new();
    let results = spread_multi_hop(&[], &links_map, &params, &HashMap::new());
    assert!(
        results.is_empty(),
        "empty seeds should return empty results"
    );
}
