//! acceptance test: AlgoParams configuration and defaults

use hippmem_core::config::AlgoParams;

// ── Defaults match the 03 §0 parameter table one by one ──

#[test]
fn default_weights_match_spec() {
    let p = AlgoParams::default();
    // Association weights (§2)
    assert!(
        (p.w_entity - 0.20).abs() < f32::EPSILON,
        "w_entity should be 0.20"
    );
    assert!(
        (p.w_semantic - 0.18).abs() < f32::EPSILON,
        "w_semantic should be 0.18"
    );
    assert!(
        (p.w_temporal - 0.10).abs() < f32::EPSILON,
        "w_temporal should be 0.10"
    );
    assert!(
        (p.w_topic - 0.10).abs() < f32::EPSILON,
        "w_topic should be 0.10"
    );
    assert!(
        (p.w_goal - 0.12).abs() < f32::EPSILON,
        "w_goal should be 0.12"
    );
    assert!(
        (p.w_event - 0.10).abs() < f32::EPSILON,
        "w_event should be 0.10"
    );
    assert!(
        (p.w_emotion - 0.05).abs() < f32::EPSILON,
        "w_emotion should be 0.05"
    );
    assert!(
        (p.w_causal - 0.10).abs() < f32::EPSILON,
        "w_causal should be 0.10"
    );
    assert!(
        (p.w_context - 0.03).abs() < f32::EPSILON,
        "w_context should be 0.03"
    );
    assert!(
        (p.w_importance - 0.02).abs() < f32::EPSILON,
        "w_importance should be 0.02"
    );
}

#[test]
fn default_multi_dim_match_spec() {
    let p = AlgoParams::default();
    assert!(
        (p.multi_dim_bonus - 0.15).abs() < f32::EPSILON,
        "multi_dim_bonus should be 0.15"
    );
    assert_eq!(p.multi_dim_min_dims, 3, "multi_dim_min_dims should be 3");
}

#[test]
fn default_edge_params_match_spec() {
    let p = AlgoParams::default();
    assert!(
        (p.strong_edge_threshold - 0.55).abs() < f32::EPSILON,
        "strong_edge_threshold should be 0.55"
    );
    assert_eq!(p.strong_edge_max, 8);
    assert_eq!(p.strong_edge_min, 3);
    assert_eq!(p.weak_edge_max, 24);
    assert!(
        (p.edge_build_min_score - 0.25).abs() < f32::EPSILON,
        "edge_build_min_score should be 0.25"
    );
    assert!(
        (p.observation_enter_max - 0.55).abs() < f32::EPSILON,
        "observation_enter_max should be 0.55"
    );
}

#[test]
fn default_init_strength_match_spec() {
    let p = AlgoParams::default();
    assert!(
        (p.init_strength_base - 0.40).abs() < f32::EPSILON,
        "init_strength_base should be 0.40"
    );
}

#[test]
fn default_activation_energy_match_spec() {
    let p = AlgoParams::default();
    assert!(
        (p.a_query_match - 0.40).abs() < f32::EPSILON,
        "a_query_match should be 0.40"
    );
    assert!(
        (p.b_context_match - 0.20).abs() < f32::EPSILON,
        "b_context_match should be 0.20"
    );
    assert!(
        (p.c_importance - 0.60).abs() < f32::EPSILON,
        "c_importance should be 0.60"
    );
    assert!(
        (p.d_freshness - 0.15).abs() < f32::EPSILON,
        "d_freshness should be 0.15"
    );
    assert!(
        (p.e_reliability - 0.10).abs() < f32::EPSILON,
        "e_reliability should be 0.10"
    );
}

#[test]
fn default_spreading_params_match_spec() {
    let p = AlgoParams::default();
    assert!(
        (p.decay_factor - 0.55).abs() < f32::EPSILON,
        "decay_factor should be 0.55"
    );
    assert!(
        (p.min_propagation_energy - 0.05).abs() < f32::EPSILON,
        "min_propagation_energy should be 0.05"
    );
    assert_eq!(p.fan_out_default, 6);
    assert_eq!(p.max_hops_default, 2);
    assert!((p.seed_energy_cap - 1.0).abs() < f32::EPSILON);
}

#[test]
fn default_hebbian_match_spec() {
    let p = AlgoParams::default();
    assert!(
        (p.hebbian_learning_rate - 0.08).abs() < f32::EPSILON,
        "hebbian_learning_rate should be 0.08"
    );
    assert_eq!(p.coactivation_create_threshold, 3);
    assert!((p.strength_max - 1.0).abs() < f32::EPSILON);
}

#[test]
fn default_decay_match_spec() {
    let p = AlgoParams::default();
    assert!(
        (p.decay_per_cycle - 0.97).abs() < f32::EPSILON,
        "decay_per_cycle should be 0.97"
    );
    assert!(
        (p.min_retained_strength - 0.12).abs() < f32::EPSILON,
        "min_retained_strength should be 0.12"
    );
    assert_eq!(p.weak_degree_limit, 32);
    assert_eq!(p.node_degree_limit, 64);
    assert_eq!(p.observation_window_ms, 1_209_600_000); // 14 days
    assert_eq!(p.stale_unactivated_ms, 2_592_000_000); // 30 days
}

#[test]
fn default_rerank_match_spec() {
    let p = AlgoParams::default();
    assert_eq!(p.rerank_top_n, 50);
    assert_eq!(p.seed_per_channel, 20);
}

#[test]
fn default_cold_start_match_spec() {
    let p = AlgoParams::default();
    assert_eq!(p.cold_start_count, 500);
    assert!(
        (p.single_semantic_penalty - 0.60).abs() < f32::EPSILON,
        "single_semantic_penalty should be 0.60"
    );
}

#[test]
fn default_temporal_params_match_spec() {
    let p = AlgoParams::default();
    assert_eq!(p.tau_temporal_days, 7);
    assert_eq!(p.tau_fresh_days, 30);
}

#[test]
fn default_thresholds_match_spec() {
    let p = AlgoParams::default();
    assert!(
        (p.low_conf_threshold - 0.35).abs() < f32::EPSILON,
        "low_conf_threshold should be 0.35"
    );
    assert!(
        (p.stale_threshold - 0.20).abs() < f32::EPSILON,
        "stale_threshold should be 0.20"
    );
    assert!(
        (p.dim_hit_threshold - 0.20).abs() < f32::EPSILON,
        "dim_hit_threshold should be 0.20"
    );
    assert!(
        (p.merge_secondary_weight - 0.30).abs() < f32::EPSILON,
        "merge_secondary_weight should be 0.30"
    );
}

#[test]
fn default_compaction_match_spec() {
    let p = AlgoParams::default();
    assert_eq!(p.summary_trigger_count, 12);
    assert_eq!(p.co_activation_keep, 16);
}

// ── figment layered override tests ──

#[test]
fn figment_default_layer_works() {
    let p = AlgoParams::default();
    assert_eq!(p.max_hops_default, 2); // default value
}

#[test]
fn figment_env_override_works() {
    // Override via environment variable
    std::env::set_var("HIPPMEM__MAX_HOPS_DEFAULT", "3");
    let p = AlgoParams::load().unwrap_or_default();
    assert_eq!(p.max_hops_default, 3);
    std::env::remove_var("HIPPMEM__MAX_HOPS_DEFAULT");
}
