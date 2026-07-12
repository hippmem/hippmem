//! AlgoParams configuration struct: centralized definition of all algorithm parameters and defaults.
//!
//! Corresponds to the 03#0 parameter table, ADR-008 configuration scheme (figment: defaults < file < env vars).

use crate::model::links::RecallChannel;
use figment::providers::Env;
use figment::Figment;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Algorithm parameter configuration: the Rust representation of the 03 §0 parameter table.
///
/// All field names match the 03 table. Defaults are compile-time constants that can be overridden by a TOML file or env vars.
/// Float fields do not derive `Eq/Hash`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlgoParams {
    // ── Association weights (§2) ──
    /// Entity dimension weight, default 0.20
    pub w_entity: f32,
    /// Semantic dimension weight, default 0.18
    pub w_semantic: f32,
    /// Temporal dimension weight, default 0.10
    pub w_temporal: f32,
    /// Topic dimension weight, default 0.10
    pub w_topic: f32,
    /// Goal dimension weight, default 0.12
    pub w_goal: f32,
    /// Event dimension weight, default 0.10
    pub w_event: f32,
    /// Emotion dimension weight, default 0.05
    pub w_emotion: f32,
    /// Causal dimension weight, default 0.10
    pub w_causal: f32,
    /// Context dimension weight, default 0.03
    pub w_context: f32,
    /// Importance dimension weight, default 0.02
    pub w_importance: f32,

    // ── Multi-dimensional cross-check (§2) ──
    /// Multi-dimension bonus, default 0.15
    pub multi_dim_bonus: f32,
    /// Minimum hit dimension count to trigger the bonus, default 3
    pub multi_dim_min_dims: u32,

    // ── Edge building (§3) ──
    /// Strong/weak edge boundary threshold, default 0.55
    pub strong_edge_threshold: f32,
    /// Maximum strong edge count, default 8
    pub strong_edge_max: u32,
    /// Minimum strong edge count, default 3
    pub strong_edge_min: u32,
    /// Maximum weak edge count, default 24
    pub weak_edge_max: u32,
    /// Minimum edge build score, default 0.25
    pub edge_build_min_score: f32,
    /// Observation zone entry upper bound, default 0.55
    pub observation_enter_max: f32,

    // ── Initial edge strength (§3) ──
    /// Initial strength of a new edge, default 0.40
    pub init_strength_base: f32,

    // ── Activation initial values (§4) ──
    /// Query match coefficient, default 0.40
    pub a_query_match: f32,
    /// Context match coefficient, default 0.20
    pub b_context_match: f32,
    /// Importance coefficient, default 0.15
    pub c_importance: f32,
    /// Freshness coefficient, default 0.15
    pub d_freshness: f32,
    /// Reliability coefficient, default 0.10
    pub e_reliability: f32,

    // ── Spreading (§4) ──
    /// Per-hop decay factor, default 0.55
    pub decay_factor: f32,
    /// Minimum propagation energy, default 0.05
    pub min_propagation_energy: f32,
    /// Per-node per-link-type fan-out cap, default 6
    pub fan_out_default: u32,
    /// Default hop count for Balanced mode, default 2
    pub max_hops_default: u32,
    /// Seed energy cap, default 1.0
    pub seed_energy_cap: f32,

    // ── Hebbian (§6) ──
    /// Hebbian learning rate, default 0.08
    pub hebbian_learning_rate: f32,
    /// Co-activation edge creation threshold, default 3
    pub coactivation_create_threshold: u32,
    /// Strength cap, default 1.0
    pub strength_max: f32,

    // ── Decay (§7) ──
    /// Decay multiplier per consolidation cycle, default 0.97
    pub decay_per_cycle: f32,
    /// Minimum retained strength, default 0.12
    pub min_retained_strength: f32,
    /// Per-node weak edge cap, default 32
    pub weak_degree_limit: u32,
    /// Per-node total out-degree cap, default 64
    pub node_degree_limit: u32,
    /// Observation window (milliseconds), default 14 days = 1_209_600_000 ms
    pub observation_window_ms: i64,
    /// Stale unactivated threshold (milliseconds), default 30 days = 2_592_000_000 ms
    pub stale_unactivated_ms: i64,

    // ── Reranking (§4.5) ──
    /// Candidate cap entering the reranker, default 50
    pub rerank_top_n: u32,
    /// Seed count cap per recall channel, default 20
    pub seed_per_channel: u32,
    /// BM25 score normalization divisor: `tanh(bm25_raw / factor)` maps unbounded BM25 scores to [0,1], default 2.0
    pub bm25_norm_factor: f32,

    // ── RRF channel precision weights (V9 §4.5.2) ──
    /// BM25 precision weight: natively carries IDF; exact term match is a strong signal
    pub rrf_w_bm25: f32,
    /// Entity precision weight: named-entity matching is high precision
    pub rrf_w_entity: f32,
    /// SemanticDense precision weight: topic-level signal; cannot distinguish answers from merely related content
    pub rrf_w_semantic_dense: f32,
    /// SemanticBinary precision weight: binary code is less precise than dense vectors
    pub rrf_w_semantic_binary: f32,
    /// Topic precision weight: coarse bag-of-words matching without IDF
    pub rrf_w_topic: f32, // default 0.3: bag-of-words without IDF, down-weighted
    /// Temporal precision weight: time-bucket overlap is the weakest signal
    pub rrf_w_temporal: f32,
    /// Goal precision weight: rule-based matching, medium precision
    pub rrf_w_goal: f32,
    /// Event precision weight: rule-based matching, medium precision
    pub rrf_w_event: f32,
    /// Causal precision weight: rule-based matching, medium precision
    pub rrf_w_causal: f32,
    /// RecentActivation precision weight: time-recency != content-relevance
    pub rrf_w_recent: f32,

    // ── Channel energy coefficients (§4.1, V9 deprecated) ──
    /// [V9 deprecated] BM25 channel initial energy coefficient, superseded by rrf_w_bm25
    pub channel_coeff_bm25: f32,
    /// SemanticDense dense-vector channel initial energy coefficient, default 1.0
    pub channel_coeff_semantic_dense: f32,
    /// SemanticBinary binary-code channel initial energy coefficient, default 1.0
    pub channel_coeff_semantic_binary: f32,
    /// EntityInverted entity inverted-index channel initial energy coefficient, default 1.0
    pub channel_coeff_entity: f32,
    /// TopicCluster topic channel initial energy coefficient, default 1.0
    pub channel_coeff_topic: f32,
    /// Temporal time-proximity channel initial energy coefficient, default 1.0
    pub channel_coeff_temporal: f32,
    /// Goal channel initial energy coefficient, default 1.0
    pub channel_coeff_goal: f32,
    /// Event channel initial energy coefficient, default 1.0
    pub channel_coeff_event: f32,
    /// Causal channel initial energy coefficient, default 1.0
    pub channel_coeff_causal: f32,
    /// RecentActivation recently-active channel initial energy coefficient, default 1.0
    pub channel_coeff_recent: f32,

    // ── Thresholds / cold start (§2.3) ──
    /// Cold-start period memory count boundary, default 500
    pub cold_start_count: u32,
    /// Single semantic dimension dominance penalty, default 0.60
    pub single_semantic_penalty: f32,
    /// temporal_score decay τ (days), default 7
    pub tau_temporal_days: u32,
    /// freshness decay τ (days), default 30
    pub tau_fresh_days: u32,
    /// LowConfidence warning threshold, default 0.35
    pub low_conf_threshold: f32,
    /// StaleFreshness warning threshold, default 0.20
    pub stale_threshold: f32,
    /// Dimension "hit" decision threshold, default 0.20
    pub dim_hit_threshold: f32,
    /// Multi-path energy merge secondary-term weight (used in spreading paths), default 0.30
    pub merge_secondary_weight: f32,
    /// RRF rank fusion parameter k, default 1.0.
    /// When k > 0, seed fusion uses RRF: score(id) = Σ_c 1/(k + rank_c(id)).
    /// When k ≤ 0, it degenerates to winner-take-all.
    /// Replaces the legacy channel_coeff and seed_merge_weight.
    pub rrf_k: f32,
    /// **[V9 deprecated]** Multi-seed fusion consensus weight, superseded by rrf_k.
    /// Field kept for compilation compatibility only; the retrieval path no longer reads it.
    pub seed_merge_weight: f32,

    // ── Compaction (§8) ──
    /// Similar-memory count that triggers summarization, default 12
    pub summary_trigger_count: u32,

    // ── Co-activation records ──
    /// Maximum neighbor count retained by ActivationState, default 16
    pub co_activation_keep: u32,
}

impl Default for AlgoParams {
    fn default() -> Self {
        Self {
            // Association weights
            w_entity: 0.20,
            w_semantic: 0.18,
            w_temporal: 0.10,
            w_topic: 0.10,
            w_goal: 0.12,
            w_event: 0.10,
            w_emotion: 0.05,
            w_causal: 0.10,
            w_context: 0.03,
            w_importance: 0.02,
            // Multi-dimensional cross-check
            multi_dim_bonus: 0.15,
            multi_dim_min_dims: 3,
            // Edge building
            strong_edge_threshold: 0.55,
            strong_edge_max: 8,
            strong_edge_min: 3,
            weak_edge_max: 24,
            edge_build_min_score: 0.25,
            observation_enter_max: 0.55,
            // Initial edge strength
            init_strength_base: 0.40,
            // Activation initial values
            a_query_match: 0.40,
            b_context_match: 0.20,
            c_importance: 0.60,
            d_freshness: 0.15,
            e_reliability: 0.10,
            // Spreading
            decay_factor: 0.55,
            min_propagation_energy: 0.05,
            fan_out_default: 6,
            max_hops_default: 2,
            seed_energy_cap: 1.0,
            // Hebbian
            hebbian_learning_rate: 0.08,
            coactivation_create_threshold: 3,
            strength_max: 1.0,
            // Decay
            decay_per_cycle: 0.97,
            min_retained_strength: 0.12,
            weak_degree_limit: 32,
            node_degree_limit: 64,
            observation_window_ms: 1_209_600_000, // 14 days
            stale_unactivated_ms: 2_592_000_000,  // 30 days
            // Reranking
            rerank_top_n: 50,
            seed_per_channel: 20,
            bm25_norm_factor: 2.0,
            // RRF channel precision weights (V9)
            // Only Topic (coarse bag-of-words) and Temporal (time buckets) are down-weighted for lacking IDF.
            // Entity/SD/BM25 etc. channels have sufficiently precise internal scoring; no extra weighting.
            rrf_w_bm25: 1.0,
            rrf_w_entity: 1.0,
            rrf_w_semantic_dense: 1.0,
            rrf_w_semantic_binary: 1.0,
            rrf_w_topic: 0.3,
            rrf_w_temporal: 0.3,
            rrf_w_goal: 1.0,
            rrf_w_event: 1.0,
            rrf_w_causal: 1.0,
            rrf_w_recent: 1.0,
            // Channel energy coefficients (V9 deprecated)
            channel_coeff_bm25: 1.0,
            channel_coeff_semantic_dense: 1.2,
            channel_coeff_semantic_binary: 0.6,
            channel_coeff_entity: 1.0,
            channel_coeff_topic: 1.0,
            channel_coeff_temporal: 1.0,
            channel_coeff_goal: 1.0,
            channel_coeff_event: 1.0,
            channel_coeff_causal: 1.0,
            channel_coeff_recent: 1.0,
            // Thresholds / cold start
            cold_start_count: 500,
            single_semantic_penalty: 0.60,
            tau_temporal_days: 7,
            tau_fresh_days: 30,
            low_conf_threshold: 0.35,
            stale_threshold: 0.20,
            dim_hit_threshold: 0.20,
            merge_secondary_weight: 0.30,
            rrf_k: 1.0,
            seed_merge_weight: 0.80, // V9 deprecated, kept for compilation
            // Compaction
            summary_trigger_count: 12,
            // Co-activation
            co_activation_keep: 16,
        }
    }
}

impl AlgoParams {
    /// Loads configuration via figment layering: defaults < env vars.
    ///
    /// Env var prefix: `HIPPMEM__` (double-underscore separates levels, e.g. `HIPPMEM__MAX_HOPS_DEFAULT=3`).
    /// TOML file support is reserved.
    #[allow(clippy::result_large_err)] // One-time initialization; Err size is irrelevant
    pub fn load() -> Result<Self, figment::Error> {
        Figment::new()
            .merge(figment::providers::Serialized::defaults(Self::default()))
            .merge(Env::prefixed("HIPPMEM__").split("__"))
            .extract()
    }

    /// Returns the initial energy coefficient for the given recall channel (§4.1).
    /// V9: returns the RRF precision weight `w_c` of channel c.
    ///
    /// Weights are determined by the inherent information precision of each channel's mechanism, not by empirical tuning.
    /// Unknown channels safely fall back to 1.0.
    pub fn rrf_channel_weight(&self, channel: RecallChannel) -> f32 {
        match channel {
            RecallChannel::Bm25 => self.rrf_w_bm25,
            RecallChannel::EntityInverted => self.rrf_w_entity,
            RecallChannel::SemanticDense => self.rrf_w_semantic_dense,
            RecallChannel::SemanticBinary => self.rrf_w_semantic_binary,
            RecallChannel::TopicCluster => self.rrf_w_topic,
            RecallChannel::Temporal => self.rrf_w_temporal,
            RecallChannel::Goal => self.rrf_w_goal,
            RecallChannel::Event => self.rrf_w_event,
            RecallChannel::Causal => self.rrf_w_causal,
            RecallChannel::RecentActivation => self.rrf_w_recent,
            _ => 1.0,
        }
    }

    /// [V9 deprecated] Per-channel energy coefficient, superseded by `rrf_channel_weight`.
    /// Retained for compilation compatibility only.
    pub fn channel_energy_coeff(&self, channel: RecallChannel) -> f32 {
        match channel {
            RecallChannel::Bm25 => self.channel_coeff_bm25,
            RecallChannel::SemanticDense => self.channel_coeff_semantic_dense,
            RecallChannel::SemanticBinary => self.channel_coeff_semantic_binary,
            RecallChannel::EntityInverted => self.channel_coeff_entity,
            RecallChannel::TopicCluster => self.channel_coeff_topic,
            RecallChannel::Temporal => self.channel_coeff_temporal,
            RecallChannel::Goal => self.channel_coeff_goal,
            RecallChannel::Event => self.channel_coeff_event,
            RecallChannel::Causal => self.channel_coeff_causal,
            RecallChannel::RecentActivation => self.channel_coeff_recent,
            // GraphSpreading is not a seed channel and needs no coefficient; future new channels safely fall back
            _ => 1.0,
        }
    }
}

// ── EmbedderConfig (V4) ──

/// Default embedding dimension.
fn default_embed_dim() -> usize {
    256
}

/// Default base URL for the OpenAI-compatible API.
fn default_openai_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

/// Embedder backend configuration: distinguished by the `provider` field.
///
/// Corresponds to 08#model-backends, V4 Embedding backend upgrade.
/// TOML example:
///
/// ```toml
/// # Deterministic fallback (default)
/// [embedder]
/// provider = "deterministic"
/// dimensions = 256
///
/// # DashScope online API
/// [embedder]
/// provider = "openai-compatible"
/// base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
/// model = "text-embedding-v4"
/// dimensions = 1024
///
/// # Offline ONNX (reserved, not yet implemented)
/// [embedder]
/// provider = "onnx"
/// model_name = "bge-small-zh-v1.5"
/// model_cache_dir = "/path/to/cache"
/// dimensions = 512
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "kebab-case")]
pub enum EmbedderConfig {
    /// Deterministic fallback backend: 256d SimHash, zero dependencies, CI default.
    Deterministic {
        /// Vector dimension, default 256.
        #[serde(default = "default_embed_dim")]
        dimensions: usize,
    },
    /// OpenAI-compatible API backend: supports DashScope / OpenAI / vLLM etc.
    #[serde(rename = "openai-compatible")]
    OpenAiCompatible {
        /// API base URL, default `"https://api.openai.com/v1"`.
        #[serde(default = "default_openai_base_url")]
        base_url: String,
        /// Model name, e.g. `"text-embedding-v4"` (DashScope) or `"text-embedding-3-small"` (OpenAI).
        model: String,
        /// API key, optional; read from env var `OPENAI_API_KEY` when not provided.
        #[serde(default)]
        api_key: Option<String>,
        /// Vector dimension, determined by the chosen model (DashScope=1024, OpenAI=1536).
        dimensions: usize,
    },
    /// Offline ONNX backend: local inference, no network required. (Reserved, not yet implemented)
    Onnx {
        /// ONNX model name, e.g. `"bge-small-zh-v1.5"`.
        model_name: String,
        /// Model cache directory.
        model_cache_dir: PathBuf,
        /// Vector dimension.
        dimensions: usize,
    },
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self::Deterministic {
            dimensions: default_embed_dim(),
        }
    }
}

impl EmbedderConfig {
    /// Returns the vector dimension specified by the current configuration.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Deterministic { dimensions } => *dimensions,
            Self::OpenAiCompatible { dimensions, .. } => *dimensions,
            Self::Onnx { dimensions, .. } => *dimensions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_are_consistent() {
        let p1 = AlgoParams::default();
        let p2 = AlgoParams::default();
        assert_eq!(p1.w_entity, p2.w_entity);
        assert_eq!(p1.max_hops_default, p2.max_hops_default);
    }

    #[test]
    fn load_without_config_file_returns_default() {
        let p = AlgoParams::load().unwrap_or_default();
        assert_eq!(p.fan_out_default, 6); // default value
    }

    // ── EmbedderConfig tests ──

    #[test]
    fn embedder_config_default_is_deterministic_256() {
        let cfg = EmbedderConfig::default();
        assert_eq!(cfg, EmbedderConfig::Deterministic { dimensions: 256 });
    }

    #[test]
    fn deserialize_deterministic_from_toml() {
        let toml_str = r#"
provider = "deterministic"
"#;
        let cfg: EmbedderConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg, EmbedderConfig::Deterministic { dimensions: 256 });
        // Round-trip: serialize then deserialize should be equivalent
        let serialized = toml::to_string(&cfg).unwrap();
        let cfg2: EmbedderConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn deserialize_deterministic_custom_dim() {
        let toml_str = r#"
provider = "deterministic"
dimensions = 512
"#;
        let cfg: EmbedderConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg, EmbedderConfig::Deterministic { dimensions: 512 });
    }

    #[test]
    fn deserialize_openai_compatible_full() {
        let toml_str = r#"
provider = "openai-compatible"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "text-embedding-v4"
api_key = "sk-test123"
dimensions = 1024
"#;
        let cfg: EmbedderConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            cfg,
            EmbedderConfig::OpenAiCompatible {
                base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".into(),
                model: "text-embedding-v4".into(),
                api_key: Some("sk-test123".into()),
                dimensions: 1024,
            }
        );
        // Round-trip
        let serialized = toml::to_string(&cfg).unwrap();
        let cfg2: EmbedderConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn deserialize_openai_compatible_minimal() {
        // Only specify provider + model + dimensions; base_url and api_key take defaults
        let toml_str = r#"
provider = "openai-compatible"
model = "text-embedding-3-small"
dimensions = 1536
"#;
        let cfg: EmbedderConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            cfg,
            EmbedderConfig::OpenAiCompatible {
                base_url: "https://api.openai.com/v1".into(),
                model: "text-embedding-3-small".into(),
                api_key: None,
                dimensions: 1536,
            }
        );
    }

    #[test]
    fn deserialize_onnx_from_toml() {
        let toml_str = r#"
provider = "onnx"
model_name = "bge-small-zh-v1.5"
model_cache_dir = "/home/user/.cache/models"
dimensions = 512
"#;
        let cfg: EmbedderConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.dimensions(), 512);
        // Round-trip
        let serialized = toml::to_string(&cfg).unwrap();
        let cfg2: EmbedderConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn embedder_config_variants_are_exhaustive() {
        // Compile-time verification: all variants can be handled via match
        fn match_config(c: &EmbedderConfig) -> &'static str {
            match c {
                EmbedderConfig::Deterministic { .. } => "deterministic",
                EmbedderConfig::OpenAiCompatible { .. } => "openai",
                EmbedderConfig::Onnx { .. } => "onnx",
            }
        }
        assert_eq!(match_config(&EmbedderConfig::default()), "deterministic");
        assert_eq!(
            match_config(&EmbedderConfig::Onnx {
                model_name: "test".into(),
                model_cache_dir: PathBuf::from("/tmp"),
                dimensions: 512,
            }),
            "onnx"
        );
    }
}
