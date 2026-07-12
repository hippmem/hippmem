//! Engine::write — write API assembly.
//!
//! Corresponds to 05#write, 09 §4.1. Wires the pure functions of the write pipeline
//! together with store persistence.

use crate::{Engine, EngineError, EngineResult, WriteMemoryInput, WriteMemoryOutput, WriteWarning};
use hippmem_core::hash::stable_hash64;
use hippmem_core::ids::{MemoryId, VectorId};
use hippmem_core::model::enums::ContentType;
use hippmem_core::model::links::SemanticSignature;
use hippmem_core::model::understanding::MemoryUnderstanding;
use hippmem_core::model::unit::{Language, MemoryContent, MemoryStage, MemoryUnit};
use hippmem_core::score::UnitScore;
use hippmem_core::time::{Clock, SystemClock};
use hippmem_model::deterministic::extract::DeterministicExtractor;
use hippmem_store::kv::InvertedIndex;
use hippmem_store::semantic::vector_index::BinaryIndex;
use hippmem_store::semantic::vector_index::VectorIndex;
use hippmem_write::edges::EdgeBuildParams;
use hippmem_write::keys::generate_keys;
use hippmem_write::staged::{raw_to_indexed, StagedWriteInput};
use hippmem_write::understanding::index_enriched_keys;

impl Engine {
    /// Writes a memory.
    pub fn write(&self, input: WriteMemoryInput) -> EngineResult<WriteMemoryOutput> {
        let memory_id = MemoryId::generate();
        write_internal(self, memory_id, input, false, None)
    }

    /// Batch write: chunks calls to the embedding API, then processes each entry.
    ///
    /// Significantly reduces API round trips compared to per-entry write
    /// (DashScope limits each batch to ≤10 entries).
    pub fn write_batch(
        &self,
        inputs: Vec<WriteMemoryInput>,
    ) -> EngineResult<Vec<WriteMemoryOutput>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        let n = inputs.len();
        const CHUNK_SIZE: usize = 10; // DashScope text-embedding-v4 batch limit

        // Chunked embedding: one API call per CHUNK_SIZE entries
        let mut embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(n);
        let texts: Vec<String> = inputs.iter().map(|inp| inp.content.clone()).collect();

        for chunk in texts.chunks(CHUNK_SIZE) {
            match self.embedder.embed_sync(chunk) {
                Ok(vectors) => {
                    for v in vectors {
                        embeddings.push(Some(v));
                    }
                }
                Err(_e) => {
                    // Embedding failure: do not block write; degrade to no dense vector
                    embeddings.resize(embeddings.len() + chunk.len(), None);
                }
            }
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        for (input, embedding) in inputs.into_iter().zip(embeddings) {
            let memory_id = MemoryId::generate();
            let output = write_internal(self, memory_id, input, false, embedding)?;
            outputs.push(output);
        }
        Ok(outputs)
    }
}

/// Write pipeline core logic (shared by write/reindex/write_batch).
///
/// - `id`: the MemoryId to use (write generates a new ID, reindex reuses the original ID).
/// - `input`: write input.
/// - `skip_memory_log`: true when called by Reindex (the record already exists in MEMORY_LOG, constitution C7).
/// - `precomputed_embedding`: precomputed vector provided for batch writes, avoiding duplicate API calls.
pub(crate) fn write_internal(
    engine: &Engine,
    memory_id: MemoryId,
    input: WriteMemoryInput,
    skip_memory_log: bool,
    precomputed_embedding: Option<Vec<f32>>,
) -> EngineResult<WriteMemoryOutput> {
    let clock = SystemClock;
    let _now = clock.now();

    // 2. Build MemoryContent
    let content = MemoryContent {
        raw: input.content.clone(),
        summary: None,
        normalized: None,
        language: Language::Zh,
        content_type: input.content_type.unwrap_or(ContentType::UserStatement),
    };

    // 3. Extract understanding (degraded backend, synchronous, uses the global JIEBA instance)
    let extractor = DeterministicExtractor;
    let (understanding, mut warnings) = match extractor.extract_sync_immediate(&content) {
        Ok(imm) => {
            let u = MemoryUnderstanding {
                entities: imm.entities,
                topics: imm.topics,
                causal_claims: imm.explicit_causals,
                goals: vec![],
                decisions: vec![],
                preferences: vec![],
                emotions: vec![],
                events: vec![],
                contradictions: vec![],
                importance: input
                    .importance_hint
                    .map(UnitScore::new)
                    .unwrap_or(imm.importance),
                confidence: UnitScore::new(0.5),
            };
            (u, vec![])
        }
        Err(_e) => (
            MemoryUnderstanding {
                entities: vec![],
                topics: vec![],
                causal_claims: vec![],
                goals: vec![],
                decisions: vec![],
                preferences: vec![],
                emotions: vec![],
                events: vec![],
                contradictions: vec![],
                importance: UnitScore::new(0.0),
                confidence: UnitScore::new(0.0),
            },
            vec![WriteWarning::ExtractorDegraded],
        ),
    };

    // 4. Embedding (config-driven Embedder backend, synchronous)
    let mut semantic = build_semantic_signature(&input.content);
    // Generate dense vector and insert into FlatVectorIndex (SemanticDense channel, 03 §4.5)
    {
        if let Some(vector) = precomputed_embedding {
            // Batch mode: use precomputed embedding, skip API call
            let vector_id = memory_id.0;
            let mut idx = engine.dense_vector_index.lock();
            let _ = idx.insert(vector_id, &vector);
            semantic.dense_embedding_ref = Some(VectorId(vector_id));
        } else {
            // Per-entry mode: standalone embedding API call
            let texts = vec![input.content.clone()];
            if let Ok(vectors) = engine.embedder.embed_sync(&texts) {
                if let Some(vector) = vectors.first() {
                    let vector_id = memory_id.0;
                    let mut idx = engine.dense_vector_index.lock();
                    let _ = idx.insert(vector_id, vector);
                    semantic.dense_embedding_ref = Some(VectorId(vector_id));
                }
            }
        }
    }
    // If embedder fails, keep dense_embedding_ref=None (SemanticDense channel empty, write not blocked)
    if semantic.dense_embedding_ref.is_none() {
        warnings.push(WriteWarning::EmbeddingDeferred);
    }

    // 4b. Insert binary_code into the binary code index (for SemanticBinary channel Hamming recall)
    {
        let bc_bytes = binary_code_to_bytes(&semantic.binary_code);
        let mut idx = engine.binary_code_index.lock();
        let _ = idx.insert(memory_id.0, &bc_bytes);
    }

    // 5. Generate AssociationKeys (4 args: content, understanding, context, semantic)
    let keys = generate_keys(&content, &understanding, &input.context, &semantic)
        .map_err(|e| EngineError::Internal(format!("generate_keys: {}", e)))?;

    // 6. Recall candidate existing memories from the store index
    let inverted = InvertedIndex::new(engine.store.db_arc());
    let candidate_ids = discover_candidates(&keys, &inverted);
    let existing_units = load_memory_units(&engine.store.db_arc(), &candidate_ids);

    // 7. Call raw_to_indexed
    let staged_input = StagedWriteInput {
        id: memory_id,
        content: content.clone(),
        understanding: understanding.clone(),
        context: input.context.clone(),
        semantic,
    };

    // Build edge params from AlgoParams (configurable, not hardcoded)
    let algo = engine.params.read();
    let edge_params = EdgeBuildParams {
        strong_threshold: algo.strong_edge_threshold,
        strong_max: algo.strong_edge_max as usize,
        weak_max: algo.weak_edge_max as usize,
        min_score: algo.edge_build_min_score,
        observation_max: algo.observation_enter_max,
        max_candidates: 30, // Limit the number of edge-building candidates to control O(n²) cost
    };
    let staged_output = raw_to_indexed(staged_input, &existing_units, &edge_params, &algo)
        .map_err(|e| EngineError::Internal(format!("raw_to_indexed: {}", e)))?;

    let unit = staged_output.unit;

    // 8. Persist to store (single redb transaction: memory_log + kv + inverted index + graph)
    let bincode_unit = bincode::serde::encode_to_vec(&unit, bincode::config::standard())
        .map_err(|e| EngineError::Internal(e.to_string()))?;
    let bincode_links =
        bincode::serde::encode_to_vec(&staged_output.created_links, bincode::config::standard())
            .map_err(|e| EngineError::Internal(e.to_string()))?;

    hippmem_store::kv::persist_memory_unit(
        engine.store.db_arc(),
        memory_id.0,
        &bincode_unit,
        &bincode_links,
        &keys.entity_keys,
        &keys.topic_keys,
        &keys.temporal_keys,
        &keys.goal_keys,
        &keys.event_keys,
        &keys.causal_keys,
        skip_memory_log,
    )
    .map_err(|e| EngineError::Store(e.to_string()))?;

    // Write to the Tantivy fulltext index (for BM25 channel recall)
    // commit is auto-batched by FulltextIndex internally based on commit_every
    {
        let tokens = hippmem_core::hash::tokenize(&input.content, "zh");
        let mut ft = engine.fulltext_index.lock();
        ft.add_document_tokenized(memory_id.0, &tokens)
            .map_err(|e| EngineError::Store(format!("Tantivy add_document: {}", e)))?;
    }

    // link_overlay was already written in persist_memory_unit within the single transaction

    // Mark strong semantic dimensions as deferred
    warnings.push(WriteWarning::StrongDimsDeferred);

    // Synchronous enrich: complete strong semantic dimensions
    let mut enriched_unit = unit.clone();
    crate::runtime::run_enrich_sync(&mut enriched_unit);

    // enriched→index closure: write the newly produced goal/event/causal keys from enrich into the inverted index
    index_enriched_keys(&enriched_unit, &inverted, memory_id.0)
        .map_err(|e| EngineError::Internal(format!("index_enriched_keys: {}", e)))?;

    let re_bincode = bincode::serde::encode_to_vec(&enriched_unit, bincode::config::standard())
        .map_err(|e| EngineError::Internal(e.to_string()))?;
    // After enrich, only update memory_kv (do not rewrite memory_log/link_overlay/inverted index)
    hippmem_store::kv::KvStore::new(engine.store.db_arc())
        .put(memory_id.0, &re_bincode)
        .map_err(|e| EngineError::Store(e.to_string()))?;

    Ok(WriteMemoryOutput {
        memory_id,
        stage_reached: MemoryStage::Indexed,
        created_links: staged_output.created_links,
        understanding,
        warnings,
    })
}

// ── Helpers ──

/// Converts binary_code [u64;2] to 16 bytes (Little Endian) for BinaryCodeIndex Hamming search.
fn binary_code_to_bytes(bc: &[u64; 2]) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&bc[0].to_le_bytes());
    bytes[8..].copy_from_slice(&bc[1].to_le_bytes());
    bytes
}

fn build_semantic_signature(text: &str) -> SemanticSignature {
    let sim0 = stable_hash64(text);
    let sim1 = stable_hash64(&format!("{}_1", text));
    let sim2 = stable_hash64(&format!("{}_2", text));
    let sim3 = stable_hash64(&format!("{}_3", text));
    let bc0 = stable_hash64(&format!("bc_0_{}", text));
    let bc1 = stable_hash64(&format!("bc_1_{}", text));

    let mut minhash = [0u32; 16];
    for (i, v) in minhash.iter_mut().enumerate() {
        *v = stable_hash64(&format!("mh_{}_{}", i, text)) as u32;
    }

    SemanticSignature {
        lexical_simhash: [sim0, sim1, sim2, sim3],
        dense_embedding_ref: None,
        binary_code: [bc0, bc1],
        topic_minhash: minhash,
    }
}

fn discover_candidates(
    keys: &hippmem_core::model::links::AssociationKeys,
    inverted: &InvertedIndex,
) -> Vec<MemoryId> {
    let mut ids = std::collections::HashSet::new();
    for ek in &keys.entity_keys {
        if let Ok(hits) = inverted.get_entity(ek) {
            for id in hits {
                ids.insert(MemoryId(id));
            }
        }
    }
    for tk in &keys.topic_keys {
        if let Ok(hits) = inverted.get_topic(tk) {
            for id in hits {
                ids.insert(MemoryId(id));
            }
        }
    }
    for tk in &keys.temporal_keys {
        if let Ok(hits) = inverted.get_temporal(tk) {
            for id in hits {
                ids.insert(MemoryId(id));
            }
        }
    }
    ids.into_iter().collect()
}

fn load_memory_units(db: &std::sync::Arc<redb::Database>, ids: &[MemoryId]) -> Vec<MemoryUnit> {
    let kv = hippmem_store::kv::KvStore::new(std::sync::Arc::clone(db));
    ids.iter()
        .filter_map(|mid| {
            kv.get(&mid.0).ok().flatten().and_then(|data| {
                bincode::serde::decode_from_slice::<MemoryUnit, _>(
                    &data,
                    bincode::config::standard(),
                )
                .ok()
                .map(|(unit, _)| unit)
            })
        })
        .collect()
}
