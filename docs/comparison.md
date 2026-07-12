# Solution Comparison

> How is HIPPMEM different from other memory/retrieval solutions? When should you choose it?

---

## One-line positioning

| Solution | One line |
|----------|----------|
| **Vector database** (Pinecone/Milvus/Qdrant/Chroma) | Store vectors → search similar vectors |
| **Graph database** (Neo4j/ArangoDB) | Store nodes and edges → graph query traversal |
| **RAG framework** (LangChain/LlamaIndex) | Stitch retriever + LLM → Q&A |
| **Memory engine** (Mem0/Graphiti/Zep) | Long-term memory for AI agents |
| **HIPPMEM** | Discover associations at write time + spreading activation at retrieval + evolves with use |

---

## Feature matrix

| Capability | Vector DB | Graph DB | Mem0 | Graphiti | **HIPPMEM** |
|------------|:---------:|:--------:|:----:|:--------:|:-----------:|
| Full-text search (BM25) | ❌ | ❌ | ✅ | ❌ | ✅ |
| Semantic search (embedding) | ✅ | ❌ | ✅ | ✅ | ✅ |
| Entity extraction | ❌ | ❌ | ✅ | ✅ | ✅ |
| Auto association edge creation | ❌ | ❌ | ❌ | ✅ | ✅ |
| Spreading activation retrieval | ❌ | ❌ | ❌ | ❌ | ✅ |
| Explanation paths (trace) | ❌ | ❌ | ❌ | ❌ | ✅ |
| Hebbian reinforcement | ❌ | ❌ | ❌ | ❌ | ✅ |
| Decay-based forgetting | ❌ | ❌ | ❌ | ❌ | ✅ |
| Deterministic degradation (zero API) | ❌ | ❌ | ❌ | ❌ | ✅ |
| gRPC interface | Partial | Partial | ✅ | ✅ | ✅ |
| Embedded in a Rust process | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multi-language SDK | ✅ | ✅ | ✅ | ✅ | gRPC |
| Distributed / cluster | ✅ | ✅ | ✅ | ❌ | ❌ |

---

## When to choose HIPPMEM?

### ✅ Suitable scenarios

| Scenario | Why |
|----------|-----|
| **Long-term memory for AI agents** | Discovers associations at write time, returns explanation paths at retrieval, evolves with use |
| **Personal knowledge base / second brain** | Works offline; private data never leaves the local machine |
| **Embedded Rust projects** | Can be embedded directly as a crate, with zero network dependencies |
| **Systems needing "explainable retrieval"** | Every result carries an activation_trace + matched_dimensions |
| **Preference tracking / decision audit** | Auto-discovers preference drift, contradictions, causal chains |
| **Environments without GPU / API access** | The deterministic degradation backend covers all core capabilities |

### ❌ Unsuitable scenarios

| Scenario | Alternative |
|----------|-------------|
| Hundred-million-scale vector similarity search | Milvus / Qdrant |
| Complex graph queries (Cypher/Gremlin) needed | Neo4j |
| Need a Python SDK + 10-minute onboarding | Mem0 |
| Need cloud hosting / ops-free | Mem0 Platform / Pinecone |
| Multi-tenant / distributed | Planned |
| Mobile / browser / WASM | Wait for later versions |

---

## Detailed comparison with Mem0

Mem0 (48K ★) is the closest competitor to HIPPMEM.

| Dimension | Mem0 | HIPPMEM |
|-----------|------|---------|
| **Language** | Python (+ hosted platform) | Rust |
| **Onboarding difficulty** | `pip install mem0ai` and go | Requires building a Rust project |
| **Core model** | Vector graph (vector + graph + KV) | Native association graph (write-time discovery + spreading activation) |
| **Association discovery timing** | Mostly at retrieval time | At write time (when context is most complete) |
| **Retrieval mechanism** | Multi-store hybrid retrieval | Multi-channel seeds + spreading activation |
| **Explainability** | Limited | activation_trace + matched_dimensions |
| **Evolution** | Adaptive memory updates | Hebbian + decay + compaction + summarization |
| **Offline** | Needs API (unless a local LLM is configured) | Deterministic degradation, zero external dependencies |
| **Deployment** | Python library or hosted platform | Embedded in a Rust binary or gRPC sidecar |
| **Evaluation** | None built-in | 5 baselines + 10 task types + 50 corpora |

**Choose Mem0**: You write Python, need to ship in 5 minutes, and can accept calling external APIs.

**Choose HIPPMEM**: You write Rust, need same-process embedding, need offline/privacy, or want deeper association discovery and explanation.

---

## Detailed comparison with Graphiti

Graphiti (by the Zep team, 24K ★) is another memory engine, featuring a time-aware knowledge graph.

| Dimension | Graphiti | HIPPMEM |
|-----------|----------|---------|
| **Core model** | Temporal knowledge graph (episodes → entities + facts) | MemoryUnit + AssociationLink + spreading activation |
| **Time handling** | First-class: facts have validity windows | Time bucketing + Temporal edges |
| **Language** | Python | Rust |
| **Graph storage** | Neo4j / FalkorDB / Kuzu | redb (in-house graph store) |
| **Graph traversal** | Relies on an external graph database | In-house spreading traversal (constitution C2) |
| **LLM dependency** | Strong (for fact extraction) | Optional (has rule-based fallback extraction) |

**Choose Graphiti**: You need to track "when a fact became true → when it became false", and don't mind depending on Neo4j + an LLM.

**Choose HIPPMEM**: You want embedded deployment, don't want to depend on an external database, or need spreading activation + explanation paths.

---

## Comparison with pure vector databases

Vector databases (Pinecone / Milvus / Qdrant / Chroma) are complementary to HIPPMEM, not replacements.

HIPPMEM **uses vector search internally** (SimHash / HNSW), but that is not its core. The core is the **association graph + spreading activation** — which vector databases do not do.

They can be combined: use a vector database to store embeddings of massive document corpora, and use HIPPMEM to manage an agent's structured long-term memory.
