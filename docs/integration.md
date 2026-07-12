# Integration Guide

> Patterns and best practices for embedding HIPPMEM into real systems.

---

## Pattern 1: Embedded in a Rust Process

The most direct integration. HIPPMEM is a dependency of your Rust project.

```toml
[dependencies]
hippmem-engine = "0.1"
```

```rust
use hippmem_engine::{Engine, EngineConfig};
use std::sync::Arc;
use parking_lot::RwLock;

// Global singleton
static MEMORY: once_cell::sync::Lazy<Arc<RwLock<Engine>>> =
    once_cell::sync::Lazy::new(|| {
        Arc::new(RwLock::new(
            Engine::open(EngineConfig::default()).unwrap()
        ))
    });

// Called from request handling
fn handle_message(msg: &str) {
    let engine = MEMORY.read();
    let results = engine.retrieve(/* ... */).unwrap();
    // ...
}
```

**Suitable for**: Rust backend services, CLI tools, TUI applications, embedded systems.

**Advantages**: Zero network overhead, type safety, compile-time checks.

**Note**: `Engine` is not `Send + Sync` (it holds an exclusive reference to redb), so it needs to be wrapped in `Arc<RwLock<Engine>>`.

---

## Pattern 2: gRPC Sidecar

HIPPMEM runs as a standalone process, and applications call it over gRPC.

```
┌──────────┐     gRPC      ┌───────────────┐
│ Your app  │ <----------> │ hippmem-server │
│ Python/Go │  :50051      │ (Rust binary)  │
└──────────┘               └───────────────┘
```

### Docker Compose Example

```yaml
# docker-compose.yml
version: "3.8"
services:
  app:
    build: .
    environment:
      - HIPPMEM_ADDR=hippmem:50051
    depends_on:
      - hippmem

  hippmem:
    image: hippmem-server:latest
    volumes:
      - ./hippmem_data:/data
    environment:
      - HIPPMEM_STORE_DIR=/data
      - HIPPMEM_LISTEN=0.0.0.0:50051
    ports:
      - "50051:50051"
```

**Suitable for**: Microservice architectures, cross-language calls, independent scaling.

**Advantages**: Language-agnostic, process isolation, independently upgradeable.

**Note**: Streaming is not yet supported; mind the serialization overhead for large result sets.

---

## Pattern 3: File Deployment (No Service Process)

The simplest deployment. CLI binary + file storage.

```bash
# Install
cp target/release/hippmem /usr/local/bin/

# Use
hippmem -s /var/lib/hippmem write -c "..." -t Decision
hippmem -s /var/lib/hippmem retrieve -q "..." -k 5
```

**Suitable for**: Personal use, shell-script integration, cron jobs.

**Advantages**: Zero configuration, no service process.

**Note**: Concurrency safety is guaranteed by the redb file lock; concurrent writes from multiple processes are not supported.

---

## Pattern 4: CI Test Integration

Run tests in CI using the deterministic degraded backend.

```yaml
# .github/workflows/test.yml
- name: Run HIPPMEM tests
  run: cargo test --workspace
  env:
    HIPPMEM_BACKEND: deterministic  # force degraded backend
```

```rust
// Use the default config in test code (deterministic 256d SimHash)
let config = EngineConfig::default();  // embedder defaults to Deterministic
let engine = Engine::open(config)?;
```

**Suitable for**: CI/CD pipelines, pre-commit checks.

**Advantages**: No API key required, deterministic and reproducible results.

---

## Pattern 5: Agent Framework Integration

HIPPMEM can serve as the Memory backend for frameworks such as LangChain / LlamaIndex.

### Concept Mapping

| Agent framework concept | HIPPMEM equivalent |
|---------------|-------------|
| `ConversationMemory` | `session_id`-scoped retrieval |
| `VectorStore` | Semantic channel (SimHash/HNSW) |
| `EntityMemory` | Entity + Topic inverted index |
| `SummaryMemory` | Consolidation summarize |
| `WorkingMemory` | `RetrieveContext.recent_memory_ids` |

### Integration (via gRPC)

```python
# langchain_hippmem.py
import grpc
import hippmem_pb2
import hippmem_pb2_grpc
from langchain.memory import BaseMemory

class HippmemMemory(BaseMemory):
    def __init__(self, addr="localhost:50051"):
        self.channel = grpc.insecure_channel(addr)
        self.stub = hippmem_pb2_grpc.HippmemStub(self.channel)

    def save_context(self, inputs, outputs):
        self.stub.Write(hippmem_pb2.WriteRequest(
            content=f"User: {inputs['input']}\nAI: {outputs['output']}",
            content_type="UserStatement",
        ))

    def load_memory_variables(self, inputs):
        resp = self.stub.Retrieve(hippmem_pb2.RetrieveRequest(
            query=inputs["input"],
            top_k=5,
            mode="Balanced",
        ))
        return {"history": [r.content for r in resp.results]}
```

---

## Performance Baseline

The following data is based on the degraded backend with 20 test memories (developer-machine SSD):

| Operation | Typical latency | Notes |
|------|---------|------|
| `write` | 5-15ms | Synchronous: index build + candidate recall + edge creation |
| `retrieve` (Balanced) | 10-30ms | Multi-channel + 2-hop spreading |
| `retrieve` (Fast) | 3-8ms | Single hop |
| `retrieve` (Deep) | 30-100ms | 3 hops + full diagnostics |
| `consolidate` (Incremental) | 1-10ms | Scales with data size |
| `consolidate` (Full) | 50-200ms | Full processing |
| `explain` | 0.5-2ms | Pure store read |

**Design target**: 100K–1M memories. We recommend running your own load tests on production-representative data.

---

## Failure Handling

| Scenario | Behavior | Recovery |
|------|------|------|
| Disk full during write | `EngineError::Store` | Clean up disk and retry |
| redb file corruption | `EngineError::Store` | Backup strategy, rebuild |
| Background worker panic | No impact on foreground API (separate task) | Auto restart |
| gRPC connection drop | tonic `Unavailable` | Client reconnects |
| API backend network failure | `WriteWarning::ModelError` | Automatic fallback to degraded backend |
| API key expired | `BackendUnavailable` | Update the key, auto-switch back to API backend |
