# HIPPMEM Documentation

> Pick a path to get started —

## 🚀 I want to use HIPPMEM

| Document | Time | Content |
|----------|------|---------|
| [Quick Start](quickstart.md) | 5 min | clone → build → write → retrieve, with expected output for every step |
| [User Guide](user-guide.md) | 20 min | Core concepts → lifecycle → scenario examples → FAQ; a complete narrative tutorial |
| [Cookbook](cookbook.md) | On demand | Scenario recipes: conversation memory, project knowledge base, preference tracking, decision audit… each with full runnable code |
| [Configuration Reference](configuration.md) | 5 min | All AlgoParams + EngineConfig fields, defaults, and tuning advice |

## 🔌 I want to integrate HIPPMEM into my system

| Document | Content |
|----------|---------|
| [API Reference](api-reference.md) | Signatures, type tables, error codes, and examples for all 7 Engine methods |
| [gRPC Guide](grpc-guide.md) | Proto overview + complete Python/Go/Node.js client examples |
| [Integration Guide](integration.md) | Patterns: embedded in a Rust agent / gRPC sidecar deployment / CI testing / file deployment |

## 🧠 I want to understand how HIPPMEM works

| Document | Content |
|----------|---------|
| [Core Concepts](concepts.md) | MemoryUnit lifecycle, association graph, spreading activation, deterministic degradation, consolidation & evolution |
| [Solution Comparison](comparison.md) | Feature-matrix comparison with Mem0 / Graphiti / Qdrant / Chroma / pure vector databases |
| [Architecture Whitepaper](architecture/whitepaper.md) | Design philosophy and first principles |
| [Data Model](architecture/data-model.md) | MemoryUnit, AssociationLink, ActivationState type definitions |
| [Algorithms](architecture/algorithms.md) | Multi-channel recall, spreading activation, Hebbian consolidation |
| [Design Overview](architecture/design.md) | Crate topology, data flow, process model |
| [Model Backends](architecture/model-backends.md) | Embedder, Extractor, Reranker, Summarizer traits and degradation |

## 🛠 I want to contribute

| Document | Content |
|----------|---------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Development setup, commit conventions, DCO requirements |
| [Architecture Decisions](architecture/decisions.md) | Key architecture decision records |
| [Glossary](glossary.md) | Terminology definitions |

## 📚 Full document index

### User documentation (`docs/`)
- `README.md` — This document (navigation hub)
- `quickstart.md` — 5-minute quick start
- `user-guide.md` — Full user guide
- `api-reference.md` — API reference
- `grpc-guide.md` — gRPC usage guide
- `cookbook.md` — Scenario cookbook
- `concepts.md` — Deep dive on core concepts
- `comparison.md` — Solution comparison
- `configuration.md` — Configuration parameter reference
- `integration.md` — Integration patterns guide
- `glossary.md` — Terminology definitions
- `llms.txt` — AI-consumable document index

### Architecture documentation (`docs/architecture/`)
- `whitepaper.md` — Design philosophy and first principles
- `data-model.md` — Core type definitions
- `algorithms.md` — Algorithm details
- `design.md` — System design overview
- `api-contract.md` — API contract
- `eval-framework.md` — Evaluation framework
- `model-backends.md` — Model backend architecture
- `decisions.md` — Architecture decision records

---

🌐 **中文文档** → [docs/zh/](zh/README.md)（快速入门、用户指南、中文 NLP 能力说明）
