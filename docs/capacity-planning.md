# Capacity Planning

> How many memories do you need? How long will 100K–1M last for your use case? This guide helps you estimate.

---

## What is one memory?

A memory in HIPPMEM is **one atomic fact**, not a document, not a conversation log, not a paragraph. Think:

- "I prefer Rust over Python for backend work" (a preference)
- "Decided to use redb because it's pure Rust with no C dependency" (a decision)
- "The engine supports 14 association edge types" (a piece of knowledge)

Each call to `engine.write()` creates exactly one memory. HIPPMEM does **no automatic chunking** — it is the caller's responsibility to decompose conversations or documents into individual facts worth remembering.

---

## A concrete example: 100 memories

The `seed_100` fixture (used in tests and examples) represents the complete knowledge profile of **one developer + one project**:

| Category | Count | Examples |
|---|---|---|
| Biography & identity | 10 | name, role, years of experience, languages spoken |
| Preferences | 15 | language preferences, tool preferences, workflow habits |
| Technical decisions | 20 | why redb, why SimHash, why bincode over JSON |
| Project knowledge | 20 | crate structure, algorithm parameters, architecture choices |
| Task states | 15 | completed CI/CD, fixed bugs, features in progress |
| Observations | 10 | benchmark results, external tool evaluations |
| Corrections & reflections | 10 | "previously used X, switched to Y because Z" |

**100 memories = everything you'd know about a person and their work after months of collaboration.**

---

## How many memories does an AI agent produce?

It depends on usage intensity. These estimates assume the agent writes only **meaningful new facts** — not every conversation turn produces a memory.

| Usage pattern | Memories/day | Typical scenario |
|---|---|---|
| **Light personal** | 10–20 | Occasional notes, preferences, reminders; a few conversations per day |
| **Regular professional** | 30–80 | Daily coding assistant, research companion, knowledge management |
| **Heavy professional** | 100–300 | Deep research sessions, document decomposition, continuous knowledge extraction |
| **Continuous agent loop** | 500–1,000 | Agent running 24/7 with high throughput |

For perspective: a single conversation turn typically produces 0–2 memories. A 50-turn work session might produce 20–60 memories. The limiting factor is not the engine — it is how many **new, distinct, worth-remembering facts** the interaction generates.

---

## How long will it last?

| Usage pattern | 100K lasts | 1M lasts |
|---|---|---|
| Light personal (~15/day) | **~18 years** | **~180 years** |
| Regular professional (~50/day) | **~5.5 years** | **~55 years** |
| Heavy professional (~200/day) | **~1.4 years** | **~14 years** |
| Continuous loop (~500/day) | ~6 months | **~5.5 years** |
| Extreme throughput (~1,000/day) | ~3 months | **~2.7 years** |

**For the vast majority of users, 100K represents years of use, and 1M is effectively a lifetime.** Only agents running at sustained, high-throughput 24/7 loops will fill 100K in months — and even then, 1M covers years.

---

## Are these hard limits?

**No.** 100K–1M is the range where HIPPMEM has been tested and where performance is known to be comfortable. Beyond 1M, the system continues to work — performance degrades gradually, not suddenly:

- **redb** (primary storage): B-tree based, degrades gracefully with size. I/O-bound.
- **Tantivy** (full-text index): designed for millions of documents natively.
- **Vector index**: performance depends on the backend and available RAM.

**Upgrading hardware extends the comfortable range.** More RAM, a faster NVMe SSD, and more CPU cores push the practical ceiling higher. 1M on a laptop might be the upper bound; 1M on a 32 GB RAM + NVMe server is mid-range.

---

## Future scalability

The 100K–1M range reflects the current single-node, single-file architecture. Several paths are available to raise this by an order of magnitude (10M+) in future releases:

| Direction | Approach | Expected gain |
|---|---|---|
| **Partitioning** | Split storage by time range or topic into multiple redb files, reducing per-file B-tree depth | 3–5× |
| **Tiered indexing** | Full indexing for recent/hot memories; lightweight indexing for cold/archived ones | 2–3× |
| **Compaction** | Merge redundant or superseded memories; consolidate old facts automatically | 2–5× (effective capacity) |
| **ANN vector index** | Replace brute-force with approximate nearest neighbor for large-scale semantic search | unlocks 10M+ vector search |
| **Format optimization** | More compact serialization, delta encoding for association keys | 1.5–2× |
| **Hardware headroom** | NVMe bandwidth still growing; RAM density per dollar improving yearly | ~2× per hardware generation |

A combination of partitioning + tiered indexing + compaction would push the comfortable range to **10M+ memories** — enough for a small business running multiple agents for a decade. None of these require architectural rewrites; they are incremental improvements on the existing design.

---

## Summary

| Concern | Reality |
|---|---|
| "Will it run out in months?" | Only in the most extreme 24/7 high-throughput scenario. For typical daily use, it lasts years. |
| "What if I need more than 1M?" | 1M is a soft design target, not a hard cap. Better hardware raises the ceiling; planned improvements target 10M+. |
| "Should I worry about this?" | If you're an individual user or a small team, you almost certainly won't hit the limit. If you're building a high-throughput production agent, plan for 1M and monitor. |
