# bench corpus (English)

Benchmark datasets and query sets for HIPPMEM evaluation. All content is in English.

## Datasets

| File | Description | Records |
|------|-------------|---------|
| `quick_bench_dataset.json` | 50-record compact benchmark | 50 |
| `quick_bench_queries.json` | 15 evaluation queries | 15 queries |
| `mem0_comparison_dataset.json` | 100-item mem0 comparison dataset | 100 |
| `mem0_comparison_queries.json` | 20 evaluation queries | 20 queries |
| `diagnostic_dataset.json` | Diagnostic dataset for P@1-miss analysis | 32 |
| `diagnostic_queries.json` | 6 failing queries for deep diagnostic | 6 queries |
| `retrieval_quality_categories.json` | 30 memories in 6 categories | 6 categories |
| `retrieval_quality_queries.json` | 10 retrieval quality queries | 10 queries |
| `user_perspective_categories.json` | User-perspective eval categories | 6 categories |
| `user_perspective_queries.json` | User-perspective eval queries | 15 queries |

## Format

Datasets use two formats:
- **BenchDataset** — `{ "entries": [{ "content_type": "...", "content": "...", "importance": 0.8, "category": "..." }] }`
- **CategoryTextSet** — `{ "categories": [{ "category": "...", "texts": [...] }] }`

Query sets use **CategoryQuerySet** format:
- `{ "queries": [{ "query": "...", "expected_categories": [...], "description": "..." }] }`

## Locale

This is the **English-language** benchmark corpus (`en/`). The Chinese corpus is at `../zh/`.
