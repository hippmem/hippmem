# Eval corpus (English)

Evaluation corpus for HIPPMEM. Contains **53 JSON fixture cases** covering all 10 task types defined in the eval framework spec, plus 3 additional corpus scenarios. The Chinese corpus is at `../zh/`.

## Task types

| Type | Description | Case count |
|------|-------------|------------|
| FactRecall | Retrieve specific factual memories | 5 |
| PreferenceRecall | Recall user preferences and tastes | 5 |
| ProjectContinuity | Track project state across sessions | 5 |
| CausalTrace | Follow causal chains through memories | 5 |
| ContradictionDetection | Detect conflicting information | 5 |
| StateChange | Handle evolving/overwritten facts | 5 |
| ImplicitAssociation | Associate without keyword overlap | 5 |
| NoiseResistance | Filter irrelevant noise memories | 5 |
| LongTailRecall | Retrieve old/rarely-accessed memories | 5 |
| ExplanationQuality | Provide causal explanations for results | 5 |

Plus 3 additional corpus scenarios: `entity-network-001`, `causal-chain-001`, `multi-dim-assoc-001`.

## Fixture format

Each `.json` file contains one `EvalCase`:
```json
{
  "case_id": "fact-recall-001",
  "task_type": "FactRecall",
  "writes": [ ... ],
  "query": { "text": "...", "mode": "Balanced", "top_k": 10, "context": {} },
  "ground_truth": { "relevant": [...], "also_acceptable": [...], ... }
}
```

## Locale

This is the **English-language** corpus (`en/`). A Chinese corpus is at `../zh/`.
