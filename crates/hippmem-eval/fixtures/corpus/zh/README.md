# 评测语料库（中文）

HIPPMEM 评测语料库。包含 **53 个 JSON fixture 用例**，覆盖评测框架规范中定义的全部 10 种任务类型，外加 3 个语料场景用例。英文语料库位于 `../en/`。

## 任务类型

| 类型 | 说明 | 用例数 |
|------|------|--------|
| FactRecall | 检索具体事实记忆 | 5 |
| PreferenceRecall | 回忆用户偏好与品味 | 5 |
| ProjectContinuity | 跨会话追踪项目状态 | 5 |
| CausalTrace | 沿记忆追溯因果链 | 5 |
| ContradictionDetection | 检测冲突信息 | 5 |
| StateChange | 处理演变/覆写的事实 | 5 |
| ImplicitAssociation | 无关键词重叠的关联 | 5 |
| NoiseResistance | 过滤无关噪声记忆 | 5 |
| LongTailRecall | 检索旧/低频访问的记忆 | 5 |
| ExplanationQuality | 为结果提供因果解释 | 5 |

外加 3 个语料场景用例：`entity-network-001`、`causal-chain-001`、`multi-dim-assoc-001`。

## Fixture 格式

每个 `.json` 文件包含一个 `EvalCase`：
```json
{
  "case_id": "fact-recall-001",
  "task_type": "FactRecall",
  "writes": [ ... ],
  "query": { "text": "...", "mode": "Balanced", "top_k": 10, "context": {} },
  "ground_truth": { "relevant": [...], "also_acceptable": [...], ... }
}
```

## 语言

这是 **中文** 语料库（`zh/`）。英文语料库位于 `../en/`。
