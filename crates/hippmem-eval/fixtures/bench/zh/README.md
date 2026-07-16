# bench 语料库（中文）

HIPPMEM 评测用的基准数据集和查询集。全部内容为中文。

## 数据集

| 文件 | 说明 | 数量 |
|------|------|------|
| `quick_bench_dataset.json` | 50 条紧凑基准数据 | 50 |
| `quick_bench_queries.json` | 15 条评测查询 | 15 条 |
| `mem0_comparison_dataset.json` | 100 条 mem0 对比数据 | 100 |
| `mem0_comparison_queries.json` | 20 条评测查询 | 20 条 |
| `diagnostic_dataset.json` | P@1 未命中诊断数据 | 32 |
| `diagnostic_queries.json` | 6 条失败查询深度诊断 | 6 条 |
| `retrieval_quality_categories.json` | 6 个类别共 30 条记忆 | 6 类 |
| `retrieval_quality_queries.json` | 10 条检索质量查询 | 10 条 |
| `user_perspective_categories.json` | 用户视角评测类别 | 6 类 |
| `user_perspective_queries.json` | 用户视角评测查询 | 15 条 |

## 格式

数据集使用两种格式：
- **BenchDataset** — `{ "entries": [{ "content_type": "...", "content": "...", "importance": 0.8, "category": "..." }] }`
- **CategoryTextSet** — `{ "categories": [{ "category": "...", "texts": [...] }] }`

查询集使用 **CategoryQuerySet** 格式：
- `{ "queries": [{ "query": "...", "expected_categories": [...], "description": "..." }] }`

## 语言

这是 **中文** 基准语料库（`zh/`）。英文语料库位于 `../en/`。
