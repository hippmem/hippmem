# HIPPMEM 用户指南

> 📖 本文为参考翻译，以 [英文原文](../user-guide.md) 为准。如有出入，请以英文版为准。

> 本指南带你从零开始掌握 HIPPMEM。建议先花 5 分钟阅读 [快速入门](quickstart.md)，再回到这里。

---

## 1. 核心概念一览

三点就够（详细解释见 [Core Concepts（英文）](../concepts.md)）：

1. **记忆不是纯文本，而是结构化的信息节点**：每条记忆自动携带实体、话题、因果断言以及指向其他记忆的关联链接。
2. **关联在写入时自动发现**：写入新记忆 → 引擎自动搜索已有记忆中的匹配项 → 打分 → 建立链接。关联在你写入的任何地方都会被创建。
3. **检索使用扩散激活**：不是「匹配关键词 → 排序」，而是「找种子 → 沿关联链接扩散 → 合并 → 解释」。工作方式类似人类的回忆过程。

---

## 2. 记忆的生命周期

```
你说了一句话
    │
    ▼
┌─────────────┐
│  raw → indexed │  同步（write() 返回时完成）
│              │  提取实体/话题/因果关系 → 构建倒排 + 全文索引 → 发现关联 → 建立链接
├─────────────┤
│  enriched    │  异步（后台 worker，几秒到几分钟）
│              │  填充 goal/preference/emotion/decision 等推理结果
├─────────────┤
│ consolidated │  异步（定时调度，默认每小时）
│              │  Hebbian 强化 / 衰减 / 压缩 / 摘要合并
└─────────────┘
```

对你来说，只需要调用 `write()` 和 `retrieve()`。中间的一切都由引擎自动完成。

---

## 3. CLI 操作

### 写入

```bash
hippmem write -c "Decided to switch from RocksDB to redb because redb compiles faster as a pure-Rust crate." -t Decision
```

输出：
```
✓ memory_id: 2152674446544667315913634290010169280 stage: Indexed links: 1
```

`content_type` 允许的值：`UserStatement` | `Decision` | `Preference` | `Event` | `TaskState` | `ProjectKnowledge` | `Reflection` | `Correction`

### 检索

```bash
hippmem retrieve -q "Why move away from RocksDB?" -k 5
```

输出：
```
1. [0.782] Decided to switch from RocksDB to redb... (dims: [Causal, EntityOverlap])
2. [0.543] The user prefers Rust for systems programming... (dims: [EntityOverlap, SemanticSimilar])
```

### 解释

```bash
hippmem explain -m 2152674446544667315913634290010169280
```

输出：
```
memory: Decided to switch from RocksDB... importance: 0.800 links: 2 corrections: 0
```

### 查看状态

```bash
hippmem inspect store-stats   # 存储统计
hippmem inspect queue         # 队列状态
```

### 巩固

```bash
hippmem consolidate           # 增量巩固
```

---

## 4. Rust 核心库 API

```rust
use hippmem_engine::{Engine, EngineConfig};

let engine = Engine::open(EngineConfig::default())?;

// 写入
let out = engine.write(WriteMemoryInput {
    content: "The team adopted Rust for the data pipeline after evaluating Go and Python.".into(),
    content_type: Some(ContentType::Decision),
    ..Default::default()
})?;

// 检索
let results = engine.retrieve(RetrieveInput {
    query: "What language is the data pipeline written in?".into(),
    top_k: 5,
    ..Default::default()
})?;

// 解释
let explanation = engine.explain(out.memory_id, None)?;

// 反馈
engine.feedback(FeedbackInput { /* ... */ })?;

// 巩固
engine.consolidate(ConsolidationScope::Incremental)?;

// 查看
engine.inspect(InspectQuery::StoreStats)?;

engine.close()?;
```

完整签名见 [API Reference（英文）](../api-reference.md)；真实场景代码见 [Cookbook（英文）](../cookbook.md)。

---

## 5. 选择 Embedder 后端

HIPPMEM 支持三种 embedding 后端，通过 `EmbedderConfig` 配置：

### 5.1 后端类型

| 提供商 | 向量维度 | 描述 | 适用场景 |
|----------|---------|------|---------|
| `deterministic`（默认） | 256d SimHash | 确定性降级模式，零依赖，无需网络，纯计算 | CI、离线、隐私保护、测试 |
| `openai-compatible` | 取决于模型 | 在线 API，高语义精度 | 生产环境，高质量检索 |
| `onnx`（预留） | 取决于模型 | 离线本地推理 | 未来：隐私 + 高精度 |

### 5.2 配置方式

**方式 1：环境变量（推荐）**

```bash
# Embedder 后端
export HIPPMEM__EMBEDDER__PROVIDER="openai-compatible"
export HIPPMEM__EMBEDDER__BASE_URL="https://api.openai.com/v1"
export HIPPMEM__EMBEDDER__MODEL="text-embedding-3-small"
export HIPPMEM__EMBEDDER__DIMENSIONS=1536

# API Key（独立于 Embedder 配置）
export OPENAI_API_KEY="sk-xxxxxxxx"
```

**方式 2：TOML 配置文件**

```toml
# hippmem.toml
[embedder]
provider = "openai-compatible"
base_url = "https://api.openai.com/v1"
model = "text-embedding-3-small"
api_key = "sk-xxxxxxxx"   # 可选；不填则从环境变量 OPENAI_API_KEY 读取
dimensions = 1536
```

**方式 3：代码中配置**

```rust
use hippmem_core::config::EmbedderConfig;
use hippmem_engine::EngineConfig;

// 确定性降级模式（默认，无需配置）
let config = EngineConfig::default();

// OpenAI API
let config = EngineConfig {
    embedder: EmbedderConfig::OpenAiCompatible {
        base_url: "https://api.openai.com/v1".into(),
        model: "text-embedding-3-small".into(),
        api_key: None,  // 从环境变量 OPENAI_API_KEY 读取
        dimensions: 1536,
    },
    ..EngineConfig::default()
};
```

**方式 4：CLI 参数**

```bash
hippmem --embedding-provider openai-compatible \
        --embedding-base-url "https://api.openai.com/v1" \
        --embedding-model "text-embedding-3-small" \
        write -c "A decision worth remembering."
```

### 5.3 API Key 配置位置

| 方式 | 配置路径 | 安全性 | 建议 |
|------|---------|--------|------|
| **环境变量** | `OPENAI_API_KEY=sk-...` | ⭐⭐⭐ 不写入文件 | **生产环境推荐** |
| TOML 文件 | `[embedder] api_key = "sk-..."` | ⭐⭐ 文件权限控制 | 开发环境方便 |
| 代码中 | `api_key: Some("sk-...".into())` | ⭐ 硬编码风险 | 仅用于测试 |

> **优先级**：CLI 参数 > 环境变量 > TOML 配置文件 > 代码默认值。

### 5.4 通道权重调优

切换到在线 API 后端后，建议调整 `SemanticDense` 通道权重，使高质量 embedding 不被 BM25 关键词匹配淹没。

**环境变量方式（推荐）**：

```bash
# API 后端用户应设置此项——让真正的语义向量发挥作用
export HIPPMEM__CHANNEL_COEFF_SEMANTIC_DENSE=1.5

# 可选：微调 BM25 权重
export HIPPMEM__CHANNEL_COEFF_BM25=0.8
```

**TOML 配置文件方式**：

```toml
[algo]
channel_coeff_semantic_dense = 1.5
channel_coeff_bm25 = 0.8
```

**代码方式**：

```rust
use hippmem_core::config::AlgoParams;

let config = EngineConfig {
    algo: AlgoParams {
        channel_coeff_semantic_dense: 1.5,
        channel_coeff_bm25: 0.8,
        ..AlgoParams::default()
    },
    embedder: EmbedderConfig::OpenAiCompatible { /* ... */ },
    ..EngineConfig::default()
};
```

> 详见 `docs/configuration.md` 中的「通道校准参数」章节。
> API Key 查找顺序：`EmbedderConfig.api_key` → 环境变量 `OPENAI_API_KEY`。
> 如果两者都不存在，`Engine::open()` 返回 `EngineError::Model("auth/missing key")`。

### 5.5 构建 Feature

使用 `openai-compatible` 后端需在构建时启用 feature：

```bash
cargo build --features api-backends
```

如果未启用 feature，指定 `openai-compatible` 将返回 `ModelError::Unavailable`。

> 默认配置（`deterministic`）无需任何 feature，零依赖即可编译——在离线/CI 环境中完全可用。

---

## 6. 评测框架

HIPPMEM 内置评测系统（`hippmem-eval`），包括：

- **5 种基线对比**：BM25 Only / Embedding Only / Hybrid / RAG Summary / HIPPMEM Full
- **10 种评测任务类型**：FactRecall / PreferenceRecall / ProjectContinuity / CausalTrace / ContradictionDetection / StateChange / ImplicitAssociation / NoiseResistance / LongTailRecall / ExplanationQuality
- **3 项核心指标**：Recall@K / Precision@K / Explanation Accuracy

```bash
cargo test -p hippmem-eval thresholds_m6
```

---

## 7. 常见问题

**Q：这和向量数据库有什么不同？**

向量数据库只做语义相似度搜索。HIPPMEM 在此基础上增加了：写入时关联发现、扩散激活检索、Hebbian 演化、可解释输出。详见 [Comparison（英文）](../comparison.md)。

**Q：可以离线使用吗？**

可以。默认的 deterministic 后端使用规则提取 + SimHash 语义；所有核心功能均可离线工作。

**Q：数据存储在哪里？**

默认在 `./hippmem_data/` 目录下：hippmem.redb（主存储）+ fulltext/（Tantivy BM25 索引）。

**Q：支持多大体量？**

设计目标为 10 万到 100 万条记忆——日常使用以年计，100 万条对个人用户足够终身使用。详细估算、硬件扩展性和极端场景分析见 [Capacity Planning](capacity-planning.md)。

**Q：如何贡献？**

阅读 [CONTRIBUTING.md（英文）](../../CONTRIBUTING.md) 了解开发环境搭建、commit 规范和 DCO 要求。
