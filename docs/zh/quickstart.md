# 快速入门（5 分钟）

> 📖 本文为参考翻译，以 [英文原文](../quickstart.md) 为准。如有出入，请以英文版为准。

> 目标：克隆仓库 → 写入一条记忆 → 检索它 → 理解为什么被检索到。

## 前提条件

- Rust 1.95+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- 默认模式：无需 GPU、无需 API Key、无需网络
- 可选：接入 OpenAI 兼容 API 以获得更高的语义精度（见 [用户指南 §5](user-guide.md#5-选择-embedder-后端)）

## 第 1 步：克隆并构建

```bash
git clone https://github.com/hippmem/hippmem.git
cd hippmem
cargo build --release
```

构建完成后，二进制文件位于 `target/release/hippmem`。

```bash
# 验证
./target/release/hippmem --version
# → hippmem 0.1.0
```

## 第 2 步：写入一条记忆

```bash
./target/release/hippmem write \
  -c "The user is a software engineer who prefers Rust and uses redb for embedded storage." \
  -t Preference
```

预期输出：

```
✓ memory_id: 12345678901234567890 stage: Indexed links: 0
```

> 如果这是第一条记忆，`links` 为 0（没有其他记忆可关联）。写入第二条后，引擎会自动发现并建立关联边。

## 第 3 步：多写几条记忆

```bash
# 写入一条相关事实
./target/release/hippmem write \
  -c "The user has contributed to multiple Rust open-source projects, including CLI tools and database bindings." \
  -t ProjectKnowledge

# 写入一条决策
./target/release/hippmem write \
  -c "Chose redb over RocksDB because redb is pure Rust, compiles quickly, and works well as an embedded store." \
  -t Decision
```

现在引擎已自动发现了关联——第二条记忆与第一条共享 "Rust" 实体，第三条与第一、二条之间存在因果关系。

## 第 4 步：检索记忆

```bash
./target/release/hippmem retrieve -q "Why did the user choose redb?" -k 5
```

预期输出（类似）：

```
1. [0.782] Chose redb over RocksDB because redb is pure Rust, compiles quickly... (dims: [Causal, EntityOverlap, SemanticSimilar])
2. [0.543] The user is a software engineer who prefers Rust and uses redb... (dims: [EntityOverlap, SemanticSimilar])
3. [0.421] The user has contributed to multiple Rust open-source projects... (dims: [Temporal, EntityOverlap])
```

每条结果携带：分数、内容摘要、匹配维度（告诉你「为什么这条被召回」）。

## 第 5 步：理解原因

```bash
# 使用第 2 步返回的 memory_id
./target/release/hippmem explain -m 12345678901234567890
```

预期输出：

```
memory: The user is a software engineer... importance: 0.800 links: 2 corrections: 0
```

引擎告诉你这条记忆关联了 2 条其他记忆，当前 importance 为 0.8。

## 第 6 步：查看引擎内部状态

```bash
./target/release/hippmem inspect store-stats
# → memories: 3 edges: 2 backlog: 0

./target/release/hippmem consolidate
# → processed: 3 decayed: 0 elapsed: 0ms
```

## 作为 Rust 库使用

在代码中调用：

```rust
use hippmem_engine::{Engine, EngineConfig, WriteMemoryInput, RetrieveInput};

let engine = Engine::open(EngineConfig::default())?;

engine.write(WriteMemoryInput {
    content: "The user prefers Rust for backend development.".into(),
    ..Default::default()
})?;

let results = engine.retrieve(RetrieveInput {
    query: "What language does the user prefer?".into(),
    top_k: 5,
    ..Default::default()
})?;

for r in &results.results {
    println!("[{:.3}] {}", r.final_score, r.memory.content.raw);
}
```

完整示例：`crates/hippmem-engine/examples/basic_usage.rs`

```bash
cargo run --example basic_usage
```

## 下一步

- [用户指南](user-guide.md) — 理解核心概念和完整工作流
- [API 参考（英文）](../api-reference.md) — 方法签名与类型定义
- [Cookbook（英文）](../cookbook.md) — 真实场景的复制粘贴配方
- [核心概念（英文）](../concepts.md) — 深入理解引擎如何「思考」
