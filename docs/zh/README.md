# HIPPMEM 文档（中文）

> 本文档为 HIPPMEM 项目的中文导航入口。英文文档为权威版本，中文翻译为参考。

## 🚀 我想使用 HIPPMEM

| 文档 | 预计时间 | 内容 |
|------|----------|------|
| [快速入门](quickstart.md) | 5 分钟 | 克隆 → 构建 → 写入 → 检索，每步都有预期输出 |
| [用户指南](user-guide.md) | 20 分钟 | 核心概念 → 生命周期 → 场景示例 → FAQ；完整的叙事教程 |
| [中文 NLP 能力说明](chinese-nlp.md) | 10 分钟 | HIPPMEM 对中文的支持详情：分词、实体提取、BM25 检索 |

> 更多文档（API 参考、Cookbook、架构等）目前仅提供英文版，请访问 [英文文档首页](../README.md)。

## 🔌 我想将 HIPPMEM 集成到系统中

| 文档（英文） | 内容 |
|-------------|------|
| [API Reference](../api-reference.md) | 7 个 Engine 方法的签名、类型表、错误码和示例 |
| [gRPC Guide](../grpc-guide.md) | Proto 概览 + Python/Go/Node.js 客户端示例 |
| [Integration Guide](../integration.md) | 集成模式：嵌入式 / gRPC sidecar / CI 测试 |

## 🧠 我想理解 HIPPMEM 的工作原理

| 文档（英文） | 内容 |
|-------------|------|
| [Core Concepts](../concepts.md) | MemoryUnit 生命周期、关联图、扩散激活、确定性衰减 |
| [Architecture Whitepaper](../architecture/whitepaper.md) | 设计哲学与第一性原理 |
| [Data Model](../architecture/data-model.md) | MemoryUnit、AssociationLink、ActivationState 类型定义 |
| [Algorithms](../architecture/algorithms.md) | 多通道召回、扩散激活、Hebbian 巩固 |

## 🛠 我想贡献

| 文档（英文） | 内容 |
|-------------|------|
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | 开发环境搭建、commit 规范、DCO 要求 |

---

📖 英文完整文档索引 → [docs/README.md](../README.md)
