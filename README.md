# HKVMigrationAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**HKVMigrationAgent** 是一个面向 HierarchicalKV 项目的自动化 CUDA → AscendC SIMT 算子迁移框架。
本 Agent 通过三阶段流水线，将 CUDA SIMT kernel 自动迁移为 AscendC SIMT kernel 实现，并同步生成对应的测试代码。

## 功能概述

| 阶段 | 模块 | 核心能力 |
|------|------|----------|
| **Phase 1** | `cuda-kernel-analyzer` Skill | 分析 CUDA kernel 功能 → 设计测试场景 → 生成 AscendC 测试 C++ 文件 |
| **Phase 2** | `hkv-codegen-workflow` SubAgent | 迭代生成 AscendC kernel 头文件，含静态规范检查与自动修复 |
| **Phase 3** | — | 编译测试正确性与性能（预留，当前环境暂不执行） |

## 项目结构

```text
HKVMigrationAgent/
├── agents/
│   ├── HKV-Migration.md           # 主编排 Agent（三阶段流水线）
│   └── hkv-codegen-workflow.md    # 代码生成子 Agent（迭代式生成+检查）
├── skills/
│   ├── cuda-kernel-analyzer/      # CUDA 分析 + 测试生成 Skill
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── ascendc-simt-patterns.md   # AscendC SIMT 编程模式
│   │       └── hkv-data-structures.md     # HKV 数据结构参考
│   └── ascendc-kernel-generator/  # AscendC Kernel 代码生成 Skill
│       ├── SKILL.md
│       └── references/
│           ├── ascendc-simt-patterns.md   # AscendC SIMT 编程模式（同上）
│           └── hkv-data-structures.md     # HKV 数据结构参考（同上）
└── README.md
```

## 快速开始

### 环境要求

- AscendC CANN 9.0+
- [OpenCode](https://opencode.ai/)（已正确安装并配置）

### 安装

```bash
# 将 agents 和 skills 部署到 OpenCode 默认配置路径
mkdir -p ~/.config/opencode/
cp -r agents/ ~/.config/opencode/
cp -r skills/ ~/.config/opencode/
```

### 使用

在 OpenCode 中，通过 `/agents` 命令切换至 `HKV-Migration`，然后输入 prompt：

```text
迁移 /path/to/HierarchicalKV/include/merlin/core_kernels/lookup.cuh 中的 lookup kernel，
输出到 /path/to/output/
```

**执行流程**：

```
1. Phase 0：确认参数（CUDA 文件路径、kernel 名称）
2. Phase 1：分析 CUDA kernel → 生成 test_lookup.cpp → 用户确认
3. Phase 2：迭代生成 lookup_kernel.h → 静态规范检查 → 自动修复
4. Phase 3：用户确认最终代码
5. Phase 4：输出迁移报告 report.md
```

## 参考算子

Agent 代码生成时参考以下已完成的 AscendC SIMT kernel 实现：

| 算子 | 路径 | 功能 |
|------|------|------|
| `clear_kernel` | `HierarchicalKV-ascend/hkv_hashtable/clear_kernel/` | 清空哈希表所有槽位 |
| `find_and_update_kernel` | `HierarchicalKV-ascend/hkv_hashtable/find_and_update_kernel/` | 查找 key 并返回 value 指针 |
| `init_table_kernel` | `HierarchicalKV-ascend/hkv_hashtable/init_table_kernel/` | 初始化哈希桶内存布局 |
| `insert_and_evict_kernel` | `HierarchicalKV-ascend/hkv_hashtable/insert_and_evict_kernel/` | 插入 key，满时按 score 淘汰 |

## CUDA → AscendC 核心映射

| CUDA | AscendC |
|------|---------|
| `__global__ void kernel(...)` | `__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...)` |
| `blockIdx.x`（kernel 内部） | `block_index`（作为参数传入） |
| `T* ptr`（全局内存参数） | `GM_ADDR ptr_gm` |
| `(T*)ptr`（内部类型转换） | `reinterpret_cast<__gm__ T*>(ptr_gm)` |
| `atomicCAS(ptr, old, new)` | `Simt::AtomicCas(ptr, old, new)` |
| `atomicAdd(ptr, val)` | `atomicAdd(ptr, val)` |

## 许可证

本项目采用 [Apache 2.0 License](LICENSE) 开源许可证。
