# HKVMigrationAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**HKVMigrationAgent** 是一个面向 HierarchicalKV 项目的自动化 CUDA → AscendC SIMT 算子迁移框架。

本仓库基于**参考算子对**进行迁移学习：
- **CUDA参考实现**：位于 `cuda_HKV_reference/`，包含原始CUDA HKV算子代码
- **AscendC参考实现**：位于 `HierarchicalKV-ascend/`，包含已完成迁移的AscendC算子

Agent通过对比学习CUDA源码与对应的AscendC实现，自动生成**待迁移算子**的AscendC代码。

## 功能概述
该agent目标是对其他HierarchicalKV-ascend未迁移的算子进行自动化迁移。为了保证算子的生成质量，分为以下三步流水，实现分析->迁移生成->验证的迭代过程。

| 阶段 | 模块 | 核心能力 |
|------|------|----------|
| **Phase 1** | `cuda-kernel-analyzer` Skill | 分析 CUDA kernel 功能 → 设计测试场景 → 生成 AscendC 测试 C++ 文件 |
| **Phase 2** | `hkv-codegen-workflow` SubAgent | 基于参考算子对，迭代生成目标 AscendC kernel 头文件 |
| **Phase 3** | — | 编译测试正确性与性能（预留） |

备注：当前阶段关注构建`hkv-codegen-workflow` SubAgent的能力，通过HierarchicalKV-ascend中固有的gtest ut校验生成算子的正确性。暂未闭环1->3的自动生成测试。
具体构建`hkv-codegen-workflow`能力的流程为对已经人工迁移的算子分为`待生成算子`和`参考算子`两类。`参考算子`作为agent学习样本，尝试去生成`待生成算子`, 通过比对agent生成算子和人工算子的差异，提高skill质量。

具体的算子划分和对应cuda实现的关系为：
### 算子映射关系

#### 待迁移算子（目标）

| CUDA算子 | 文件位置 | 功能描述 | AscendC目标算子 |
|----------|----------|----------|----------------|
| `tlp_update_score_kernel` | `update_score.cuh` | 批量更新指定key的score | `assign_scores_kernel` |
| `find_or_insert_ptr_kernel_lock_key` | `find_ptr_or_insert.cuh` | 查找key返回指针，不存在则插入 | `find_or_insert_ptr_kernel` |
| `tlp_lookup_ptr_kernel_with_filter` | `lookup_ptr.cuh` | 查找key并返回value指针 | `find_ptr_kernel` |
| `tlp_v2_upsert_kernel_with_io` | `upsert.cuh` | 插入或更新key-value | `insert_or_assign_kernel` |

#### 参考算子（已完成映射）

| CUDA算子 | AscendC算子 | 功能 |
|----------|-------------|------|
| `tlp_lookup_ptr_kernel_with_filter` (update分支) | `find_and_update_kernel` | 查找key并更新score |
| `tlp_v1_upsert_and_evict_kernel_unique` | `insert_and_evict_kernel` | 插入key，满时淘汰 |
| `clear_kernel` | `clear_kernel` | 清空哈希表槽位 |
| `dump_kernel` | `dump_kernel` | 导出key-value-score |
| `create_atomic_keys/scores` | `init_table_kernel` | 初始化哈希桶 |
| `rehash_kernel_for_fast_mode` | `rehash_kernel` | 重哈希 |

## 项目结构

```text
HKVMigrationAgent/
├── cuda_HKV_reference/            # CUDA参考算子实现
│   ├── find_ptr_or_insert.cuh     # find_or_insert_ptr kernel
│   ├── upsert.cuh                 # insert_or_assign kernel
│   ├── update_score.cuh           # assign_scores kernel
│   ├── lookup_ptr.cuh             # find_ptr kernel
│   ├── upsert_and_evict.cuh       # insert_and_evict参考
│   └── ...
├── HierarchicalKV-ascend/         # AscendC参考/目标算子实现
│   └── hkv_hashtable/
│       ├── find_and_update_kernel/     # 参考：查找+更新
│       ├── insert_and_evict_kernel/    # 参考：插入+淘汰
│       ├── find_or_insert_ptr_kernel_lock_key/  # 目标：查找或插入
│       ├── upsert_with_io_kernel/      # 目标：插入或更新
│       └── ...
├── agents/
│   ├── HKV-Migration.md           # 主编排 Agent
│   └── hkv-codegen-workflow.md    # 代码生成子 Agent
├── skills/
│   ├── cuda-kernel-analyzer/      # CUDA分析Skill
│   └── ascendc-kernel-generator/  # AscendC生成Skill
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

基于**参考算子对**生成**目标算子**的AscendC实现：

```text
参考：cuda_HKV_reference/upsert_and_evict.cuh  →  HierarchicalKV-ascend/hkv_hashtable/insert_and_evict_kernel/
目标：cuda_HKV_reference/upsert.cuh            →  生成 insert_or_assign_kernel
```

在 OpenCode 中，通过 `/agents` 命令切换至 `HKV-Migration`，然后输入：

```text
参考 insert_and_evict_kernel 的实现，迁移 upsert.cuh 中的 tlp_v2_upsert_kernel_with_io，
输出到 HierarchicalKV-ascend/hkv_hashtable/upsert_with_io_kernel/
```

**执行流程**：

```
1. Phase 0：确认参数（参考算子、目标CUDA kernel路径）
2. Phase 1：分析目标CUDA kernel → 生成测试文件 → 用户确认
3. Phase 2：对比参考算子对，迭代生成目标AscendC kernel
4. Phase 3：用户确认最终代码
5. Phase 4：输出迁移报告
```

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
