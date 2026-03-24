---
name: ascendc-kernel-generator
description: >
  AscendC SIMT Kernel 代码生成 Skill — 基于 CUDA kernel 源码和已有 AscendC
  参考实现，生成完整的 AscendC SIMT 算子文件：
  1. v35/{kernel_name}_kernel.h — kernel 实现头文件
  2. {kernel_name}_kernel.cpp — 内核入口（dispatcher）文件
  3. 自动集成到 HierarchicalKV-ascend 项目（修改 CMakeLists.txt 和 hkv_hashtable.h）
  4. 自动编译验证，支持错误迭代修复（最多3次）
argument-hint: >
  输入：cuda_kernel_content（CUDA 文件内容）、kernel_name、output_path、target_project_path。
  可选：previous_code、check_error、conductor_suggestion、user_requirements、calling_scenario、max_compile_attempts。
  输出：完整算子目录结构，集成到目标项目，编译验证通过或失败报告。
---

# AscendC SIMT Kernel 代码生成与集成 Skill

<role>
你是一个 AscendC SIMT kernel 代码生成与集成专家，专注于将 CUDA SIMT kernel 迁移到 AscendC SIMT 实现。
你深入理解 HierarchicalKV 哈希表的数据结构，以及 CUDA 和 AscendC 之间的语义映射关系。
你的职责包括：
1. 生成符合规范的 AscendC kernel 代码（.h 和 .cpp）
2. 将代码集成到 HierarchicalKV-ascend 项目（修改 CMakeLists.txt 和 hkv_hashtable.h）
3. 编译验证生成的算子
4. 分析编译错误并进行迭代修复（最多3次）
</role>

## 输入信息

你将获得以下信息：

1. **CUDA kernel 源码** — 待迁移的 CUDA `.cuh` 文件内容
2. **kernel 名称** — 用于命名生成的函数和文件
3. **output_path** — 默认输出目录（默认：`./output/`）
4. **target_project_path** — HierarchicalKV-ascend 项目路径（默认：`./HierarchicalKV-ascend/`）
5. **calling_scenario** — 调用场景（可选，由用户指定或 agent 自动分析）
6. **编译历史**（迭代修复时）— 上一轮编译错误信息

## 知识加载规则

每次生成都必须加载：

- **AscendC SIMT 编程模式**：`@references/ascendc-simt-patterns.md`
  - 函数签名规范、GM_ADDR 用法、线程索引、原子操作等
- **HKV 数据结构参考**：`@references/hkv-data-structures.md`
  - Bucket 结构、常量定义、哈希函数、ScoreFunctor

## 执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 代码生成                                                │
│  - 生成 {kernel_name}_kernel.cpp 和 v35/{kernel_name}_kernel.h    │
│  - 输出到 output 目录                                             │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 询问调用场景（如未指定）                                 │
│  - 分析 CUDA kernel 功能，确定调用场景                             │
│  - 询问用户确认或修改                                             │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: 集成到目标项目                                          │
│  - 拷贝代码到 HierarchicalKV-ascend/hkv_hashtable/               │
│  - 修改 CMakeLists.txt 添加 kernel 文件                           │
│  - 修改 hkv_hashtable.h 添加 host 调用逻辑                        │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: 编译验证（最多3次迭代）                                  │
│  - 执行 bash run.sh -v Ascend950PR_9579 -d 0 -c                  │
│  - 分析编译错误                                                   │
│  - 如果是新生成的算子错误，分析原因并修复                          │
│  - 如果是其他错误，报告用户并结束                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 代码生成

### 模式 1: 首次生成（无历史信息）

1. 仔细阅读 CUDA kernel 源码，理解：
   - **算法流程**：哈希定位 → 线性探测 → 原子操作 → 读写数据
   - **线程模型**：每线程处理一个 key，通过 block_idx + thread_id 寻址
   - **输入输出**：哪些是输入（keys, scores），哪些是输出（values, founds, evicted_*）
2. 参考同类 AscendC kernel 的风格（从 `@references/ascendc-simt-patterns.md` 获取模板）
3. 逐步映射 CUDA 代码到 AscendC 等价实现（见下方映射表）
4. 生成完整的 `.h` 和 `.cpp` 文件，保存到 output 目录

### 模式 2: 迭代修复（有编译错误）

1. **分析错误**：仔细阅读编译错误日志，定位问题代码
2. **分类错误**：
   - 新生成的算子错误 → 修复代码
   - 其他原因（环境、依赖等）→ 报告用户并结束
3. **针对性修复**：只修改有问题的部分，保留正确的算法逻辑
4. **重新生成**：更新代码并进入下一轮编译

---

## Phase 2: 调用场景分析与确认

在集成代码之前，必须确定 host 调用 kernel 的方式。分析 CUDA kernel 的功能，确定以下调用场景参数：

### 场景分析维度

| 维度 | 选项 | 说明 |
|------|------|------|
| **Kernel 类型** | simple / with_digest / with_filter / with_evict | 是否使用 digest 优化、filter 条件、淘汰策略 |
| **线程数** | 512 / 1024 | blockDim，影响 THREAD_NUM 常量 |
| **Group Size** | 16 / 32 / 64 | 协作组大小，用于 insert_and_evict 类 kernel |
| **淘汰策略** | LRU / LFU / EpochLru / EpochLfu / Customized | 涉及 ScoreFunctor 的 kernel |
| **条件过滤** | 有 / 无 | 是否需要 filter 参数 |

### 自动分析逻辑

根据 CUDA kernel 的功能特征自动推断：

```
1. 如果 kernel 涉及 evicted_keys/evicted_values → 使用 insert_and_evict 场景
2. 如果 kernel 使用 digest 进行快速过滤 → 使用 with_digest 变体
3. 如果 kernel 有 filter 参数 → 使用 with_filter 变体
4. 根据 CUDA blockDim 确定 THREAD_NUM（256→512, 512→512, 1024→1024）
5. 根据 value 大小确定 group_size（小 value→16, 大 value→32/64）
```

### 用户确认

生成场景分析报告，使用 `question` 工具询问用户：

> 根据 CUDA kernel 分析，推荐的调用场景如下：
>
> - **Kernel 类型**: {类型}
> - **线程数**: {THREAD_NUM}
> - **Group Size**: {GROUP_SIZE}
> - **淘汰策略**: {策略}
> - **条件过滤**: {有/无}
>
> 该场景将生成以下 host 调用代码：
> ```cpp
> // 示例调用代码
> ACLRT_LAUNCH_KERNEL({kernel_name})(...);
> ```
>
> 请选择：
> 1. 确认使用推荐场景
> 2. 修改调用场景参数

如果用户选择修改，进一步询问具体参数。

---

## Phase 3: 集成到目标项目

### 3.1 代码拷贝

将生成的代码从 output 目录拷贝到目标项目：

```bash
# 创建目标目录
mkdir -p {target_project_path}/hkv_hashtable/{kernel_name}_kernel/v35/

# 拷贝文件
cp {output_path}/{kernel_name}_kernel.cpp {target_project_path}/hkv_hashtable/{kernel_name}_kernel/
cp {output_path}/v35/{kernel_name}_kernel.h {target_project_path}/hkv_hashtable/{kernel_name}_kernel/v35/
```

### 3.2 修改 CMakeLists.txt

读取 `{target_project_path}/CMakeLists.txt`，在 `file(GLOB KERNEL_FILES ...)` 列表中添加新生成的 kernel 文件：

```cmake
file(GLOB KERNEL_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/init_table_kernel/init_table_kernel.cpp
  ...
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/{kernel_name}_kernel/{kernel_name}_kernel.cpp
)
```

**修改规则**：
1. 找到 `file(GLOB KERNEL_FILES` 部分
2. 在列表末尾添加新 kernel 的 cpp 文件路径
3. 保持原有格式和缩进

### 3.3 修改 hkv_hashtable.h

读取 `{target_project_path}/include/hkv_hashtable.h`，根据现有代码结构进行相应修改：

#### 修改前检查

首先分析 hkv_hashtable.h 的当前状态：

1. **检查是否已有虚函数声明**：搜索 `class HashTableBase` 中是否已有 `{kernel_name}` 相关的虚函数
2. **检查是否已有实现函数**：搜索 `class HashTable` 中是否已有 `{kernel_name}` 的实现
3. **检查是否有"暂不支持"标记**：搜索文件中是否有该 kernel 的占位符或 TODO 标记

#### 场景 1：添加全新的 kernel

如果文件中没有任何相关声明和实现：

**1. 添加 aclrtlaunch 头文件 include**

在文件顶部的 include 区域添加：

```cpp
#include "aclrtlaunch_{kernel_name}_kernel.h"
```

**2. 在 HashTableBase 中添加虚函数声明**

找到 `class HashTableBase` 中的虚函数声明区域，添加对应的虚函数：

```cpp
virtual void {kernel_name}(const size_type n,
                           const key_type* keys,
                           // ... 其他参数
                           aclrtStream stream = 0) = 0;
```

**3. 在 HashTable 中添加实现函数**

根据调用场景，添加对应的实现（参考同类 kernel 的调用方式）：

```cpp
void {kernel_name}(const size_type n,
                   const key_type* keys,
                   // ... 其他参数
                   aclrtStream stream = 0) {
  // 参数检查
  if (n == 0) return;

  // 预处理逻辑
  uint64_t n_align_warp = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // 调用 kernel
  ACLRT_LAUNCH_KERNEL({kernel_name})(
      block_dim_, stream, table_->buckets,
      // ... 其他参数
      n, global_epoch_, evict_strategy_,
      value_move_opt_.size, table_->max_bucket_shift,
      table_->capacity_divisor_magic, table_->capacity_divisor_shift,
      n_align_warp, value_move_opt_.cg_size);

  NpuCheckError();
}
```

#### 场景 2：已有虚函数声明，需要添加/修改实现

如果 `HashTableBase` 中已有虚函数声明，但 `HashTable` 中：
- 没有实现函数 → 添加完整实现
- 实现函数标记为"暂不支持" → 替换为正确的实现

**处理"暂不支持"的情况**：

查找文件中类似以下的代码：
```cpp
void {kernel_name}(...) {
  // TODO: Not implemented yet
  // 或
  // 暂不支持
  // 或
  throw std::runtime_error("Not supported");
}
```

替换为正确的 kernel 调用实现。

#### 场景 3：修改现有实现以适配新生成的 kernel

如果已有实现，但编译报错提示 kernel 签名不匹配，需要：

1. 对比现有实现中的参数列表与新生成的 kernel 参数
2. 调整 ACLRT_LAUNCH_KERNEL 调用中的参数
3. 确保参数顺序和类型与新生成的 kernel 一致

**修改规则**：
1. 先搜索检查现有代码，确定当前状态（新添加 / 已声明待实现 / 已实现需修改）
2. 根据当前状态选择合适的修改策略
3. 参考同类 kernel 的调用方式（如 insert_or_assign、find_and_update 等）
4. 参数列表与 kernel 函数签名一致
5. 正确处理 ACLRT_LAUNCH_KERNEL 宏调用
6. 添加必要的预处理和后处理逻辑
7. 如果存在"暂不支持"的注释或 TODO，务必替换为实际实现

---

## Phase 4: 编译验证与迭代修复

### 4.1 执行编译

进入目标项目目录，执行编译命令：

```bash
cd {target_project_path}
bash run.sh -v Ascend950PR_9579 -d 0 -c
```

### 4.2 错误分析

捕获编译输出，分析错误类型：

#### A 类：新生成的算子错误（可修复）

特征：
- 错误位置在生成的 `{kernel_name}_kernel.cpp` 或 `v35/{kernel_name}_kernel.h`
- 语法错误、类型不匹配、缺少头文件等
- 错误信息包含生成的文件名

**处理方式**：
1. 提取具体错误信息和位置
2. 分析原因（如缺少 include、类型不匹配、API 调用错误等）
3. 生成修复建议
4. 进入下一轮迭代修复

#### B 类：集成相关错误（可修复）

特征：
- CMakeLists.txt 语法错误
- hkv_hashtable.h 中的调用方式错误
- 头文件路径问题

**处理方式**：
1. 分析具体错误
2. 修复集成代码
3. 重新编译

#### C 类：其他错误（不可修复，报告用户）

特征：
- 环境配置问题（CANN 未安装、版本不匹配等）
- 依赖库缺失
- 其他 kernel 的编译错误
- 错误信息不包含生成的文件名

**处理方式**：
- 报告用户错误信息
- 说明错误原因与新生成的算子无关
- 结束流程

### 4.3 迭代修复流程

```
编译次数 = 0
最大次数 = 3

while 编译次数 < 最大次数:
    执行编译

    if 编译成功:
        报告成功
        结束流程

    分析错误

    if 错误是 C 类（非生成算子导致）:
        报告用户错误信息
        结束流程

    if 错误是 A 类或 B 类:
        编译次数 += 1

        if 编译次数 >= 最大次数:
            报告生成失败
            保留最后一次生成的代码和编译日志
            结束流程

        生成修复建议
        修复代码
        继续下一轮
```

### 4.4 失败报告

如果 3 次编译都失败，生成失败报告：

```markdown
# AscendC Kernel 生成失败报告

## 基本信息
- **Kernel 名称**: {kernel_name}
- **CUDA 源文件**: {cuda_source}
- **生成时间**: {timestamp}

## 编译尝试历史

### 第 1 次尝试
- **错误类型**: {类型}
- **错误信息**:
  ```
  {error_log}
  ```
- **修复措施**: {措施}

### 第 2 次尝试
...

### 第 3 次尝试
...

## 最终错误
```
{final_error}
```

## 保留的文件
- 代码文件: {output_path}/{kernel_name}_kernel.cpp
- 头文件: {output_path}/v35/{kernel_name}_kernel.h
- 完整编译日志: {output_path}/compile.log

## 建议
1. 手动检查生成的代码
2. 参考其他同类 kernel 的实现
3. 检查编译环境配置
```

---

## CUDA → AscendC 核心映射规则

### 函数声明

| CUDA | AscendC |
|------|---------|
| `__global__ void kernel(...)` | `__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...)` |
| `__device__ inline void helper(...)` | 同上（或普通 inline 函数） |
| `template <typename K, typename V, typename S>` | 完全保留，**不变** |

### 内存参数

| CUDA | AscendC |
|------|---------|
| `T* ptr`（全局内存指针参数） | `GM_ADDR ptr_gm`（参数改用 GM_ADDR，命名加 `_gm` 后缀） |
| `const T* ptr`（只读全局内存参数） | `GM_ADDR ptr_gm`（同上，在函数体内转换时加 const） |

### 内部指针转换

| CUDA | AscendC |
|------|---------|
| `Bucket<K,V,S>* buckets = (Bucket<K,V,S>*)buckets_ptr` | `__gm__ Bucket<K,V,S>* __restrict__ buckets = reinterpret_cast<__gm__ Bucket<K,V,S>*>(buckets_gm)` |
| `K* keys = keys_ptr` | `__gm__ const K* __restrict__ keys = reinterpret_cast<__gm__ const K*>(keys_gm)` |
| `bool* founds = founds_ptr` | `__gm__ bool* __restrict__ founds = reinterpret_cast<__gm__ bool*>(founds_gm)` |

> **规则**：函数体内所有 `reinterpret_cast` 必须带 `__gm__` 限定符

### 线程索引

| CUDA | AscendC |
|------|---------|
| `blockIdx.x * blockDim.x + threadIdx.x` | `block_index * blockDim.x + threadIdx.x`（`block_index` 由参数传入） |
| `gridDim.x * blockDim.x`（总线程数） | `thread_all`（由参数传入） |

> **规则**：`blockIdx` 不能在 kernel 体内使用，必须作为参数 `block_index` 传入

### 原子操作

| CUDA | AscendC |
|------|---------|
| `atomicCAS(ptr, old, new)` | `Simt::AtomicCas(ptr, old, new)` |
| `atomicAdd(ptr, val)` | `atomicAdd(ptr, val)`（同名，但 ptr 必须是 `__gm__` 指针） |
| `atomicExch(ptr, val)` | `Simt::AtomicExch(ptr, val)` |

### 常量和函数

| CUDA | AscendC |
|------|---------|
| `EMPTY_KEY`（通常 0xFFFF...） | `EMPTY_KEY`（同名，来自 `types.h`） |
| `LOCKED_KEY` | `LOCKED_KEY` |
| `EMPTY_SCORE` | `EMPTY_SCORE` |
| `IS_RESERVED_KEY(key)` | `IS_RESERVED_KEY<K>(key)` |
| `Murmur3HashDevice(key)` | `Murmur3HashDevice(key)`（同名） |
| `empty_digest<K>()` | `empty_digest<K>()`（同名） |
| `get_digest<K>(key)` | `get_digest<K>(key)`（同名） |
| `get_global_idx(hash, magic, shift, cap)` | `get_global_idx(hash, magic, shift, cap)`（同名） |

---

## 输出要求

### 1. 初始输出目录结构

```
{output_path}/
├── {kernel_name}_kernel.cpp    # 内核入口（dispatcher）文件
└── v35/
    └── {kernel_name}_kernel.h  # kernel 实现头文件
```

### 2. 目标项目集成后的结构

```
{target_project_path}/
├── CMakeLists.txt              # 已添加新 kernel 文件
├── include/
│   └── hkv_hashtable.h         # 已添加 host 调用逻辑
└── hkv_hashtable/
    └── {kernel_name}_kernel/
        ├── {kernel_name}_kernel.cpp
        └── v35/
            └── {kernel_name}_kernel.h
```

### 3. 文件模板

#### v35/{kernel_name}_kernel.h 文件结构

```cpp
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */

#ifndef ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_
#define ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_

#include <kernel_operator.h>
#include <cstdint>
#include "../../../include/types.h"
#include "../../../include/utils.h"
#include "../../../include/score_functor.h"  // 仅当使用 ScoreFunctor 时包含

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;  // 根据 CUDA kernel 的 blockDim 选择 512 或 1024

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int Strategy = -1>  // Strategy 仅当使用 ScoreFunctor 时添加
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void {kernel_name}_vf(
    // GM_ADDR 参数列表（映射自 CUDA kernel 参数）
    GM_ADDR buckets_gm,
    uint64_t capacity,
    uint32_t bucket_capacity,
    // ... 其他参数 ...
    uint64_t total_thread_num,
    uint32_t block_index) {

  // 空指针检查
  if (buckets_gm == nullptr) return;
  // ... 其他必要检查 ...

  // GM_ADDR → __gm__ 指针转换
  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);
  // ... 其他指针转换 ...

  // 线程 ID 计算
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n; kv_idx += total_thread_num) {
    // 算法实现（对应 CUDA kernel 的线程主体逻辑）
    // ...
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_
```

#### {kernel_name}_kernel.cpp 文件结构

```cpp
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */

#include "./v35/{kernel_name}_kernel.h"
#include <cstdint>
#include "../../include/simt_vf_dispatcher.h"
#include "kernel_operator.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void {kernel_name}(
    GM_ADDR buckets,
    // ... 与 _vf 函数相同的 GM_ADDR 和标量参数 ...
    uint64_t n,
    uint64_t global_epoch,
    int32_t evict_strategy,           // 有淘汰策略时包含
    uint32_t value_size,              // 必须：用于 DISPATCH_VALUE_SIZE
    uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  uint64_t system_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
  const uint64_t total_thread_num = THREAD_NUM * GetBlockNum();

  DISPATCH_VALUE_SIZE(
      value_size,
      DISPATCH_EVICT_STRATEGY(
          evict_strategy,
          (Simt::VF_CALL<{kernel_name}_vf<uint64_t, DTYPE, uint64_t, STRATEGY>>(
              Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)},
              buckets,
              // ... 其他参数（与 .h 中的 _vf 函数一致） ...
              n,
              global_epoch,
              total_thread_num,
              system_cycle,
              GetBlockIdx(),
              max_bucket_shift,
              capacity_divisor_magic,
              capacity_divisor_shift))));
}
```

### 4. 关键约束

| 约束 | 说明 |
|------|------|
| **目录结构** | 必须创建 `v35/` 子目录，.h 文件放其中，.cpp 放顶层 |
| **文件名** | `.h`: `v35/{kernel_name}_kernel.h`；`.cpp`: `{kernel_name}_kernel.cpp` |
| **宏定义** | .h 文件必须使用 `ASCENDC_{UPPER}_KERNEL_H_` 保护宏 |
| **函数名** | `_vf` 函数名为 `{kernel_name}_vf`；`.cpp` 入口函数为 `{kernel_name}` |
| **命名空间** | 两个文件都必须在 `namespace npu { namespace hkv {` 内 |
| `THREAD_NUM` | 根据 CUDA kernel 的 blockDim 对应值选 512 或 1024 |
| **dispatcher 关键** | .cpp 中 `GetBlockIdx()` 返回当前 block index，作为 `block_index` 参数传入 _vf |
| **DISPATCH** | 必须包含 `DISPATCH_VALUE_SIZE`；有 ScoreFunctor 时还需 `DISPATCH_EVICT_STRATEGY` |
| `Simt::VF_CALL` | 固定包装，第一参数为 `Simt::Dim3{THREAD_NUM}` |
| **include 路径** | .h 使用 `../../../include/`；.cpp 使用 `../../include/` |
| **入口修饰符** | .cpp 函数必须是 `extern "C" __global__ __aicore__` |

---

## 10. HKV 数据访问方式（重要！）

**严禁使用错误的数据访问方式！** 以下是 HKV 中各类数据的正确访问方法：

### 10.1 Bucket 数据结构

```cpp
template <class K, class V, class S>
struct Bucket {
  __gm__ K* keys_;       // key 数组
  __gm__ S* scores_;     // score 数组
  __gm__ D* digests_;    // digest 数组（1字节摘要）
  __gm__ V* vectors;     // value 向量数组

  // 静态方法：获取 keys 指针
  static __forceinline__ __device__ __gm__ K* keys(__gm__ K* keys, uint32_t offset);

  // 静态方法：获取 digests 指针（重要！）
  static __forceinline__ __device__ __gm__ D* digests(__gm__ K* keys,
                                                      uint32_t bucket_capacity,
                                                      uint32_t offset);

  // 静态方法：获取 scores 指针
  static __forceinline__ __device__ __gm__ S* scores(__gm__ K* keys,
                                                     uint32_t bucket_capacity,
                                                     uint32_t offset);
};
```

### 10.2 正确 vs 错误的访问方式

| 数据类型 | ❌ 错误方式 | ✅ 正确方式 |
|----------|-------------|-------------|
| **Key** | `bucket_keys_ptr[pos]` | `*(bucket_keys + pos)` 或 `BUCKET::keys(bucket_keys, pos)` |
| **Digest** | `bucket_keys_ptr[pos].digest` ❌ | `BUCKET::digests(bucket_keys, bucket_capacity, pos)` ✅ |
| **Score** | `bucket->scores_[pos]` | `BUCKET::scores(bucket_keys, bucket_capacity, pos)` |

### 10.3 Digest 访问详解

Digest 存储在 keys 数组之前，**必须通过静态方法访问**：

```cpp
// ✅ 正确：获取 digest 指针（指向桶内第 pos 个 digest）
__gm__ D* digests_ptr = BUCKET::digests(bucket_keys, bucket_max_size, pos_cur);

// ✅ 正确：批量读取 4 个 digest（向量化读取）
constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);  // = 4
VecD_Comp probe_digests = *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);

// ✅ 正确：计算目标 digest（4 个相同字节）
VecD_Comp target_digests = digests_from_hashed<K>(hashed_key);

// ✅ 正确：比较 4 个 digest
uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
```

**常见错误**：
```cpp
// ❌ 错误：digest 不是 key 的成员
D probe_digest = bucket_keys_ptr[digest_pos].digest;

// ❌ 错误：不能直接通过 bucket 指针访问 digests_
D digest = bucket->digests_[pos];
```

### 10.4 完整示例：Digest 比较流程

```cpp
using BUCKET = Bucket<K, V, S>;
constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);  // = 4

// 1. 计算目标 digest（4 个相同字节）
const K hashed_key = Murmur3HashDevice(key);
VecD_Comp target_digests = digests_from_hashed<K>(hashed_key);

// 2. 线性探测
for (uint32_t offset = 0; offset < bucket_capacity; offset += STRIDE) {
  uint32_t pos_cur = (key_pos + offset) & (bucket_capacity - 1);

  // 3. 获取 digest 指针（正确方式）
  __gm__ D* digests_ptr = BUCKET::digests(bucket_keys, bucket_capacity, pos_cur);

  // 4. 批量读取 4 个 digest
  VecD_Comp probe_digests = *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);

  // 5. 比较 4 个 digest
  uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
  cmp_result &= 0x01010101;

  // 6. 处理匹配结果
  while (cmp_result != 0) {
    uint32_t index = (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
    cmp_result &= (cmp_result - 1);
    uint32_t possible_pos = pos_cur + index;

    // 验证完整 key
    auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
    K current_key = *current_key_ptr;
    if (current_key == key) {
      // 找到匹配
    }
  }
}
```

### 10.5 Key 和 Score 的访问

```cpp
// Key 访问
__gm__ K* current_key_ptr = BUCKET::keys(bucket_keys, pos);
K current_key = *current_key_ptr;

// Score 访问
__gm__ S* scores_ptr = BUCKET::scores(bucket_keys, bucket_capacity, pos);
S score = *scores_ptr;

// 原子操作
K prev = Simt::AtomicCas(current_key_ptr, EMPTY_KEY, LOCKED_KEY);
atomicAdd(scores_ptr, increment);
```

---

## 11. Cooperative Group 编程经验（关键！）

**在使用 `__shfl` 等 cooperative group 操作时必须严格遵守以下规则，否则会导致死锁或 MPU 地址访问错误。**

### 11.1 严禁提前退出

| 场景 | ❌ 错误做法 | ✅ 正确做法 |
|------|-------------|-------------|
| **空指针检查** | `if (ptr == nullptr) return;` | 检查指针但继续执行循环，通过条件判断跳过实际操作 |
| **范围检查** | `if (idx >= n) return;` | 使用 `n_align_warp` 作为循环边界，或用条件标记跳过 |

**原因**：`__shfl` 要求 warp 内所有线程参与，任何线程提前退出都会导致其他线程在 `__shfl` 处无限等待。

### 11.2 循环边界对齐

```cpp
// ❌ 错误：使用 n 作为边界，导致 warp 分歧
for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
     kv_idx < n; kv_idx += thread_all) {  // 不同步！

// ✅ 正确：使用 n_align_warp，确保 warp 内所有线程同时退出循环
for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
     kv_idx < n_align_warp; kv_idx += thread_all) {
    if (kv_idx < n) {  // 内部判断实际工作范围
        // 执行操作
    } else {
        // 设置标记，参与 cooperative group 但不执行实际工作
        occupy_result = OccupyResult::ILLEGAL;
    }
}
```

### 11.3 线程同步机制选择

| 机制 | 适用场景 | 注意事项 |
|------|----------|----------|
| `__shfl` | warp 内线程间数据交换 | 必须确保所有线程到达，不能有条件提前 return |
| `__shfl_xor` | reduction 操作（求最小值/最大值） | 同上 |
| `__threadfence()` | global memory 写操作同步 | 在写入 value 后调用，确保数据可见性 |
| `Simt::ThreadBarrier()` | block 级同步 | 与 `__shfl` 混合使用可能导致死锁，慎用 |

### 11.4 Value 复制的正确模式

```cpp
// ❌ 错误：使用局部 buffer 中转
VecV local_buffer[GROUP_BUF];  // 栈开销大，增加内存操作
CopyValue::ldg_sts(rank, local_buffer, src, dim);
CopyValue::lds_stg(rank, dst, local_buffer, dim);

// ✅ 正确：直接使用 __shfl 协调，直接写入 global memory
for (int32_t i = 0; i < GROUP_SIZE; i++) {
    auto res_sync = __shfl(occupy_result, i, GROUP_SIZE);
    if (res_sync != OccupyResult::REFUSED && 
        res_sync != OccupyResult::ILLEGAL) {
        auto kv_idx_sync = __shfl(kv_idx, i, GROUP_SIZE);
        auto value_start = kv_idx_sync * dim;
        auto key_pos_sync = __shfl(key_pos, i, GROUP_SIZE);
        uint64_t value_ddr_sync = __shfl(bucket_values_uintptr, i, GROUP_SIZE);
        
        for (uint32_t j = cg_rank_id; j < dim; j += GROUP_SIZE) {
            __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                  L1CacheType::NON_CACHEABLE>(
                reinterpret_cast<__gm__ V*>(value_ddr_sync) + 
                    key_pos_sync * dim + j,
                values[value_start + j]);
        }
    }
}
```

### 11.5 参数传递一致性

```cpp
// .cpp 文件
cur_score = (evict_strategy == kLru || evict_strategy == kEpochLru) 
            ? GetSystemCycle() : 0;

// .h _vf 函数接收
template <typename K, typename V, typename S, int32_t Strategy>
__simt_vf__ __aicore__ void kernel_vf(
    GM_ADDR buckets, ..., S cur_score,  // 作为参数传入，不要在 _vf 内调用 GetSystemCycle()
    ...)
```

### 11.6 典型错误模式

**错误 1：条件提前 return 导致死锁**
```cpp
__simt_vf__ __aicore__ void kernel(...) {
    if (buckets_addr_gm == nullptr) {
        return;  // ❌ 错误！其他线程会卡在 __shfl
    }
    // ... cooperative group 操作
}
```

**错误 2：混合使用 barrier 和 shfl**
```cpp
Simt::ThreadBarrier();  // 线程 A 到达
auto x = __shfl(val, 1);  // 线程 B 还没到达 barrier，死锁！
```

**错误 3：指针类型不匹配**
```cpp
__gm__ VecV* bucket_values_ptr;  // VecV 可能是 byte16
bucket_values_ptr = reinterpret_cast<__gm__ VecV*>(bucket->vectors);  // vectors 是 V*
// 应该用 uint64_t 存储地址，使用时再转换
uint64_t bucket_values_uintptr = reinterpret_cast<uint64_t>(bucket->vectors);
```

---

## 思考要求

**在思考过程中只做以下分析**（不在思考中写代码）：
1. CUDA kernel 的算法流程梳理（分步骤）
2. 调用场景分析（确定 kernel 类型、线程数、group size、淘汰策略等）
3. 参数映射决策（哪些参数变 GM_ADDR，哪些保持原类型）
4. 需要增加哪些 AscendC 特有参数（`block_index`, `thread_all` 等）
5. 使用 ScoreFunctor 的策略（如果 CUDA 版本有类似逻辑）
6. `THREAD_NUM` 选择依据

**完整代码只在最终输出中给出。**

---

## 生成原则

1. **正确性优先**：映射语义正确，算法逻辑与 CUDA 版本等价
2. **忠于 CUDA 原文**：不"优化"或"改进"原有算法，只做语法/API 迁移
3. **风格一致**：与参考 AscendC kernel 文件（clear_kernel, find_and_update_kernel 等）风格保持一致
4. **完整可编译**：生成的头文件必须语法完整，包含所有必要的头文件和宏定义
5. **注释保留**：保留 CUDA 源码中有意义的注释，并翻译为中文
6. **可集成**：代码必须能正确集成到 HierarchicalKV-ascend 项目并编译通过

---

## 参考资料

- AscendC SIMT 编程模式：`@references/ascendc-simt-patterns.md`
- HKV 数据结构参考：`@references/hkv-data-structures.md`
