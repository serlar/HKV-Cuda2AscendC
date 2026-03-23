---
name: cuda-kernel-analyzer
description: >
  HKV CUDA Kernel 分析 Skill — 读取或接收 CUDA SIMT kernel 代码，深入理解其功能、
  数据结构和算法逻辑，设计测试用例，并生成对应的 AscendC 测试 C++ 文件。
  输入来源支持两种方式：.cuh 文件路径 或 直接粘贴的 kernel 代码片段。
  测试文件遵循 test_find_and_update.cpp 中 GoogleTest 风格的模式。
argument-hint: >
  必需：kernel_name（算子名称）、output_path（输出目录）。
  来源二选一（优先级：代码片段 > 文件路径）：
    - cuda_kernel_code：直接粘贴的 CUDA kernel 代码字符串
    - cuda_kernel_file：CUDA .cuh 文件的绝对路径
  输出：analysis.md（功能分析报告）、test_<kernel_name>.cpp（AscendC 测试文件）。
---

# CUDA Kernel 分析与测试生成 Skill

<role>
你是一个 HKV CUDA/AscendC 迁移专家。你的任务是：
1. 深入阅读并分析 CUDA kernel 源码，理解其算子功能
2. 设计全面的测试用例（覆盖正常、边界、异常场景）
3. 生成符合 HierarchicalKV-ascend 测试规范的 AscendC C++ 测试文件

参考格式来自：`HierarchicalKV-ascend/tests/test_find_and_update.cpp`
</role>

---

## Step 1: 获取并分析 CUDA kernel

### 1.1 确定输入来源

按以下优先级判断 CUDA 源码的获取方式：

```
if cuda_kernel_code 非空:
    # 方式A：直接使用传入的代码片段
    kernel_source = cuda_kernel_code
    source_desc   = "用户提供的代码片段"
elif cuda_kernel_file 非空:
    # 方式B：从文件读取
    读取 cuda_kernel_file 的完整内容
    kernel_source = 文件内容
    source_desc   = cuda_kernel_file 路径
else:
    # 两者都未提供，向用户报错
    报告："请提供 cuda_kernel_code（代码片段）或 cuda_kernel_file（文件路径）"
    终止
```

> **代码片段输入注意事项**：
> - 片段可以是单个函数、多个函数或整个文件内容，不做格式限制
> - 若片段中包含多个 `__global__` 函数，以 `kernel_name` 参数指定的函数名为主分析目标；
>   其余函数作为辅助背景信息（如 helper 函数、配套的 unlock kernel 等）
> - 若片段引用了未定义的外部函数（如 `find_and_lock_when_vacant`），
>   在分析报告中注明"依赖外部 helper，行为推断自函数名和调用上下文"

### 1.2 阅读并分析 kernel 源码

对 `kernel_source` 重点分析：

1. **函数签名**：
   - 函数名、模板参数（K/V/S 类型）
   - 所有输入/输出参数及其含义
   - `__global__`、`__device__` 等 CUDA 修饰符

2. **算子功能**：
   - 哈希表操作类型（查找/插入/清空/初始化/淘汰等）
   - 核心算法流程（哈希定位 → 线性探测 → 写入/读取）
   - 对哈希桶（Bucket）的读写方式

3. **线程并行模型**：
   - 线程并行粒度（每线程处理一个 key、一个 bucket slot 等）
   - 使用了哪些原子操作（`atomicCAS`, `atomicAdd` 等）
   - CUDA block/grid 配置策略

4. **数据结构使用**：
   - `Bucket<K, V, S>` 的哪些字段被读写（`keys_`, `scores_`, `digests_`, `vectors`）
   - 特殊常量：`EMPTY_KEY`, `LOCKED_KEY`, `RECLAIM_KEY`, `EMPTY_SCORE`
   - 哈希函数：`Murmur3HashDevice` 的使用

5. **返回/输出语义**：
   - 输出数组（`founds`, `values`, `scores`, `evicted_*` 等）的含义
   - 成功/失败条件

### 1.3 生成分析报告

将分析结果写入 `<output_path>/analysis.md`，格式如下：

```markdown
# {kernel_name} CUDA Kernel 分析

## 来源
- 输入方式：<"代码片段" 或 "文件: {cuda_kernel_file}">
- 主分析函数：`{cuda_函数名}`
- 辅助函数（如有）：<列举片段中其他 __global__ 或关键 helper 函数>

## 功能概述
<一段话描述算子做了什么>

## 函数签名
```cpp
<完整的 CUDA 函数原型>
```

## 参数说明
| 参数名 | 类型 | 方向 | 说明 |
|--------|------|------|------|
| ...    | ...  | 输入/输出 | ... |

## 算法流程
1. <步骤 1>
2. <步骤 2>
...

## 并行模型
- 并行粒度：每线程处理 <N> 个 key
- 原子操作：<列举>
- 线程总数计算：<说明>

## 关键 CUDA → AscendC 映射点
| CUDA | AscendC 等价 |
|------|-------------|
| `__global__` | `__simt_vf__ __aicore__` |
| `blockIdx.x` | `block_index`（传参） |
| `blockDim.x * blockIdx.x + threadIdx.x` | `block_index * blockDim.x + threadIdx.x` |
| `atomicCAS` | `Simt::AtomicCas` |
| <其他> | <对应 AscendC API> |
```

---

## Step 2: 设计测试用例

根据 Step 1 的分析，为 `{kernel_name}` 设计以下类别的测试用例：

### 必须覆盖的测试场景

#### 基础功能测试（每个 kernel 必有）

| 测试名 | 说明 |
|--------|------|
| `basic_function` | 正常情况下的基本功能验证（插入后操作，验证预期结果） |
| `empty_table` | 在空表上操作（验证不崩溃，结果符合空表语义） |
| `zero_keys` | n=0 时调用（验证边界处理，不崩溃） |
| `single_key` | n=1 时调用（最小正常情况） |

#### 场景特定测试

根据 kernel 功能，选择适用的：

| 适用 kernel | 测试名 | 说明 |
|-------------|--------|------|
| 查找类（find/lookup） | `partial_keys_exist` | 部分 key 存在、部分不存在时的查找结果 |
| 查找类 | `find_after_clear` | 清表后查找，所有 key 应不存在 |
| 插入类（insert/upsert） | `insert_duplicate_keys` | 插入重复 key，验证覆盖语义 |
| 插入类 | `insert_until_full` | 填满哈希桶后的插入行为 |
| 淘汰类（evict） | `eviction_triggered` | 触发淘汰时验证淘汰 key 正确 |
| 清空类（clear） | `clear_then_insert` | 清空后重新插入，验证状态恢复正常 |
| 初始化类（init） | `init_state_check` | 初始化后验证 EMPTY_KEY/EMPTY_SCORE 填充正确 |

#### 数据规模测试

| 测试名 | 说明 |
|--------|------|
| `random_keys` | 使用随机 key（`create_random_keys` 辅助函数） |
| `large_scale` | 大规模数据（key_num = 64K+） |
| `small_dim` | 小 dim（dim=4） |
| `large_dim` | 大 dim（dim=128 或 256） |

---

## Step 3: 生成 AscendC 测试文件

### 3.1 文件结构模板

生成的测试文件必须严格遵循以下结构（参考 `test_find_and_update.cpp`）：

```cpp
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * ... Apache License header ...
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;

// 测试夹具类
class {KernelName}Test : public ::testing::Test {
 protected:
  static constexpr size_t hbm_for_values = 1UL << 30;
  static constexpr size_t init_capacity = 128UL * 1024;

  void SetUp() override {
    init_env();
    size_t total_mem = 0;
    size_t free_mem = 0;
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem), ACL_ERROR_NONE);
    ASSERT_GT(free_mem, hbm_for_values)
        << "free HBM is not enough free:" << free_mem
        << " need:" << hbm_for_values;
    ASSERT_EQ(aclrtCreateStream(&stream_), ACL_ERROR_NONE);
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      aclrtDestroyStream(stream_);
      stream_ = nullptr;
    }
  }

  template <typename T>
  T* alloc_device_mem(size_t count) {
    T* ptr = nullptr;
    EXPECT_EQ(aclrtMalloc(reinterpret_cast<void**>(&ptr),
                          count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    return ptr;
  }

  template <typename T>
  void free_device_mem(T* ptr) {
    if (ptr != nullptr) EXPECT_EQ(aclrtFree(ptr), ACL_ERROR_NONE);
  }

  template <typename T>
  void copy_to_device(T* dst, const T* src, size_t count) {
    EXPECT_EQ(aclrtMemcpy(dst, count * sizeof(T), src, count * sizeof(T),
                          ACL_MEMCPY_HOST_TO_DEVICE), ACL_ERROR_NONE);
  }

  template <typename T>
  void copy_to_host(T* dst, const T* src, size_t count) {
    EXPECT_EQ(aclrtMemcpy(dst, count * sizeof(T), src, count * sizeof(T),
                          ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);
  }

  aclrtStream stream_ = nullptr;
};

// ===== 各测试用例 =====

// 测试 1：基本功能测试
TEST_F({KernelName}Test, basic_function) {
  // ... 按照设计的测试用例实现 ...
}

// ... 其余测试用例 ...
```

### 3.2 关键规范

| 规范项 | 要求 |
|--------|------|
| **测试类名** | `{KernelName}Test`（驼峰式，与 kernel 名对应） |
| **SetUp/TearDown** | 必须包含 `init_env()`、ACL stream 创建/销毁 |
| **HBM 内存检查** | SetUp 中必须用 `aclrtGetMemInfo` 检查 HBM 可用内存 |
| **内存管理** | 所有 device 内存通过 `alloc_device_mem`/`free_device_mem` 管理 |
| **数据拷贝** | 必须使用 `copy_to_device`/`copy_to_host` 辅助函数 |
| **辅助函数** | 使用 `test_util.h` 中的 `create_continuous_keys` 和 `create_random_keys` |
| **流同步** | 每次 kernel 调用后必须 `aclrtSynchronizeStream(stream_)` |
| **测试粒度** | 每个测试用例只验证一个场景，测试名反映场景 |
| **注释** | 每个测试用例前有中文注释说明测试目的 |
| **资源释放** | 每个测试用例结尾必须调用 `free_device_mem` 释放所有分配的 device 内存 |

### 3.3 生成文件

将完整测试文件内容写入 `<output_path>/test_<kernel_name>.cpp`。

---

## Step 4: 验证（静态检查，禁止跳过）

对生成的测试文件进行以下静态检查：

```
✅ 包含 gtest/gmock 头文件
✅ 包含 acl/acl.h 和 hkv_hashtable.h
✅ 包含 test_util.h
✅ 有 {KernelName}Test 测试夹具类
✅ SetUp 中有 init_env()
✅ SetUp 中有 aclrtGetMemInfo 检查
✅ SetUp 中有 aclrtCreateStream
✅ TearDown 中有 aclrtDestroyStream
✅ 至少包含 basic_function、empty_table、zero_keys、single_key 4 个测试
✅ 每个 TEST_F 都有 aclrtSynchronizeStream 调用
✅ 每个 TEST_F 都有完整的 free_device_mem 调用
```

若检查失败，修复问题后**重新生成**（最多重试 2 次）。

---

## Step 5: 用户确认（必须执行）

验证通过后，**必须使用 `question` 工具**展示生成内容并请求确认：

展示：
1. `analysis.md` 中的功能概述和参数说明
2. 测试用例列表（每个测试的名称和说明）
3. `test_<kernel_name>.cpp` 的完整内容

询问用户：
> CUDA 算子分析和测试文件生成完成，请查看：
>
> 请选择：
> 1. 接受，继续代码生成
> 2. <请输入修改意见>

**处理回复**：
- 用户接受 → skill 完成，返回成功
- 用户要求修改 → 结合反馈重新生成（返回 Step 2 或 Step 3）

---

## 关键约束

| 约束 | 说明 |
|------|------|
| 测试文件必须 self-contained | 不能引用项目外部头文件（除了 acl, gtest, hkv_hashtable, test_util） |
| 测试场景必须真实 | 基于 CUDA kernel 的实际功能设计，不能凭空捏造 |
| 禁止跳过确认 | 必须通过 `question` 工具获得用户确认 |
| 分析报告必须写入文件 | `analysis.md` 必须保存，供代码生成阶段参考 |

---

## 参考资料

- AscendC SIMT 编程模式：`@references/ascendc-simt-patterns.md`
- HKV 数据结构参考：`@references/hkv-data-structures.md`
