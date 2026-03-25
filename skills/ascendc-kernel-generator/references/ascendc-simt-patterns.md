# AscendC SIMT Kernel 编程模式参考

本文档总结了 HierarchicalKV-ascend 项目中 AscendC SIMT kernel 的编程规范和常用模式，
基于已完成的参考实现（`clear_kernel`, `find_and_update_kernel`, `init_table_kernel`, `insert_and_evict_kernel`）归纳而来。

---

## 1. 函数签名规范

### 1.1 CUDA vs AscendC 函数签名映射

| CUDA | AscendC |
|------|---------|
| `__global__ void kernel(...)` | `__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...)` |
| `__device__ inline void helper(...)` | 同上（或普通 inline 函数） |
| `template <typename K, typename V, typename S>` | 完全保留，**不变** |

### 1.2 标准签名模板

```cpp
template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void {kernel_name}_vf(
    GM_ADDR buckets_addr_gm,          // 哈希桶数组（全局内存）
    GM_ADDR buckets_size_addr_gm,     // 桶大小数组（如需）
    uint64_t capacity,                // 哈希表总容量（桶数 × 桶大小）
    uint32_t bucket_capacity,         // 单桶最大 key 数
    uint32_t dim,                     // value 向量维度
    GM_ADDR keys_addr_gm,             // 输入 key 数组
    // ... 其他 GM_ADDR 参数 ...
    uint64_t n,                       // 本次操作的 key 数量
    uint64_t thread_all,              // 总线程数（gridDim × blockDim）
    uint64_t global_epoch,            // 当前 epoch（用于 score 计算）
    uint32_t block_index,             // 当前 block 索引（对应 CUDA blockIdx.x）
    uint32_t max_bucket_shift,        // 桶索引位移量（log2(bucket_capacity)）
    uint64_t capacity_divisor_magic,  // 快速取模魔数
    uint64_t capacity_divisor_shift   // 快速取模位移
)
```

### 1.3 关键修饰符说明

| 修饰符 | 含义 |
|--------|------|
| `__simt_vf__` | 标识函数为 AscendC SIMT VF（Vector Function）kernel |
| `__aicore__` | 声明运行在 AI Core 上（对应 CUDA 的 `__global__` 或 `__device__`） |
| `LAUNCH_BOUND(N)` | 声明 kernel 最大线程数，N 为 `THREAD_NUM`（通常 512 或 1024） |
| `inline` | 必须声明为内联 |

### 1.4 THREAD_NUM 选择规则

| CUDA kernel blockDim | AscendC THREAD_NUM |
|---------------------|--------------------|
| 256 | 512（向上取整到 AscendC 支持值） |
| 512 | 512 |
| 1024 | 1024 |

惯例：
- 涉及 key-value 遍历操作（find/insert/evict）：`THREAD_NUM = 512`
- 简单批量初始化/清空操作：`THREAD_NUM = 1024`

---

## 2. 全局内存参数（GM_ADDR）

### 2.1 基本类型映射

| CUDA | AscendC |
|------|---------|
| `T* ptr`（参数） | `GM_ADDR ptr_gm` |
| `Bucket<K,V,S>* buckets` | `GM_ADDR buckets_gm` |
| `const K* keys` | `GM_ADDR keys_gm` |
| `V** value_ptrs` | `GM_ADDR value_ptrs_gm` |

### 2.2 参数声明规则

所有来自主机分配（aclrtMalloc）的全局内存指针，在参数中必须声明为 `GM_ADDR` 类型：

```cpp
// ✅ 正确
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void my_kernel_vf(
    GM_ADDR buckets_gm,
    GM_ADDR keys_gm,
    GM_ADDR values_gm,
    uint64_t n,
    ...
)

// ❌ 错误
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void my_kernel_vf(
    Bucket<K,V,S>* buckets,  // 不能直接用指针类型
    K* keys,
    ...
)
```

### 2.3 内部指针转换规则

在函数体内，将 `GM_ADDR` 转换为带 `__gm__` 限定符的指针：

```cpp
// 读写型全局指针
__gm__ Bucket<K, V, S>* __restrict__ buckets =
    reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);

// 只读型全局指针（对应 CUDA 的 const T*）
__gm__ const K* __restrict__ keys =
    reinterpret_cast<__gm__ const K*>(keys_gm);

// 标量输出
__gm__ bool* __restrict__ founds =
    reinterpret_cast<__gm__ bool*>(founds_gm);

// 指针数组（如 V** value_ptrs）
__gm__ V* __gm__* __restrict__ value_ptrs =
    reinterpret_cast<__gm__ V* __gm__*>(value_ptrs_gm);

// 整型指针
__gm__ int32_t* __restrict__ buckets_size =
    reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
```

> **核心规则**：函数体内所有全局内存指针赋值必须带 `__gm__` 限定符，禁止省略。

### 2.4 参数封装模式差异

| 模式 | CUDA | AscendC |
|------|------|---------|
| 结构体封装 | `Table*` 包含所有元数据 | 元数据展开为独立参数 |
| 元数据访问 | `table->bucket_max_size` | 通过 `GM_ADDR` 传入独立参数 |

**CUDA 结构体访问 vs AscendC 独立参数**：

```cpp
// CUDA: 通过 Table* 访问元数据
template <class K, class V, class S>
__global__ void clear_kernel(Table<K,V,S>* table, Bucket<K,V,S>* buckets, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t bucket_max_size = table->bucket_max_size;  // 从结构体读取
    for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
        int bkt_idx = t / bucket_max_size;
        table->buckets_size[bkt_idx] = 0;  // 修改结构体内数据
    }
}

// AscendC: 元数据作为独立参数传入
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void clear_kernel_vf(
    GM_ADDR buckets_gm, GM_ADDR buckets_size_gm,
    uint64_t table_capacity,           // 原 table->capacity
    uint32_t bucket_max_size,          // 原 table->bucket_max_size
    uint64_t thread_all, uint32_t block_index) {
    // 直接访问传入的参数，无需通过结构体
    for (size_t t = tid; t < table_capacity; t += thread_all) {
        size_t bkt_idx = t / bucket_max_size;
        buckets_size[bkt_idx] = 0;
    }
}
```

---

## 3. 线程索引模式

### 3.1 CUDA vs AscendC 线程索引映射

| CUDA | AscendC |
|------|---------|
| `blockIdx.x`（kernel 内部使用） | `block_id` 或 `block_index`（作为参数传入） |
| `blockDim.x` | `blockDim.x`（AscendC 保留） |
| `threadIdx.x` | `threadIdx.x`（AscendC 保留） |
| `gridDim.x * blockDim.x` | `total_thread_num`（作为参数传入） |

### 3.2 基础线程 ID 计算

```cpp
// AscendC（block_index 作为参数传入，不使用 blockIdx）
uint64_t kv_idx = (uint64_t)block_index * blockDim.x + threadIdx.x;

// 步进式遍历（处理 n > total_thread_num 的情况）
for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
     kv_idx < n; kv_idx += total_thread_num) {
    // 每线程处理一个 key
}
```

### 3.3 对比 CUDA

```cpp
// CUDA（不允许在 AscendC 中使用）
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
```

---

## 4. 原子操作

### 4.1 CUDA vs AscendC 原子操作映射

| CUDA | AscendC |
|------|---------|
| `atomicCAS(ptr, old, new)` | `Simt::AtomicCas(ptr, old, new)` |
| `atomicAdd(ptr, val)` | `atomicAdd(ptr, val)`（同名，但 ptr 必须是 `__gm__` 指针） |
| `atomicExch(ptr, val)` | `Simt::AtomicExch(ptr, val)` |
| `compare_exchange_strong` | `Simt::AtomicCas` |
| `store/release` | `Simt::AtomicExch` |

### 4.2 Compare-And-Swap

```cpp
// 抢占 key 位置（将 EMPTY_KEY 替换为 LOCKED_KEY）
K prev = Simt::AtomicCas(current_key_ptr, EMPTY_KEY, LOCKED_KEY);
if (prev == EMPTY_KEY) {
    // 成功抢占，可以写入
}

// 抢占已有 key（确保同一 key 不被多线程同时写）
K prev = Simt::AtomicCas(current_key_ptr, existing_key, LOCKED_KEY);
if (prev == existing_key) {
    // 成功抢占
}
```

### 4.3 锁定/解锁模式对比

```cpp
// CUDA: 锁定 key
K expected_key = key;
bool result = current_key->compare_exchange_strong(
    expected_key, static_cast<K>(LOCKED_KEY),
    cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);

// AscendC: 锁定 key
K try_key = Simt::AtomicCas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
if (try_key == key) {
    // 成功锁定
}

// CUDA: 解锁
key_address->store(key, cuda::std::memory_order_release);

// AscendC: 解锁
(void)Simt::AtomicExch(current_key_ptr, key);
```

### 4.4 Add

```cpp
// 递增桶大小
atomicAdd(bucket_size, 1);

// 获取淘汰位置
uint64_t evicted_idx = atomicAdd(d_evicted_counter, 1UL);
```

---

## 5. 线程同步与协作

### 5.1 CUDA vs AscendC 协作映射

| CUDA | AscendC |
|------|---------|
| `cg::tiled_partition<N>(cg::this_thread_block())` | 无需创建，直接使用 `__shfl` |
| `g.thread_rank()` | `threadIdx.x % GROUP_SIZE` |
| `g.ballot(cond)` | 需手动实现或使用 `__shfl` 传播 |
| `g.shfl(val, lane)` | `__shfl(val, lane, GROUP_SIZE)` |
| `g.shfl_xor(val, mask)` | `__shfl_xor(val, mask, GROUP_SIZE)` |
| `__syncthreads()` | `AscendC::Simt::ThreadBarrier()` |
| `__threadfence()` | `__threadfence()` |

### 5.2 协作组模式对比

```cpp
// CUDA: 创建协作组
auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
int rank = g.thread_rank();
int val_from_lane0 = g.shfl(my_val, 0);

// AscendC: 直接使用 shuffle
int rank = threadIdx.x % GROUP_SIZE;
int val_from_lane0 = __shfl(my_val, 0, GROUP_SIZE);
```

---

## 6. 数据加载指令

### 6.1 CUDA vs AscendC 数据加载映射

| CUDA | AscendC |
|------|---------|
| `__ldg(ptr)` | `ldg_l2nc_l1c(ptr)` |
| `*(ptr)` | `*(ptr)` 或 `ldg_l2nc_l1c(ptr)` |
| `__ldcg(ptr)` | `__ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(ptr)` |

### 6.2 使用示例

```cpp
// 使用 ldg_l2nc_l1c 进行优化加载
__gm__ D* digests_ptr = BUCKET::digests(bucket_keys, bucket_capacity, pos_cur);
VecD_Comp probe_digests = ldg_l2nc_l1c(reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr));
```

---

## 7. C++ 特性差异

### 7.1 特性对比

| 特性 | CUDA | AscendC |
|------|------|---------|
| Placement new | `new (ptr) Type{value}` | 直接赋值（`AtomicKey`/`AtomicScore` 为简单类型） |
| 模板参数推导 | 自动推导 | 通过 `DISPATCH_VALUE_SIZE` 等宏分发 |
| 结构体方法 | `bucket->keys(i)` | 直接访问成员 `bucket->keys_[i]` |

### 7.2 Placement new vs 直接赋值

```cpp
// CUDA: 使用 placement new 初始化原子类型
template <class K, class V, class S>
__global__ void create_atomic_keys(Bucket<K,V,S>* buckets, ...) {
    for (size_t i = 0; i < bucket_max_size; i++) {
        new (buckets[start + tid].keys(i))
            AtomicKey<K>{static_cast<K>(EMPTY_KEY)};  // placement new
    }
}

// AscendC: 直接赋值
__simt_vf__ __aicore__ inline void create_atomic_keys_vf(...) {
    // AtomicKey 定义为简单类型别名，直接赋值
    bucket->keys_[i] = EMPTY_KEY;
    bucket->scores_[i] = EMPTY_SCORE;
}
```

---

## 8. 哈希定位模式（标准模板）

```cpp
// 1. 计算哈希值
const K hashed_key = Murmur3HashDevice(key);

// 2. 快速取模定位全局槽位（capacity_divisor_magic/shift 避免除法）
uint64_t global_idx = get_global_idx(hashed_key,
                                      capacity_divisor_magic,
                                      capacity_divisor_shift,
                                      capacity);

// 3. 分解为桶索引和桶内位置
uint32_t key_pos = global_idx & (bucket_capacity - 1);   // 桶内初始位置
uint64_t bkt_idx = global_idx >> max_bucket_shift;        // 桶索引

// 4. 获取桶指针
__gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
__gm__ K* bucket_keys = bucket->keys_;
__gm__ V* bucket_values = bucket->vectors;
__gm__ S* bucket_scores = bucket->scores_;

// 5. 线性探测
for (uint32_t offset = 0; offset < bucket_capacity; offset++) {
    uint32_t current_pos = (key_pos + offset) % bucket_capacity;
    auto current_key_ptr = bucket_keys + current_pos;
    K current_key = *current_key_ptr;
    // 判断：找到目标 key / 找到空位 / 继续探测
}
```

---

## 9. 空指针保护

在 kernel 函数开头，必须对关键指针参数做空值保护：

```cpp
// 示例：find_and_update_kernel_vf
if (buckets_gm == nullptr) return;
if (keys_gm == nullptr) return;
if (value_ptrs_gm == nullptr) return;
```

---

## 10. INVALID_KEY_POS 模式

用于标识"未找到目标位置"的哨兵值：

```cpp
constexpr uint32_t INVALID_KEY_POS = UINT32_MAX;  // 定义在 types.h 中

uint32_t target_pos = INVALID_KEY_POS;

// 线性探测后
if (target_pos == INVALID_KEY_POS) {
    // 未找到目标位置，根据 kernel 语义决定处理方式
}
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

## 12. 完整的函数头文件模板

```cpp
/**
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
// 仅在使用 ScoreFunctor 时包含：
// #include "../../../include/score_functor.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void {kernel_name}_vf(
    // 参数列表
) {
  // 空指针检查
  // ...

  // GM_ADDR → __gm__ 指针转换
  // ...

  // 主循环（步进式线程遍历）
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n; kv_idx += total_thread_num) {
    // 每线程处理逻辑
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_
```

---

## 13. 已完成参考 kernel 一览

| Kernel | 文件路径 | 核心操作 |
|--------|----------|----------|
| `clear_kernel_vf` | `hkv_hashtable/clear_kernel/v35/clear_kernel.h` | 遍历所有槽位，写入 EMPTY_KEY/EMPTY_SCORE，重置桶大小 |
| `find_and_update_kernel_vf` | `hkv_hashtable/find_and_update_kernel/v35/find_and_update_kernel.h` | 哈希定位 → 线性探测 → 返回 value 指针和 found 标志 |
| `init_table_kernel` (多函数) | `hkv_hashtable/init_table_kernel/v35/init_table_kernel.h` | 初始化桶指针、填充 EMPTY_KEY/EMPTY_SCORE |
| `insert_and_evict_kernel_vf` | `hkv_hashtable/insert_and_evict_kernel/v35/insert_and_evict_kernel.h` | 插入 key，满时按 score 淘汰最旧 key |

在生成新 kernel 时，优先参考**功能最相似**的已有实现作为代码风格模板。

---

## 14. `.cpp` Dispatcher 文件模板

每个算子目录除 `v35/<kernel>.h` 外，还需在顶层生成 `<kernel>.cpp`，作为 AscendC 内核入口（对应 CUDA 的 `__global__` 启动点）。

### 14.1 标准模板（含淘汰策略）

```cpp
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    uint32_t value_size,              // 必须：用于类型分发
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
              // ... 其他 GM_ADDR 参数 ...
              n,
              update_score,
              global_epoch,
              total_thread_num,
              system_cycle,
              GetBlockIdx(),
              max_bucket_shift,
              capacity_divisor_magic,
              capacity_divisor_shift))));
}
```

### 14.2 简单模板（无淘汰策略，如 clear_kernel）

```cpp
#include "./v35/{kernel_name}_kernel.h"
#include "../../include/simt_vf_dispatcher.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void {kernel_name}(
    GM_ADDR buckets,
    GM_ADDR buckets_size,
    size_t bucket_max_size,
    size_t table_capacity,
    uint32_t value_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const uint64_t thread_all = THREAD_NUM * GetBlockNum();

  DISPATCH_VALUE_SIZE(
      value_size,
      (Simt::VF_CALL<{kernel_name}_vf<uint64_t, DTYPE, uint64_t>>(
          Simt::Dim3{THREAD_NUM},
          buckets, buckets_size, bucket_max_size,
          table_capacity, thread_all, GetBlockIdx())));
}
```

### 14.3 关键规则

| 规则 | 说明 |
|------|------|
| **入口函数名** | `{kernel_name}`（不带 `_vf` 后缀） |
| **修饰符** | `extern "C" __global__ __aicore__`（固定写法） |
| **block_index** | 在 `.cpp` 内用 `GetBlockIdx()` 获取，传入 `_vf` |
| **total_thread_num** | `THREAD_NUM * GetBlockNum()`，传入 `_vf` 的 `thread_all` 参数 |
| **DISPATCH_VALUE_SIZE** | 始终需要，将 `value_size` 映射为 `DTYPE` 模板参数 |
| **DISPATCH_EVICT_STRATEGY** | 仅当 kernel 使用 ScoreFunctor / 淘汰策略时包含 |
| **DISPATCH_GROUP_SIZE** | 仅当 kernel 有协作组（insert_and_evict 类）时包含 |
| **Simt::VF_CALL** | 固定包装，第一参数为 `Simt::Dim3{THREAD_NUM}` |
| **include 路径** | `./v35/{kernel_name}_kernel.h`、`../../include/simt_vf_dispatcher.h` |
| **system_cycle** | 仅当 _vf 函数需要时：`static_cast<uint64_t>(AscendC::GetSystemCycle())` |

---

## 15. 常见陷阱与解决方案

### 15.1 指针转换遗漏 `__gm__` 限定符

**问题**：全局内存指针转换后缺少 `__gm__` 限定符

**正确做法**：
```cpp
// ✅ 正确
__gm__ Bucket<K,V,S>* buckets = reinterpret_cast<__gm__ Bucket<K,V,S>*>(buckets_gm);

// ❌ 错误
Bucket<K,V,S>* buckets = reinterpret_cast<Bucket<K,V,S>*>(buckets_gm);
```

### 15.2 线程索引计算错误

**问题**：使用 CUDA 内置变量 `blockIdx.x`

**正确做法**：
```cpp
// ✅ 正确：从参数获取
uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;

// ❌ 错误
uint64_t kv_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 15.3 协作组大小不匹配

**问题**：`GROUP_SIZE` 与 CUDA 版本不一致，导致 shuffle 数据错位

**解决**：确保 `GROUP_SIZE` 与 CUDA 代码中的 `tiled_partition<N>` 的 N 一致
