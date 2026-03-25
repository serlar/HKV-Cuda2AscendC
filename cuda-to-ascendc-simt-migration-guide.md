# CUDA SIMT 到 AscendC SIMT 迁移经验指南

本文档总结 CUDA HierarchicalKV (HKV) 算子迁移到 AscendC SIMT 算子的经验，包含编程语言层面差异和 HKV 操作层面差异。

---

## 一、CUDA HKV 到 AscendC HKV 算子映射关系

| CUDA HKV Kernel | 文件位置 | 功能描述 | AscendC HKV Kernel | 文件位置 |
|----------------|----------|----------|-------------------|----------|
| `tlp_update_score_kernel` | `update_score.cuh` | 为指定 key 批量更新 score，使用 filter 模式加速查找 | `assign_scores_kernel_with_filter_vf` | `assign_scores_kernel/v35/assign_scores_kernel_with_filter.h` |
| `tlp_lookup_ptr_kernel_with_filter` (update_score=true 分支) | `lookup_ptr.cuh:26` | 查找 key 并返回 value 指针，同时更新 score（如 LRU/LFU） | `find_and_update_kernel_with_filter_vf` | `find_and_update_kernel/v35/find_and_update_kernel_with_filter.h` |
| `tlp_lookup_ptr_kernel_with_filter` (update_score=false 分支) | `lookup_ptr.cuh:26` | 纯查找 key 并返回 value 指针，不更新 score | `find_ptr_with_digest_kernel_vf` | `find_ptr_kernel/v35/find_ptr_with_digest_kernel.h` |
| `tlp_v1_upsert_and_evict_kernel_unique` | `upsert_and_evict.cuh:27` | 插入 key-value，桶满时按 score 淘汰旧 key，返回被淘汰的数据 | `insert_and_evict_kernel_with_digest_vf` | `insert_and_evict_kernel/v35/insert_and_evict_kernel.h` |
| `find_or_insert_ptr_kernel_lock_key` | `find_ptr_or_insert.cuh` | 查找 key 返回指针，不存在则插入新 key，原子锁定保证一致性 | `find_or_insert_ptr_kernel_v2_vf` | `find_or_insert_ptr_kernel/v35/find_or_insert_ptr_kernel_v2.h` |
| `tlp_v2_upsert_kernel_with_io` | `upsert.cuh` | 插入或更新 key-value，已存在则更新，不存在则插入（upsert 语义） | `insert_or_assign_kernel_with_digest_vf` | `insert_or_assign_kernel/v35/insert_or_assign_kernel.h` |
| `clear_kernel` | `core_kernels.cuh:643` | 清空哈希表所有槽位，重置 digests、keys 和 buckets_size | `clear_kernel_vf` | `clear_kernel/v35/clear_kernel.h` |
| `dump_kernel` (多个版本) | `core_kernels.cuh:832` | 导出哈希表中所有或符合条件的 key-value-score 数据 | `dump_kernel_vf` | `dump_kernel/v35/dump_kernel.h` |
| `create_atomic_keys` | `core_kernels.cuh:57` | 初始化 bucket 的 keys 和 digests 为 EMPTY_KEY/empty_digest | `create_atomic_keys_vf` | `init_table_kernel/v35/init_table_kernel.h` |
| `create_atomic_scores` | `core_kernels.cuh:71` | 初始化 bucket 的 scores 为 EMPTY_SCORE | `create_atomic_scores_vf` | `init_table_kernel/v35/init_table_kernel.h` |
| `allocate_bucket_vectors` | `core_kernels.cuh:84` | 为 bucket 分配 value 向量内存 | `allocate_bucket_vectors_vf` | `init_table_kernel/v35/init_table_kernel.h` |
| `allocate_bucket_others` | `core_kernels.cuh:91` | 为 bucket 分配 digests、keys、scores 内存 | `allocate_bucket_others_vf` | `init_table_kernel/v35/init_table_kernel.h` |
| `rehash_kernel_for_fast_mode` | `core_kernels.cuh:519` | 表容量翻倍后重新哈希所有 key 到新位置 | `rehash_kernel_vf` | `rehash_kernel/v35/rehash_kernel.h` |
| `defragmentation_for_rehash` | `core_kernels.cuh:433` | 重哈希后的碎片整理，将 key 移动到正确位置 | `defragmentation_for_rehash_vf` | `rehash_kernel/v35/rehash_kernel.h` |

### 映射规律总结

1. **纯查找类算子**（无 Score 更新）：`find_ptr_with_digest_kernel_vf`
   - 无 Strategy 模板参数
   - 无 `update_score` 参数
   - 使用 `ldg_l2nc_l1c` 加载数据，无原子锁操作

2. **查找+更新类算子**（有 Score 更新）：`find_and_update_kernel_with_filter_vf`
   - 有 Strategy 模板参数（用于 ScoreFunctor）
   - 有 `update_score` 参数控制是否更新
   - 使用 `AtomicCas` 锁定/解锁

3. **插入淘汰类算子**：`insert_and_evict_kernel_with_digest_vf`
   - 使用协作组（GROUP_SIZE）进行并行淘汰
   - 有 `evicted_keys/values/scores` 输出参数
   - 使用 `DISPATCH_GROUP_SIZE` 宏分发

---

## 二、编程语言层面差异

### 2.1 Kernel 修饰符与函数签名

| CUDA | AscendC |
|------|---------|
| `__global__ void kernel(...)` | `__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...)` |
| `blockIdx.x`（kernel 内部使用） | `block_id` 或 `block_index`（作为参数传入） |
| `blockDim.x` | `blockDim.x`（AscendC 保留） |
| `threadIdx.x` | `threadIdx.x`（AscendC 保留） |
| `gridDim.x * blockDim.x` | `total_thread_num`（作为参数传入） |

**迁移示例**：

```cpp
// CUDA
__global__ void my_kernel(Bucket<K,V,S>* buckets, const K* keys, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    // ...
}

// AscendC
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void my_kernel_vf(
    GM_ADDR buckets_gm, GM_ADDR keys_gm, uint64_t n,
    uint64_t total_thread_num, uint32_t block_id) {
    uint64_t tid = block_id * blockDim.x + threadIdx.x;
    // ...
}
```

### 2.2 全局内存指针与参数传递模式

#### 基本类型映射

| CUDA | AscendC |
|------|---------|
| `T* ptr`（参数） | `GM_ADDR ptr_gm` |
| `Bucket<K,V,S>* buckets` | `GM_ADDR buckets_gm` |
| `const K* keys` | `GM_ADDR keys_gm` |
| `V** value_ptrs` | `GM_ADDR value_ptrs_gm` |

#### 参数封装模式差异

| 模式 | CUDA | AscendC |
|------|------|---------|
| 结构体封装 | `Table*` 包含所有元数据 | 元数据展开为独立参数 |
| 元数据访问 | `table->bucket_max_size` | 通过 `GM_ADDR` 传入独立参数 |
| 典型算子 | `clear_kernel`, `dump_kernel` | `clear_kernel_vf`, `dump_kernel_vf` |

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

**转换规则**：

```cpp
// CUDA 参数声明
Bucket<K,V,S>* buckets;
const K* keys;
V** value_ptrs;

// AscendC 参数声明
GM_ADDR buckets_gm;
GM_ADDR keys_gm;
GM_ADDR value_ptrs_gm;

// AscendC 内部转换
__gm__ Bucket<K,V,S>* buckets = reinterpret_cast<__gm__ Bucket<K,V,S>*>(buckets_gm);
__gm__ const K* keys = reinterpret_cast<__gm__ const K*>(keys_gm);
__gm__ V* __gm__* value_ptrs = reinterpret_cast<__gm__ V* __gm__*>(value_ptrs_gm);
```

**重要规则**：函数体内所有全局内存指针赋值必须带 `__gm__` 限定符！

### 2.3 原子操作

| CUDA | AscendC |
|------|---------|
| `atomicCAS(ptr, old, new)` | `Simt::AtomicCas(ptr, old, new)` |
| `atomicAdd(ptr, val)` | `atomicAdd(ptr, val)` |
| `compare_exchange_strong` | `Simt::AtomicCas` |
| `store/release` | `Simt::AtomicExch` |
| `atomicExch` | `Simt::AtomicExch` |

**锁定/解锁模式对比**：

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

### 2.4 线程同步与协作

| CUDA | AscendC |
|------|---------|
| `cg::tiled_partition<N>(cg::this_thread_block())` | 无需创建，直接使用 `__shfl` |
| `g.thread_rank()` | `threadIdx.x % GROUP_SIZE` |
| `g.ballot(cond)` | 需手动实现或使用 `__shfl` 传播 |
| `g.shfl(val, lane)` | `__shfl(val, lane, GROUP_SIZE)` |
| `g.shfl_xor(val, mask)` | `__shfl_xor(val, mask, GROUP_SIZE)` |
| `__syncthreads()` | `AscendC::Simt::ThreadBarrier()` |
| `__threadfence()` | `__threadfence()` |

**协作组模式对比**：

```cpp
// CUDA: 创建协作组
auto g = cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
int rank = g.thread_rank();
int val_from_lane0 = g.shfl(my_val, 0);

// AscendC: 直接使用 shuffle
int rank = threadIdx.x % GROUP_SIZE;
int val_from_lane0 = __shfl(my_val, 0, GROUP_SIZE);
```

### 2.5 数据加载指令

| CUDA | AscendC |
|------|---------|
| `__ldg(ptr)` | `ldg_l2nc_l1c(ptr)` |
| `*(ptr)` | `*(ptr)` 或 `ldg_l2nc_l1c(ptr)` |
| `__ldcg(ptr)` | `__ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(ptr)` |

### 2.6 线程数配置模式

| 模式 | CUDA | AscendC |
|------|------|---------|
| 运行时决定 | 调用时指定 block/grid 大小 | 不支持 |
| 编译时固定 | 模板参数或宏定义 | `constexpr uint32_t THREAD_NUM = 512/1024/2048` |
| 典型值 | 动态调整 | clear: 1024, find: 512, dump: 2048 |

**说明**：
- CUDA 的 `clear_kernel`、`dump_kernel` 等辅助算子通常在调用时决定线程数
- AscendC 使用编译时常量 `THREAD_NUM`，根据算子复杂度选择 512/1024/2048

### 2.7 C++ 特性差异

| 特性 | CUDA | AscendC |
|------|------|---------|
| Placement new | `new (ptr) Type{value}` | 直接赋值（`AtomicKey`/`AtomicScore` 为简单类型） |
| 模板参数推导 | 自动推导 | 通过 `DISPATCH_VALUE_SIZE` 等宏分发 |
| 结构体方法 | `bucket->keys(i)` | 直接访问成员 `bucket->keys_[i]` |

**Placement new vs 直接赋值**：

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

## 三、HKV 操作层面差异

### 3.1 线程处理模式

| CUDA 模式 | AscendC 模式 | 适用算子 |
|-----------|--------------|----------|
| TLP (1线程1key) | SIMT (1线程1key) | `find_ptr`, `find_and_update` |
| TLP + GROUP (协作组) | SIMT + `__shfl` 协作 | `insert_and_evict`, `insert_or_assign` |

**关键差异**：
- CUDA 使用 `cg::tiled_partition<GROUP_SIZE>` 显式创建协作组
- AscendC 使用 `__shfl` 系列指令在指定 GROUP 内交换数据
- AscendC 需要手动计算 group rank：`threadIdx.x % GROUP_SIZE`

### 3.2 Score 处理与淘汰策略

| CUDA | AscendC |
|------|---------|
| `ScoreFunctor<K,V,S,Strategy>` 模板参数 | 相同 |
| `ScoreFunctor::desired_when_missed()` | 相同 |
| `ScoreFunctor::update_with_digest()` | `ScoreFunctor::update_with_digest()` 或 `update_score_only()` |
| `global_epoch` 参数 | 相同 |
| `system_cycle` | `AscendC::GetSystemCycle()` |

**映射规律**：
- 纯查找（不更新 score）：无 Strategy 模板参数
- 查找+更新 score：有 Strategy 参数，使用 `update_score_only()`
- 插入操作：使用 `update_with_digest()`

### 3.3 Digest 查找逻辑

**相同点**：
- 都使用 `VecD_Comp`（4字节）进行批量 digest 比较
- 都使用 `vcmpeq4` / `__vcmpeq4` 进行向量化比较
- 都使用 FFS（Find First Set）指令提取匹配位置

**差异**：

| CUDA | AscendC |
|------|---------|
| `__ffs(cmp_result)` | `Simt::Ffs(static_cast<int32_t>(cmp_result))` |
| `BUCKET::digests(bucket_keys_ptr, capacity, pos)` | 相同 |
| `bucket_keys_ptr[possible_pos]` | `ldg_l2nc_l1c(bucket_keys_ptr + possible_pos)` |

**重要：Digest 访问必须使用正确 API**：

```cpp
// ✅ 正确：获取 digest 指针
__gm__ D* digests_ptr = BUCKET::digests(bucket_keys, bucket_capacity, pos_cur);
VecD_Comp probe_digests = ldg_l2nc_l1c(reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr));

// ❌ 错误：digest 不是 key 的成员
D probe_digest = bucket_keys_ptr[digest_pos].digest;  // 错误！
```

### 3.4 OccupyResult 状态处理

**相同的状态枚举**（定义在 `types.h` 中）：
- `OccupyResult::INITIAL` - 初始状态
- `OccupyResult::DUPLICATE` - 找到已有 key
- `OccupyResult::OCCUPIED_EMPTY` - 找到空位
- `OccupyResult::EVICT` - 淘汰旧 key
- `OccupyResult::REFUSED` - 分数不足被拒绝
- `OccupyResult::OCCUPIED_RECLAIMED` - 回收 key 位置
- `OccupyResult::ILLEGAL` - 非法 key

### 3.5 返回值处理

| 操作 | CUDA | AscendC |
|------|------|---------|
| 返回 value 指针 | `values[kv_idx] = bucket_values_ptr + key_pos * dim` | 相同 |
| 返回 score | `scores[kv_idx] = *bucket_scores_ptr` | 相同 |
| 返回 found 标志 | `founds[kv_idx] = found` | 相同 |
| 淘汰计数 | `atomicAdd(evicted_counter, 1)` | 相同 |

### 3.6 锁机制差异（rehash_kernel）

| 机制 | CUDA | AscendC |
|------|------|---------|
| **Bucket 锁** | `Mutex` + `lock<Mutex, TILE_SIZE>()` / `unlock()` | `Simt::AtomicCas` 原子操作 |
| **协作组投票** | `g.ballot(current_key == EMPTY_KEY)` | `__shfl` 传播 + 手动判断 |
| **Key 移动** | `move_key_to_new_bucket` 辅助函数 | 内联展开 |
| **碎片整理** | `defragmentation_for_rehash` device 函数 | 内联在 kernel 中 |

**关键差异**：
- CUDA 使用显式的 `Mutex` 锁保护整个 bucket，确保重哈希过程中并发安全
- AscendC 使用 `Simt::AtomicCas` 原子操作锁定单个 key 位置，粒度更细
- 两者都使用**两阶段**处理：先移动 key 到新 bucket，再进行碎片整理

### 3.7 初始化操作差异（init_table_kernel）

| 操作 | CUDA | AscendC |
|------|------|---------|
| **内存分配** | Kernel 内调用 `allocator->alloc()` | Host 端统一分配，kernel 只初始化值 |
| **Key 初始化** | `new (bucket->keys(i)) AtomicKey<K>{EMPTY_KEY}` | 直接赋值 `bucket->keys_[i] = EMPTY_KEY` |
| **Score 初始化** | `new (bucket->scores(i)) AtomicScore<S>{EMPTY_SCORE}` | 直接赋值 `bucket->scores_[i] = EMPTY_SCORE` |
| **Vector 分配** | `bucket->vectors = address` | 相同 |
| **指针偏移计算** | Kernel 内计算 `digests_`, `keys_`, `scores_` 偏移 | Host 端计算后传入 |

**关键差异**：
- CUDA 使用 placement new 初始化 `AtomicKey`/`AtomicScore` 对象，支持构造函数
- AscendC 直接赋值，因为 `AtomicKey`/`AtomicScore` 通常定义为简单类型别名（如 `using AtomicKey = K`）
- AscendC 将内存分配和指针计算逻辑放在 Host 端，kernel 只负责写入初始值

---

## 四、AscendC 算子调用方式（Dispatcher 模式）

### 4.1 文件结构

每个算子包含两个文件：
1. **Kernel 实现**：`v35/{kernel_name}_kernel.h`（含 `_vf` 后缀的函数）
2. **Dispatcher**：`{kernel_name}_kernel.cpp`（`extern "C" __global__ __aicore__` 入口）

### 4.2 Dispatcher 模板

```cpp
// {kernel_name}_kernel.cpp
#include "./v35/{kernel_name}_kernel.h"
#include "../../include/simt_vf_dispatcher.h"
#include "kernel_operator.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void {kernel_name}(
    GM_ADDR buckets,
    // ... 其他 GM_ADDR 和标量参数 ...
    uint32_t value_size,              // 必须：用于类型分发
    int32_t evict_strategy) {         // 可选：用于淘汰策略分发

  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const uint64_t thread_all = THREAD_NUM * GetBlockNum();

  // 简单算子：仅 value_size 分发
  DISPATCH_VALUE_SIZE(
      value_size,
      (Simt::VF_CALL<{kernel_name}_vf<uint64_t, DTYPE, uint64_t>>(
          Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)},
          buckets, /* ... 其他参数 ... */, thread_all, GetBlockIdx())));

  // 复杂算子：多层分发
  DISPATCH_GROUP_SIZE(
      group_size,
      DISPATCH_VALUE_SIZE(
          value_size,
          DISPATCH_EVICT_STRATEGY(
              evict_strategy,
              (Simt::VF_CALL<{kernel_name}_vf<uint64_t, DTYPE, uint64_t, STRATEGY, GROUP_SIZE>>(
                  Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)},
                  buckets, /* ... 其他参数 ... */, thread_all, GetBlockIdx())))));
}
```

### 4.3 分发宏说明

#### `DISPATCH_VALUE_SIZE(value_size, FUNC)`
- 将 `value_size`（1/2/4/8/16）映射为 `DTYPE` 模板参数
- `DTYPE` 对应：`int8_t`, `int16_t`, `int32_t`, `int64_t`, `int4`
- **所有算子都必须使用**

#### `DISPATCH_EVICT_STRATEGY(strategy, FUNC)`
- 将 `strategy` 映射为 `STRATEGY` 模板参数
- 支持：kLru, kLfu, kEpochLru, kEpochLfu, kCustomized
- **涉及 Score 更新的算子需要使用**

#### `DISPATCH_GROUP_SIZE(group_size, FUNC)`
- 将 `group_size`（2/4/8/16/32）映射为 `GROUP_SIZE` 模板参数
- **涉及协作组的算子需要使用**（如 insert_and_evict）

### 4.4 Host 端调用方式

```cpp
// 包含 aclrtlaunch 头文件
#include "aclrtlaunch_find_ptr_with_digest_kernel.h"

// 调用 kernel
ACLRT_LAUNCH_KERNEL(find_ptr_with_digest_kernel)(
    block_dim_,           // block 维度
    stream,               // CANN stream
    table_->buckets,      // GM_ADDR 参数...
    table_->capacity,
    options_.max_bucket_size,
    options_.dim,
    keys, values, scores, founds, n, global_epoch_,
    value_size_,          // 用于 DISPATCH_VALUE_SIZE
    table_->max_bucket_shift,
    table_->capacity_divisor_magic,
    table_->capacity_divisor_shift
);
```

### 4.5 参数传递规则

| 参数类型 | 说明 |
|----------|------|
| `GM_ADDR` | 全局内存指针（设备内存地址） |
| `uint64_t` | 容量、数量等 64 位整数 |
| `uint32_t` | 维度、大小等 32 位整数 |
| `value_size` | 必须传入，用于类型分发（`sizeof(V)`） |
| `evict_strategy` | 涉及 Score 时传入，用于策略分发 |
| `group_size` | 涉及协作组时传入 |

---

## 五、迁移检查清单

### 5.1 必须修改的项目

- [ ] Kernel 修饰符改为 `__simt_vf__ __aicore__ LAUNCH_BOUND(N)`
- [ ] 全局内存参数改为 `GM_ADDR`，内部转换为 `__gm__ T*`
- [ ] `blockIdx.x` 改为从参数获取 `block_id`
- [ ] `atomicCAS` 改为 `Simt::AtomicCas`
- [ ] `atomicAdd` 保留，但确认命名空间
- [ ] `__syncthreads()` 改为 `Simt::ThreadBarrier()`
- [ ] 协作组操作改为 `__shfl` 系列

### 5.2 需要确认的项目

- [ ] `ScoreFunctor` 是否需要 `Strategy` 模板参数
- [ ] 是否需要 `update_score` 参数控制
- [ ] `THREAD_NUM` 选择（查找类通常 512，复杂操作 1024）
- [ ] `__ldg` (CUDA) → `ldg_l2nc_l1c` (AscendC) 转换
- [ ] 是否需要 `system_cycle` 参数
- [ ] 是否需要 `DISPATCH_GROUP_SIZE` 宏
- [ ] 协作组大小 `GROUP_SIZE` 是否与 CUDA 一致

### 5.3 HKV 特有常量

这些常量在两个平台保持一致：
- `EMPTY_KEY`, `LOCKED_KEY`, `RECLAIM_KEY`
- `EMPTY_SCORE`, `MAX_SCORE`
- `INVALID_KEY_POS` (UINT32_MAX)

---

## 六、常见陷阱与解决方案

### 6.1 Digest 访问错误

**问题**：生成代码可能错误地访问 `bucket_keys[pos].digest`

**原因**：digest 不是 key 的成员，而是独立存储的

**正确做法**：
```cpp
// ✅ 正确：使用 BUCKET::digests API
__gm__ D* digests_ptr = BUCKET::digests(bucket_keys, bucket_capacity, pos_cur);
VecD_Comp probe_digests = *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);

// ❌ 错误
digest = bucket_keys[pos].digest;  // 编译错误！
```

### 6.2 指针转换遗漏 `__gm__` 限定符

**问题**：全局内存指针转换后缺少 `__gm__` 限定符

**正确做法**：
```cpp
// ✅ 正确
__gm__ Bucket<K,V,S>* buckets = reinterpret_cast<__gm__ Bucket<K,V,S>*>(buckets_gm);

// ❌ 错误
Bucket<K,V,S>* buckets = reinterpret_cast<Bucket<K,V,S>*>(buckets_gm);
```

### 6.3 线程索引计算错误

**问题**：使用 CUDA 内置变量 `blockIdx.x`

**正确做法**：
```cpp
// ✅ 正确：从参数获取
uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;

// ❌ 错误
uint64_t kv_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 6.4 协作组大小不匹配

**问题**：`GROUP_SIZE` 与 CUDA 版本不一致，导致 shuffle 数据错位

**解决**：确保 `GROUP_SIZE` 与 CUDA 代码中的 `tiled_partition<N>` 的 N 一致

### 6.5 ScoreFunctor 使用错误

| 场景 | 应使用函数 |
|------|-----------|
| 纯查找不更新 score | 无 ScoreFunctor |
| 查找时更新已有 key 的 score | `ScoreFunctor::update_score_only()` |
| 插入时设置新 key 的 score | `ScoreFunctor::update_with_digest()` |

### 6.6 Dispatcher 参数遗漏

**常见问题**：
- 遗漏 `value_size` 参数 → 编译错误（无法类型分发）
- 遗漏 `evict_strategy` → ScoreFunctor 模板参数不匹配
- 遗漏 `group_size` → 协作组大小无法确定

---

## 七、参考文件路径

### CUDA HKV
- 根目录：`HierarchicalKV/include/merlin/core_kernels/`
- 关键文件：
  - `lookup_ptr.cuh` - 查找相关 kernel
  - `upsert.cuh` - 插入更新相关 kernel
  - `upsert_and_evict.cuh` - 插入淘汰相关 kernel
  - `update_score.cuh` - 更新 score 相关 kernel

### AscendC HKV
- 根目录：`HierarchicalKV-ascend/hkv_hashtable/`
- 关键文件：
  - `find_ptr_kernel/v35/find_ptr_with_digest_kernel.h`
  - `find_and_update_kernel/v35/find_and_update_kernel_with_filter.h`
  - `insert_and_evict_kernel/v35/insert_and_evict_kernel.h`
  - `insert_or_assign_kernel/v35/insert_or_assign_kernel.h`
- 工具头文件：
  - `HierarchicalKV-ascend/include/types.h` - 类型定义
  - `HierarchicalKV-ascend/include/simt_vf_dispatcher.h` - 分发宏
  - `HierarchicalKV-ascend/include/score_functor.h` - ScoreFunctor

---

## 八、总结

### 迁移核心原则

1. **语言层**：理解 CUDA `__global__` 与 AscendC `__simt_vf__ __aicore__` 的映射
2. **内存层**：所有全局内存参数使用 `GM_ADDR`，内部转换为 `__gm__ T*`
3. **线程层**：`blockIdx.x` 改为参数传入，协作组改为 `__shfl` 实现
4. **功能层**：保持 HKV 核心逻辑不变（digest 查找、CAS 锁定、ScoreFunctor）
5. **调用层**：理解 Dispatcher 模式，正确使用分发宏

### 调试建议

1. 先确保 kernel 生成代码能通过编译
2. 检查 `__gm__` 限定符是否正确添加
3. 验证 `THREAD_NUM` 和 `GROUP_SIZE` 与 CUDA 版本一致
4. 使用 `ldg_l2nc_l1c` 替代直接内存访问，提高性能
5. 检查 Dispatcher 中所有必需参数是否传入

---

*文档版本：1.0*
*最后更新：2026-03-25*

