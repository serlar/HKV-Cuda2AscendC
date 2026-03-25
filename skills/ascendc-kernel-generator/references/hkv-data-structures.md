# HKV 数据结构与算子 API 参考

本文档总结了 HierarchicalKV 项目中核心数据结构、常量、工具函数以及算子 API 的依赖关系，
供 AscendC kernel 代码生成时参考。

---

## 1. Bucket<K, V, S> 结构体

`Bucket` 是哈希表的核心存储单元，定义在 `include/types.h`：

```cpp
template <class K, class V, class S>
struct Bucket {
  __gm__ K* keys_;       // 存储 key 数组（bucket_capacity 个）
  __gm__ S* scores_;     // 存储 score 数组（bucket_capacity 个）
  __gm__ D* digests_;    // key 的 digest（1 字节摘要，用于快速过滤）
  __gm__ V* vectors;     // value 向量数组（bucket_capacity × dim 个 float）
};
```

### 1.1 访问模式

```cpp
// 获取 bucket 指针（从 GM_ADDR 转换）
__gm__ Bucket<K, V, S>* __restrict__ buckets =
    reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);

// 定位到第 bkt_idx 个桶
__gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;

// 读取桶内的指针（访问 keys/scores/digests/vectors）
__gm__ K* bucket_keys = bucket->keys_;
__gm__ S* bucket_scores = bucket->scores_;
__gm__ D* bucket_digests = bucket->digests_;
__gm__ V* bucket_values = bucket->vectors;

// 读取具体 key
K current_key = *(bucket_keys + slot_pos);
// 或者
K current_key = buckets[bkt_idx].keys_[slot_pos];
```

### 1.2 静态方法访问（推荐）

```cpp
using BUCKET = Bucket<K, V, S>;

// 获取 keys 指针
__gm__ K* keys_ptr = BUCKET::keys(bucket_keys, pos);

// 获取 digests 指针（重要！digest 存储在 keys 数组之前）
__gm__ D* digests_ptr = BUCKET::digests(bucket_keys, bucket_capacity, pos);

// 获取 scores 指针
__gm__ S* scores_ptr = BUCKET::scores(bucket_keys, bucket_capacity, pos);
```

---

## 2. 关键常量

定义在 `include/types.h`：

```cpp
// Key 特殊值
constexpr uint64_t EMPTY_KEY    = 0xFFFFFFFFFFFFFFFF;  // 空槽标记
constexpr uint64_t LOCKED_KEY   = 0xFFFFFFFFFFFFFFFD;  // 正在被写入（临时锁）
constexpr uint64_t RECLAIM_KEY  = 0xFFFFFFFFFFFFFFFE;  // 待回收

// Score 特殊值
constexpr uint64_t EMPTY_SCORE = 0;
constexpr uint64_t MAX_SCORE   = 0xFFFFFFFFFFFFFFFF;

// 非法 key 位置（用于标记"未找到"）
constexpr uint32_t INVALID_KEY_POS = UINT32_MAX;

// Digest 类型
using D = uint8_t;  // 每个 key 对应 1 字节 digest
```

---

## 3. 关键谓词函数

```cpp
// 判断 key 是否为系统保留值（EMPTY / LOCKED / RECLAIM）
// 在插入/查找时跳过保留 key
template <typename K>
__forceinline__ __device__ bool IS_RESERVED_KEY(K key);
// 使用示例：
if (IS_RESERVED_KEY<K>(key)) continue;  // 跳过该 key

// 判断 key 是否为空（EMPTY 或 RECLAIM）
template <typename K>
__forceinline__ __device__ bool IS_VACANT_KEY(K key);
```

---

## 4. Digest 函数与访问

### 4.1 Digest 计算函数

```cpp
// 计算 key 的 digest（1 字节 fingerprint，用于快速过滤）
template <typename K>
__forceinline__ __device__ D get_digest(K key);

// 空 key 对应的 digest
template <typename K>
__forceinline__ __device__ D empty_digest();

// 从 hashed key 生成 4 字节 digest（用于批量比较）
template <typename K>
__forceinline__ __device__ VecD_Comp digests_from_hashed(K hashed_key);
```

### 4.2 Digest 访问方式（重要！）

**严禁使用错误的数据访问方式！**

| 数据类型 | ❌ 错误方式 | ✅ 正确方式 |
|----------|-------------|-------------|
| **Key** | `bucket_keys_ptr[pos].digest` ❌ | `*(bucket_keys + pos)` 或 `BUCKET::keys(bucket_keys, pos)` ✅ |
| **Digest** | `bucket_keys_ptr[pos].digest` ❌ | `BUCKET::digests(bucket_keys, bucket_capacity, pos)` ✅ |
| **Score** | `bucket->scores_[pos]` | `BUCKET::scores(bucket_keys, bucket_capacity, pos)` |

### 4.3 Digest 访问详解

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
cmp_result &= 0x01010101;
```

**常见错误**：
```cpp
// ❌ 错误：digest 不是 key 的成员
D probe_digest = bucket_keys_ptr[digest_pos].digest;

// ❌ 错误：不能直接通过 bucket 指针访问 digests_
D digest = bucket->digests_[pos];
```

### 4.4 完整示例：Digest 比较流程

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

---

## 5. 哈希与索引工具

定义在 `include/utils.h`：

```cpp
// Murmur3 哈希函数（对 key 做混淆，改善分布均匀性）
__forceinline__ __device__ uint64_t Murmur3HashDevice(uint64_t key);

// 快速取模：将哈希值映射到 [0, capacity) 范围内的全局槽位索引
// capacity_divisor_magic 和 capacity_divisor_shift 在 CPU 侧预计算传入
__forceinline__ __device__ uint64_t get_global_idx(
    uint64_t hashed_key,
    uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift,
    uint64_t capacity);
```

### 5.1 桶索引分解（关键公式）

```cpp
// 从全局槽位索引分解为桶索引和桶内位置
// bucket_capacity 必须是 2 的幂次
uint64_t global_idx = get_global_idx(...);

uint32_t key_pos = global_idx & (bucket_capacity - 1);  // 桶内位置（初始探测点）
uint64_t bkt_idx = global_idx >> max_bucket_shift;       // 桶索引
// 其中 max_bucket_shift = log2(bucket_capacity)
```

---

## 6. ScoreFunctor

定义在 `include/score_functor.h`，用于计算和更新 key 的 score（淘汰优先级）：

```cpp
// 实例化：使用 Strategy 参数选择策略（-1 为默认策略）
using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

// 计算"未命中时的期望 score"（当 key 不在表中时使用）
S score = ScoreFunctor::desired_when_missed(
    scores,       // 输入 score 数组
    kv_idx,       // 当前处理的 key 索引
    global_epoch, // 当前 epoch
    system_cycle  // 系统时钟周期（可选）
);

// 更新已有 key 的 score（找到 key 时调用）
ScoreFunctor::update_score_only(
    bucket_keys,    // 桶的 key 数组
    target_pos,     // key 在桶内的位置
    scores,         // 输入 score 数组
    kv_idx,         // 当前处理的 key 索引
    score,          // 期望 score 值
    bucket_capacity,
    false           // is_new_key
);

// 写入 digest 并更新新 key 的 score（插入新 key 时调用）
ScoreFunctor::update_with_digest(
    bucket_keys,
    target_pos,
    scores,
    kv_idx,
    score,
    bucket_capacity,
    get_digest<K>(key),  // key 的 digest
    true                 // is_new_key
);
```

### 6.1 ScoreFunctor 使用场景

| 场景 | 应使用函数 |
|------|-----------|
| 纯查找不更新 score | 无 ScoreFunctor |
| 查找时更新已有 key 的 score | `ScoreFunctor::update_score_only()` |
| 插入时设置新 key 的 score | `ScoreFunctor::update_with_digest()` |

**参数说明**：
- `global_epoch`: 当前 epoch，用于基于时间的淘汰策略（如 EpochLru）
- `system_cycle`: 系统时钟周期，通过 `AscendC::GetSystemCycle()` 获取

---

## 7. Simt 命名空间（AscendC 原子操作）

```cpp
// 定义在 AscendC 运行时，通过 kernel_operator.h 引入
namespace Simt {

// Compare-And-Swap：将 *ptr 从 expected 原子替换为 desired
// 返回 *ptr 在操作前的值
template <typename T>
T AtomicCas(__gm__ T* ptr, T expected, T desired);

// Exchange：原子地将 *ptr 设为 val，返回旧值
template <typename T>
T AtomicExch(__gm__ T* ptr, T val);

// Find First Set：返回最低位 1 的位置（1-based）
int32_t Ffs(int32_t val);

// 线程屏障（block 级同步）
void ThreadBarrier();

}  // namespace Simt

// 全局原子加（不在 Simt 命名空间内）
template <typename T>
T atomicAdd(__gm__ T* ptr, T val);
```

---

## 8. OccupyResult 状态处理

定义在 `include/types.h` 中的状态枚举：

```cpp
enum class OccupyResult : uint32_t {
  INITIAL = 0,        // 初始状态
  DUPLICATE = 1,      // 找到已有 key
  OCCUPIED_EMPTY = 2, // 找到空位
  EVICT = 3,          // 淘汰旧 key
  REFUSED = 4,        // 分数不足被拒绝
  OCCUPIED_RECLAIMED = 5, // 回收 key 位置
  ILLEGAL = 6         // 非法 key
};
```

---

## 9. 锁机制差异

### 9.1 rehash_kernel 锁机制

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

### 9.2 通用锁定/解锁模式

```cpp
// 锁定 key 位置
K prev = Simt::AtomicCas(current_key_ptr, EMPTY_KEY, LOCKED_KEY);
if (prev == EMPTY_KEY) {
    // 成功锁定空位，可以插入
}

// 解锁（释放锁定）
(void)Simt::AtomicExch(current_key_ptr, actual_key);
```

---

## 10. 初始化操作差异（init_table_kernel）

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

## 11. AscendC 算子调用方式（Dispatcher 模式）

> **注意**：完整的 Dispatcher 模板和分发宏说明请参考 `ascendc-simt-patterns.md` 第 10 节。
> 本文档仅列出关键规则和算子特定参数。

### 11.1 分发宏使用场景

| 宏 | 适用算子 | 说明 |
|----|----------|------|
| `DISPATCH_VALUE_SIZE` | **所有算子** | 必须，用于将 `value_size` 映射为 `DTYPE` |
| `DISPATCH_EVICT_STRATEGY` | `find_and_update`, `insert_and_evict`, `insert_or_assign` | 涉及 Score 更新的算子 |
| `DISPATCH_GROUP_SIZE` | `insert_and_evict`, `dump_kernel` | 使用协作组的算子 |

### 11.2 算子特定参数

| 参数 | 类型 | 适用算子 | 说明 |
|------|------|----------|------|
| `update_score` | `bool` | `find_and_update` | 是否更新已有 key 的 score |
| `global_epoch` | `uint64_t` | 带 Strategy 的算子 | 当前 epoch，用于 score 计算 |
| `system_cycle` | `uint64_t` | 部分带 Strategy 算子 | `AscendC::GetSystemCycle()` 获取 |
| `is_train_mode` | `bool` | `find_ptr` 等 | 训练模式标志（部分算子有） |

---

## 12. 测试辅助函数（test_util.h）

在生成测试文件时使用：

```cpp
namespace test_util {

// 初始化 ACL 环境（必须在所有 ACL 操作前调用）
void init_env();

// 生成连续 key（从 start_key 开始，每个 key 递增 1）
// 同时填充 host_scores 和 host_values（如果非 nullptr）
template <typename K, typename S, typename V, size_t dim>
void create_continuous_keys(
    K* host_keys,    // 输出：key 数组
    S* host_scores,  // 输出：score 数组（可为 nullptr）
    V* host_values,  // 输出：value 数组（可为 nullptr）
    size_t num,      // key 数量
    K start_key = 0  // 起始 key 值
);

// 生成随机 key（保证不含保留 key）
template <typename K, typename S, typename V>
void create_random_keys(
    size_t dim,
    K* host_keys,
    S* host_scores,  // 可为 nullptr
    V* host_values,  // 可为 nullptr
    size_t num
);

}  // namespace test_util
```

---

## 13. HashTable 公共接口（hkv_hashtable.h）

测试中通过 `HashTable<K, V>` 的公共接口调用各 kernel：

```cpp
namespace npu::hkv {

struct HashTableOptions {
    size_t init_capacity;        // 初始桶总数
    size_t max_capacity;         // 最大桶总数
    size_t max_hbm_for_vectors;  // value 向量使用的最大 HBM（字节）
    size_t dim;                  // value 向量维度
    bool io_by_cpu;              // 是否通过 CPU 拷贝 I/O（测试中通常 false）
};

template <typename K, typename V, typename S = uint64_t>
class HashTable {
 public:
    void init(const HashTableOptions& options);
    size_t size() const;

    // 各算子接口（测试中调用）
    void insert_or_assign(size_t n, K* keys, V* values, S* scores, aclrtStream stream);
    void find_and_update(size_t n, K* keys, V** value_ptrs, bool* founds, S* scores, aclrtStream stream, bool update_score);
    void insert_and_evict(size_t n, K* keys, V* values, S* scores, K* evicted_keys, V* evicted_values, S* evicted_scores, size_t* evicted_num, aclrtStream stream);
    void clear(aclrtStream stream);
};

}  // namespace npu::hkv
```
