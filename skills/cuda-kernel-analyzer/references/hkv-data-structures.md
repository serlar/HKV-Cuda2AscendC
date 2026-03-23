# HKV 数据结构与常量参考

本文档总结了 HierarchicalKV-ascend 项目中核心数据结构、常量和工具函数，
供 AscendC kernel 代码生成时参考，来源于 `HierarchicalKV-ascend/include/` 目录。

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

### 访问模式

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

## 4. Digest 函数

```cpp
// 计算 key 的 digest（1 字节 fingerprint，用于快速过滤）
template <typename K>
__forceinline__ __device__ D get_digest(K key);

// 空 key 对应的 digest
template <typename K>
__forceinline__ __device__ D empty_digest();
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

### 桶索引分解（关键公式）

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

> **何时使用 ScoreFunctor**：当 kernel 涉及 score 的读写/更新（如 find_and_update, insert_and_evict）时使用。仅做清空或初始化的 kernel 不需要。

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

}  // namespace Simt

// 全局原子加（不在 Simt 命名空间内）
template <typename T>
T atomicAdd(__gm__ T* ptr, T val);
```

---

## 8. 测试辅助函数（test_util.h）

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

## 9. HashTable 公共接口（hkv_hashtable.h）

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
