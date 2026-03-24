/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCENDC_FIND_OR_INSERT_PTR_KERNEL_LOCK_KEY_H_
#define ASCENDC_FIND_OR_INSERT_PTR_KERNEL_LOCK_KEY_H_

#include <kernel_operator.h>
#include <cstdint>
#include "../../../include/types.h"
#include "../../../include/utils.h"
#include "../../../include/score_functor.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;
constexpr uint32_t STRIDE_S = 4;
constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(uint64_t);

template <typename K = uint64_t, typename V = byte4, typename S = uint64_t,
          int Strategy = -1>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void find_or_insert_ptr_kernel_lock_key_vf(
    GM_ADDR buckets_gm, GM_ADDR buckets_size_gm, uint64_t capacity,
    uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys_gm,
    GM_ADDR value_ptrs_gm, GM_ADDR scores_gm, GM_ADDR key_ptrs_gm, uint64_t n,
    GM_ADDR founds_gm, uint64_t global_epoch, uint64_t total_thread_num,
    uint64_t system_cycle, uint32_t block_index, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
  // 空指针检查
  if (buckets_gm == nullptr || buckets_size_gm == nullptr) {
    return;
  }

  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  // GM_ADDR 转换为 __gm__ 指针
  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_gm);
  __gm__ V* __gm__* __restrict__ value_ptrs =
      reinterpret_cast<__gm__ V* __gm__*>(value_ptrs_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  __gm__ K* __gm__* __restrict__ key_ptrs =
      reinterpret_cast<__gm__ K* __gm__*>(key_ptrs_gm);
  __gm__ bool* __restrict__ founds = reinterpret_cast<__gm__ bool*>(founds_gm);

  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  // 主循环：每个线程处理一个 key
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n; kv_idx += total_thread_num) {
    K key{static_cast<K>(EMPTY_KEY)};
    S score{static_cast<S>(EMPTY_SCORE)};
    OccupyResult occupy_result{OccupyResult::INITIAL};
    VecD_Comp target_digests{0};
    __gm__ V* bucket_values_ptr{nullptr};
    __gm__ K* bucket_keys_ptr{nullptr};
    __gm__ int32_t* bucket_size_ptr{nullptr};
    uint32_t key_pos = 0;
    uint32_t bucket_size = 0;

    // 初始化：读取 key，计算哈希值
    key = keys[kv_idx];
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                              system_cycle);

    if (IS_RESERVED_KEY<K>(key)) {
      if (key_ptrs != nullptr) {
        key_ptrs[kv_idx] = nullptr;
      }
      if (value_ptrs != nullptr) {
        value_ptrs[kv_idx] = nullptr;
      }
      if (founds != nullptr) {
        founds[kv_idx] = false;
      }
      continue;
    }

    const K hashed_key = Murmur3HashDevice(key);
    target_digests = digests_from_hashed<K>(hashed_key);
    uint64_t global_idx =
        get_global_idx(hashed_key, capacity_divisor_magic,
                       capacity_divisor_shift, capacity);
    key_pos = get_start_position(global_idx, bucket_capacity);
    uint64_t bkt_idx = global_idx >> max_bucket_shift;
    bucket_size_ptr = buckets_size + bkt_idx;
    bucket_size = *bucket_size_ptr;
    __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
    bucket_keys_ptr = bucket->keys_;
    bucket_values_ptr = bucket->vectors;

    // 阶段1：使用 digest 进行线性探测，查找 key 或空位
    for (uint32_t offset = 0; offset < bucket_capacity + STRIDE;
         offset += STRIDE) {
      if (occupy_result != OccupyResult::INITIAL) break;

      uint32_t pos_cur = align_to<STRIDE>(key_pos);
      pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

      __gm__ D* digests_ptr =
          BUCKET::digests(bucket_keys_ptr, bucket_capacity, pos_cur);
      VecD_Comp probe_digests =
          *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);

      // 检查是否有匹配的 digest（可能包含目标 key）
      uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
      cmp_result &= 0x01010101;

      while (cmp_result != 0) {
        uint32_t index =
            (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        uint32_t possible_pos = pos_cur + index;

        __gm__ K* current_key_ptr =
            BUCKET::keys(bucket_keys_ptr, possible_pos);
        K expected_key = key;
        // 尝试锁定 key
        K prev_key = Simt::AtomicCas(current_key_ptr, expected_key,
                                     static_cast<K>(LOCKED_KEY));

        if (prev_key == key) {
          // 找到已存在的 key
          occupy_result = OccupyResult::DUPLICATE;
          key_pos = possible_pos;
          ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                           kv_idx, score, bucket_capacity,
                                           get_digest<K>(key), false);
          // 解锁 key
          Simt::AtomicExch(current_key_ptr, key);
          break;
        }
      }

      if (occupy_result == OccupyResult::DUPLICATE) {
        break;
      }

      // 如果桶未满，尝试找空位
      if (bucket_size < bucket_capacity) {
        VecD_Comp empty_digests_ = empty_digests<K>();
        cmp_result = vcmpeq4(probe_digests, empty_digests_);
        cmp_result &= 0x01010101;

        while (cmp_result != 0) {
          uint32_t index =
              (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          uint32_t possible_pos = pos_cur + index;

          if (offset == 0 && possible_pos < key_pos) continue;

          __gm__ K* current_key_ptr =
              BUCKET::keys(bucket_keys_ptr, possible_pos);
          K expected_key = static_cast<K>(EMPTY_KEY);
          K prev_key = Simt::AtomicCas(current_key_ptr, expected_key,
                                       static_cast<K>(LOCKED_KEY));

          if (prev_key == static_cast<K>(EMPTY_KEY)) {
            // 成功抢占空位
            occupy_result = OccupyResult::OCCUPIED_EMPTY;
            key_pos = possible_pos;
            ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                             kv_idx, score, bucket_capacity,
                                             get_digest<K>(key), true);
            atomicAdd(bucket_size_ptr, 1);
            // 解锁 key
            Simt::AtomicExch(current_key_ptr, key);
            break;
          }
        }

        if (occupy_result == OccupyResult::OCCUPIED_EMPTY) {
          break;
        }
      }
    }

    // 阶段2：如果未找到 key 且桶已满，尝试淘汰最低 score 的 key
    while (occupy_result == OccupyResult::INITIAL) {
      __gm__ S* bucket_scores_ptr =
          BUCKET::scores(bucket_keys_ptr, bucket_capacity, 0);
      S min_score = static_cast<S>(MAX_SCORE);
      int32_t min_pos = -1;

      // 遍历查找最小 score
      for (uint32_t i = 0; i < bucket_capacity; i++) {
        S temp_score = bucket_scores_ptr[i];
        if (temp_score < min_score) {
          __gm__ K* verify_key_ptr = BUCKET::keys(bucket_keys_ptr, i);
          K verify_key = *verify_key_ptr;
          if (verify_key != static_cast<K>(LOCKED_KEY) &&
              verify_key != static_cast<K>(EMPTY_KEY)) {
            min_score = temp_score;
            min_pos = static_cast<int32_t>(i);
          }
        }
      }

      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                system_cycle);
      if (score <= min_score) {
        occupy_result = OccupyResult::REFUSED;
        break;
      }

      if (min_pos < 0) {
        occupy_result = OccupyResult::REFUSED;
        break;
      }

      __gm__ K* min_score_key_ptr =
          BUCKET::keys(bucket_keys_ptr, static_cast<uint32_t>(min_pos));
      K expected_key = *min_score_key_ptr;

      if (expected_key != static_cast<K>(LOCKED_KEY) &&
          expected_key != static_cast<K>(EMPTY_KEY)) {
        bool result = Simt::AtomicCas(min_score_key_ptr, expected_key,
                                      static_cast<K>(LOCKED_KEY)) ==
                      expected_key;
        if (result) {
          __gm__ S* min_score_ptr =
              BUCKET::scores(bucket_keys_ptr, bucket_capacity,
                            static_cast<uint32_t>(min_pos));
          S verify_score = *min_score_ptr;
          if (verify_score <= min_score) {
            key_pos = static_cast<uint32_t>(min_pos);
            ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                             kv_idx, score, bucket_capacity,
                                             get_digest<K>(key), true);
            if (expected_key == static_cast<K>(RECLAIM_KEY)) {
              occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
              atomicAdd(bucket_size_ptr, 1);
            } else {
              occupy_result = OccupyResult::EVICT;
            }
          } else {
            // 分数已变，解锁并重新尝试
            Simt::AtomicExch(min_score_key_ptr, expected_key);
          }
        }
      }

      // 避免无限循环
      if (occupy_result == OccupyResult::INITIAL) {
        // 短暂等待后重试
        __threadfence();
      }
    }

    // 设置输出
    if (occupy_result == OccupyResult::REFUSED) {
      if (value_ptrs != nullptr) {
        value_ptrs[kv_idx] = nullptr;
      }
      if (key_ptrs != nullptr) {
        key_ptrs[kv_idx] = nullptr;
      }
    } else {
      if (value_ptrs != nullptr) {
        value_ptrs[kv_idx] = bucket_values_ptr + key_pos * dim;
      }
      if (key_ptrs != nullptr) {
        __gm__ K* key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
        key_ptrs[kv_idx] = key_address;
      }
    }

    if (founds != nullptr) {
      founds[kv_idx] = (occupy_result == OccupyResult::DUPLICATE);
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_OR_INSERT_PTR_KERNEL_LOCK_KEY_H_
