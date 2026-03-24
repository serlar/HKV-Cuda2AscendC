/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef ASCENDC_UPSERT_WITH_IO_KERNEL_H_
#define ASCENDC_UPSERT_WITH_IO_KERNEL_H_

#include <cstdint>
#include "kernel_operator.h"
#include "../../../include/types.h"
#include "../../../include/utils.h"
#include "../../../include/score_functor.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;
constexpr int32_t GROUP_SIZE = 16;
constexpr int32_t EVICT_GROUP_SIZE = 16;

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int32_t Strategy = -1>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void upsert_with_io_kernel_vf(
    GM_ADDR buckets_addr_gm, GM_ADDR buckets_size_addr_gm, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, GM_ADDR keys_addr_gm,
    GM_ADDR values_addr_gm, GM_ADDR scores_gm, S cur_score, uint64_t n,
    uint32_t thread_all, uint64_t global_epoch, uint32_t block_index,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, uint64_t n_align_warp) {
  // Check pointers individually but don't return early
  // All threads must participate in cooperative group operations
  if (buckets_addr_gm == nullptr) {
    return;
  }
  if (buckets_size_addr_gm == nullptr) {
    return;
  }
  if (keys_addr_gm == nullptr) {
    return;
  }
  if (values_addr_gm == nullptr) {
    return;
  }

  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_addr_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_addr_gm);
  __gm__ V* __restrict__ values = reinterpret_cast<__gm__ V*>(values_addr_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  
  S score = static_cast<S>(EMPTY_SCORE);
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  uint32_t key_pos = 0;
  K key = 0;
  __gm__ K* bucket_keys = nullptr;
  uint64_t bucket_values_uintptr = 0;
  __gm__ S* bucket_scores = nullptr;
  __gm__ int32_t* bucket_size = nullptr;

  // Use n_align_warp to ensure all threads in a warp participate
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += thread_all) {
    VecD_Comp target_digests{0};
    OccupyResult occupy_result{OccupyResult::INITIAL};

    // 1. Each thread processes one key
    if (kv_idx < n) {
      key = keys[kv_idx];
      if (IS_RESERVED_KEY<K>(key)) {
        occupy_result = OccupyResult::ILLEGAL;
      }
      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                cur_score);

      // 2. Compute key hash and locate position
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                           capacity_divisor_shift, capacity);
      key_pos = global_idx & (bucket_max_size - 1);
      uint64_t bkt_idx = global_idx >> (max_bucket_shift);

      bucket_size = buckets_size + bkt_idx;
      int32_t cur_bucket_size = *bucket_size;
      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys = bucket->keys_;
      bucket_values_uintptr = reinterpret_cast<uint64_t>(bucket->vectors);
      bucket_scores = bucket->scores_;

      // 3. Traverse bucket to find key/empty slot
      for (uint32_t offset = 0; offset < bucket_max_size + STRIDE;
           offset += STRIDE) {
        if (occupy_result != OccupyResult::INITIAL) {
          break;
        }
        uint32_t pos_cur = align_to<STRIDE>(key_pos);
        pos_cur = (pos_cur + offset) & (bucket_max_size - 1);

        __gm__ D* digests_ptr =
            BUCKET::digests(bucket_keys, bucket_max_size, pos_cur);
        VecD_Comp probe_digests =
            *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);

        // 3.1 Compare 4 digests at once
        uint32_t possible_pos = 0;
        uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index =
              (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key = Simt::AtomicCas(current_key_ptr, key, 
                                         static_cast<K>(LOCKED_KEY));
          // 3.2 Found key, try to lock
          if (try_key == key) {
            occupy_result = OccupyResult::DUPLICATE;
            key_pos = possible_pos;
            ScoreFunctor::update_score_only(bucket_keys, key_pos, scores,
                                            kv_idx, score, bucket_max_size,
                                            false);
            break;
          }
        } while (true);

        // 3.3 Found, exit loop
        if (occupy_result == OccupyResult::DUPLICATE) {
          break;
        } else if (cur_bucket_size == static_cast<int32_t>(bucket_max_size)) {
          // 3.4 Not found and bucket full, continue to next batch
          continue;
        }

        // 3.5 Not found, bucket not full, look for empty slot
        VecD_Comp empty_digests_ = empty_digests<K>();
        cmp_result = vcmpeq4(probe_digests, empty_digests_);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index =
              (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;
          if (offset == 0 && possible_pos < key_pos) {
            continue;
          }

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key = Simt::AtomicCas(current_key_ptr, 
                                         static_cast<K>(EMPTY_KEY),
                                         static_cast<K>(LOCKED_KEY));
          // 3.6 Found empty slot, try to occupy
          if (try_key == static_cast<K>(EMPTY_KEY)) {
            occupy_result = OccupyResult::OCCUPIED_EMPTY;
            key_pos = possible_pos;
            ScoreFunctor::update_with_digest(bucket_keys, key_pos, scores,
                                             kv_idx, score, bucket_max_size,
                                             get_digest<K>(key), true);
            atomicAdd(bucket_size, 1);
            break;
          }
        } while (true);

        // 3.7 Occupied empty slot, exit loop, otherwise continue
        if (occupy_result == OccupyResult::OCCUPIED_EMPTY) {
          break;
        }
      }
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }

    // 4. Eviction phase if not found and bucket is full
    // Possible results from above:
    // * OccupyResult::DUPLICATE - found existing key
    // * OccupyResult::OCCUPIED_EMPTY - occupied empty slot
    // * OccupyResult::INITIAL - failed to occupy
    auto cg_rank_id = threadIdx.x % EVICT_GROUP_SIZE;
    
    // Iterate through group threads, each may need eviction
    for (int32_t i = 0; i < EVICT_GROUP_SIZE; i++) {
      auto res_sync = __shfl(occupy_result, i, EVICT_GROUP_SIZE);
      while (res_sync == OccupyResult::INITIAL) {
        S min_score = MAX_SCORE;
        uint32_t min_pos = key_pos;
        
        // 4.1 Traverse bucket to find minimum score
        uint64_t bucket_scores_sync = __shfl(
            reinterpret_cast<uint64_t>(bucket_scores), i, EVICT_GROUP_SIZE);
        uint64_t bucket_keys_sync = __shfl(
            reinterpret_cast<uint64_t>(bucket_keys), i, EVICT_GROUP_SIZE);
        
        for (uint32_t current_pos = cg_rank_id; current_pos < bucket_max_size;
             current_pos += EVICT_GROUP_SIZE) {
          S current_score = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                  L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ S*>(bucket_scores_sync) + current_pos);
          K current_key = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ K*>(bucket_keys_sync) + current_pos);
          if (current_score < min_score &&
              current_key != static_cast<K>(LOCKED_KEY) &&
              current_key != static_cast<K>(EMPTY_KEY)) {
            min_score = current_score;
            min_pos = current_pos;
          }
        }
        
        // Reduction to find global minimum
        for (int32_t offset = EVICT_GROUP_SIZE / 2; offset > 0; offset /= 2) {
          S other_score = __shfl_xor(min_score, offset, EVICT_GROUP_SIZE);
          uint32_t other_pos = __shfl_xor(min_pos, offset, EVICT_GROUP_SIZE);
          if (other_score < min_score) {
            min_score = other_score;
            min_pos = other_pos;
          }
        }

        // Thread i tries to evict
        if (cg_rank_id == i) {
          // 4.2 Score too low, cannot evict
          if (score < min_score) {
            occupy_result = OccupyResult::REFUSED;
          } else {
            // 4.3 Score sufficient, try to evict
            auto current_key_ptr = bucket_keys + min_pos;
            auto current_key =
                __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                      L1CacheType::NON_CACHEABLE>(current_key_ptr);
            if (current_key != static_cast<K>(LOCKED_KEY) &&
                current_key != static_cast<K>(EMPTY_KEY)) {
              auto try_key = Simt::AtomicCas(current_key_ptr, current_key,
                                             static_cast<K>(LOCKED_KEY));
              // 4.4 Eviction successful
              if (try_key == current_key) {
                // 4.4.1 Verify score hasn't decreased
                if (__ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                          L1CacheType::NON_CACHEABLE>(bucket_scores +
                                                      min_pos) <= min_score) {
                  key_pos = min_pos;
                  ScoreFunctor::update_with_digest(
                      bucket_keys, key_pos, scores, kv_idx, score,
                      bucket_max_size, get_digest<K>(key), true);
                  if (try_key == static_cast<K>(RECLAIM_KEY)) {
                    occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
                    atomicAdd(bucket_size, 1);
                  } else {
                    occupy_result = OccupyResult::EVICT;
                  }
                } else {
                  // 4.4.2 Score increased, eviction failed, restore key
                  (void)Simt::AtomicExch(current_key_ptr, current_key);
                }
              }
              // 4.5 Eviction failed, retry
            }
          }
        }
        res_sync = __shfl(occupy_result, i, EVICT_GROUP_SIZE);
      }
    }

    // 5. Copy values using cooperative groups
    cg_rank_id = threadIdx.x % GROUP_SIZE;
    for (int32_t i = 0; i < GROUP_SIZE; i++) {
      auto res_sync = __shfl(occupy_result, i, GROUP_SIZE);
      if ((res_sync != OccupyResult::REFUSED &&
           res_sync != OccupyResult::ILLEGAL)) {
        auto kv_idx_sync = __shfl(kv_idx, i, GROUP_SIZE);
        auto value_start = kv_idx_sync * dim;

        auto key_pos_sync = __shfl(key_pos, i, GROUP_SIZE);
        uint64_t value_ddr_sync = __shfl(bucket_values_uintptr, i, GROUP_SIZE);
        auto bucket_value_start = key_pos_sync * dim;
        
        for (uint32_t j = cg_rank_id; j < dim; j += GROUP_SIZE) {
          __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ V*>(value_ddr_sync) + bucket_value_start + j,
              values[value_start + j]);
        }
      }
    }

    __threadfence();

    // 6. Release lock and write key
    if (occupy_result != OccupyResult::REFUSED &&
        occupy_result != OccupyResult::ILLEGAL) {
      (void)Simt::AtomicExch(bucket_keys + key_pos, key);
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_UPSERT_WITH_IO_KERNEL_H_
