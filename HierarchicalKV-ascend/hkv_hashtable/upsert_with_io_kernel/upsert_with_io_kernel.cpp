/**
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

#include "./v35/upsert_with_io_kernel.h"
#include <cstdint>
#include "../../include/simt_vf_dispatcher.h"
#include "kernel_operator.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void upsert_with_io_kernel(
    GM_ADDR buckets, GM_ADDR buckets_size, uint64_t capacity,
    uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys, GM_ADDR values,
    GM_ADDR scores, uint64_t n, uint64_t global_epoch,
    int32_t evict_strategy, uint32_t value_size,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, uint64_t n_align_warp) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const uint64_t thread_all = THREAD_NUM * GetBlockNum();
  
  // Calculate cur_score based on eviction strategy
  uint64_t cur_score =
      (evict_strategy == npu::hkv::EvictStrategyInternal::kLru ||
       evict_strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
          ? static_cast<uint64_t>(GetSystemCycle())
          : 0;

  DISPATCH_VALUE_SIZE(
      value_size,
      DISPATCH_EVICT_STRATEGY(
          evict_strategy,
          (Simt::VF_CALL<upsert_with_io_kernel_vf<uint64_t, DTYPE, uint64_t,
                                                  STRATEGY>>(
              Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)},
              buckets, buckets_size, capacity, bucket_capacity, dim,
              keys, values, scores, cur_score, n,
              thread_all, global_epoch, GetBlockIdx(),
              max_bucket_shift, capacity_divisor_magic,
              capacity_divisor_shift, n_align_warp))));
}
