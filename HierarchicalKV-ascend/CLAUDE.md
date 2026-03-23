# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Port of [NVIDIA HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV) to Huawei's Ascend NPU platform. A high-performance key-value hash table library for recommendation system embedding tables, implemented using AscendC (Ascend's GPU-like kernel language). Lives in the `npu::hkv::` namespace.

## Build Commands

All builds require Huawei CANN toolkit installed at `ASCEND_CANN_PACKAGE_PATH` (default: `/usr/local/Ascend/ascend-toolkit/latest`).

### Using run.sh (recommended)

```bash
# Build and run with default settings (RUN_MODE=npu, SOC_VERSION=Ascend950PR_9579)
bash run.sh

# Build only, CPU simulation mode
bash run.sh -r cpu -c

# Build with tests, NPU simulation
bash run.sh -r sim -t 1

# Full options: -v SOC_VERSION, -d DEVICE_ID, -r RUN_MODE (cpu|sim|npu), -b BUILD_TYPE, -t ENABLE_TEST (0|1), -c (compile only)
bash run.sh -r npu -t 1 -v Ascend910_9589
```

### Manual CMake

```bash
# Source CANN environment first
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

cmake -B build -DRUN_MODE=cpu -DSOC_VERSION=Ascend910_9589 -DENABLE_TEST=1
cmake --build build -j 16
cmake --install build
```

### Running tests

```bash
cd build && ctest -V

# Run a specific test binary directly
./build/tests/test_main --gtest_filter="TestSuiteName.TestName"
./build/tests/test_aclnn_helper
./build/tests/test_score_functor/test_score_functor
```

## Architecture

### Core Data Model

- `Bucket<K,V,S>` (`include/types.h`): fundamental storage unit. Keys + digests (8-bit Murmur3 hash for fast comparison) are stored in HBM. Digests live at negative offsets from the keys pointer; scores at positive offsets. Values are stored in a separate contiguous array.
- `Table<K,V,S>` (`include/types.h`): array of `Bucket*` pointers + metadata (capacity, bucket count, max capacity, etc.)
- `HashTableOptions` (`include/hashtable_options.h`): user-facing configuration (capacity, max_capacity, dim, eviction strategy, IO block size, etc.)

### Public API (`include/hkv_hashtable.h`)

`HashTable<K,V,S,Strategy>` is the concrete implementation (template, header-only). Key methods:
- `insert_or_assign()`: inserts; auto-triggers rehash when load factor exceeds ~75%
- `insert_and_evict()`: inserts and evicts low-scoring entries; only in pure-HBM mode
- `find_or_insert()`: atomic find-or-insert returning device pointers to values
- `find()`, `find_and_update()`: lookups with optional score update
- `assign_scores()`: bulk score override
- `clear()`: reset table
- `reserve()`: expand capacity (calls `double_capacity()` + `rehash_kernel`)
- `export_batch()`: iterate all stored entries in chunks
- `save()` / `load()`: checkpoint via `BaseKVFile` / `LocalKVFile`
- `fast_load_factor()`: approximate load factor sampling up to 1024 buckets

`HashTableBase<K,V,S>` is the pure virtual interface.

### Eviction Strategies (`include/score_functor.h`)

Five strategies defined by `EvictStrategy` enum: `kLru`, `kLfu`, `kEpochLru`, `kEpochLfu`, `kCustomized`. Each is a `ScoreFunctor<K,V,S,Strategy>` specialization implementing `desired_when_missed()`, `update()`, `update_with_digest()`, `update_score_only()`.

### AscendC Kernel Organization (`hkv_hashtable/`)

Each operation has its own subdirectory with `.cpp` kernel files compiled by the CANN AscendC compiler. Kernels are launched from host code via auto-generated `aclrtlaunch_*` headers using `ACLRT_LAUNCH_KERNEL()`.

`dump_kernel.cpp` is special: compiled with `set_source_files_properties(... LANGUAGE ASC)` into a separate `kernel_lib` shared library (with `-DUSE_DUMP_KERNEL_ASC`), because `export_batch()` needs to transfer ownership of device memory addresses.

### Kernel Dispatch Pattern (`include/simt_vf_dispatcher.h`)

Runtime `value_size` / `group_size` are converted to compile-time template args via `DISPATCH_VALUE_SIZE()` and `DISPATCH_GROUP_SIZE()` macros. Large value sizes (`is_large_size=true`, > threshold) use `_with_thread_1024` kernel variants. `GetValueMoveOpt()` in `hkv_hashtable.h` computes the optimal memory access width (8 or 16 bytes) and CTA group size based on vector byte size.

### Memory Management

- `DefaultAllocator` (`include/allocator.h`): wraps `aclrtMalloc`/`aclrtFree` for HBM, plus pinned and regular host allocations.
- `BucketMemoryPoolManager` (`include/bucket_memory_pool_manager.h`): implements `IBucketAddressProvider`. Pre-allocates a single large contiguous HBM block to avoid fragmentation during table expansion. Controlled by `use_bucket_memory_pool_` flag.
- Three memory pool types used by `HashTable`: `DeviceMemoryPool`, `HostMemoryPool`, and the `BucketMemoryPoolManager`.

### Platform Compatibility (`include/cuda2npu.h`)

Shim mapping CUDA-style annotations to AscendC equivalents:
- `__device__` → `__aicore__` (on CCE/device builds) or nothing (on host)
- `__gm__` → AscendC global memory qualifier
- `__forceinline__` → `inline`

This allows headers to be shared between host-side C++ and device-side AscendC kernel code.

## Build Modes

| `RUN_MODE` | Description | Compiler flags |
|---|---|---|
| `cpu` | CPU simulation via `tikicpulib` | `-g -O0 -std=c++17` |
| `sim` | NPU simulator | `-g -O2 -mllvm -disable-machine-licm` |
| `npu` | Real Ascend hardware | `-g -O2 -mllvm -disable-machine-licm` |

## Tests

Tests are in `tests/` and use Google Test (bundled in `3rdparty/gtest/`). Enable with `-DENABLE_TEST=1`. Test binaries: `test_main` (all core tests), `test_aclnn_helper` (separate due to `TEST_MEM` macro), `test_score_functor/test_score_functor` (kernel-level ScoreFunctor tests).
