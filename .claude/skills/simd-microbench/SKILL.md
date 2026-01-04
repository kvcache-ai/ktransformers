---
name: simd-microbench
description: Write and run microbenchmarks to validate SIMD operator performance. Use when verifying throughput, comparing implementations, or ensuring CPU resources are fully utilized. Keywords: benchmark, microbench, performance, throughput, roofline, profiling.
allowed-tools: [Read, Write, Edit, Glob, Grep, Bash]
model: sonnet
---

# SIMD Microbenchmarking for KTransformers

## Core Principle

> **Never merge SIMD code without benchmarks. If performance doesn't hit targets, redesign.**

## Project Test Structure

```
kt-kernel/operators/amx/test/
├── avx-test.cpp          # AVX-512 bandwidth benchmark
├── nvfp4-moe-test.cpp    # NVFP4 functionality + perf
├── amx-test.cpp          # AMX matrix multiply tests
└── build scripts (*.sh)

kt-kernel/bench/
├── bench_moe_amx.py      # Python benchmark harness
├── bench_fp8_moe.py      # FP8 quantization bench
└── multi_bench_moe.py    # Multi-config testing
```

## Quick Start: C++ Microbenchmark

### Template for New Benchmark

Create `kt-kernel/operators/amx/test/your-kernel-test.cpp`:

```cpp
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <omp.h>

// Configuration
constexpr size_t DATA_SIZE = 100ULL * 1024 * 1024 * 1024;  // 100GB for bandwidth
constexpr size_t ALIGNMENT = 64;
constexpr int WARMUP_ITERS = 10;
constexpr int TEST_ITERS = 100;

// Allocate aligned memory
template <typename T>
T* alloc_aligned(size_t count) {
    void* ptr = std::aligned_alloc(ALIGNMENT, count * sizeof(T));
    if (!ptr) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }
    return static_cast<T*>(ptr);
}

// Your kernel to benchmark
void your_kernel(const float* __restrict__ src,
                 float* __restrict__ dst,
                 size_t n) {
    constexpr size_t VEC_WIDTH = 16;

    #pragma omp parallel for
    for (size_t i = 0; i < n; i += VEC_WIDTH) {
        __m512 v = _mm512_load_ps(src + i);
        // ... your computation ...
        _mm512_store_ps(dst + i, v);
    }
}

// Baseline for comparison
void baseline_kernel(const float* src, float* dst, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i];  // or your scalar version
    }
}

int main() {
    const size_t n = DATA_SIZE / sizeof(float);

    float* src = alloc_aligned<float>(n);
    float* dst = alloc_aligned<float>(n);

    // Initialize
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        src[i] = static_cast<float>(i % 1000) / 1000.0f;
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        your_kernel(src, dst, n);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_ITERS; ++i) {
        your_kernel(src, dst, n);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double time_per_iter = elapsed / TEST_ITERS;

    // Calculate metrics
    size_t bytes_accessed = n * sizeof(float) * 2;  // read + write
    double bandwidth_gbps = (bytes_accessed * TEST_ITERS) / elapsed / 1e9;

    printf("=== Benchmark Results ===\n");
    printf("Data size:     %.2f GB\n", n * sizeof(float) / 1e9);
    printf("Time per iter: %.3f ms\n", time_per_iter * 1000);
    printf("Bandwidth:     %.2f GB/s\n", bandwidth_gbps);
    printf("Iterations:    %d\n", TEST_ITERS);

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < std::min(n, (size_t)1000); ++i) {
        float expected = /* your expected value */;
        if (std::abs(dst[i] - expected) > 1e-5) {
            printf("Mismatch at %zu: got %f, expected %f\n", i, dst[i], expected);
            correct = false;
            break;
        }
    }
    printf("Correctness:   %s\n", correct ? "PASS" : "FAIL");

    free(src);
    free(dst);
    return correct ? 0 : 1;
}
```

### Build Script

Create `kt-kernel/operators/amx/test/build_your_test.sh`:

```bash
#!/bin/bash
set -e

g++ -O3 -march=native -fopenmp \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl \
    your-kernel-test.cpp \
    -o your-kernel-test

echo "Build complete. Run with: ./your-kernel-test"
```

## Measuring Different Metrics

### Memory Bandwidth (GB/s)
```cpp
// For memory-bound kernels
size_t bytes = n * sizeof(T) * accesses_per_element;
double bandwidth = bytes / elapsed_seconds / 1e9;
```

### Compute Throughput (GFLOPS)
```cpp
// For compute-bound kernels
size_t flops = n * flops_per_element;
double gflops = flops / elapsed_seconds / 1e9;
```

### Throughput vs Latency
```cpp
// Throughput: How fast can we process bulk data?
// Latency: How fast is a single operation?

// For latency measurement (dependency chain):
__m512 v = _mm512_setzero_ps();
auto start = rdtsc();
for (int i = 0; i < ITERS; ++i) {
    v = _mm512_add_ps(v, v);  // Depends on previous result
}
auto end = rdtsc();
double cycles_per_op = (double)(end - start) / ITERS;
```

## Performance Targets

### Memory Bandwidth
| Memory Type | Theoretical | Practical |
|-------------|-------------|-----------|
| DDR4-3200 | 51 GB/s | ~40 GB/s |
| DDR5-5600 | 90 GB/s | ~70 GB/s |
| DDR5-6400 | 102 GB/s | ~80 GB/s |

### Compute (per core, AVX-512)
| Operation | Theoretical | Notes |
|-----------|-------------|-------|
| FP32 FMA | 64 FLOPS/cycle | 2 FMAs × 16 floats × 2 (mul+add) |
| INT8 VNNI | 256 ops/cycle | 4 multiplies per DPBUSD |

### Speedup Expectations
| Implementation | vs Scalar |
|----------------|-----------|
| SSE (128-bit) | 2-4× |
| AVX2 (256-bit) | 4-8× |
| AVX-512 (512-bit) | 8-16× |
| AVX-512 + AMX | 10-20× |

## Common Issues and Fixes

### 1. Compiler Optimizing Away Work
```cpp
// BAD: Compiler may delete unused results
for (int i = 0; i < ITERS; ++i) {
    result = compute(data);
}

// GOOD: Prevent optimization
for (int i = 0; i < ITERS; ++i) {
    result = compute(data);
    asm volatile("" : "+m"(result));  // Compiler barrier
}

// Or use volatile store:
volatile float sink;
sink = _mm512_reduce_add_ps(result);
```

### 2. Cache Effects
```cpp
// BAD: Data fits in cache, unrealistic
constexpr size_t N = 1024;  // 4KB

// GOOD: Exceed L3 cache
constexpr size_t N = 256 * 1024 * 1024;  // 1GB
```

### 3. Frequency Scaling
```bash
# Before benchmarking:
sudo cpupower frequency-set -g performance

# Pin to specific cores:
taskset -c 0-7 ./benchmark

# Check current frequency:
watch -n 0.5 "cat /proc/cpuinfo | grep MHz"
```

### 4. NUMA Effects
```cpp
// Use first-touch policy
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
    data[i] = 0;  // Each thread touches its portion
}

// Or use numactl:
// numactl --membind=0 ./benchmark
```

### 5. Thermal Throttling
```bash
# Monitor temperature:
watch -n 1 "sensors | grep Core"

# Check for throttling:
sudo turbostat --show Busy%,Bzy_MHz,PkgWatt -i 1
```

## Comparison Benchmark Template

```cpp
void benchmark_compare() {
    // ... setup ...

    struct Impl {
        const char* name;
        void (*func)(const float*, float*, size_t);
    };

    Impl impls[] = {
        {"scalar", scalar_kernel},
        {"avx2", avx2_kernel},
        {"avx512", avx512_kernel},
        {"avx512_unroll4", avx512_unroll4_kernel},
    };

    printf("%-20s %12s %12s %10s\n", "Implementation", "Time (ms)", "GB/s", "Speedup");
    printf("%-20s %12s %12s %10s\n", "-------------", "--------", "----", "-------");

    double baseline_time = 0;

    for (const auto& impl : impls) {
        // Warmup
        for (int i = 0; i < WARMUP; ++i) impl.func(src, dst, n);

        // Measure
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) impl.func(src, dst, n);
        auto t1 = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        double gbps = (n * sizeof(float) * 2) / (time_ms / 1000) / 1e9;

        if (baseline_time == 0) baseline_time = time_ms;
        double speedup = baseline_time / time_ms;

        printf("%-20s %12.3f %12.2f %10.2fx\n", impl.name, time_ms, gbps, speedup);
    }
}
```

## Python Benchmark (for high-level operators)

```python
#!/usr/bin/env python3
import time
import numpy as np
import torch

# Import your kernel
from kt_kernel import your_kernel

def benchmark_kernel(func, *args, warmup=10, iters=100):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Synchronize if using GPU
    if hasattr(torch.cuda, 'synchronize'):
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        func(*args)
        if hasattr(torch.cuda, 'synchronize'):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000

    return mean_ms, std_ms

# Usage
input_data = torch.randn(batch, seq_len, hidden_dim)
mean, std = benchmark_kernel(your_kernel, input_data)
print(f"Time: {mean:.3f} +/- {std:.3f} ms")
```

## Profiling Tools

### perf stat
```bash
perf stat -e cycles,instructions,cache-misses,cache-references \
    ./your-benchmark
```

### perf record + flamegraph
```bash
perf record -g ./your-benchmark
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

### Intel VTune (if available)
```bash
vtune -collect hotspots ./your-benchmark
vtune -report summary -r r000hs
```

## Checklist

Before merging:
- [ ] Benchmark exceeds theoretical minimum threshold
- [ ] Tested on representative data sizes
- [ ] Compared against baseline implementation
- [ ] Verified correctness alongside performance
- [ ] Documented expected performance in comments
- [ ] Tested with different thread counts

## When to Use This Skill

Invoke when:
- Validating new SIMD kernel performance
- Comparing implementation alternatives
- Investigating performance regressions
- Setting up CI performance tests

## Related Skills
- `simd-operator-design`: Design the kernel first
- `simd-intrinsics-lookup`: Find optimal instructions
