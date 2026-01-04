---
name: simd-operator-design
description: Design high-performance SIMD operators for x86 architecture. Use when developing new computational kernels, vectorized operations, or low-level performance-critical code. Keywords: SIMD, AVX, AVX2, AVX-512, vectorization, kernel design, intrinsics.
allowed-tools: [Read, Write, Edit, Glob, Grep, Bash, WebFetch, WebSearch]
model: sonnet
---

# SIMD Operator Design for KTransformers

## Project Context

KTransformers uses x86 SIMD instructions (AVX-512, AMX) to accelerate LLM inference. All kernel code follows specific patterns for maintainability and performance.

## Code Organization

### Directory Structure
```
kt-kernel/operators/amx/
├── la/                          # Low-level linear algebra kernels
│   ├── amx_kernels.hpp         # Core GEMM kernels (AMX tiles)
│   ├── amx_buffers.hpp         # Buffer management + quantization
│   ├── nvfp4_kernel.hpp        # NVFP4 LUT-based multiply
│   ├── utils.hpp               # BF16/FP32 conversion utilities
│   └── pack.hpp                # Weight packing/reordering
├── moe.hpp                     # MOE layer implementation
├── fp8-moe.hpp                 # FP8 quantized MOE
├── nvfp4-moe.hpp               # NVFP4 quantized MOE
└── test/                       # Unit tests and benchmarks
    ├── avx-test.cpp
    └── nvfp4-moe-test.cpp
```

### Where to Put New Code
| Code Type | Location | Naming |
|-----------|----------|--------|
| Low-level SIMD kernel | `operators/amx/la/` | `*_kernel.hpp` |
| Buffer/data management | `operators/amx/la/` | `*_buffers.hpp` |
| High-level operator | `operators/amx/` | `*-moe.hpp` or `*.hpp` |
| Unit test | `operators/amx/test/` | `*-test.cpp` |

## Design Workflow

### Step 1: Analyze the Computation

Before writing SIMD code:
1. Understand the mathematical operation
2. Identify data types (FP32, BF16, INT8, FP4, etc.)
3. Determine memory access patterns
4. Calculate theoretical peak throughput

### Step 2: Choose Instruction Set

| ISA | Vector Width | Best For |
|-----|-------------|----------|
| AVX-512 | 512-bit (16 floats) | General vectorization |
| AVX-512 VNNI | 512-bit | INT8 dot products |
| AMX | 16x16 tiles | Large matrix multiplies |
| AVX-512 BF16 | 512-bit | BF16 conversion |

Runtime detection:
```cpp
#ifdef HAVE_AMX
    // AMX path
#else
    // AVX-512 fallback
#endif
```

### Step 3: Design the Kernel

#### Memory Alignment
Always use 64-byte alignment for AVX-512/AMX:
```cpp
alignas(64) float buffer[SIZE];
alignas(64) static const uint8_t LUT[64] = {...};
```

#### Loop Structure
```cpp
void kernel(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    constexpr size_t VEC_WIDTH = 16;  // AVX-512: 16 floats
    size_t i = 0;

    // Main vectorized loop
    for (; i + VEC_WIDTH <= n; i += VEC_WIDTH) {
        __m512 v = _mm512_load_ps(src + i);
        // ... compute ...
        _mm512_store_ps(dst + i, v);
    }

    // Tail handling with masking (preferred for AVX-512)
    if (i < n) {
        __mmask16 mask = (1u << (n - i)) - 1;
        __m512 v = _mm512_maskz_load_ps(mask, src + i);
        // ... compute ...
        _mm512_mask_store_ps(dst + i, mask, v);
    }
}
```

#### Template Metaprogramming Style
Use compile-time dispatch for quantization types:
```cpp
// Forward declaration
template <class QA, class QB>
struct GemmKernel;

// Specialization for INT8 x INT8
template <>
struct GemmKernel<int8_t, int8_t> {
    static void run(const int8_t* a, const int8_t* b, int32_t* c) {
        // Implementation
    }
};

// Dispatch macro
#define DISPATCH_QTYPES(QA, QB, ...) \
    [&] { \
        if constexpr (std::is_same_v<QA, int8_t> && std::is_same_v<QB, int8_t>) { \
            return GemmKernel<int8_t, int8_t>::run(__VA_ARGS__); \
        } \
    }()
```

## Common Patterns

### Pattern 1: LUT-based Computation (NVFP4 style)
For non-linear operations, use lookup tables:
```cpp
// Define LUT (64-byte aligned for AVX-512 vpermb)
alignas(64) static const uint8_t RESULT_LUT[64] = {
    // Pre-computed results
};

// Use vpermb for parallel lookup
__m512i indices = ...;  // 64 indices in range [0,63]
__m512i results = _mm512_permutexvar_epi8(indices, _mm512_load_si512(RESULT_LUT));
```

### Pattern 2: Horizontal Reduction
```cpp
float horizontal_sum(__m512 v) {
    // Reduce 512 -> 256 -> 128 -> scalar
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(low, high);

    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                               _mm256_extractf128_ps(sum256, 1));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// Or use built-in (simpler but check performance):
float sum = _mm512_reduce_add_ps(v);
```

### Pattern 3: BF16 <-> FP32 Conversion
```cpp
// FP32 to BF16 (32 floats -> 32 bf16)
static inline void fp32_to_bf16(__m512* src0, __m512* src1, __m512i* dst) {
#if defined(__AVX512BF16__)
    *dst = (__m512i)_mm512_cvtne2ps_pbh(*src1, *src0);
#else
    // Manual: shift right 16 bits with rounding
    __m512i i0 = _mm512_castps_si512(*src0);
    __m512i i1 = _mm512_castps_si512(*src1);
    // Add rounding bias
    i0 = _mm512_add_epi32(i0, _mm512_set1_epi32(0x7FFF));
    i1 = _mm512_add_epi32(i1, _mm512_set1_epi32(0x7FFF));
    // Shift and pack
    i0 = _mm512_srli_epi32(i0, 16);
    i1 = _mm512_srli_epi32(i1, 16);
    *dst = _mm512_packus_epi32(i0, i1);
#endif
}
```

### Pattern 4: Tiled Matrix Multiply (AMX)
```cpp
template <int TILE_M = 16, int TILE_N = 16, int TILE_K = 32>
struct TiledGemm {
    static void config() {
#ifdef HAVE_AMX
        // Configure tile registers
        _tile_loadconfig(&tile_config);
#endif
    }

    static void compute(const int8_t* A, const int8_t* B, int32_t* C,
                        int lda, int ldb, int ldc) {
#ifdef HAVE_AMX
        _tile_loadd(0, A, lda);  // Load A into TMM0
        _tile_loadd(1, B, ldb);  // Load B into TMM1
        _tile_loadd(2, C, ldc);  // Load C into TMM2
        _tile_dpbssd(2, 0, 1);   // C += A * B
        _tile_stored(2, C, ldc); // Store result
#endif
    }
};
```

## Performance Considerations

### Memory Bandwidth
- DDR4-3200: ~40 GB/s practical
- DDR5-5600: ~80 GB/s practical
- Ensure kernel is compute-bound, not memory-bound

### Cache Blocking
```cpp
constexpr int K_BLOCK = 512;  // Fits in L2 cache
constexpr int M_BLOCK = 64;   // Fits in L1 cache

for (int k = 0; k < K; k += K_BLOCK) {
    for (int m = 0; m < M; m += M_BLOCK) {
        // Process block
    }
}
```

### Avoid These
1. Unaligned loads/stores when alignment is possible
2. Gather/scatter when sequential access works
3. Scalar operations inside vector loops
4. Branch mispredictions in hot loops

## Checklist Before Implementation

- [ ] Understand the mathematical operation
- [ ] Choose appropriate instruction set (AVX-512 vs AMX)
- [ ] Design memory layout (alignment, blocking)
- [ ] Plan for tail handling
- [ ] Consider multi-threading with OpenMP
- [ ] Write unit test in `operators/amx/test/`
- [ ] Benchmark against baseline

## When to Use This Skill

Invoke when:
- Designing a new SIMD kernel
- Vectorizing existing scalar code
- Implementing a quantized operation
- Adding new data type support

## Related Skills
- `simd-intrinsics-lookup`: Find specific intrinsics
- `simd-microbench`: Validate kernel performance
