---
name: simd-intrinsics-lookup
description: Look up and select appropriate x86 SIMD intrinsics from Intel Intrinsics Guide. Use when finding specific vector instructions, comparing instruction performance, or selecting optimal intrinsics for a given operation. Keywords: intrinsics, Intel Intrinsics Guide, AVX, AVX2, AVX-512, SIMD instructions, vector operations.
allowed-tools: [WebSearch, WebFetch, Read]
model: haiku
---

# SIMD Intrinsics Lookup

## Quick Reference

### Intel Intrinsics Guide
**URL**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

Search tips:
- By operation: "add", "multiply", "permute"
- By data type: "epi32" (int32), "ps" (float), "pd" (double)
- By ISA: "AVX-512", "AVX2", "FMA"

## Naming Convention

```
_mm<width>_<operation>_<type>

width:    (empty)=128, 256, 512
operation: add, mul, load, store, permute, etc.
type:     ps=float, pd=double, epi32=int32, epi8=int8, si512=raw bits
```

Examples:
- `_mm512_add_ps` = 512-bit float add
- `_mm256_mullo_epi32` = 256-bit int32 multiply (low bits)
- `_mm512_permutexvar_epi8` = 512-bit byte permute

## Common Operations Quick Reference

### Load/Store

| Operation | SSE (128) | AVX (256) | AVX-512 (512) |
|-----------|-----------|-----------|---------------|
| Aligned load | `_mm_load_ps` | `_mm256_load_ps` | `_mm512_load_ps` |
| Unaligned load | `_mm_loadu_ps` | `_mm256_loadu_ps` | `_mm512_loadu_ps` |
| Aligned store | `_mm_store_ps` | `_mm256_store_ps` | `_mm512_store_ps` |
| Masked load | - | - | `_mm512_maskz_load_ps` |
| Masked store | - | - | `_mm512_mask_store_ps` |
| Broadcast scalar | `_mm_set1_ps` | `_mm256_broadcast_ss` | `_mm512_set1_ps` |
| Stream store | `_mm_stream_ps` | `_mm256_stream_ps` | `_mm512_stream_ps` |

### Arithmetic (Float)

| Operation | SSE | AVX | AVX-512 |
|-----------|-----|-----|---------|
| Add | `_mm_add_ps` | `_mm256_add_ps` | `_mm512_add_ps` |
| Subtract | `_mm_sub_ps` | `_mm256_sub_ps` | `_mm512_sub_ps` |
| Multiply | `_mm_mul_ps` | `_mm256_mul_ps` | `_mm512_mul_ps` |
| Divide | `_mm_div_ps` | `_mm256_div_ps` | `_mm512_div_ps` |
| FMA (a*b+c) | - | `_mm256_fmadd_ps` | `_mm512_fmadd_ps` |
| Max | `_mm_max_ps` | `_mm256_max_ps` | `_mm512_max_ps` |
| Min | `_mm_min_ps` | `_mm256_min_ps` | `_mm512_min_ps` |
| Sqrt | `_mm_sqrt_ps` | `_mm256_sqrt_ps` | `_mm512_sqrt_ps` |
| Reciprocal | `_mm_rcp_ps` | `_mm256_rcp_ps` | `_mm512_rcp14_ps` |

### Arithmetic (Integer)

| Operation | SSE | AVX2 | AVX-512 |
|-----------|-----|------|---------|
| Add int32 | `_mm_add_epi32` | `_mm256_add_epi32` | `_mm512_add_epi32` |
| Add int8 | `_mm_add_epi8` | `_mm256_add_epi8` | `_mm512_add_epi8` |
| Multiply int32 | `_mm_mullo_epi32` | `_mm256_mullo_epi32` | `_mm512_mullo_epi32` |
| Multiply int16 | `_mm_mullo_epi16` | `_mm256_mullo_epi16` | `_mm512_mullo_epi16` |
| VNNI dot (u8*i8) | - | - | `_mm512_dpbusd_epi32` |

### Shuffle/Permute

| Operation | SSE | AVX | AVX-512 |
|-----------|-----|-----|---------|
| Shuffle float | `_mm_shuffle_ps` | `_mm256_shuffle_ps` | `_mm512_shuffle_ps` |
| Permute float | - | `_mm256_permutevar8x32_ps` | `_mm512_permutexvar_ps` |
| Permute int32 | - | `_mm256_permutevar8x32_epi32` | `_mm512_permutexvar_epi32` |
| Permute bytes | - | - | `_mm512_permutexvar_epi8` |
| Blend | `_mm_blend_ps` | `_mm256_blend_ps` | `_mm512_mask_blend_ps` |

### Conversion

| From -> To | Intrinsic | Notes |
|------------|-----------|-------|
| int32 -> float | `_mm512_cvtepi32_ps` | |
| float -> int32 | `_mm512_cvttps_epi32` | Truncate |
| float -> int32 | `_mm512_cvtps_epi32` | Round |
| int8 -> int32 | `_mm512_cvtepi8_epi32` | Sign extend |
| uint8 -> int32 | `_mm512_cvtepu8_epi32` | Zero extend |
| fp32 -> bf16 | `_mm512_cvtne2ps_pbh` | AVX512-BF16 |
| bf16 -> fp32 | shift left 16 bits | Manual |

### Comparison (AVX-512 returns mask)

| Operation | AVX-512 |
|-----------|---------|
| Equal | `_mm512_cmpeq_epi32_mask` |
| Greater than | `_mm512_cmpgt_epi32_mask` |
| Less than | `_mm512_cmplt_epi32_mask` |
| Float compare | `_mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)` |

### Reduction

| Operation | AVX-512 |
|-----------|---------|
| Sum all | `_mm512_reduce_add_ps` |
| Max all | `_mm512_reduce_max_ps` |
| Min all | `_mm512_reduce_min_ps` |

### Bit Manipulation

| Operation | AVX-512 |
|-----------|---------|
| AND | `_mm512_and_si512` / `_mm512_and_ps` |
| OR | `_mm512_or_si512` / `_mm512_or_ps` |
| XOR | `_mm512_xor_si512` / `_mm512_xor_ps` |
| Shift left | `_mm512_slli_epi32` |
| Shift right | `_mm512_srli_epi32` (logical) / `_mm512_srai_epi32` (arithmetic) |

## AVX-512 Masked Operations

AVX-512 uses mask registers (k0-k7) for conditional operations:

```cpp
__mmask16 mask = 0b1111111100000000;  // Upper 8 elements active

// Masked load (zeros inactive elements)
__m512 v = _mm512_maskz_load_ps(mask, ptr);

// Masked load (keeps original in inactive elements)
__m512 v = _mm512_mask_load_ps(original, mask, ptr);

// Masked compute
__m512 result = _mm512_mask_add_ps(original, mask, a, b);

// Compare returns mask
__mmask16 cmp_mask = _mm512_cmpgt_ps_mask(a, b);
```

## Performance Tips

### Latency vs Throughput
- **Latency**: Cycles until result is ready
- **Throughput**: Operations per cycle (inverse of CPI)

Typical Skylake-X performance:
| Operation | Latency | Throughput |
|-----------|---------|------------|
| `_mm512_add_ps` | 4 | 0.5 |
| `_mm512_mul_ps` | 4 | 0.5 |
| `_mm512_fmadd_ps` | 4 | 0.5 |
| `_mm512_div_ps` | 14 | 5 |
| `_mm512_sqrt_ps` | 19 | 6 |
| `_mm512_load_ps` | 5 (L1) | 0.5 |
| `_mm512_permutexvar_epi8` | 3 | 1 |

### Instruction Selection Guidelines

1. **Prefer FMA over separate mul+add**
   ```cpp
   // Bad: 2 instructions, longer dependency
   __m512 t = _mm512_mul_ps(a, b);
   __m512 r = _mm512_add_ps(t, c);

   // Good: 1 instruction, same latency as mul
   __m512 r = _mm512_fmadd_ps(a, b, c);
   ```

2. **Use masked operations for tail handling**
   ```cpp
   // AVX-512: no scalar cleanup needed
   __mmask16 tail_mask = (1u << remaining) - 1;
   __m512 v = _mm512_maskz_load_ps(tail_mask, ptr);
   ```

3. **Avoid lane-crossing shuffles when possible**
   - In-lane: `_mm256_shuffle_ps` (fast)
   - Cross-lane: `_mm256_permutevar8x32_ps` (slower)

4. **Use `vpermb` for LUT lookups**
   ```cpp
   // 64-entry byte lookup in one instruction
   __m512i result = _mm512_permutexvar_epi8(indices, lut);
   ```

## KTransformers-Specific Intrinsics

### VNNI (INT8 Dot Product)
```cpp
// acc += sum(a[i] * b[i]) for 4 pairs of int8
// a: uint8_t[4], b: int8_t[4] -> int32 accumulator
__m512i acc = _mm512_dpbusd_epi32(acc, a_u8, b_i8);
```

### BF16 (Brain Float 16)
```cpp
// Convert 32 FP32 to 32 BF16 (packed in 512 bits)
__m512bh bf16 = _mm512_cvtne2ps_pbh(fp32_hi, fp32_lo);  // AVX512-BF16

// Without AVX512-BF16: manual shift
__m512i as_int = _mm512_castps_si512(fp32);
__m512i bf16_bits = _mm512_srli_epi32(as_int, 16);  // Truncate
```

### AMX (Advanced Matrix Extensions)
```cpp
// Configure tiles
_tile_loadconfig(&config);

// Load tiles (16x64 bytes max per tile)
_tile_loadd(0, src_a, stride_a);  // Load A to TMM0
_tile_loadd(1, src_b, stride_b);  // Load B to TMM1

// Compute: C += A * B
_tile_dpbssd(2, 0, 1);  // signed * signed -> int32

// Store result
_tile_stored(2, dst, stride_c);
```

## How to Search Intel Intrinsics Guide

1. **By operation name**: "fmadd", "permute", "gather"
2. **By assembly mnemonic**: "vfmadd", "vpermb", "vgatherd"
3. **By feature flag**: Check CPUID requirements
4. **Filter by technology**: AVX-512, AVX2, etc.

## When to Use This Skill

Invoke when:
- Finding the right intrinsic for an operation
- Comparing performance of alternatives
- Checking CPUID requirements
- Understanding instruction semantics

## Related Skills
- `simd-operator-design`: Full kernel design
- `simd-microbench`: Performance validation
