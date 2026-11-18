// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/tinyblas_cpu.h
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
//
//                                ██████╗ ██╗   █████╗ ██████╗
//         ██████╗██╗██╗ ██╗██═██╗██╔══██╗██║  ██╔══██╗██╔═══╝
//         ╚═██╔═╝██║███▄██║██ ██║██████╔╝██║  ███████║██████╗
//           ██║  ██║██▀███║╚███╔╝██╔══██╗██║  ██╔══██║╔═══██║
//           ██║  ██║██║ ██║ ███║ ██████╔╝████╗██║  ██║██████║
//           ╚═╝  ╚═╝╚═╝ ╚═╝ ╚══╝ ╚═════╝ ╚═══╝╚═╝  ╚═╝╚═════╝
//
//                   BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#pragma once

#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
// #include "log.h"
#include "flags.h"
#include "sgemm.h"

#include <vector>
// #include <cosmo.h>

// Pull in the CPU vec-dot declarations (including ggml_vec_dot_mxfp4_q8_0,
// used by the MXFP4 case in tinyblas_cpu_sgemm.inc) and ggml_compute_params.
// The pinned third_party/llama.cpp is a flat b3173-lineage tree, so these
// live under llama.cpp/ in the include root.
#if defined(__riscv)
#include "llama.cpp/ggml-quants.h"
#include "kt_ggml_compute_params.h"
#endif

#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"

#define ROW_ALIGN 64
#define MATRIX_ALIGN 4096
#define MAX_ALIGN 4096

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

// ---------------------------------------------------------------------------
// RISC-V Vector Extension (RVV 1.0) support.  VLEN=128 hardware maps a
// vfloat32m1_t to 4 lanes, the same tiling SSE uses; larger VLEN still works
// because the RVV intrinsics below always pass an explicit vl.
// ---------------------------------------------------------------------------
#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include "rv64_vec_math.hpp"
#define KT_RVV 1
// RVV has 32 architectural vector registers; use the wide tiling like AVX512.
#define KT_RVV_VECTOR_REGS 32
#endif

// llama.cpp b3173+ removed ggml_task_type; kt-kernel's llamafile operators
// still pass GGML_TASK_TYPE_COMPUTE to llamafile_sgemm.  Provide the enum so
// the shared tinyblas sources compile against either ggml generation.
#if defined(__riscv) && !defined(GGML_TASK_TYPE_COMPUTE)
#define GGML_TASK_TYPE_INIT 0
#define GGML_TASK_TYPE_COMPUTE 1
#define GGML_TASK_TYPE_FINALIZE 2
#endif

#if 0
#define NOT_SUPPORTED tinyBLAS_not_supported(__FILE__, __LINE__)
#else
#define NOT_SUPPORTED false
#endif
#define WANT_QUANTIZATION false

namespace {

bool tinyBLAS_not_supported(const char* file, int line) {
    // tinylogf("%s:%d: tinyBLAS not supported\n", file, line);
    return false;
}

inline float unhalf(ggml_fp16_t d) {
    return GGML_FP16_TO_FP32(d);
}
inline float unhalf(ggml_bf16_t d) {
    return GGML_BF16_TO_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MATRIX MEMORY INDEXING

#define NCA 1
#define NCB 2
#define NCC 4

#define INDEX(A, lda, j, i) (CONFIG & NC##A ? ((T##A**)A)[j] + i : A + lda * (j) + i)

////////////////////////////////////////////////////////////////////////////////////////////////////
// GGML TYPE TRAITS

template <typename T>
struct ggml_type_trait;
template <>
struct ggml_type_trait<float> {
    static constexpr ggml_type id = GGML_TYPE_F32;
};
template <>
struct ggml_type_trait<ggml_bf16_t> {
    static constexpr ggml_type id = GGML_TYPE_BF16;
};
template <>
struct ggml_type_trait<ggml_fp16_t> {
    static constexpr ggml_type id = GGML_TYPE_F16;
};
template <>
struct ggml_type_trait<block_q8_0> {
    static constexpr ggml_type id = GGML_TYPE_Q8_0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m128 add(__m128 x, __m128 y) {
    return _mm_add_ps(x, y);
}
inline __m128 sub(__m128 x, __m128 y) {
    return _mm_sub_ps(x, y);
}
inline __m128 mul(__m128 x, __m128 y) {
    return _mm_mul_ps(x, y);
}
#endif  // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m256 add(__m256 x, __m256 y) {
    return _mm256_add_ps(x, y);
}
inline __m256 sub(__m256 x, __m256 y) {
    return _mm256_sub_ps(x, y);
}
inline __m256 mul(__m256 x, __m256 y) {
    return _mm256_mul_ps(x, y);
}
#endif  // __AVX__

#if defined(__AVX512F__)
inline __m512 add(__m512 x, __m512 y) {
    return _mm512_add_ps(x, y);
}
inline __m512 sub(__m512 x, __m512 y) {
    return _mm512_sub_ps(x, y);
}
inline __m512 mul(__m512 x, __m512 y) {
    return _mm512_mul_ps(x, y);
}
#endif  // __AVX512F__

#if defined(__ARM_NEON)
inline float32x4_t add(float32x4_t x, float32x4_t y) {
    return vaddq_f32(x, y);
}
inline float32x4_t sub(float32x4_t x, float32x4_t y) {
    return vsubq_f32(x, y);
}
inline float32x4_t mul(float32x4_t x, float32x4_t y) {
    return vmulq_f32(x, y);
}
#endif  // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline float16x8_t add(float16x8_t x, float16x8_t y) {
    return vaddq_f16(x, y);
}
inline float16x8_t sub(float16x8_t x, float16x8_t y) {
    return vsubq_f16(x, y);
}
inline float16x8_t mul(float16x8_t x, float16x8_t y) {
    return vmulq_f16(x, y);
}
#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

/**
 * Computes a * b + c with error correction.
 *
 * @see W. Kahan, "Further remarks on reducing truncation errors,"
 *    Communications of the ACM, vol. 8, no. 1, p. 40, Jan. 1965,
 *    doi: 10.1145/363707.363723.
 */
template <typename T, typename U>
inline U madder(T a, T b, U c, U* e) {
    U y = sub(mul(a, b), *e);
    U t = add(c, y);
    *e = sub(sub(t, c), y);
    return t;
}

#ifdef __ARM_NEON
inline float32x4_t badder(float32x4_t a, float b, float32x4_t c, float32x4_t* e) {
    float32x4_t y = sub(vmulq_n_f32(a, b), *e);
    float32x4_t t = add(c, y);
    *e = sub(sub(t, c), y);
    return t;
}
#endif

#if defined(__FMA__) || (defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__)))
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
#endif
#if defined(__AVX512F__)
template <>
inline __m512 madd(__m512 a, __m512 b, __m512 c) {
    return _mm512_fmadd_ps(a, b, c);
}
#endif
#endif

#if defined(__ARM_FEATURE_FMA)
template <>
inline float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vfmaq_f32(c, a, b);
}
#if 0  // todo: this specialization chops gcc 12.3 performance in half
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER) && 0
template <>
inline float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    return vfmaq_f16(c, b, a);
}
#endif
#endif
#endif

#if defined(__AVX512BF16__)
template <>
inline __m512 madd(__m512bh x, __m512bh y, __m512 z) {
    return _mm512_dpbf16_ps(z, x, y);
}
template <>
inline __m512 madder(__m512bh x, __m512bh y, __m512 z, __m512* _) {
    return _mm512_dpbf16_ps(z, x, y);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// RISC-V RVV VECTORIZED ARITHMETIC OPERATIONS
// vfloat32m1_t is the tinyBLAS vector workhorse on rv64.  The add/sub/mul/
// madd/madder/hsum overloads below mirror the AVX512 ones.  All RVV intrinsics
// need an active vector length; we always use the full register (vlmax).
#if defined(KT_RVV)
namespace rvvops {
inline size_t vl() { return __riscv_vsetvlmax_e32m1(); }
inline vfloat32m1_t add(vfloat32m1_t x, vfloat32m1_t y) { return rvv::vadd_f32(x, y, vl()); }
inline vfloat32m1_t sub(vfloat32m1_t x, vfloat32m1_t y) { return rvv::vsub_f32(x, y, vl()); }
inline vfloat32m1_t mul(vfloat32m1_t x, vfloat32m1_t y) { return rvv::vmul_f32(x, y, vl()); }
inline vfloat32m1_t madd(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t c) {
    return rvv::vfmacc_f32(c, a, b, vl());
}
inline vfloat32m1_t madder(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t c, vfloat32m1_t* e) {
    vfloat32m1_t p = rvv::vmul_f32(a, b, vl());
    vfloat32m1_t y = rvv::vsub_f32(p, *e, vl());
    vfloat32m1_t t = rvv::vadd_f32(c, y, vl());
    *e = rvv::vsub_f32(rvv::vsub_f32(t, c, vl()), y, vl());
    return t;
}
}  // namespace rvvops
#endif  // KT_RVV

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#if defined(__ARM_NEON)
inline float hsum(float32x4_t x) {
    return vaddvq_f32(x);
}
#endif  // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
inline float hsum(float16x8_t x) {
    // todo: this works great on clang but it produces terrible code on gcc 12.3
    return vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_high_f16(x))));
}
#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m128 x) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
#else
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
#endif
    return _mm_cvtss_f32(x);
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}
#endif  // __AVX__

#if defined(__AVX512F__)
inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#endif  // __AVX512F__

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

template <typename T, typename U>
T load(const U*);

template <>
inline float load(const float* p) {
    return *p;
}
template <>
inline float load(const ggml_fp16_t* p) {
    return unhalf(*p);
}
template <>
inline float load(const ggml_bf16_t* p) {
    return unhalf(*p);
}

#if defined(__ARM_NEON)
template <>
inline float32x4_t load(const float* p) {
    return vld1q_f32(p);
}
template <>
inline float32x4_t load(const ggml_bf16_t* p) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16((const unsigned short*)p), 16));
}
#if !defined(_MSC_VER)
template <>
inline float16x8_t load(const ggml_fp16_t* p) {
    return vld1q_f16((const float16_t*)p);
}
template <>
inline float32x4_t load(const ggml_fp16_t* p) {
    return vcvt_f32_f16(vld1_f16((const float16_t*)p));
}
#endif  // _MSC_VER
#endif  // __ARM_NEON

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m128 load(const float* p) {
    return _mm_loadu_ps(p);
}
#endif  // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 load(const float* p) {
    return _mm256_loadu_ps(p);
}
#endif  // __AVX__

#if defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 load(const ggml_bf16_t* p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)), 16));
}
#endif  // __AVX2__

#if defined(__F16C__)
template <>
inline __m256 load(const ggml_fp16_t* p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)p));
}
#endif  // __F16C__

#if defined(__AVX512F__)
template <>
inline __m512 load(const float* p) {
    return _mm512_loadu_ps(p);
}
template <>
inline __m512 load(const ggml_fp16_t* p) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)p));
}
template <>
inline __m512 load(const ggml_bf16_t* p) {
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}
#endif  // __AVX512F__

#if defined(__AVX512BF16__)
template <>
inline __m512bh load(const ggml_bf16_t* p) {
    return (__m512bh)_mm512_loadu_ps((const float*)p);
}
template <>
inline __m512bh load(const float* p) {
    return _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
#endif  // __AVX512BF16__

#if defined(KT_RVV)
// Route the generic templates (madd/madder) to the RVV overloads.
template <>
inline vfloat32m1_t madd(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t c) {
    return rvvops::madd(a, b, c);
}
template <>
inline vfloat32m1_t madder(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t c, vfloat32m1_t* e) {
    return rvvops::madder(a, b, c, e);
}

inline float hsum(vfloat32m1_t x) {
    return rvv::vhsum_f32m1(x);
}

template <>
inline vfloat32m1_t load(const float* p) {
    return rvv::vload_f32(p, __riscv_vsetvlmax_e32m1());
}
template <>
inline vfloat32m1_t load(const ggml_bf16_t* p) {
    // widen bf16 -> f32: (uint32)u16 << 16 reinterpreted as float
    size_t vl = __riscv_vsetvlmax_e32m1();
    vuint16m1_t u = __riscv_vle16_v_u16m1((const uint16_t*)p, vl);
    vuint32m2_t w = __riscv_vwaddu_vx_u32m2(u, 0, vl);
    w = __riscv_vsll_vx_u32m2(w, 16, vl);
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vget_v_u32m2_u32m1(w, 0));
}
template <>
inline vfloat32m1_t load(const ggml_fp16_t* p) {
    // fp16 -> f32 without zvfh: integer bit manipulation (IEEE half decode).
    // f16: sign[15] exp[14:10] mant[9:0] -> f32: sign exp(m+112) mant<<13.
    // sign<<16 puts bit15->bit31 cleanly; exp/mant use <<13 so bit10->bit23.
    // exp 31 (inf/NaN) is reconstructed explicitly below (the +112 rebias would
    // wrap the exponent field). exp 0 (zero/denormal) is flushed to zero, with
    // the sign preserved so -0.0 stays -0.0.
    size_t vl = __riscv_vsetvlmax_e32m1();
    vuint16m1_t h = __riscv_vle16_v_u16m1((const uint16_t*)p, vl);
    vuint32m2_t w = __riscv_vwaddu_vx_u32m2(h, 0, vl);
    // Extract the sign bit BEFORE stripping it from w; then mask w so the
    // (w<<13) exp window is not contaminated by the sign (bit15 -> bit28).
    vuint32m2_t sign = __riscv_vand_vx_u32m2(__riscv_vsll_vx_u32m2(w, 16, vl), 0x80000000u, vl);
    w = __riscv_vand_vx_u32m2(w, 0x7fffu, vl);
    vuint32m2_t expm1 = __riscv_vand_vx_u32m2(__riscv_vsll_vx_u32m2(w, 13, vl), 0x7f800000u, vl);
    vuint32m2_t mant = __riscv_vand_vx_u32m2(__riscv_vsll_vx_u32m2(w, 13, vl), 0x007fffffu, vl);
    // biased exp = (h_exp + 112) << 23 = (h_exp<<23) + (112<<23); expm1 already
    // holds h_exp<<23, so adding (112<<23) rebias; mant stays in place.
    vuint32m2_t bits = __riscv_vor_vv_u32m2(sign, __riscv_vadd_vx_u32m2(expm1, 0x38000000u, vl), vl);
    bits = __riscv_vor_vv_u32m2(bits, mant, vl);
    const vuint32m2_t h_exp = __riscv_vand_vx_u32m2(w, 0x7c00, vl);
    // exp == 31: sign | 0x7f800000 | mant (inf when mant==0, NaN otherwise).
    vbool16_t infnan = __riscv_vmseq_vx_u32m2_b16(h_exp, 0x7c00, vl);
    vuint32m2_t special = __riscv_vor_vv_u32m2(sign, __riscv_vor_vx_u32m2(mant, 0x7f800000u, vl), vl);
    bits = __riscv_vmerge_vvm_u32m2(bits, special, infnan, vl);
    // exp == 0: flush zero/denormals to zero, preserving the sign bit.
    vbool16_t denorm = __riscv_vmseq_vx_u32m2_b16(h_exp, 0, vl);
    bits = __riscv_vmerge_vvm_u32m2(bits, sign, denorm, vl);
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vget_v_u32m2_u32m1(bits, 0));
}
#endif  // KT_RVV

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT OUTPUT STREAMING

inline void store(float* p, float f) {
    *p = f;
}

inline void store(ggml_fp16_t* p, float f) {
    *p = GGML_FP32_TO_FP16(f);
}

inline void store(ggml_bf16_t* p, float f) {
    *p = GGML_FP32_TO_BF16(f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

template <int CONFIG, int KN, typename D, typename V, typename TA, typename TB, typename TC>
class tinyBLAS {
   public:
    tinyBLAS(long k, const TA* A, long lda, const TB* B, long ldb, TC* C, long ldc, int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(long m, long n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

   private:
    NOINLINE void mnpack(long m0, long m, long n0, long n) {
        long mc, nc, mp, np;

#if VECTOR_REGISTERS == 32
        if (!FLAG_precise) {
            switch ((MIN(m - m0, 5) << 4) | MIN(n - n0, 5)) {
                case 0x55:
                    mc = 5;
                    nc = 5;
                    gemm<5, 5, false>(m0, m, n0, n);
                    break;
                case 0x54:
                case 0x53:
                case 0x52:
                case 0x45:
                case 0x44:
                case 0x43:
                case 0x42:
                case 0x35:
                case 0x34:
                case 0x33:
                case 0x32:
                case 0x25:
                case 0x24:
                case 0x23:
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm<2, 2, false>(m0, m, n0, n);
                    break;
                case 0x51:
                case 0x41:
                case 0x31:
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, false>(m0, m, n0, n);
                    break;
                case 0x15:
                case 0x14:
                case 0x13:
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, false>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, false>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        } else {
            switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 3)) {
                case 0x43:
                    mc = 4;
                    nc = 3;
                    gemm<4, 3, true>(m0, m, n0, n);
                    break;
                case 0x42:
                case 0x33:
                case 0x32:
                case 0x23:
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm<2, 2, true>(m0, m, n0, n);
                    break;
                case 0x41:
                case 0x31:
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, true>(m0, m, n0, n);
                    break;
                case 0x13:
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, true>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, true>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        }
#endif

#if VECTOR_REGISTERS == 16
        if (!FLAG_precise) {
            switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 3)) {
                case 0x43:
                    mc = 4;
                    nc = 3;
                    gemm<4, 3, false>(m0, m, n0, n);
                    break;
                case 0x42:
                case 0x33:
                case 0x32:
                case 0x23:
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm<2, 2, false>(m0, m, n0, n);
                    break;
                case 0x41:
                case 0x31:
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, false>(m0, m, n0, n);
                    break;
                case 0x13:
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, false>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, false>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        } else {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 2)) {
                case 0x32:
                    mc = 3;
                    nc = 2;
                    gemm<3, 2, true>(m0, m, n0, n);
                    break;
                case 0x23:
                    mc = 2;
                    nc = 3;
                    gemm<2, 3, true>(m0, m, n0, n);
                    break;
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm<2, 2, true>(m0, m, n0, n);
                    break;
                case 0x31:
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, true>(m0, m, n0, n);
                    break;
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, true>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, true>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        }
#endif

        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN, int PRECISE>
    NOINLINE void gemm(long m0, long m, long n0, long n) {
#if defined(KT_RVV)
        // GCC forbids RVV scalable vector types as array elements, so the
        // accumulators live in a thread-local scratch buffer.  The stride is
        // derived from the *runtime* VLMAX (not a fixed 4 lanes), so this is
        // correct on VLEN > 128 hardware too.  Callers guarantee k % vl == 0.
        constexpr int KT_NACC = RN * RM;
        static thread_local std::vector<unsigned char> kt_acc_buf;
        const size_t vl = __riscv_vsetvlmax_e32m1();
        const size_t acc_bytes = (((vl * sizeof(float)) + 15) / 16) * 16;
        const size_t acc_buf_bytes = (size_t)KT_NACC * acc_bytes;
        if (kt_acc_buf.size() < 2 * acc_buf_bytes)
            kt_acc_buf.resize(2 * acc_buf_bytes);
        unsigned char* CvBuf = kt_acc_buf.data();
        unsigned char* CeBuf = CvBuf + acc_buf_bytes;
        long ytiles = RM > 1 ? (m - m0) / RM : 1;
        long xtiles = RN > 1 ? (n - n0) / RN : 1;
        long tiles = xtiles * ytiles;
        long duty = (tiles + nth - 1) / nth;
        long start = duty * ith;
        long end = start + duty;
        if (end > tiles)
            end = tiles;
        for (long job = start; job < end; ++job) {
            long ii = m0 + job / xtiles * RM;
            long jj = n0 + job % xtiles * RN;
            for (int t = 0; t < KT_NACC; ++t) {
                __riscv_vse32_v_f32m1((float*)(CvBuf + (size_t)t * acc_bytes),
                                      __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
                __riscv_vse32_v_f32m1((float*)(CeBuf + (size_t)t * acc_bytes),
                                      __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
            }
            for (long l = 0; l < k; l += (long)vl)
#pragma GCC unroll 100
                for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                    for (int i = 0; i < RM; ++i) {
                        V a = load<V>(INDEX(A, lda, ii + i, l));
                        V b = load<V>(INDEX(B, ldb, jj + j, l));
                        float* cvp = (float*)(CvBuf + (size_t)(j * RM + i) * acc_bytes);
                        float* cep = (float*)(CeBuf + (size_t)(j * RM + i) * acc_bytes);
                        V cv = __riscv_vle32_v_f32m1(cvp, vl);
                        V ce = __riscv_vle32_v_f32m1(cep, vl);
                        if (PRECISE)
                            cv = madder(a, b, cv, &ce);
                        else
                            cv = madd(a, b, cv);
                        __riscv_vse32_v_f32m1(cvp, cv, vl);
                        if (PRECISE)
                            __riscv_vse32_v_f32m1(cep, ce, vl);
                    }
#pragma GCC unroll 100
            for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                for (int i = 0; i < RM; ++i)
                    store(INDEX(C, ldc, jj + j, ii + i),
                          hsum(__riscv_vle32_v_f32m1(
                              (const float*)(CvBuf + (size_t)(j * RM + i) * acc_bytes), vl)));
        }
#else
        long ytiles = RM > 1 ? (m - m0) / RM : 1;
        long xtiles = RN > 1 ? (n - n0) / RN : 1;
        long tiles = xtiles * ytiles;
        long duty = (tiles + nth - 1) / nth;
        long start = duty * ith;
        long end = start + duty;
        if (end > tiles)
            end = tiles;
        for (long job = start; job < end; ++job) {
            long ii = m0 + job / xtiles * RM;
            long jj = n0 + job % xtiles * RN;
            D Cv[RN][RM] = {};
            D Ce[RN][RM] = {};
            for (long l = 0; l < k; l += KN)
#pragma GCC unroll 100
                for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                    for (int i = 0; i < RM; ++i)
                        if (PRECISE)
                            Cv[j][i] = madder(load<V>(INDEX(A, lda, ii + i, l)),  //
                                              load<V>(INDEX(B, ldb, jj + j, l)),  //
                                              Cv[j][i], &Ce[j][i]);
                        else
                            Cv[j][i] = madd(load<V>(INDEX(A, lda, ii + i, l)),  //
                                            load<V>(INDEX(B, ldb, jj + j, l)),  //
                                            Cv[j][i]);
#pragma GCC unroll 100
            for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                for (int i = 0; i < RM; ++i)
                    store(INDEX(C, ldc, jj + j, ii + i), hsum(Cv[j][i]));
        }
#endif
    }

    const TA* const A;
    const TB* const B;
    TC* const C;
    const long k;
    const long lda;
    const long ldb;
    const long ldc;
    const int ith;
    const int nth;
};

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#if defined(__ARM_FEATURE_DOTPROD)
template <int CONFIG, typename TA, typename TB, typename TC>
class tinyBLAS_Q0_ARM {
   public:
    tinyBLAS_Q0_ARM(long k, const TA* A, long lda, const TB* B, long ldb, TC* C, long ldc, int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(long m, long n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

   private:
    NOINLINE void mnpack(long m0, long m, long n0, long n) {
        long mc, nc, mp, np;

        if (!FLAG_precise) {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3)) {
                case 0x33:
                    mc = 3;
                    nc = 3;
                    gemm<3, 3, false>(m0, m, n0, n);
                    break;
                case 0x32:
                case 0x23:
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm<2, 2, false>(m0, m, n0, n);
                    break;
                case 0x31:
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, false>(m0, m, n0, n);
                    break;
                case 0x13:
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, false>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, false>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        } else {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3)) {
                case 0x33:
                    mc = 3;
                    nc = 3;
                    gemm<3, 3, true>(m0, m, n0, n);
                    break;
                case 0x32:
                case 0x23:
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm<2, 2, true>(m0, m, n0, n);
                    break;
                case 0x31:
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, true>(m0, m, n0, n);
                    break;
                case 0x13:
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, true>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, true>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        }

        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN, int PRECISE>
    NOINLINE void gemm(long m0, long m, long n0, long n) {
        long ytiles = RM > 1 ? (m - m0) / RM : 1;
        long xtiles = RN > 1 ? (n - n0) / RN : 1;
        long tiles = xtiles * ytiles;
        long duty = (tiles + nth - 1) / nth;
        long start = duty * ith;
        long end = start + duty;
        if (end > tiles)
            end = tiles;
        for (long job = start; job < end; ++job) {
            long ii = m0 + job / xtiles * RM;
            long jj = n0 + job % xtiles * RN;
            float32x4_t Cv[RN][RM] = {};
            float32x4_t Ce[RN][RM] = {};
            for (int l = 0; l < k; ++l)
#pragma GCC unroll 100
                for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                    for (int i = 0; i < RM; ++i) {
                        float32x4_t a = vcvtq_f32_s32(vdotq_s32(
                            vdotq_s32(vdupq_n_s32(0), load_lo(INDEX(A, lda, ii + i, l)),
                                      load_lo(INDEX(B, ldb, jj + j, l))),
                            load_hi(INDEX(A, lda, ii + i, l)), load_hi(INDEX(B, ldb, jj + j, l))));
                        float b = unhalf(INDEX(A, lda, ii + i, l)->d) *
                                  unhalf(INDEX(B, ldb, jj + j, l)->d);
                        if (PRECISE)
                            Cv[j][i] = badder(a, b, Cv[j][i], &Ce[j][i]);
                        else
                            Cv[j][i] = vmlaq_n_f32(Cv[j][i], a, b);
                    }
#pragma GCC unroll 100
            for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                for (int i = 0; i < RM; ++i)
                    store(INDEX(C, ldc, jj + j, ii + i), hsum(Cv[j][i]));
        }
    }

    inline int8x16_t load_lo(const block_q8_0* b) {
        return vld1q_s8(b->qs);
    }

    inline int8x16_t load_hi(const block_q8_0* b) {
        return vld1q_s8(b->qs + 16);
    }

    inline int8x16_t load_lo(const block_q4_0* b) {
        return vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs), vdupq_n_u8(0x0f))),
                        vdupq_n_s8(0x8));
    }

    inline int8x16_t load_hi(const block_q4_0* b) {
        return vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)), vdupq_n_s8(0x8));
    }

    const TA* const A;
    const TB* const B;
    TC* const C;
    const long k;
    const long lda;
    const long ldb;
    const long ldc;
    const int ith;
    const int nth;
};
#endif  // __ARM_FEATURE_DOTPROD

#if defined(__AVX2__) || defined(__AVX512F__)
template <int CONFIG, typename TA, typename TB, typename TC>
class tinyBLAS_Q0_AVX2 {
   public:
    tinyBLAS_Q0_AVX2(long k, const TA* A, long lda, const TB* B, long ldb, TC* C, long ldc, int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(long m, long n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

   private:
    void mnpack(long m0, long m, long n0, long n) {
        long mc, nc, mp, np;

#if VECTOR_REGISTERS == 32
        if (!FLAG_precise) {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3)) {
                case 0x33:
                    mc = 3;
                    nc = 3;
                    gemm<3, 3, false>(m0, m, n0, n);
                    break;
                case 0x32:
                case 0x23:
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm<2, 2, false>(m0, m, n0, n);
                    break;
                case 0x31:
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, true>(m0, m, n0, n);
                    break;
                case 0x13:
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, true>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, true>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        } else {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3)) {
                case 0x33:
                    mc = 3;
                    nc = 3;
                    gemm<3, 3, true>(m0, m, n0, n);
                    break;
                case 0x32:
                case 0x23:
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm<2, 2, true>(m0, m, n0, n);
                    break;
                case 0x31:
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, true>(m0, m, n0, n);
                    break;
                case 0x13:
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, true>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, true>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        }
#endif

#if VECTOR_REGISTERS == 16
        if (!FLAG_precise) {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 2)) {
                case 0x32:
                    mc = 3;
                    nc = 2;
                    gemm<3, 2, false>(m0, m, n0, n);
                    break;
                case 0x23:
                    mc = 2;
                    nc = 3;
                    gemm<2, 3, false>(m0, m, n0, n);
                    break;
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm<2, 2, false>(m0, m, n0, n);
                    break;
                case 0x31:
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, false>(m0, m, n0, n);
                    break;
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, false>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, false>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        } else {
            switch ((MIN(m - m0, 2) << 4) | MIN(n - n0, 1)) {
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm<2, 1, true>(m0, m, n0, n);
                    break;
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm<1, 2, true>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm<1, 1, true>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        }
#endif

        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN, int PRECISE>
    NOINLINE void gemm(long m0, long m, long n0, long n) {
        long ytiles = RM > 1 ? (m - m0) / RM : 1;
        long xtiles = RN > 1 ? (n - n0) / RN : 1;
        long tiles = xtiles * ytiles;
        long duty = (tiles + nth - 1) / nth;
        long start = duty * ith;
        long end = start + duty;
        if (end > tiles)
            end = tiles;
        for (long job = start; job < end; ++job) {
            long ii = m0 + job / xtiles * RM;
            long jj = n0 + job % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            __m256 Ce[RN][RM] = {};
            for (long l = 0; l < k; ++l)
#pragma GCC unroll 100
                for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                    for (int i = 0; i < RM; ++i) {
                        __m256 a = _mm256_set1_ps(unhalf(INDEX(A, lda, ii + i, l)->d) *
                                                  unhalf(INDEX(B, ldb, jj + j, l)->d));
                        __m256 b = updot(_mm256_sign_epi8(load(INDEX(A, lda, ii + i, l)),
                                                          load(INDEX(A, lda, ii + i, l))),
                                         _mm256_sign_epi8(load(INDEX(B, ldb, jj + j, l)),
                                                          load(INDEX(A, lda, ii + i, l))));
                        if (PRECISE)
                            Cv[j][i] = madder(a, b, Cv[j][i], &Ce[j][i]);
                        else
                            Cv[j][i] = madd(a, b, Cv[j][i]);
                    }
#pragma GCC unroll 100
            for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                for (int i = 0; i < RM; ++i)
                    store(INDEX(C, ldc, jj + j, ii + i), hsum(Cv[j][i]));
        }
    }

    inline __m256i load(const block_q8_0* b) {
        return _mm256_loadu_si256((const __m256i*)b->qs);
    }

    inline __m256i load(const block_q4_0* b) {
        __m128i x = _mm_loadu_si128((const __m128i*)b->qs);
        return _mm256_sub_epi8(_mm256_and_si256(_mm256_set1_epi8(15),
                                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                                        _mm_srli_epi16(x, 4), 1)),
                               _mm256_set1_epi8(8));
    }

    inline __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#else
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
        return _mm256_cvtepi32_ps(res);
    }

    const TA* const A;
    const TB* const B;
    TC* const C;
    const long k;
    const long lda;
    const long ldb;
    const long ldc;
    const int ith;
    const int nth;
};
#endif  // __AVX2__

// RISC-V RVV quantized (Q4_0/Q8_0 x Q8_0) matrix multiplication.  Follows the
// same blocked-tile scheme as tinyBLAS_Q0_AVX2: per (RM,RN) tile accumulate
// scale * dot(int8 dequant(a), int8 b) over k blocks.  Uses vint8m1_t/vint16m2_t
// widening dot via vmv/vmacc on 32-bit, then converts to float.
#if defined(KT_RVV)
template <int CONFIG, typename TA, typename TB, typename TC>
class tinyBLAS_Q0_RVV {
   public:
    tinyBLAS_Q0_RVV(long k, const TA* A, long lda, const TB* B, long ldb, TC* C, long ldc, int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(long m, long n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

   private:
    void mnpack(long m0, long m, long n0, long n) {
        long mc, nc, mp, np;
        // Conservative 1x2 / 2x1 tiling: 32 arch registers but LMUL=2
        // intermediates, keep tiles small for register pressure safety.
        // NOTE: every branch here must genuinely reduce the problem range;
        // recursing with an empty m or n range (m == m0 || n == n0) would
        // loop forever, so bail out early on degenerate ranges.
        if (m <= m0 || n <= n0)
            return;
        if (MIN(m - m0, 2) >= 2 && MIN(n - n0, 2) >= 2) {
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
        } else if (MIN(m - m0, 2) >= 2) {
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
        } else if (MIN(n - n0, 2) >= 2) {
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
        } else if (m - m0 >= 1 && n - n0 >= 1) {
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
        } else {
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        if (mp == m0 && np == n0)
            return;  // no progress possible; avoids infinite recursion
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    NOINLINE void gemm(long m0, long m, long n0, long n) {
        long ytiles = RM > 1 ? (m - m0) / RM : 1;
        long xtiles = RN > 1 ? (n - n0) / RN : 1;
        long tiles = xtiles * ytiles;
        long duty = (tiles + nth - 1) / nth;
        long start = duty * ith;
        long end = start + duty;
        if (end > tiles) end = tiles;
        for (long job = start; job < end; ++job) {
            long ii = m0 + job / xtiles * RM;
            long jj = n0 + job % xtiles * RN;
            float Cv[RN][RM] = {};
            for (long l = 0; l < k; ++l)
                for (int j = 0; j < RN; ++j)
                    for (int i = 0; i < RM; ++i)
                        Cv[j][i] += dot(INDEX(A, lda, ii + i, l), INDEX(B, ldb, jj + j, l));
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    store(INDEX(C, ldc, jj + j, ii + i), Cv[j][i]);
        }
    }

    // dequantize int8 q8_0 block (32 elems) to f32 with scale, dot with b
    inline float dot(const block_q8_0* a, const block_q8_0* b) {
        const size_t vlmax8 = __riscv_vsetvlmax_e8m1();
        vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, vlmax8);
        const int8_t* pa = a->qs;
        const int8_t* pb = b->qs;
        for (int off = 0; off < 32; off += (int)vlmax8) {
            size_t vl = __riscv_vsetvl_e8m1(32 - off);
            vint8m1_t va = __riscv_vle8_v_i8m1(pa + off, vl);
            vint8m1_t vb = __riscv_vle8_v_i8m1(pb + off, vl);
            vint16m2_t prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
            acc = __riscv_vwredsum_vs_i16m2_i32m1(prod, acc, vl);
        }
        long s = __riscv_vmv_x_s_i32m1_i32(acc);
        return unhalf(a->d) * unhalf(b->d) * (float)s;
    }

    // q4_0 block: nibbles in [0,15] -> signed by subtracting 8
    inline float dot(const block_q4_0* a, const block_q8_0* b) {
        const size_t vlmax8 = __riscv_vsetvlmax_e8m1();
        vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, vlmax8);
        const uint8_t* pa = a->qs;
        const int8_t* pb = b->qs;
        for (int off = 0; off < 16; off += (int)vlmax8) {
            size_t vl = __riscv_vsetvl_e8m1(16 - off);
            vuint8m1_t raw = __riscv_vle8_v_u8m1(pa + off, vl);
            vint8m1_t lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(
                                                    __riscv_vand_vx_u8m1(raw, 0x0f, vl)),
                                                8, vl);
            vint8m1_t hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(
                                                    __riscv_vsrl_vx_u8m1(raw, 4, vl)),
                                                8, vl);
            vint8m1_t vb_lo = __riscv_vle8_v_i8m1(pb + off, vl);
            vint8m1_t vb_hi = __riscv_vle8_v_i8m1(pb + 16 + off, vl);
            vint16m2_t p1 = __riscv_vwmul_vv_i16m2(lo, vb_lo, vl);
            vint16m2_t p2 = __riscv_vwmul_vv_i16m2(hi, vb_hi, vl);
            vint16m2_t p = __riscv_vadd_vv_i16m2(p1, p2, vl);
            acc = __riscv_vwredsum_vs_i16m2_i32m1(p, acc, vl);
        }
        long s = __riscv_vmv_x_s_i32m1_i32(acc);
        return unhalf(a->d) * unhalf(b->d) * (float)s;
    }

    const TA* const A;
    const TB* const B;
    TC* const C;
    const long k;
    const long lda;
    const long ldb;
    const long ldc;
    const int ith;
    const int nth;
};
#endif  // KT_RVV

}  // namespace
