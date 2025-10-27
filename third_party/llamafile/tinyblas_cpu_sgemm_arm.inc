// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/tinyblas_cpu_sgemm.inc
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

#include "tinyblas_cpu.h"
#include <arm_neon.h>
#include <ostream>
#include <iostream>
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

namespace {

template <typename TC>
void SgemmHelperN1Neon2(long m, long n, long k, const float16_t* A, long lda, const float16_t* B, long ldb,
                        TC* C, long ldc, int ith, int nth) {
    // A m * k    B n * k    c n * m
    const long NVL = 8;
    long kk = k / (NVL * 4);
    kk = kk * (NVL * 4);
    long length = (m / nth) + (ith < (m % nth) ? 1 : 0);
    long startRow = ith * (m / nth) + (ith < (m % nth) ? ith : (m % nth));
    long endRow = startRow + length;
    for (long i = startRow; i < endRow; i ++) {
        const float16_t* tA = A + i * lda;
        float32x4_t c0 = vdupq_n_f32(0);
        float32x4_t c1 = vdupq_n_f32(0);
        float32x4_t c2 = vdupq_n_f32(0);
        float32x4_t c3 = vdupq_n_f32(0);
        float32x4_t c4 = vdupq_n_f32(0);
        float32x4_t c5 = vdupq_n_f32(0);
        float32x4_t c6 = vdupq_n_f32(0);
        float32x4_t c7 = vdupq_n_f32(0);
        for (long j = 0; j < kk; j += NVL * 4) {
            __builtin_prefetch(tA + 192, 0, 0);
            float16x8_t a0 = vld1q_f16(tA + j);
            float16x8_t b0 = vld1q_f16(B + j);
            c0 = vfmlalq_low_f16(c0, a0, b0);
            c1 = vfmlalq_high_f16(c1, a0, b0);
            float16x8_t a1 = vld1q_f16(tA + j + NVL);
            float16x8_t b1 = vld1q_f16(B + j + NVL);
            c2 = vfmlalq_low_f16(c2, a1, b1);
            c3 = vfmlalq_high_f16(c3, a1, b1);
            float16x8_t a2 = vld1q_f16(tA + j + NVL * 2);
            float16x8_t b2 = vld1q_f16(B + j + NVL * 2);
            c4 = vfmlalq_low_f16(c4, a2, b2);
            c5 = vfmlalq_high_f16(c5, a2, b2);
            float16x8_t a3 = vld1q_f16(tA + j + NVL * 3);
            float16x8_t b3 = vld1q_f16(B + j + NVL * 3);
            c6 = vfmlalq_low_f16(c6, a3, b3);
            c7 = vfmlalq_high_f16(c7, a3, b3);
        }
        if (k - kk >= NVL * 2) {
            float16x8_t a0 = vld1q_f16(tA + kk);
            float16x8_t b0 = vld1q_f16(B + kk);
            c0 = vfmlalq_low_f16(c0, a0, b0);
            c1 = vfmlalq_high_f16(c1, a0, b0);
            float16x8_t a1 = vld1q_f16(tA + kk + NVL);
            float16x8_t b1 = vld1q_f16(B + kk + NVL);
            c2 = vfmlalq_low_f16(c2, a1, b1);
            c3 = vfmlalq_high_f16(c3, a1, b1);
            kk += NVL * 2;
        }
        if (k - kk >= NVL) {
            float16x8_t a = vld1q_f16(tA + kk);
            float16x8_t b = vld1q_f16(B + kk);
            c0 = vfmlalq_low_f16(c0, a, b);
            c1 = vfmlalq_high_f16(c1, a, b);
            kk += NVL;
        }
        TC sum = 0.0f;
        for (long j = kk; j < k; j ++) {
            sum += (float32_t)tA[j] * (float32_t)B[j];
        }
        c0 = vaddq_f32(c0, c1);
        c2 = vaddq_f32(c2, c3);
        c4 = vaddq_f32(c4, c5);
        c6 = vaddq_f32(c6, c7);
        c0 = vaddq_f32(c0, c2);
        c4 = vaddq_f32(c4, c6);
        sum += vaddvq_f32(c0) + vaddvq_f32(c4);
        C[i] = sum;
    }
    return;
}

template <typename TC>
void SgemmHelperN1(long m, long n, long k, const ggml_fp16_t* A_, long lda, const ggml_fp16_t* B_, long ldb,
                   TC* C, long ldc, int ith, int nth) {
    // A m * k    B n * k    c n * m
    float16_t *A = (float16_t*)A_;
    float16_t *B = (float16_t*)B_;
    long rowsPerThread = m / nth;
    long startRow = ith * rowsPerThread;
    long endRow = (ith == nth - 1) ? m : startRow + rowsPerThread;
    for (long i = startRow; i < endRow; i ++) {
        TC sum = 0.0f;
        for (long j = 0; j < k; j ++) {
            sum += (float32_t)A[i * lda + j] * (float32_t)B[j];
        }
        C[i] = sum;
    }
    return;
}

template <typename TC>
bool llamafile_sgemm_impl(long m, long n, long k, const void* A, long lda, const void* B, long ldb, TC* C, long ldc, int ith, int nth, int task, int Atype, int Btype, int Ctype, int precision) {
    // std::cout << "tinyBLAS tinyBLAS NOT_SUPPORTED FP16  55, n: " << n << ", m: " << m << ", k: " << k << ", FLAG_precise: " << FLAG_precise << "\n"<<std::endl;
    switch (Atype) {
        case GGML_TYPE_F32: {
            if (Btype != GGML_TYPE_F32)
                return NOT_SUPPORTED;
#if defined(__AVX512F__)
            if (k % 16)
                return NOT_SUPPORTED;
            tinyBLAS<0, 16, __m512, __m512, float, float, TC> tb{
                k, (const float*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#elif defined(__AVX__) || defined(__AVX2__)
            if (k % 8)
                return NOT_SUPPORTED;
            tinyBLAS<0, 8, __m256, __m256, float, float, TC> tb{
                k, (const float*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#elif defined(__ARM_NEON)
            if (k % 4)
                return NOT_SUPPORTED;
            tinyBLAS<0, 4, float32x4_t, float32x4_t, float, float, TC> tb{
                k, (const float*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#else
            return NOT_SUPPORTED;
#endif
        }

        case GGML_TYPE_BF16: {
#if defined(__AVX512BF16__)
            if (k % 32)
                return NOT_SUPPORTED;
            if (Btype == GGML_TYPE_F32 && n < 2) {
                tinyBLAS<0, 16, __m512, __m512, ggml_bf16_t, float, TC> tb{
                    k, (const ggml_bf16_t*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
                tb.matmul(m, n, task);
                return true;
            }
            if (Btype == GGML_TYPE_F32)
                return WANT_QUANTIZATION;
            if (Btype != GGML_TYPE_BF16)
                return NOT_SUPPORTED;
            if (!FLAG_precise) {
                tinyBLAS<0, 32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, TC> tb{
                    k, (const ggml_bf16_t*)A, lda, (const ggml_bf16_t*)B, ldb, C, ldc, ith, nth};
                tb.matmul(m, n, task);
                return true;
            } else {
                tinyBLAS<0, 16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, TC> tb{
                    k, (const ggml_bf16_t*)A, lda, (const ggml_bf16_t*)B, ldb, C, ldc, ith, nth};
                tb.matmul(m, n, task);
                return true;
            }
#elif defined(__AVX512F__)
            if (k % 16)
                return NOT_SUPPORTED;
            tinyBLAS<0, 16, __m512, __m512, ggml_bf16_t, float, TC> tb{
                k, (const ggml_bf16_t*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#elif defined(__AVX2__)
            if (k % 8)
                return NOT_SUPPORTED;
            if (Btype != GGML_TYPE_F32)
                return NOT_SUPPORTED;
            tinyBLAS<0, 8, __m256, __m256, ggml_bf16_t, float, TC> tb{
                k, (const ggml_bf16_t*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#elif defined(__ARM_NEON) && !defined(_MSC_VER)
            if (k % 4)
                return NOT_SUPPORTED;
            if (Btype != GGML_TYPE_F32)
                return NOT_SUPPORTED;
            tinyBLAS<0, 4, float32x4_t, float32x4_t, ggml_bf16_t, float, TC> tb{
                k, (const ggml_bf16_t*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#else
            return NOT_SUPPORTED;
#endif
        }

        case GGML_TYPE_F16: {
#if defined(__AVX512F__)
            if (k % 16)
                return NOT_SUPPORTED;
            if (Btype == GGML_TYPE_F32 && n < 2) {
                tinyBLAS<0, 16, __m512, __m512, ggml_fp16_t, float, TC> tb{
                    k, (const ggml_fp16_t*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
                tb.matmul(m, n, task);
                return true;
            }
            if (Btype == GGML_TYPE_F32)
                return WANT_QUANTIZATION;
            if (Btype != GGML_TYPE_F16)
                return NOT_SUPPORTED;
            tinyBLAS<0, 16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, TC> tb{
                k, (const ggml_fp16_t*)A, lda, (const ggml_fp16_t*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#elif (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
            // if (X86_CHECK(F16C)) {
            if (k % 8)
                return NOT_SUPPORTED;
            if (Btype == GGML_TYPE_F32 && n < 2) {
                tinyBLAS<0, 8, __m256, __m256, ggml_fp16_t, float, TC> tb{
                    k, (const ggml_fp16_t*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
                tb.matmul(m, n, task);
                return true;
            }
            if (Btype == GGML_TYPE_F32)
                return WANT_QUANTIZATION;
            if (Btype != GGML_TYPE_F16)
                return NOT_SUPPORTED;
            tinyBLAS<0, 8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, TC> tb{
                k, (const ggml_fp16_t*)A, lda, (const ggml_fp16_t*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
            // } else {
            //     return NOT_SUPPORTED;
            // }
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
            if (n < 2 && !FLAG_precise) {
                // TODO(jart): Why is ggml_vec_dot_f16_unroll() so fast at matvec?
                if (Btype == GGML_TYPE_F16 && task == GGML_TASK_TYPE_COMPUTE) {
                    SgemmHelperN1Neon2<TC>(m, n, k, (const float16_t*)A, lda, (const float16_t*)B, ldb, C, ldc, ith, nth);
                    // SgemmHelperN1<TC>(m, n, k, (const ggml_fp16_t*)A, lda, (const ggml_fp16_t*)B, ldb, C, ldc, ith, nth);
                    return true;
                }
                return NOT_SUPPORTED;
            }
            if (precision == GGML_PREC_F32) {
                if (k % 4)
                    return NOT_SUPPORTED;
                if (Btype != GGML_TYPE_F32)
                    return NOT_SUPPORTED;
                tinyBLAS<0, 4, float32x4_t, float32x4_t, ggml_fp16_t, float, TC> tb{
                    k, (const ggml_fp16_t*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
                tb.matmul(m, n, task);
                return true;
            } else {
                if (k % 8)
                    return NOT_SUPPORTED;
                if (Btype == GGML_TYPE_F32)
                    return WANT_QUANTIZATION;
                if (Btype != GGML_TYPE_F16)
                    return NOT_SUPPORTED;
                tinyBLAS<0, 8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, TC> tb{
                    k, (const ggml_fp16_t*)A, lda, (const ggml_fp16_t*)B, ldb, C, ldc, ith, nth};
                tb.matmul(m, n, task);
                return true;
            }
#elif defined(__ARM_NEON) && !defined(_MSC_VER)
            if (n < 2 && !FLAG_precise) {
                // TODO(jart): Why is ggml_vec_dot_f16_unroll() so fast at matvec?
                // printf("tinyBLAS tinyBLAS NOT_SUPPORTED FP16 225, m: %ld, n: %ld, k: %ld\n", m, n, k);
                if (Btype == GGML_TYPE_F16 && task == GGML_TASK_TYPE_COMPUTE) {
                    SgemmHelperN1Neon2<TC>(m, n, k, (const float16_t*)A, lda, (const float16_t*)B, ldb, C, ldc, ith, nth);
                    // SgemmHelperN1<TC>(m, n, k, (const ggml_fp16_t*)A, lda, (const ggml_fp16_t*)B, ldb, C, ldc, ith, nth);
                    return true;
                }
                std::cout << "tinyBLAS tinyBLAS NOT_SUPPORTED FP16 231, n: " << n << ", m: " << m << ", k: " << m << ", FLAG_precise: " << FLAG_precise << "\n"<<std::endl;
                return NOT_SUPPORTED;
            }
            if (k % 4) {
                // std::cout << "tinyBLAS tinyBLAS NOT_SUPPORTED FP16  215" <<std::endl;
                return NOT_SUPPORTED;
            }
            if (Btype != GGML_TYPE_F32) {
                // std::cout << "tinyBLAS tinyBLAS NOT_SUPPORTED FP16  218" <<std::endl;
                return NOT_SUPPORTED;
            }
            // std::cout << "tinyBLAS tinyBLAS true FP16" <<std::endl;
            tinyBLAS<0, 4, float32x4_t, float32x4_t, ggml_fp16_t, float, TC> tb{
                k, (const ggml_fp16_t*)A, lda, (const float*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#else
            // std::cout << "tinyBLAS tinyBLAS NOT_SUPPORTED FP16" <<std::endl;
            return NOT_SUPPORTED;
#endif
        }

        case GGML_TYPE_Q8_0: {
            if (Btype == GGML_TYPE_F32)
                return WANT_QUANTIZATION;
            if (Btype != GGML_TYPE_Q8_0)
                return NOT_SUPPORTED;
#if defined(__AVX2__) || defined(__AVX512F__)
            tinyBLAS_Q0_AVX2<0, block_q8_0, block_q8_0, TC> tb{
                k, (const block_q8_0*)A, lda, (const block_q8_0*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#elif defined(__ARM_FEATURE_DOTPROD)
            tinyBLAS_Q0_ARM<0, block_q8_0, block_q8_0, TC> tb{
                k, (const block_q8_0*)A, lda, (const block_q8_0*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#else
            return NOT_SUPPORTED;
#endif
        }

        case GGML_TYPE_Q4_0: {
            if (Btype == GGML_TYPE_F32)
                return WANT_QUANTIZATION;
            if (Btype != GGML_TYPE_Q8_0)
                return NOT_SUPPORTED;
#if defined(__AVX2__) || defined(__AVX512F__)
            tinyBLAS_Q0_AVX2<0, block_q4_0, block_q8_0, TC> tb{
                k, (const block_q4_0*)A, lda, (const block_q8_0*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#elif defined(__ARM_FEATURE_DOTPROD)
            tinyBLAS_Q0_ARM<0, block_q4_0, block_q8_0, TC> tb{
                k, (const block_q4_0*)A, lda, (const block_q8_0*)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n, task);
            return true;
#else
            return NOT_SUPPORTED;
#endif
        }

        default:
            return NOT_SUPPORTED;
    }

    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)ith;
    (void)nth;
    (void)Atype;
    (void)Btype;
    (void)precision;
}

}  // namespace

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 1,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32,
 *                     GGML_PREC_DEFAULT);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @param precision may be used to control the internal compute type
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(long m, long n, long k, const void* A, long lda, const void* B, long ldb, void* C, long ldc, int ith, int nth, int task, int Atype, int Btype, int Ctype, int precision) {
    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(nth > 0);
    assert(ith < nth);

#if QK_K == 256
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) && (defined(__FMA__) || (defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))))
    // if (X86_CHECK(AVX2) && X86_CHECK(FMA)) {
    if (Btype == GGML_TYPE_Q8_K && Ctype == GGML_TYPE_F32){
        if (iqk_mul_mat(m, n, k * QK_K, Atype, A, B, (float*)C, ldc, ith, nth)) {
            return true;
        }
    }
    if ((Btype == GGML_TYPE_Q8_0 || Btype == GGML_TYPE_Q8_1) && Ctype == GGML_TYPE_F32) {
        // assert(QK8_0 == QK8_1 == QK4_0 == QK4_1 == QK5_0 == QK5_1 == 32);
        assert((QK8_0 == 32) && (QK8_1 == 32) && (QK4_0 == 32) && (QK4_1 == 32) && (QK5_0 == 32) && (QK5_1 == 32));
        if (iqk_mul_mat(m, n, k * QK8_0, Atype, A, B, (float*)C, ldc, ith, nth)) {
            return true;
        }
    }
    // }
#endif
#elif defined __aarch64__ && defined __ARM_FEATURE_DOTPROD && !defined _MSC_VER
    if (Btype == GGML_TYPE_Q8_K && Ctype == GGML_TYPE_F32) {
        if (iqk_mul_mat(m, n, k * QK_K, Atype, A, k, Btype, B, k, (float*)C, ldc, ith, nth)) {
            return true;
        }
    }
    if ((Btype == GGML_TYPE_Q8_0 || Btype == GGML_TYPE_Q8_1) && Ctype == GGML_TYPE_F32) {
        // assert(QK8_0 == QK8_1 == QK4_0 == QK4_1 == QK5_0 == QK5_1 == 32);
        assert((QK8_0 == 32) && (QK8_1 == 32) && (QK4_0 == 32) && (QK4_1 == 32) && (QK5_0 == 32) && (QK5_1 == 32));
        if (iqk_mul_mat(m, n, k * QK8_0, Atype, A, k, Btype, B, k, (float*)C, ldc, ith, nth)) {
            return true;
        }
    }
#endif
#endif

    switch (Ctype) {
        case GGML_TYPE_F32:
            return llamafile_sgemm_impl(m, n, k, A, lda, B, ldb, (float*)C, ldc, ith, nth, task, Atype,
                                        Btype, Ctype, precision);
        default:
            return NOT_SUPPORTED;
    }
}
