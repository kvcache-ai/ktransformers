#pragma once
#include <cstddef>
#ifndef _BATCH_GEMM_KERNEL_API_
#define _BATCH_GEMM_KERNEL_API_
#include "../api/common.h"
#ifdef __cplusplus
extern "C" {
#endif
void decode_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                               const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc, const size_t m,
                               const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                               const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob, const float beta,
                               int32_t* c, const size_t ldc, const int32_t* oc);

void prefill_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc, const size_t m,
                                const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                                const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob, const float beta,
                                int32_t* c, const size_t ldc, const int32_t* oc);

void decode_int4_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                    const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
                                    const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                                    const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb,
                                    const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                                    const int32_t* oc);

void prefill_int4_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                     const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
                                     const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                                     const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb,
                                     const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                                     const int32_t* oc);
void reorder_B_gemm(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transb, const size_t k,
                    const size_t n, const size_t ldb, const void* b, void* b_reordered);
size_t get_reorder_B_size(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transb, const size_t k,
                          const size_t n);

#ifdef __cplusplus
}
#endif
#endif /*** _BATCH_GEMM_KERNEL_API_ ***/