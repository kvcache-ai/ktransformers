#include "../api/mat_kernel.h"

#include <cassert>

namespace {
constexpr int kInt4ElementDivisor = 2;
constexpr int kInt8ElementDivisor = 1;
}  // namespace
extern "C" {
void decode_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                               const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc, const size_t m,
                               const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                               const int8_t oa, const void* b, const size_t ldb, const int8_t ob, const float beta,
                               int32_t* c, const size_t ldc, const int32_t* oc);

void prefill_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc, const size_t m,
                                const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                                const int8_t oa, const void* b, const size_t ldb, const int8_t ob, const float beta,
                                int32_t* c, const size_t ldc, const int32_t* oc);

void decode_int4_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                    const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
                                    const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                                    const size_t lda, const int8_t oa, const void* b, const size_t ldb, const int8_t ob,
                                    const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void prefill_int4_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                     const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
                                     const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                                     const size_t lda, const int8_t oa, const void* b, const size_t ldb,
                                     const int8_t ob, const float beta, int32_t* c, const size_t ldc,
                                     const int32_t* oc);

// Stub implementation for non-BLIS builds - INT4 prefill not supported
void prefill_int4_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                     const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
                                     const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                                     const size_t lda, const int8_t oa, const void* b, const size_t ldb,
                                     const int8_t ob, const float beta, int32_t* c, const size_t ldc,
                                     const int32_t* oc) {
  (void)layout; (void)transa; (void)transb; (void)offsetc;
  (void)m; (void)n; (void)k; (void)alpha; (void)a; (void)lda; (void)oa;
  (void)b; (void)ldb; (void)ob; (void)beta; (void)c; (void)ldc; (void)oc;
  // INT4 prefill requires BLIS/AOCL which is not available in this build
  // This stub prevents linker errors; the function should not be called
}

// Stub implementations for non-BLIS builds - INT8 kernels not supported
void decode_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                               const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc, const size_t m,
                               const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                               const int8_t oa, const void* b, const size_t ldb, const int8_t ob, const float beta,
                               int32_t* c, const size_t ldc, const int32_t* oc) {
  (void)layout; (void)transa; (void)transb; (void)offsetc;
  (void)m; (void)n; (void)k; (void)alpha; (void)a; (void)lda; (void)oa;
  (void)b; (void)ldb; (void)ob; (void)beta; (void)c; (void)ldc; (void)oc;
  // INT8 decode requires BLIS/AOCL which is not available in this build
  // This stub prevents linker errors; the function should not be called
}

void prefill_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc, const size_t m,
                                const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                                const int8_t oa, const void* b, const size_t ldb, const int8_t ob, const float beta,
                                int32_t* c, const size_t ldc, const int32_t* oc) {
  (void)layout; (void)transa; (void)transb; (void)offsetc;
  (void)m; (void)n; (void)k; (void)alpha; (void)a; (void)lda; (void)oa;
  (void)b; (void)ldb; (void)ob; (void)beta; (void)c; (void)ldc; (void)oc;
  // INT8 prefill requires BLIS/AOCL which is not available in this build
  // This stub prevents linker errors; the function should not be called
}

void decode_int4_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                    const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
                                    const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                                    const size_t lda, const int8_t oa, const void* b, const size_t ldb,
                                    const int8_t ob, const float beta, int32_t* c, const size_t ldc,
                                    const int32_t* oc) {
  (void)layout; (void)transa; (void)transb; (void)offsetc;
  (void)m; (void)n; (void)k; (void)alpha; (void)a; (void)lda; (void)oa;
  (void)b; (void)ldb; (void)ob; (void)beta; (void)c; (void)ldc; (void)oc;
  // INT4 decode requires BLIS/AOCL which is not available in this build
  // This stub prevents linker errors; the function should not be called
}

// Stub for AOCL/BLIS B-matrix reordering function
// Called from GemmKernelInt8::BufferB::from_mat when if_pack=true and plain=false
size_t get_reorder_B_size(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transb,
                          const size_t k, const size_t n_block) {
  (void)layout; (void)transb; (void)k; (void)n_block;
  // Return a dummy size - this should not be called in non-AOCL builds
  return k * n_block;
}

void reorder_B_gemm(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transb,
                    const size_t k, const size_t n_block, const size_t ldb, const void* b_src, void* b_dst) {
  (void)layout; (void)transb; (void)k; (void)n_block; (void)ldb; (void)b_src; (void)b_dst;
  // B-matrix reordering requires AOCL/BLIS which is not available in this build
  // This stub prevents linker errors; the function should not be called
}
}

MatKernelSelection select_kernel_for_int4(MatKernelVariant variant) {
  switch (variant) {
    case MatKernelVariant::Decode:
      return {decode_int4_cblas_gemm_s8s8s32, kInt4ElementDivisor};
    case MatKernelVariant::Prefill:
      return {prefill_int4_cblas_gemm_s8s8s32, kInt4ElementDivisor};
  }
  return {nullptr, 0};
}

MatKernelSelection select_kernel_for_int8(MatKernelVariant variant) {
  switch (variant) {
    case MatKernelVariant::Decode:
      return {decode_cblas_gemm_s8s8s32, kInt8ElementDivisor};
    case MatKernelVariant::Prefill:
      return {prefill_cblas_gemm_s8s8s32, kInt8ElementDivisor};
  }
  return {nullptr, 0};
}