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