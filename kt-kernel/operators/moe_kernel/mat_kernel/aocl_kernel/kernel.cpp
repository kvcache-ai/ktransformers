#include <stdexcept>

#include "../batch_gemm_api.hpp"
#include "blis.h"

namespace {

char ToAoclOrder(KERNEL_CBLAS_LAYOUT layout) {
  switch (layout) {
    case KernelCblasRowMajor:
      return 'r';
    case KernelCblasColMajor:
      return 'c';
  }
  throw std::invalid_argument("Unsupported KERNEL_CBLAS_LAYOUT value");
}

char ToAoclTranspose(KERNEL_CBLAS_TRANSPOSE transpose) {
  switch (transpose) {
    case KernelCblasNoTrans:
      return 'n';
    case KernelCblasTrans:
      return 't';
    case KernelCblasConjTrans:
    case KernelCblasConjNoTrans:
      break;
  }
  throw std::invalid_argument("Unsupported KERNEL_CBLAS_TRANSPOSE value");
}

}  // namespace

// 映射表，layout 从KERNEL_CBLAS_ORDER 映射到'r'或者'c',以及将KERNEL_CBLAS_TRANSPOSE映射到'n'或者't'
#ifdef __cplusplus
extern "C" {
#endif
void decode_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                               const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc, const size_t m,
                               const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                               const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob, const float beta,
                               int32_t* c, const size_t ldc, const int32_t* oc) {
  const char order = ToAoclOrder(layout);
  const char op_a = ToAoclTranspose(transa);
  const char op_b = ToAoclTranspose(transb);
  (void)offsetc;
  aocl_gemm_s8s8s32os32(order, op_a, op_b, static_cast<dim_t>(m), static_cast<dim_t>(n), static_cast<dim_t>(k),
                        static_cast<int32_t>(alpha), static_cast<const int8_t*>(a), static_cast<dim_t>(lda), 'n',
                        static_cast<const int8_t*>(b), static_cast<dim_t>(ldb), 'r', static_cast<int32_t>(beta), c,
                        static_cast<dim_t>(ldc), nullptr);
}

void prefill_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc, const size_t m,
                                const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                                const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob, const float beta,
                                int32_t* c, const size_t ldc, const int32_t* oc) {
  const char order = ToAoclOrder(layout);
  const char op_a = ToAoclTranspose(transa);
  const char op_b = ToAoclTranspose(transb);
  (void)offsetc;
  aocl_gemm_s8s8s32os32(order, op_a, op_b, static_cast<dim_t>(m), static_cast<dim_t>(n), static_cast<dim_t>(k),
                        static_cast<int32_t>(alpha), static_cast<const int8_t*>(a), static_cast<dim_t>(lda), 'n',
                        static_cast<const int8_t*>(b), static_cast<dim_t>(ldb), 'r', static_cast<int32_t>(beta), c,
                        static_cast<dim_t>(ldc), nullptr);
}

void prefill_int4_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                     const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
                                     const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                                     const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb,
                                     const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                                     const int32_t* oc) {
  throw std::runtime_error("int4 not support prefill");
}

void decode_int4_cblas_gemm_s8s8s32(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                                    const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc,
                                    const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                                    const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb,
                                    const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                                    const int32_t* oc) {
  throw std::runtime_error("int4 not support decode");
}

void reorder_B_gemm(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transb, const size_t k,
                    const size_t n, const size_t ldb, const void* b, void* b_reordered) {
  const char order = ToAoclOrder(layout);
  const char op_b = ToAoclTranspose(transb);
  aocl_reorder_s8s8s32os32(order, op_b, 'B', static_cast<const int8_t*>(b), static_cast<int8_t*>(b_reordered), k, n,
                           ldb);
}

size_t get_reorder_B_size(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transb, const size_t k,
                          const size_t n) {
  return aocl_get_reorder_buf_size_s8s8s32os32(ToAoclOrder(layout), ToAoclTranspose(transb), 'B', k, n);
}

#ifdef __cplusplus
}
#endif