#include "batch_gemm_api.hpp"
#ifdef __cplusplus
extern "C" {
#endif
void decode_cblas_gemm_s8s8s32(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                               const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k,
                               const float alpha, const void* a, const size_t lda, const BLASINT8 oa, const void* b,
                               const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                               const int32_t* oc) {
  BLASINT8* ptrA = (BLASINT8*)a;
  BLASINT8* ptrB = (BLASINT8*)b;
  int32_t* ptrC = c;
  size_t split_n = n / N_SIZE;

  for (size_t n_block = 0; n_block < split_n; n_block++) {
    BLASINT8* cur_ptrA = ptrA;
    BLASINT8* cur_ptrB = ptrB + n_block * (N_SIZE * k);
    int32_t* cur_ptrC = ptrC + n_block * N_SIZE;
    gemm_kernel_1x8(cur_ptrA, cur_ptrB, cur_ptrC, ldc, k, COMP_SV_LEN);
  }
}

void decode_int4_cblas_gemm_s8s8s32(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa,
                                    const CBLAS_TRANSPOSE transb, const CBLAS_OFFSET offsetc, const size_t m,
                                    const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                                    const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                                    const float beta, int32_t* c, const size_t ldc, const int32_t* oc) {
  BLASINT8* ptrA = (BLASINT8*)a;
  BLASINT8* ptrB = (BLASINT8*)b;
  int32_t* ptrC = c;
  size_t split_n = n / N_SIZE;

  for (size_t n_block = 0; n_block < split_n; n_block++) {
    BLASINT8* cur_ptrA = ptrA;
    BLASINT8* cur_ptrB = ptrB + n_block * (N_SIZE * (k / 2));
    int32_t* cur_ptrC = ptrC + n_block * N_SIZE;
    gemm_kernel_1x8_int4(cur_ptrA, cur_ptrB, cur_ptrC, (ldc / 2), (k / 2), COMP_SV_LEN);
  }
}

#ifdef __cplusplus
}
#endif