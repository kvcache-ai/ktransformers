#ifndef __GEMM_INTFOUR_KERNELS_H__
#define __GEMM_INTFOUR_KERNELS_H__

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef int8_t BLASINT8;
typedef uint8_t BLASUINT8;

typedef struct {
  void (*small_kernel)(const size_t, const size_t, const size_t, const float, const void*, const size_t, const BLASINT8,
                       const void*, const size_t, const BLASINT8, const float, int32_t*, const size_t, const int32_t*);
  void (**gemm_kernels)(const void*, const void*, int32_t*, size_t, int64_t, int64_t);
  void (*pack_a_fun)(void*, const void*, size_t, size_t, size_t, const BLASINT8);
  void (*pack_b_fun)(void*, const void*, size_t, size_t, size_t, const BLASINT8);
  void (*beta_func)(int32_t*, const int32_t*, float, size_t, size_t, size_t);
  void (*post_ops_func)(float, const int32_t*, int32_t*, size_t, size_t, size_t);
  size_t (*a_indexing)(size_t, size_t, size_t);
  size_t (*b_indexing)(size_t, size_t, size_t);
  size_t (*move_oc)(size_t, size_t);

} int_gemm_funcs;

#ifndef COMP_SV_LEN
#error "COMP_SV_LEN is not defined"
#endif

#define KERNEL_M_STEP 4
#define KERNEL_N_STEP 4
#define KERNEL_K_STEP COMP_SV_LEN

#define M_BLOCK 256
#if ((M_BLOCK % KERNEL_M_STEP) != 0)
#error "M_BLOCK % KERNEL_M_STEP != 0"
#endif
#define N_BLOCK 256
#if ((N_BLOCK % KERNEL_N_STEP) != 0)
#error "N_BLOCK % KERNEL_N_STEP != 0"
#endif
#define K_BLOCK 512
#if ((K_BLOCK % KERNEL_K_STEP) != 0)
#error "K_BLOCK % KERNEL_K_STEP != 0"
#endif

#define ALIGNMENT 4096

#define EXTERNAL_API __attribute__((visibility("default")))
#define UNUSED(arg) ((void)(arg))

// general pipeline
void gemm_impl_8bit(int_gemm_funcs* arg, size_t m, size_t n, size_t k, float alpha, const void* a, size_t lda,
                    const BLASINT8 oa, const void* b, size_t ldb, const BLASINT8 ob, float beta, int32_t* c, size_t ldc,
                    const int32_t* oc, size_t small_switch);
// s8 kernel
void pack_b_s8_n(void* bufferB, const void* curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb,
                 const BLASINT8 ob);

void pack_a_s8_n(void* bufferA, const void* curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda,
                 const BLASINT8 oa);

void pack_b_s8_t(void* bufferB, const void* curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb,
                 const BLASINT8 ob);

void pack_a_s8_t(void* bufferA, const void* curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda,
                 const BLASINT8 oa);

// u8 kernels
void pack_b_u8_n(void* bufferB, const void* curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb,
                 const BLASINT8 ob);

void pack_a_u8_n(void* bufferA, const void* curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda,
                 const BLASINT8 oa);

void pack_b_u8_t(void* bufferB, const void* curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb,
                 const BLASINT8 ob);

void pack_a_u8_t(void* bufferA, const void* curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda,
                 const BLASINT8 oa);

// beta kernels
void beta_cf_s8(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc);
void beta_cc_s8(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc);
void beta_cr_s8(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc);
void beta_rf_s8(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc);
void beta_rc_s8(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc);
void beta_rr_s8(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc);

void beta_cf_s8_opt(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size,
                    size_t ldc);
void beta_cc_s8_opt(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size,
                    size_t ldc);
void beta_cr_s8_opt(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size,
                    size_t ldc);
void beta_rf_s8_opt(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size,
                    size_t ldc);
void beta_rc_s8_opt(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size,
                    size_t ldc);
void beta_rr_s8_opt(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size,
                    size_t ldc);

// post-ops kernels
void post_ops(float alpha, const int32_t* bufferC, int32_t* current_c_ptr, size_t m, size_t n_block, size_t ldc);
void post_ops_opt(float alpha, const int32_t* bufferC, int32_t* current_c_ptr, size_t m, size_t n_block, size_t ldc);

// drivers
void gemm_driver(int_gemm_funcs* arg, size_t m, size_t n, size_t k, float alpha, const void* a, size_t lda,
                 const BLASINT8 oa, const void* b, size_t ldb, const BLASINT8 ob, float beta, int32_t* c, size_t ldc,
                 const int32_t* oc);
void gemm_driver_opt(int_gemm_funcs* arg, size_t m, size_t n, size_t k, float alpha, const void* a, size_t lda,
                     const BLASINT8 oa, const void* b, size_t ldb, const BLASINT8 ob, float beta, int32_t* c,
                     size_t ldc, const int32_t* oc);

// matrix multiplication kernels

// s8s8s32 kernels
void gemm_kernel_s8s8s32_4x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_4x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_4x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_4x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_s8s8s32_3x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_3x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_3x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_3x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_s8s8s32_2x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_2x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_2x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_2x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_s8s8s32_1x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_1x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_1x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8s8s32_1x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

// u8u8s32 kernels

void gemm_kernel_u8u8s32_4x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_4x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_4x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_4x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_u8u8s32_3x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_3x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_3x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_3x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_u8u8s32_2x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_2x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_2x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_2x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_u8u8s32_1x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_1x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_1x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8u8s32_1x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

// s8u8s32 kernels
void gemm_kernel_s8u8s32_4x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_4x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_4x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_4x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_s8u8s32_3x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_3x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_3x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_3x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_s8u8s32_2x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_2x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_2x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_2x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_s8u8s32_1x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_1x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_1x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_s8u8s32_1x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
// u8s8s32 kernels
void gemm_kernel_u8s8s32_4x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_4x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_4x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_4x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_u8s8s32_3x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_3x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_3x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_3x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_u8s8s32_2x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_2x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_2x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_2x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

void gemm_kernel_u8s8s32_1x4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_1x3(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_1x2(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);
void gemm_kernel_u8s8s32_1x1(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len);

// small kernels
// s8s8s32 kernels
void small_kernel_s8s8s32_nn_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_nt_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_tn_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_tt_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_nn_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_nt_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_tn_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_tt_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_nn_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_nt_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_tn_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8s8s32_tt_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

// s8u8s32 kernels

void small_kernel_s8u8s32_nn_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_nt_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_tn_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_tt_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_nn_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_nt_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_tn_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_tt_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_nn_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_nt_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_tn_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_s8u8s32_tt_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

// u8s8s32 kernels
void small_kernel_u8s8s32_nn_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_nt_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_tn_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_tt_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_nn_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_nt_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_tn_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_tt_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_nn_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_nt_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_tn_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8s8s32_tt_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

// u8u8s32 kernels
void small_kernel_u8u8s32_nn_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_nt_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_tn_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_tt_f(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_nn_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_nt_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_tn_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_tt_c(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_nn_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_nt_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_tn_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void small_kernel_u8u8s32_tt_r(const size_t m, const size_t n, const size_t k, const float alpha, const void* a,
                               const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb, const BLASINT8 ob,
                               const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

#ifdef __cplusplus
}
#endif
#endif
