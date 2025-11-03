
#pragma once
/*** gemm helper ***/
#include "../../api/common.h"
#ifdef __cplusplus
extern "C" {
#endif
#define COMP_SV_LEN 32
#define K_SIZE COMP_SV_LEN
#define M_SIZE 1
#define N_SIZE 8

#define INDEXING_B(row_idx, col_idx, ldb) ((col_idx) * (ldb) + row_idx)

#define PROCESS_ACCUM(reg_idx, z_reg_idx, tmp_reg, dst, p) \
  "mov w" #reg_idx                                         \
  ", #0\n"                                                 \
  "saddv d" #reg_idx ", " #p ", z" #z_reg_idx              \
  ".s\n"                                                   \
  "fmov " #tmp_reg ", d" #reg_idx                          \
  "\n"                                                     \
  "add x" #reg_idx ", x" #reg_idx ", " #tmp_reg            \
  "\n"                                                     \
  "str w" #reg_idx ", [%[" #dst "]], #4\n"

#define INT4_CP_MASK_SHIFT_1x8(src_reg, dst_reg, mask_reg1, mask_reg2, shift) \
  "movprfx z" #dst_reg ", z" #src_reg                                         \
  "\n"                                                                        \
  "lsl z" #dst_reg ".b, p0/m, z" #dst_reg ".b, #" #shift                      \
  "\n"                                                                        \
  "and z" #src_reg ".b, p0/m, z" #src_reg ".b, z" #mask_reg1 ".b\n"

void pack_b_1x8(void* bufferB, const void* cur_b_ptr, size_t n, size_t k, size_t ldb, const BLASINT8 ob);
void pack_b_1x8_int4(void* bufferB, const void* cur_b_ptr, size_t n, size_t k, size_t ldb, const BLASINT8 ob);

void gemm_kernel_1x8(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                     int64_t sv_len);

void gemm_kernel_1x8_int4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                          int64_t sv_len);

#ifdef __cplusplus
}
#endif
/*** gemm helper ***/