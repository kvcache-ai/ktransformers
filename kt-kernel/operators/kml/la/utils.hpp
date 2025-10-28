#pragma once
// #include <arm_sve.h>
#include <cstdint>
#include <cstring>

// static inline void sve_32xbf16_to_32xfp32(const bfloat16_t *src, float *dst0, float *dst1) {
// #ifdef __ARM_FEATURE_SVE
//   // 全真谓词，对应每个 16‑bit 元素
//   svbool_t pg_h = svptrue_b16();
//   // 每次循环处理 svcnth(pg_h) 个 BF16 元素，svcnth(pg_h)==VL/16
//   size_t offset = 0;
//   // 我们要生产两段 FP32 输出，每段长度 svcntw(pg_h)==VL/32
//   // SVE 向量寄存宽度 VL 可以任意 128–2048，但代码与之无关

//   // Load first half BF16→FP32
//   svbfloat16_t vb0 = svld1(pg_h, &src[offset]);                       // load BF16
//   svfloat32_t vf0 = svcvt_f32_bf16_z(pg_h, vb0);                      // widen→FP32
//   svst1(pg_h, &dst0[offset/2], vf0);                                  // store

//   offset += svcnth(pg_h);  // 移到第二批 BF16 元素

//   // Load second half BF16→FP32
//   svbfloat16_t vb1 = svld1(pg_h, &src[offset]);
//   svfloat32_t vf1 = svcvt_f32_bf16_z(pg_h, vb1);
//   svst1(pg_h, &dst1[offset/2], vf1);
// #else
// // fallback: scalar or NEON
// #endif
// }

// 简单截断模式：直接丢弃低 16 位
static inline uint16_t float_to_bf16_trunc(float f) {
  uint32_t u;
  // 按位拷贝，避免 strict‑aliasing UB
  memcpy(&u, &f, sizeof(u));   // :contentReference[oaicite:3]{index=3}
  return (uint16_t)(u >> 16);  // 截断得到高 16 位 :contentReference[oaicite:4]{index=4}
}

static inline void convert_32fp32_to_32bf16_pure_c(const float* src, uint16_t* dst) {
  // src 已偏移至 token_nth * hidden_size
  for (int e = 0; e < 32; e++) {  // 共 32 个元素
    // 选择截断或四舍五入
    dst[e] = float_to_bf16_trunc(src[e]);
  }
}

// 把 32 个 bf16 元素转换成 32 个 fp32 元素

static inline void convert_32bf16_to_32fp32_pure_c(const uint16_t* src, float* dst) {
  for (int e = 0; e < 32; e++) {
    uint32_t temp = ((uint32_t)src[e]) << 16;  // 将 BF16 左移 16 位
    memcpy(&dst[e], &temp, sizeof(float));     // 将结果复制到 FP32 变量中
  }
}

// template <typename T> T *offset_pointer(T *ptr, std::size_t byte_offset) {
//   return reinterpret_cast<T *>(reinterpret_cast<char *>(ptr) + byte_offset);
// }

/*** gemm helper ***/
#include <kblas.h>

#include <iostream>
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