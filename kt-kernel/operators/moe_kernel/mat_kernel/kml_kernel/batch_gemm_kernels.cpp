#include "../batch_gemm_api.hpp"
#include "utils.hpp"
#ifdef __cplusplus
extern "C" {
#endif
void pack_b_1x8(void* bufferB, const void* cur_b_ptr, size_t n, size_t k, size_t ldb, const BLASINT8 ob) {
  BLASINT8* bufferB_typed = (BLASINT8*)bufferB;
  BLASINT8* cur_b_typed = (BLASINT8*)cur_b_ptr;

  size_t split_n = n / N_SIZE;
  size_t split_k = k / K_SIZE;

  // TODO::vectorization
  for (size_t np = 0; np < split_n; np++) {
    for (size_t n_idx = 0; n_idx < N_SIZE; n_idx++) {
      for (size_t kp = 0; kp < split_k; kp++) {
        for (size_t k_idx = 0; k_idx < K_SIZE; k_idx++) {
          bufferB_typed[np * (N_SIZE * k) + kp * (K_SIZE * N_SIZE) + n_idx * K_SIZE + k_idx] =
              cur_b_typed[INDEXING_B((kp * K_SIZE + k_idx), (np * N_SIZE + n_idx), ldb)] + ob;
        }
      }
    }
  }
}

void pack_b_1x8_int4(void* bufferB, const void* cur_b_ptr, size_t n, size_t k, size_t ldb, const BLASINT8 ob) {
  uint8_t* bufferB_typed = (uint8_t*)bufferB;
  uint8_t* cur_b_typed = (uint8_t*)cur_b_ptr;

#define RHS_MASK 0x0F
#define LHS_MASK 0xF0

  size_t split_n = n / N_SIZE;
  size_t split_k = k / K_SIZE;

  // TODO::vectorization
  for (size_t np = 0; np < split_n; np++) {
    for (size_t n_idx = 0; n_idx < N_SIZE; n_idx++) {
      for (size_t kp = 0; kp < split_k; kp++) {
        for (size_t k_idx = 0; k_idx < K_SIZE; k_idx += 2) {
          uint8_t b01 = cur_b_typed[INDEXING_B((kp * K_SIZE + k_idx / 2), (np * N_SIZE + n_idx), ldb)];
          uint8_t b23 = cur_b_typed[INDEXING_B((kp * K_SIZE + k_idx / 2 + K_SIZE / 2), (np * N_SIZE + n_idx), ldb)];
          uint8_t b02 = (b01 & LHS_MASK) | ((b23 & LHS_MASK) >> 4);
          uint8_t b13 = (b23 & RHS_MASK) | ((b01 & RHS_MASK) << 4);
          bufferB_typed[np * (N_SIZE * k) + kp * (K_SIZE * N_SIZE) + n_idx * K_SIZE + k_idx] = b02;
          bufferB_typed[np * (N_SIZE * k) + kp * (K_SIZE * N_SIZE) + n_idx * K_SIZE + k_idx + 1] = b13;
        }
      }
    }
  }
}

void gemm_kernel_1x8(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                     int64_t sv_len) {
  int64_t run_k_depth = k_depth;
  int64_t run_sv_len = sv_len;
  int64_t run_2sv_len = 2 * sv_len;
  int64_t move_lhs = sv_len;
  int64_t move_rhs = N_SIZE * sv_len;
  int32_t* dst_ptr = accum_ptr;
  ldc -= N_SIZE;
  ldc *= 4;

  asm volatile(

      "ptrue p0.b, all\n"
      "ld1b {z0.b}, p0/z, [%[rhs_ptr], #0, MUL VL]\n"
      "dup z16.s, #0\n"
      "ld1b {z1.b}, p0/z, [%[rhs_ptr], #1, MUL VL]\n"
      "dup z17.s, #0\n"
      "ld1b {z2.b}, p0/z, [%[rhs_ptr], #2, MUL VL]\n"
      "dup z18.s, #0\n"
      "ld1b {z3.b}, p0/z, [%[rhs_ptr], #3, MUL VL]\n"
      "dup z19.s, #0\n"
      "ld1b {z4.b}, p0/z, [%[rhs_ptr], #4, MUL VL]\n"
      "dup z20.s, #0\n"
      "ld1b {z5.b}, p0/z, [%[rhs_ptr], #5, MUL VL]\n"
      "dup z21.s, #0\n"
      "ld1b {z6.b}, p0/z, [%[rhs_ptr], #6, MUL VL]\n"
      "dup z22.s, #0\n"
      "ld1b {z7.b}, p0/z, [%[rhs_ptr], #7, MUL VL]\n"
      "dup z23.s, #0\n"
      "ld1b {z8.b}, p0/z, [%[lhs_ptr], #0, MUL VL]\n"
      "subs %[run_k_depth], %[run_k_depth], %[run_sv_len]\n"
      "add %[lhs_ptr], %[lhs_ptr], %[move_lhs]\n"
      "add %[rhs_ptr], %[rhs_ptr], %[move_rhs]\n"

      "ble 1f\n"

      "cmp %[run_k_depth], %[run_2sv_len]\n"
      "blt 2f\n"

      "3:\n"
      "ld1b {z9.b}, p0/z, [%[lhs_ptr], #0, MUL VL]\n"
      "add %[lhs_ptr], %[lhs_ptr], %[move_lhs]\n"
      "sdot z16.s, z8.b, z0.b\n"
      "ld1b {z0.b}, p0/z, [%[rhs_ptr], #0, MUL VL]\n"
      "sdot z17.s, z8.b, z1.b\n"
      "ld1b {z1.b}, p0/z, [%[rhs_ptr], #1, MUL VL]\n"
      "sdot z18.s, z8.b, z2.b\n"
      "ld1b {z2.b}, p0/z, [%[rhs_ptr], #2, MUL VL]\n"
      "sdot z19.s, z8.b, z3.b\n"
      "ld1b {z3.b}, p0/z, [%[rhs_ptr], #3, MUL VL]\n"
      "sdot z20.s, z8.b, z4.b\n"
      "ld1b {z4.b}, p0/z, [%[rhs_ptr], #4, MUL VL]\n"
      "sdot z21.s, z8.b, z5.b\n"
      "ld1b {z5.b}, p0/z, [%[rhs_ptr], #5, MUL VL]\n"
      "sdot z22.s, z8.b, z6.b\n"
      "ld1b {z6.b}, p0/z, [%[rhs_ptr], #6, MUL VL]\n"
      "sdot z23.s, z8.b, z7.b\n"
      "ld1b {z7.b}, p0/z, [%[rhs_ptr], #7, MUL VL]\n"
      "add %[rhs_ptr], %[rhs_ptr], %[move_rhs]\n"
      "sub %[run_k_depth], %[run_k_depth], %[run_2sv_len]\n"

      "ld1b {z8.b}, p0/z, [%[lhs_ptr], #0, MUL VL]\n"
      "add %[lhs_ptr], %[lhs_ptr], %[move_lhs]\n"
      "sdot z16.s, z9.b, z0.b\n"
      "ld1b {z0.b}, p0/z, [%[rhs_ptr], #0, MUL VL]\n"
      "sdot z17.s, z9.b, z1.b\n"
      "ld1b {z1.b}, p0/z, [%[rhs_ptr], #1, MUL VL]\n"
      "sdot z18.s, z9.b, z2.b\n"
      "ld1b {z2.b}, p0/z, [%[rhs_ptr], #2, MUL VL]\n"
      "sdot z19.s, z9.b, z3.b\n"
      "ld1b {z3.b}, p0/z, [%[rhs_ptr], #3, MUL VL]\n"
      "sdot z20.s, z9.b, z4.b\n"
      "ld1b {z4.b}, p0/z, [%[rhs_ptr], #4, MUL VL]\n"
      "sdot z21.s, z9.b, z5.b\n"
      "ld1b {z5.b}, p0/z, [%[rhs_ptr], #5, MUL VL]\n"
      "sdot z22.s, z9.b, z6.b\n"
      "ld1b {z6.b}, p0/z, [%[rhs_ptr], #6, MUL VL]\n"
      "sdot z23.s, z9.b, z7.b\n"
      "ld1b {z7.b}, p0/z, [%[rhs_ptr], #7, MUL VL]\n"
      "add %[rhs_ptr], %[rhs_ptr], %[move_rhs]\n"
      "cmp %[run_k_depth], %[run_2sv_len]\n"
      "bge 3b\n"

      "cmp %[run_k_depth], #0\n"
      "ble 1f\n"

      "2:\n"
      "subs %[run_k_depth], %[run_k_depth], %[run_sv_len]\n"
      "sdot z16.s, z8.b, z0.b\n"
      "ld1b {z0.b}, p0/z, [%[rhs_ptr], #0, MUL VL]\n"
      "sdot z17.s, z8.b, z1.b\n"
      "ld1b {z1.b}, p0/z, [%[rhs_ptr], #1, MUL VL]\n"
      "sdot z18.s, z8.b, z2.b\n"
      "ld1b {z2.b}, p0/z, [%[rhs_ptr], #2, MUL VL]\n"
      "sdot z19.s, z8.b, z3.b\n"
      "ld1b {z3.b}, p0/z, [%[rhs_ptr], #3, MUL VL]\n"
      "sdot z20.s, z8.b, z4.b\n"
      "ld1b {z4.b}, p0/z, [%[rhs_ptr], #4, MUL VL]\n"
      "sdot z21.s, z8.b, z5.b\n"
      "ld1b {z5.b}, p0/z, [%[rhs_ptr], #5, MUL VL]\n"
      "sdot z22.s, z8.b, z6.b\n"
      "ld1b {z6.b}, p0/z, [%[rhs_ptr], #6, MUL VL]\n"
      "sdot z23.s, z8.b, z7.b\n"
      "ld1b {z7.b}, p0/z, [%[rhs_ptr], #7, MUL VL]\n"
      "add %[rhs_ptr], %[rhs_ptr], %[move_rhs]\n"
      "ld1b {z8.b}, p0/z, [%[lhs_ptr], #0, MUL VL]\n"
      "add %[lhs_ptr], %[lhs_ptr], %[move_lhs]\n"
      "bgt 2b\n"

      "1:\n"
      "sdot z16.s, z8.b, z0.b\n"
      "sdot z17.s, z8.b, z1.b\n"
      "sdot z18.s, z8.b, z2.b\n"
      "sdot z19.s, z8.b, z3.b\n"
      "sdot z20.s, z8.b, z4.b\n"
      "sdot z21.s, z8.b, z5.b\n"
      "sdot z22.s, z8.b, z6.b\n"
      "sdot z23.s, z8.b, z7.b\n"

      PROCESS_ACCUM(0, 16, x16, dst_ptr, p0) PROCESS_ACCUM(1, 17, x17, dst_ptr, p0)
          PROCESS_ACCUM(2, 18, x18, dst_ptr, p0) PROCESS_ACCUM(3, 19, x19, dst_ptr, p0)
              PROCESS_ACCUM(4, 20, x16, dst_ptr, p0) PROCESS_ACCUM(5, 21, x17, dst_ptr, p0)
                  PROCESS_ACCUM(6, 22, x18, dst_ptr, p0) PROCESS_ACCUM(7, 23, x19, dst_ptr, p0)

      : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [run_k_depth] "+r"(run_k_depth), [dst_ptr] "+wr"(dst_ptr)
      : [run_sv_len] "r"(run_sv_len), [run_2sv_len] "r"(run_2sv_len), [move_lhs] "r"(move_lhs),
        [move_rhs] "r"(move_rhs), [ldc] "r"(ldc), [accum_ptr] "r"(accum_ptr)
      : "cc", "memory", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14",
        "w15", "x16", "x17", "x18", "x19", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11",
        "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",
        "z28", "z29", "z30", "z31");
}

// A:  z8  ~ z11
// B:  z0  ~ z7
// C:  z16 ~ z23
// M:  z12 z13
// MB: z14

void gemm_kernel_1x8_int4(const void* lhs_ptr, const void* rhs_ptr, int32_t* accum_ptr, size_t ldc, int64_t k_depth,
                          int64_t sv_len) {
  int64_t run_k_depth = k_depth;
  int64_t run_sv_len = sv_len;
  int64_t run_2sv_len = 2 * sv_len;
  int64_t move_lhs = 2 * sv_len;
  int64_t move_rhs = N_SIZE * sv_len;
  int32_t* dst_ptr = accum_ptr;
  ldc -= N_SIZE;
  ldc *= 4;

  asm volatile(
        "ptrue p0.b, all\n"
        "mov z12.b, #0xF0\n" //mask high
        "mov z13.b, #0x0F\n" //mask low
        "ld1b {z0.b}, p0/z, [%[rhs_ptr], #0, MUL VL]\n"
        "dup z16.s, #0\n"
        "ld1b {z1.b}, p0/z, [%[rhs_ptr], #1, MUL VL]\n"
        "dup z17.s, #0\n"
        "ld1b {z2.b}, p0/z, [%[rhs_ptr], #2, MUL VL]\n"
        "dup z18.s, #0\n"
        "ld1b {z3.b}, p0/z, [%[rhs_ptr], #3, MUL VL]\n"
        "dup z19.s, #0\n"
        "ld1b {z4.b}, p0/z, [%[rhs_ptr], #4, MUL VL]\n"
        "dup z20.s, #0\n"
        "ld1b {z5.b}, p0/z, [%[rhs_ptr], #5, MUL VL]\n"
        "dup z21.s, #0\n"
        "ld1b {z6.b}, p0/z, [%[rhs_ptr], #6, MUL VL]\n"
        "dup z22.s, #0\n"
        "ld1b {z7.b}, p0/z, [%[rhs_ptr], #7, MUL VL]\n"
        "dup z23.s, #0\n"

        "ld1b {z8.b}, p0/z, [%[lhs_ptr], #0, MUL VL]\n"
        "ld1b {z9.b}, p0/z, [%[lhs_ptr], #1, MUL VL]\n"
        "subs %[run_k_depth], %[run_k_depth], %[run_sv_len]\n"
        "add %[lhs_ptr], %[lhs_ptr], %[move_lhs]\n"
        "add %[rhs_ptr], %[rhs_ptr], %[move_rhs]\n"

        "ble 1f\n"

        "cmp %[run_k_depth], %[run_2sv_len]\n"
        "blt 2f\n"

        "3:\n"
            "ld1b {z10.b}, p0/z, [%[lhs_ptr], #0, MUL VL]\n"
            "ld1b {z11.b}, p0/z, [%[lhs_ptr], #1, MUL VL]\n"
            "add %[lhs_ptr], %[lhs_ptr], %[move_lhs]\n"
            INT4_CP_MASK_SHIFT_1x8(0, 14, 12, 13, 4)
            "sdot z16.s, z8.b, z0.b\n"
            "sdot z16.s, z9.b, z14.b\n"
            "ld1b {z0.b}, p0/z, [%[rhs_ptr], #0, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(1, 14, 12, 13, 4)
            "sdot z17.s, z8.b, z1.b\n"
            "sdot z17.s, z9.b, z14.b\n"
            "ld1b {z1.b}, p0/z, [%[rhs_ptr], #1, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(2, 14, 12, 13, 4)
            "sdot z18.s, z8.b, z2.b\n"
            "sdot z18.s, z9.b, z14.b\n"
            "ld1b {z2.b}, p0/z, [%[rhs_ptr], #2, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(3, 14, 12, 13, 4)
            "sdot z19.s, z8.b, z3.b\n"
            "sdot z19.s, z9.b, z14.b\n"
            "ld1b {z3.b}, p0/z, [%[rhs_ptr], #3, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(4, 14, 12, 13, 4)
            "sdot z20.s, z8.b, z4.b\n"
            "sdot z20.s, z9.b, z14.b\n"
            "ld1b {z4.b}, p0/z, [%[rhs_ptr], #4, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(5, 14, 12, 13, 4)
            "sdot z21.s, z8.b, z5.b\n"
            "sdot z21.s, z9.b, z14.b\n"
            "ld1b {z5.b}, p0/z, [%[rhs_ptr], #5, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(6, 14, 12, 13, 4)
            "sdot z22.s, z8.b, z6.b\n"
            "sdot z22.s, z9.b, z14.b\n"
            "ld1b {z6.b}, p0/z, [%[rhs_ptr], #6, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(7, 14, 12, 13, 4)
            "sdot z23.s, z8.b, z7.b\n"
            "sdot z23.s, z9.b, z14.b\n"
            "ld1b {z7.b}, p0/z, [%[rhs_ptr], #7, MUL VL]\n"
            "add %[rhs_ptr], %[rhs_ptr], %[move_rhs]\n"
            "sub %[run_k_depth], %[run_k_depth], %[run_2sv_len]\n"

            "ld1b {z8.b}, p0/z, [%[lhs_ptr], #0, MUL VL]\n"
            "ld1b {z9.b}, p0/z, [%[lhs_ptr], #1, MUL VL]\n"
            "add %[lhs_ptr], %[lhs_ptr], %[move_lhs]\n"
            INT4_CP_MASK_SHIFT_1x8(0, 14, 12, 13, 4)
            "sdot z16.s, z10.b, z0.b\n"
            "sdot z16.s, z11.b, z14.b\n"
            "ld1b {z0.b}, p0/z, [%[rhs_ptr], #0, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(1, 14, 12, 13, 4)
            "sdot z17.s, z10.b, z1.b\n"
            "sdot z17.s, z11.b, z14.b\n"
            "ld1b {z1.b}, p0/z, [%[rhs_ptr], #1, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(2, 14, 12, 13, 4)
            "sdot z18.s, z10.b, z2.b\n"
            "sdot z18.s, z11.b, z14.b\n"
            "ld1b {z2.b}, p0/z, [%[rhs_ptr], #2, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(3, 14, 12, 13, 4)
            "sdot z19.s, z10.b, z3.b\n"
            "sdot z19.s, z11.b, z14.b\n"
            "ld1b {z3.b}, p0/z, [%[rhs_ptr], #3, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(4, 14, 12, 13, 4)
            "sdot z20.s, z10.b, z4.b\n"
            "sdot z20.s, z11.b, z14.b\n"
            "ld1b {z4.b}, p0/z, [%[rhs_ptr], #4, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(5, 14, 12, 13, 4)
            "sdot z21.s, z10.b, z5.b\n"
            "sdot z21.s, z11.b, z14.b\n"
            "ld1b {z5.b}, p0/z, [%[rhs_ptr], #5, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(6, 14, 12, 13, 4)
            "sdot z22.s, z10.b, z6.b\n"
            "sdot z22.s, z11.b, z14.b\n"
            "ld1b {z6.b}, p0/z, [%[rhs_ptr], #6, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(7, 14, 12, 13, 4)
            "sdot z23.s, z10.b, z7.b\n"
            "sdot z23.s, z11.b, z14.b\n"
            "ld1b {z7.b}, p0/z, [%[rhs_ptr], #7, MUL VL]\n"
            "add %[rhs_ptr], %[rhs_ptr], %[move_rhs]\n"
            "cmp %[run_k_depth], %[run_2sv_len]\n"
        "bge 3b\n"

        "cmp %[run_k_depth], #0\n"
        "ble 1f\n"

        "2:\n"
            "subs %[run_k_depth], %[run_k_depth], %[run_sv_len]\n"
            INT4_CP_MASK_SHIFT_1x8(0, 14, 12, 13, 4)
            "sdot z16.s, z8.b, z0.b\n"
            "sdot z16.s, z9.b, z14.b\n"
            "ld1b {z0.b}, p0/z, [%[rhs_ptr], #0, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(1, 14, 12, 13, 4)
            "sdot z17.s, z8.b, z1.b\n"
            "sdot z17.s, z9.b, z14.b\n"
            "ld1b {z1.b}, p0/z, [%[rhs_ptr], #1, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(2, 14, 12, 13, 4)
            "sdot z18.s, z8.b, z2.b\n"
            "sdot z18.s, z9.b, z14.b\n"
            "ld1b {z2.b}, p0/z, [%[rhs_ptr], #2, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(3, 14, 12, 13, 4)
            "sdot z19.s, z8.b, z3.b\n"
            "sdot z19.s, z9.b, z14.b\n"
            "ld1b {z3.b}, p0/z, [%[rhs_ptr], #3, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(4, 14, 12, 13, 4)
            "sdot z20.s, z8.b, z4.b\n"
            "sdot z20.s, z9.b, z14.b\n"
            "ld1b {z4.b}, p0/z, [%[rhs_ptr], #4, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(5, 14, 12, 13, 4)
            "sdot z21.s, z8.b, z5.b\n"
            "sdot z21.s, z9.b, z14.b\n"
            "ld1b {z5.b}, p0/z, [%[rhs_ptr], #5, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(6, 14, 12, 13, 4)
            "sdot z22.s, z8.b, z6.b\n"
            "sdot z22.s, z9.b, z14.b\n"
            "ld1b {z6.b}, p0/z, [%[rhs_ptr], #6, MUL VL]\n"
            INT4_CP_MASK_SHIFT_1x8(7, 14, 12, 13, 4)
            "sdot z23.s, z8.b, z7.b\n"
            "sdot z23.s, z9.b, z14.b\n"
            "ld1b {z7.b}, p0/z, [%[rhs_ptr], #7, MUL VL]\n"
            "add %[rhs_ptr], %[rhs_ptr], %[move_rhs]\n"
            "ld1b {z8.b}, p0/z, [%[lhs_ptr], #0, MUL VL]\n"
            "ld1b {z9.b}, p0/z, [%[lhs_ptr], #1, MUL VL]\n"
            "add %[lhs_ptr], %[lhs_ptr], %[move_lhs]\n"
        "bgt 2b\n"

        "1:\n"
            INT4_CP_MASK_SHIFT_1x8(0, 14, 12, 13, 4)
            "sdot z16.s, z8.b, z0.b\n"
            "sdot z16.s, z9.b, z14.b\n"
            INT4_CP_MASK_SHIFT_1x8(1, 14, 12, 13, 4)
            "sdot z17.s, z8.b, z1.b\n"
            "sdot z17.s, z9.b, z14.b\n"
            INT4_CP_MASK_SHIFT_1x8(2, 14, 12, 13, 4)
            "sdot z18.s, z8.b, z2.b\n"
            "sdot z18.s, z9.b, z14.b\n"
            INT4_CP_MASK_SHIFT_1x8(3, 14, 12, 13, 4)
            "sdot z19.s, z8.b, z3.b\n"
            "sdot z19.s, z9.b, z14.b\n"
            INT4_CP_MASK_SHIFT_1x8(4, 14, 12, 13, 4)
            "sdot z20.s, z8.b, z4.b\n"
            "sdot z20.s, z9.b, z14.b\n"
            INT4_CP_MASK_SHIFT_1x8(5, 14, 12, 13, 4)
            "sdot z21.s, z8.b, z5.b\n"
            "sdot z21.s, z9.b, z14.b\n"
            INT4_CP_MASK_SHIFT_1x8(6, 14, 12, 13, 4)
            "sdot z22.s, z8.b, z6.b\n"
            "sdot z22.s, z9.b, z14.b\n"
            INT4_CP_MASK_SHIFT_1x8(7, 14, 12, 13, 4)
            "sdot z23.s, z8.b, z7.b\n"
            "sdot z23.s, z9.b, z14.b\n"

        PROCESS_ACCUM(0, 16, x16, dst_ptr, p0)
        PROCESS_ACCUM(1, 17, x17, dst_ptr, p0)
        PROCESS_ACCUM(2, 18, x18, dst_ptr, p0)
        PROCESS_ACCUM(3, 19, x19, dst_ptr, p0)
        PROCESS_ACCUM(4, 20, x16, dst_ptr, p0)
        PROCESS_ACCUM(5, 21, x17, dst_ptr, p0)
        PROCESS_ACCUM(6, 22, x18, dst_ptr, p0)
        PROCESS_ACCUM(7, 23, x19, dst_ptr, p0)

        :
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [run_k_depth] "+r"(run_k_depth),
        [dst_ptr] "+wr"(dst_ptr)
        :
        [run_sv_len] "r"(run_sv_len), [run_2sv_len] "r"(run_2sv_len),
        [move_lhs] "r"(move_lhs), [move_rhs] "r"(move_rhs), [ldc] "r"(ldc),
        [accum_ptr] "r"(accum_ptr)
        :
        "cc", "memory",
        "w0","w1","w2","w3","w4","w5","w6","w7",
        "w8","w9","w10","w11","w12","w13","w14","w15",    
        "x16","x17","x18","x19",
        "z0","z1","z2","z3","z4","z5","z6","z7",
        "z8","z9","z10","z11","z12","z13","z14","z15",
        "z16","z17","z18","z19","z20","z21","z22","z23",
        "z24","z25","z26","z27","z28","z29","z30","z31"
    );
}

#ifdef __cplusplus
}
#endif