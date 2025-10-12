#include <stdint.h>
#include "integer_gemm_kernels.h"
#include "helping_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ADD_SUFFIX(name) ADD_M_N_SIZES_MACRO(ADD_TYPES_MACRO(name, LHS_TYPE, RHS_TYPE), M_SIZE, N_SIZE)

#define LD1B_PTR(reg_name, p, ptr, idx) "ld1b {" #reg_name ".b}, " #p "/z, [%[" #ptr "], #" #idx ", MUL VL]\n"
#define COMPUTE_ADDP(out, in1, in2) "addp " #out ".s, " #in1 ".s, " #in2 ".s\n"
#if (defined(LHS_INT) && defined(RHS_INT)) || (defined(LHS_UINT) && defined(RHS_UINT))
#define COMPUTE_DOT_TYPED(out, in1, in2, lhs_type, rhs_type) #lhs_type "dot " #out ".s, " #in1 ".b, " #in2 ".b\n"
#else
  #ifdef LHS_INT
  #define COMPUTE_DOT_TYPED(out, in1, in2, lhs_type, rhs_type) "usdot " #out ".s, " #in2 ".b, " #in1 ".b\n"
  #else
  #define COMPUTE_DOT_TYPED(out, in1, in2, lhs_type, rhs_type) "usdot " #out ".s, " #in1 ".b, " #in2 ".b\n"
  #endif
#endif

#define COMPUTE_DOT_TYPED_MACRO(out, in1, in2, LHS_TYPE, RHS_TYPE) COMPUTE_DOT_TYPED(out, in1, in2, LHS_TYPE, RHS_TYPE)
#define COMPUTE_DOT(out, in1, in2) COMPUTE_DOT_TYPED_MACRO(out, in1, in2, LHS_TYPE, RHS_TYPE)

#if (N_SIZE > 4)
#error "N_SIZE can't be greater than 4"
#endif

#if (M_SIZE > 4)
#error "M_SIZE can't be greater than 4"
#endif

#define LOAD_Z0(p, ptr) LD1B_PTR(z0, p, ptr, 0)
#define LOAD_Z8(p, ptr) LD1B_PTR(z8, p, ptr, 0)

#if (N_SIZE > 1)
  #define LOAD_Z1(p, ptr) LD1B_PTR(z1, p, ptr, 1)
  #define LOAD_Z9(p, ptr) LD1B_PTR(z9, p, ptr, 1)
#else
  #define LOAD_Z1(p, ptr)
  #define LOAD_Z9(p, ptr)
#endif

#if (N_SIZE > 2)
  #define LOAD_Z2(p, ptr) LD1B_PTR(z2, p, ptr, 2)
  #define LOAD_Z10(p, ptr) LD1B_PTR(z10, p, ptr, 2)
#else
#define LOAD_Z2(p, ptr)
#define LOAD_Z10(p, ptr)
#endif

#if (N_SIZE > 3)
  #define LOAD_Z3(p, ptr) LD1B_PTR(z3, p, ptr, 3)
  #define LOAD_Z11(p, ptr) LD1B_PTR(z11, p, ptr, 3)
#else
  #define LOAD_Z3(p, ptr)
  #define LOAD_Z11(p, ptr)
#endif

#define LOAD_Z4(p, ptr) LD1B_PTR(z4, p, ptr, 0)
#define LOAD_Z12(p, ptr) LD1B_PTR(z12, p, ptr, 0)

#if (M_SIZE > 1)
  #define LOAD_Z5(p, ptr) LD1B_PTR(z5, p, ptr, 1)
  #define LOAD_Z13(p, ptr) LD1B_PTR(z13, p, ptr, 1)
#else
  #define LOAD_Z5(p, ptr)
  #define LOAD_Z13(p, ptr)
#endif

#if (M_SIZE > 2)
  #define LOAD_Z6(p, ptr) LD1B_PTR(z6, p, ptr, 2)
  #define LOAD_Z14(p, ptr) LD1B_PTR(z14, p, ptr, 2)
#else
  #define LOAD_Z6(p, ptr)
  #define LOAD_Z14(p, ptr)
#endif

#if (M_SIZE > 3)
  #define LOAD_Z7(p, ptr) LD1B_PTR(z7, p, ptr, 3)
  #define LOAD_Z15(p, ptr) LD1B_PTR(z15, p, ptr, 3)
#else
  #define LOAD_Z7(p, ptr)
  #define LOAD_Z15(p, ptr)
#endif

// macros for dot multiplication
#define ACCUMULATE_Z16(lhs, rhs) COMPUTE_DOT(z16, lhs, rhs)

#if (N_SIZE > 1)
  #define ACCUMULATE_Z17(lhs, rhs) COMPUTE_DOT(z17, lhs, rhs)
#else
  #define ACCUMULATE_Z17(lhs, rhs)
#endif

#if (N_SIZE > 2)
  #define ACCUMULATE_Z18(lhs, rhs) COMPUTE_DOT(z18, lhs, rhs)
#else
  #define ACCUMULATE_Z18(lhs, rhs)
#endif

#if (N_SIZE > 3)
  #define ACCUMULATE_Z19(lhs, rhs) COMPUTE_DOT(z19, lhs, rhs)
#else
  #define ACCUMULATE_Z19(lhs, rhs)
#endif

#if (M_SIZE > 1)
  #define ACCUMULATE_Z20(lhs, rhs) COMPUTE_DOT(z20, lhs, rhs)
  #if (N_SIZE > 1)
    #define ACCUMULATE_Z21(lhs, rhs) COMPUTE_DOT(z21, lhs, rhs)
  #else
    #define ACCUMULATE_Z21(lhs, rhs)
  #endif

  #if (N_SIZE > 2)
    #define ACCUMULATE_Z22(lhs, rhs) COMPUTE_DOT(z22, lhs, rhs)
  #else
    #define ACCUMULATE_Z22(lhs, rhs)
  #endif

  #if (N_SIZE > 3)
    #define ACCUMULATE_Z23(lhs, rhs) COMPUTE_DOT(z23, lhs, rhs)
  #else
    #define ACCUMULATE_Z23(lhs, rhs)
  #endif

#else
  #define ACCUMULATE_Z20(lhs, rhs)
  #define ACCUMULATE_Z21(lhs, rhs)
  #define ACCUMULATE_Z22(lhs, rhs)
  #define ACCUMULATE_Z23(lhs, rhs)
#endif

#if (M_SIZE > 2)
  #define ACCUMULATE_Z24(lhs, rhs) COMPUTE_DOT(z24, lhs, rhs)

  #if (N_SIZE > 1)
    #define ACCUMULATE_Z25(lhs, rhs) COMPUTE_DOT(z25, lhs, rhs)
  #else
    #define ACCUMULATE_Z25(lhs, rhs)
  #endif

  #if (N_SIZE > 2)
    #define ACCUMULATE_Z26(lhs, rhs) COMPUTE_DOT(z26, lhs, rhs)
  #else
    #define ACCUMULATE_Z26(lhs, rhs)
  #endif

  #if (N_SIZE > 3)
    #define ACCUMULATE_Z27(lhs, rhs) COMPUTE_DOT(z27, lhs, rhs)
  #else
    #define ACCUMULATE_Z27(lhs, rhs)
  #endif

#else
  #define ACCUMULATE_Z24(lhs, rhs)
  #define ACCUMULATE_Z25(lhs, rhs)
  #define ACCUMULATE_Z26(lhs, rhs)
  #define ACCUMULATE_Z27(lhs, rhs)
#endif

#if (M_SIZE > 3)
  #define ACCUMULATE_Z28(lhs, rhs) COMPUTE_DOT(z28, lhs, rhs)

  #if (N_SIZE > 1)
    #define ACCUMULATE_Z29(lhs, rhs) COMPUTE_DOT(z29, lhs, rhs)
  #else
    #define ACCUMULATE_Z29(lhs, rhs)
  #endif

  #if (N_SIZE > 2)
    #define ACCUMULATE_Z30(lhs, rhs) COMPUTE_DOT(z30, lhs, rhs)
  #else
    #define ACCUMULATE_Z30(lhs, rhs)
  #endif

  #if (N_SIZE > 3)
    #define ACCUMULATE_Z31(lhs, rhs) COMPUTE_DOT(z31, lhs, rhs)
  #else
    #define ACCUMULATE_Z31(lhs, rhs)
  #endif

#else
  #define ACCUMULATE_Z28(lhs, rhs)
  #define ACCUMULATE_Z29(lhs, rhs)
  #define ACCUMULATE_Z30(lhs, rhs)
  #define ACCUMULATE_Z31(lhs, rhs)
#endif

#define MOVE_LHS_PTR(ptr) "add %[" #ptr "], %[" #ptr "], %[move_lhs]\n"
#define MOVE_RHS_PTR(ptr) "add %[" #ptr "], %[" #ptr "], %[move_rhs]\n"

#define PROCESS_ACCUM(reg_idx, z_reg_idx, tmp_reg, dst, p)                     \
  "ldr w" #reg_idx ", [%[" #dst "]]\n"                                         \
  "saddv d" #reg_idx ", " #p ", z" #z_reg_idx ".s\n"                           \
  "fmov " #tmp_reg ", d" #reg_idx "\n"                                         \
  "add x" #reg_idx ", x" #reg_idx ", " #tmp_reg "\n"                           \
  "str w" #reg_idx ", [%[" #dst "]], #4\n"

// function logic
void ADD_SUFFIX(gemm_kernel)(const void *lhs_ptr, const void *rhs_ptr,
                             int32_t *accum_ptr, size_t ldc, int64_t k_depth,
                             int64_t sv_len) {
  int64_t run_k_depth = k_depth;
  int64_t run_sv_len = sv_len;
  int64_t run_2sv_len = 2 * sv_len;
  int64_t move_lhs = M_SIZE * sv_len;
  int64_t move_rhs = N_SIZE * sv_len;
  int32_t* dst_ptr = accum_ptr;
  ldc -= M_SIZE;
  ldc *= 4;
  asm volatile(
    // predicate for operating on lhs and rhs is always true
    "ptrue p0.b, all\n"
    // Clear accumulators
    LOAD_Z0(p0, rhs_ptr)
    "dup z16.s, #0\n"
    LOAD_Z1(p0, rhs_ptr)
    "dup z17.s, #0\n"
    LOAD_Z4(p0, lhs_ptr)
    "dup z18.s, #0\n"
    LOAD_Z5(p0, lhs_ptr)
    "dup z19.s, #0\n"
    LOAD_Z6(p0, lhs_ptr)
    "dup z20.s, #0\n"
    LOAD_Z7(p0, lhs_ptr)
    "dup z21.s, #0\n"
    LOAD_Z2(p0, rhs_ptr)
    "dup z22.s, #0\n"
    LOAD_Z3(p0, rhs_ptr)
    "dup z23.s, #0\n"
    "subs %[run_k_depth], %[run_k_depth], %[run_sv_len]\n"
    "dup z24.s, #0\n"
    "mov x16, %[dst_ptr]\n"
    "dup z25.s, #0\n"
    "dup z26.s, #0\n"
    "dup z27.s, #0\n"
    MOVE_LHS_PTR(lhs_ptr)
    "dup z28.s, #0\n"
    MOVE_RHS_PTR(rhs_ptr)
    "dup z29.s, #0\n"
    "dup z30.s, #0\n"
    "dup z31.s, #0\n"

    "ble 1f\n"

    "cmp %[run_k_depth], %[run_2sv_len]\n"
    "blt 2f\n"

    "3:\n"
        LOAD_Z12(p0, lhs_ptr)
        ACCUMULATE_Z16(z4,z0)
        ACCUMULATE_Z17(z4,z1)
        LOAD_Z13(p0, lhs_ptr)
        ACCUMULATE_Z18(z4,z2)
        ACCUMULATE_Z19(z4,z3)
        LOAD_Z8(p0, rhs_ptr)
        ACCUMULATE_Z20(z5,z0)
        ACCUMULATE_Z21(z5,z1)
        LOAD_Z9(p0, rhs_ptr)
        ACCUMULATE_Z22(z5,z2)
        ACCUMULATE_Z23(z5,z3)
        LOAD_Z10(p0,rhs_ptr)
        ACCUMULATE_Z24(z6,z0)
        ACCUMULATE_Z25(z6,z1)
        LOAD_Z11(p0,rhs_ptr)
        ACCUMULATE_Z26(z6,z2)
        MOVE_RHS_PTR(rhs_ptr)
        "prfw pldl1keep, p0, [%[rhs_ptr], #4, MUL VL]\n"
        ACCUMULATE_Z27(z6,z3)
        LOAD_Z14(p0, lhs_ptr)
        ACCUMULATE_Z28(z7,z0)
        ACCUMULATE_Z29(z7,z1)
        LOAD_Z15(p0,lhs_ptr)
        ACCUMULATE_Z30(z7,z2)
        MOVE_LHS_PTR(lhs_ptr)
        "prfw pldl1keep, p0, [%[lhs_ptr], #4, MUL VL]\n"
        ACCUMULATE_Z31(z7,z3)

        LOAD_Z4(p0, lhs_ptr)
        ACCUMULATE_Z16(z12,z8)
        ACCUMULATE_Z17(z12,z9)
        LOAD_Z5(p0, lhs_ptr)
        ACCUMULATE_Z18(z12,z10)
        ACCUMULATE_Z19(z12,z11)
        LOAD_Z6(p0, lhs_ptr)
        ACCUMULATE_Z20(z13,z8)
        ACCUMULATE_Z21(z13,z9)
        LOAD_Z0(p0, rhs_ptr)
        "sub %[run_k_depth], %[run_k_depth], %[run_2sv_len]\n"
        ACCUMULATE_Z22(z13,z10)
        ACCUMULATE_Z23(z13,z11)
        LOAD_Z1(p0, rhs_ptr)
        ACCUMULATE_Z24(z14,z8)
        ACCUMULATE_Z25(z14,z9)
        LOAD_Z2(p0,rhs_ptr)
        ACCUMULATE_Z26(z14,z10)
        ACCUMULATE_Z27(z14,z11)
        LOAD_Z3(p0, rhs_ptr)
        ACCUMULATE_Z28(z15,z8)
        MOVE_RHS_PTR(rhs_ptr)
        "prfw pldl1keep, p0, [%[rhs_ptr], #4, MUL VL]\n"
        ACCUMULATE_Z29(z15,z9)
        LOAD_Z7(p0, lhs_ptr)
        "cmp %[run_k_depth], %[run_2sv_len]\n"
        ACCUMULATE_Z30(z15, z10)
        MOVE_LHS_PTR(lhs_ptr)
        "prfw pldl1keep, p0, [%[lhs_ptr], #4, MUL VL]\n"
        ACCUMULATE_Z31(z15,z11)
    "bge 3b\n"

    "cmp %[run_k_depth], #0\n"
    "ble 1f\n"

    "2:\n"
        "subs %[run_k_depth], %[run_k_depth], %[run_sv_len]\n"
        ACCUMULATE_Z16(z4,z0)
        ACCUMULATE_Z17(z4,z1)
        ACCUMULATE_Z18(z4,z2)
        ACCUMULATE_Z19(z4,z3)
        LOAD_Z4(p0,lhs_ptr)
        ACCUMULATE_Z20(z5,z0)
        ACCUMULATE_Z21(z5,z1)
        ACCUMULATE_Z22(z5,z2)
        ACCUMULATE_Z23(z5,z3)
        LOAD_Z5(p0,lhs_ptr)
        ACCUMULATE_Z24(z6,z0)
        ACCUMULATE_Z25(z6,z1)
        ACCUMULATE_Z26(z6,z2)
        ACCUMULATE_Z27(z6,z3)
        LOAD_Z6(p0,lhs_ptr)
        ACCUMULATE_Z28(z7,z0)
        LOAD_Z0(p0,rhs_ptr)
        ACCUMULATE_Z29(z7,z1)
        LOAD_Z1(p0,rhs_ptr)
        ACCUMULATE_Z30(z7,z2)
        LOAD_Z2(p0,rhs_ptr)
        ACCUMULATE_Z31(z7,z3)
        LOAD_Z3(p0,rhs_ptr)
        MOVE_RHS_PTR(rhs_ptr)
        LOAD_Z7(p0,lhs_ptr)
        MOVE_LHS_PTR(lhs_ptr)
    "bgt 2b\n"

    "1:\n"
    ACCUMULATE_Z16(z4,z0)
    ACCUMULATE_Z17(z4,z1)
    ACCUMULATE_Z18(z4,z2)
    ACCUMULATE_Z19(z4,z3)
    ACCUMULATE_Z20(z5,z0)
    ACCUMULATE_Z21(z5,z1)
    ACCUMULATE_Z22(z5,z2)
    ACCUMULATE_Z23(z5,z3)
    ACCUMULATE_Z24(z6,z0)
    ACCUMULATE_Z25(z6,z1)
    ACCUMULATE_Z26(z6,z2)
    ACCUMULATE_Z27(z6,z3)
    ACCUMULATE_Z28(z7,z0)
    ACCUMULATE_Z29(z7,z1)
    ACCUMULATE_Z30(z7,z2)
    ACCUMULATE_Z31(z7,z3)

#if (N_SIZE > 0)
#if (M_SIZE > 0)
    PROCESS_ACCUM(0, 16, x16, dst_ptr, p0)
#endif
#if (M_SIZE > 1)
    PROCESS_ACCUM(4, 20, x17, dst_ptr, p0)
#endif
#if (M_SIZE > 2)
    PROCESS_ACCUM(8, 24, x18, dst_ptr, p0)
#endif
#if (M_SIZE > 3)
    PROCESS_ACCUM(12, 28, x17, dst_ptr, p0)
#endif
#endif
    "add %[dst_ptr], %[dst_ptr], %[ldc]\n"

#if (N_SIZE > 1)
#if (M_SIZE > 0)
    PROCESS_ACCUM(1, 17, x16, dst_ptr, p0)
#endif
#if (M_SIZE > 1)
    PROCESS_ACCUM(5, 21, x17,dst_ptr,p0)
#endif
#if (M_SIZE > 2)
    PROCESS_ACCUM(9, 25, x18, dst_ptr, p0)
#endif
#if (M_SIZE > 3)
    PROCESS_ACCUM(13, 29, x17, dst_ptr, p0)
#endif
#endif
    "add %[dst_ptr], %[dst_ptr], %[ldc]\n"


#if (N_SIZE > 2)
#if (M_SIZE > 0)
        PROCESS_ACCUM(2, 18, x16, dst_ptr, p0)
#endif
#if (M_SIZE > 1)
    PROCESS_ACCUM(6, 22, x17,dst_ptr,p0)
#endif
#if (M_SIZE > 2)
    PROCESS_ACCUM(10,26,x18,dst_ptr,p0)
#endif
#if (M_SIZE > 3)
    PROCESS_ACCUM(14,30,x17,dst_ptr,p0)
#endif
#endif
    "add %[dst_ptr], %[dst_ptr], %[ldc]\n"


#if (N_SIZE > 3)
#if (M_SIZE > 0)
        PROCESS_ACCUM(3, 19, x16, dst_ptr, p0)
#endif
#if (M_SIZE > 1)
    PROCESS_ACCUM(7, 23, x17,dst_ptr,p0)
#endif
#if (M_SIZE > 2)
    PROCESS_ACCUM(11,27,x18,dst_ptr,p0)
#endif
#if (M_SIZE > 3)
    PROCESS_ACCUM(15,31,x17,dst_ptr,p0)
#endif
#endif

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
