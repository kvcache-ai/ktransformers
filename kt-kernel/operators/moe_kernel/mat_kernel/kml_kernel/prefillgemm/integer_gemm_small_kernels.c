#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "integer_gemm_kernels.h"
#include "helping_macros.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if defined(OC_FIX)
    #define OC_TYPE f
    #define OC_IDX(mi, nn) 0
#elif defined(OC_COL)
    #define OC_TYPE c
    #define OC_IDX(mi, ni) mi
#else // OC_ROW
    #define OC_TYPE r
    #define OC_IDX(mi, ni) ni
#endif // OC_T

#if defined(TRANSA)
    #if defined TRANSB
        #define ADD_SUFFIX(name) ADD_TRANSP_MACRO(name, t, t, OC_TYPE)
    #elif defined(NOTRANSB)
        #define ADD_SUFFIX(name) ADD_TRANSP_MACRO(name, t, n, OC_TYPE)
    #else
        #error "Neither TRANSB or NOTRANSB is defined"
    #endif
#elif defined(NOTRANSA)
    #if defined TRANSB
        #define ADD_SUFFIX(name) ADD_TRANSP_MACRO(name, n, t, OC_TYPE)
    #elif defined(NOTRANSB)
        #define ADD_SUFFIX(name) ADD_TRANSP_MACRO(name, n, n, OC_TYPE)
    #else
        #error "Neither TRANSB or NOTRANSB is defined"
    #endif
#else
    #error "Neither TRANSA or NOTRANSA is defined"
#endif

#if (defined (LHS_INT) && defined(RHS_INT)) || (defined (LHS_UINT) && defined(RHS_UINT))
    #define COMPUTE_DOT_TYPED(out, in1, in2, lhs_type, rhs_type) #lhs_type "dot " #out ".s, " #in1 ".b, " #in2 ".b\n"
#else
    #if defined(LHS_INT)
        #define COMPUTE_DOT_TYPED(out, in1, in2, lhs_type, rhs_type) "usdot " #out ".s, " #in2 ".b," #in1 ".b\n"
    #else // LHS_UINT
        #define COMPUTE_DOT_TYPED(out, in1, in2, lhs_type, rhs_type) "usdot " #out ".s, " #in1 ".b," #in2 ".b\n"
    #endif // LHS_INT
#endif // LHS_INT

#define COMPUTE_DOT_TYPED_MACRO(out, in1, in2, LHS_TYPE, RHS_TYPE) COMPUTE_DOT_TYPED(out, in1, in2, LHS_TYPE, RHS_TYPE)
#define COMPUTE_DOT(out, in1, in2) COMPUTE_DOT_TYPED_MACRO(out, in1, in2, LHS_TYPE, RHS_TYPE)

static inline double compute_dot(size_t k, const void *a, const BLASINT8* oa, 
            const void *b, const BLASINT8* ob, int64_t sv_len) {
    int32_t accum = 0;
    int64_t run_k_depth = k;
    int64_t run_sv_len = sv_len;
    const void* lhs_ptr = a;
    const void* rhs_ptr = b;
    asm volatile(
        "dup z4.s, #0\n"
        "ptrue p0.b, all\n"
        "ld1b {z0.b}, p0/z, [%[oa]]\n"
        "ld1b {z1.b}, p0/z, [%[ob]]\n"
        "1:\n"
            "whilelt p1.b, xzr, %[run_k_depth]\n"
            "ld1b {z2.b}, p1/z, [%[lhs_ptr]]\n"
            "ld1b {z3.b}, p1/z, [%[rhs_ptr]]\n"
            "add z2.b, p1/m, z2.b, z0.b\n"
            "add z3.b, p1/m, z3.b, z1.b\n"
            "add %[lhs_ptr], %[lhs_ptr], %[run_sv_len]\n"
            "add %[rhs_ptr], %[rhs_ptr], %[run_sv_len]\n"
            COMPUTE_DOT(z4, z2, z3)
            "subs %[run_k_depth], %[run_k_depth], #1\n"
        "bgt 1b\n"
        "ptrue p2.s, all\n"
        "saddv d1, p2, z4.s\n"
        "fmov x2, d1\n"
        "add %[accum], %[accum], x2\n"
        : // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [run_k_depth] "+wr"(run_k_depth),
        [accum] "+wr"(accum)
            : // inputs
            [run_sv_len] "r"(run_sv_len), [oa] "r"(oa), [ob] "r"(ob)
            : // clobbers
            "cc", "memory",
            "d1", "x2",
            "z0", "z1", "z2", "z3", "z4", "p0", "p1", "p2"
        );
    return (double) accum;
}

#if !defined(TRANSA) || defined(TRANSB)
// performs transposition (n0 * n1) -> (n1 * n0) assuming col major
static inline void simplest_transpose(const void *in, void *out, size_t n0, size_t ld0, size_t n1) {
    // since we care only about size, we can use signed type always
    BLASINT8* typed_in = (BLASINT8*) in;
    BLASINT8* typed_out = (BLASINT8*) out;
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < n0; ++j) {
            typed_out[i + j * n1] = typed_in[j + i * ld0];
        }
    }
}
#endif // !defined(TRANSA) || defined(TRANSB)

// A in row-major, B in col-major
void ADD_SUFFIX(small_kernel)(const size_t m, const size_t n, const size_t k, const float alpha,
                             const void *a, const size_t lda, const BLASINT8 oa,
                             const void *b, const size_t ldb, const BLASINT8 ob,
                             const float beta, int32_t *c, const size_t ldc, const int32_t *oc) {
    double double_alpha = (double) alpha;
    // we use typed pointers only for indexing, so we don't care about signess
#ifdef TRANSA
    BLASINT8* a_typed = (BLASINT8*) a;
    const size_t used_lda = lda;
#else
    BLASINT8* a_typed = (BLASINT8*) aligned_alloc(128, sizeof(BLASINT8) * m * k);
    simplest_transpose(a, a_typed, m, lda, k);
    const size_t used_lda = k;
#endif // TRANSA
#ifndef TRANSB
    BLASINT8* b_typed = (BLASINT8*) b;
    const size_t used_ldb = ldb;
#else // TRANSB
    BLASINT8* b_typed = (BLASINT8*) aligned_alloc(128, sizeof(BLASINT8) * k * n);
    simplest_transpose(b, b_typed, n, ldb, k);
    const size_t used_ldb = k;
#endif // TRANSB
    BLASINT8 oa_buf[KERNEL_K_STEP];
    BLASINT8 ob_buf[KERNEL_K_STEP];
    for (size_t i = 0; i < KERNEL_K_STEP; ++i) {
        oa_buf[i] = oa;
        ob_buf[i] = ob;
    }
    // printf("\n========\n");
    for (size_t mi = 0; mi < m; ++mi) {
        for (size_t ni = 0; ni < n; ++ni) {
            // printf("mi = %lu, ni = %lu, oc_idx = %lu\n", mi, ni, OC_IDX(mi, ni));
            double tmp = compute_dot(k, a_typed + mi * used_lda, oa_buf, b_typed + ni * used_ldb, ob_buf, KERNEL_K_STEP);
            c[mi + ni * ldc] = round(tmp * double_alpha + ((double)(beta * ((float)c[mi + ni * ldc]) + oc[OC_IDX(mi, ni)])));
        }
    }
#ifdef TRANSA
    free(a_typed);
#endif // TRANSA
#ifdef TRANSB
    free(b_typed);
#endif // TRANSB
}

#ifdef __cplusplus
} // extern "C"
#endif /* __cplusplus */
