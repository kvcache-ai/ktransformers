#include <stdint.h>
#include <stdio.h>
//#include "cblas.h"
#include "integer_gemm_kernels.h"
#include "helping_macros.h"


/* matrix saved in rows or cols */
typedef enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

/* matrix transpose or conjugate transpose */
typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113, // conjugate transpose
    CblasConjNoTrans = 114
} CBLAS_TRANSPOSE;

typedef CBLAS_ORDER CBLAS_LAYOUT;

typedef enum CBLAS_OFFSET {
    CblasRowOffset = 171,
    CblasColOffset = 172,
    CblasFixOffset = 173
} CBLAS_OFFSET;



#ifdef __cplusplus
extern "C" {
    #endif /* __cplusplus */

#define ADD_KERNEL_SUFF(name, m_size, n_size) ADD_M_N_SIZES_MACRO(ADD_TYPES_MACRO(name, LHS_TYPE, RHS_TYPE), m_size, n_size)

static void (*gemm_kernels[])(const void*, const void*, int32_t*, size_t, int64_t, int64_t) = {
    ADD_KERNEL_SUFF(gemm_kernel, 1, 1), ADD_KERNEL_SUFF(gemm_kernel, 1, 2),ADD_KERNEL_SUFF(gemm_kernel, 1, 3), ADD_KERNEL_SUFF(gemm_kernel, 1, 4),
    ADD_KERNEL_SUFF(gemm_kernel, 2, 1), ADD_KERNEL_SUFF(gemm_kernel, 2, 2),ADD_KERNEL_SUFF(gemm_kernel, 2, 3), ADD_KERNEL_SUFF(gemm_kernel, 2, 4),
    ADD_KERNEL_SUFF(gemm_kernel, 3, 1), ADD_KERNEL_SUFF(gemm_kernel, 3, 2),ADD_KERNEL_SUFF(gemm_kernel, 3, 3), ADD_KERNEL_SUFF(gemm_kernel, 3, 4),
    ADD_KERNEL_SUFF(gemm_kernel, 4, 1), ADD_KERNEL_SUFF(gemm_kernel, 4, 2),ADD_KERNEL_SUFF(gemm_kernel, 4, 3), ADD_KERNEL_SUFF(gemm_kernel, 4, 4)
};

static void (*pack_b_funs[])(void*, const void*, size_t, size_t, size_t, const BLASINT8) = {
    ADD_PACK_B_N_SUFF(pack_b),
    ADD_PACK_B_T_SUFF(pack_b)
};

static void (*pack_a_funs[])(void*, const void*, size_t, size_t, size_t, const BLASINT8) = {
    ADD_PACK_A_N_SUFF(pack_a),
    ADD_PACK_A_T_SUFF(pack_a)
};

static void (*small_kernels[])(const size_t, const size_t, const size_t, const float,
    const void *, const size_t, const BLASINT8,
    const void *, const size_t, const BLASINT8,
    const float, int32_t *, const size_t, const int32_t *) = {
        ADD_TRANSP_MACRO(small_kernel, n, n, f), ADD_TRANSP_MACRO(small_kernel, n, t, f),
        ADD_TRANSP_MACRO(small_kernel, t, n, f), ADD_TRANSP_MACRO(small_kernel, t, t, f),
        ADD_TRANSP_MACRO(small_kernel, n, n, c), ADD_TRANSP_MACRO(small_kernel, n, t, c),
        ADD_TRANSP_MACRO(small_kernel, t, n, c), ADD_TRANSP_MACRO(small_kernel, t, t, c),
        ADD_TRANSP_MACRO(small_kernel, n, n, r), ADD_TRANSP_MACRO(small_kernel, n, t, r),
        ADD_TRANSP_MACRO(small_kernel, t, n, r), ADD_TRANSP_MACRO(small_kernel, t, t, r),
    };

static void (*beta_funcs[])(int32_t*, const int32_t*, float, size_t, size_t, size_t) = {
    beta_cf_s8, beta_cc_s8, beta_cr_s8, beta_rf_s8, beta_rc_s8, beta_rr_s8,
    beta_cf_s8_opt, beta_cc_s8_opt, beta_cr_s8_opt, beta_rf_s8_opt, beta_rc_s8_opt, beta_rr_s8_opt
};

static void (*post_op_kernels[])(float alpha, const int32_t* bufferC, int32_t* current_c_ptr, size_t m, size_t n_block, size_t ldc) = {
    post_ops, post_ops_opt
};

static size_t row_major_idx(size_t m, size_t n, size_t ld) {
    return ld * m + n;
}

static size_t col_major_idx(size_t m, size_t n, size_t ld) {
    return m + ld * n;
}

static size_t (*compute_idx[])(size_t m, size_t n, size_t ld) = {
    col_major_idx,
    row_major_idx
};

static size_t mov_oc_fix(size_t mi, size_t ni) {
    UNUSED(mi);
    UNUSED(ni);
    return 0;
}
static size_t mov_oc_col(size_t mi, size_t ni){
    UNUSED(ni);
    return mi;
}

static size_t mov_oc_row(size_t mi, size_t ni) {
    UNUSED(mi);
    return ni;
}

static size_t (*move_oc[])(size_t, size_t) = {
    mov_oc_fix, mov_oc_col, mov_oc_row
};

EXTERNAL_API void ADD_TYPES_SUFF(prefill_cblas_gemm)(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const CBLAS_OFFSET offsetc,
    const size_t m, const size_t n, const size_t k, const float alpha,
    const void *a, const size_t lda, const BLASINT8 oa,
    const void *b, const size_t ldb, const BLASINT8 ob,
    const float beta, int32_t *c, const size_t ldc, const int32_t *oc) {

    int opt_offset = ((alpha == 1.0f) && (beta == 0.0f || beta == 1.0f)) ? 1 : 0;
    if(Layout == CblasColMajor) {
        int beta_offset = (offsetc == CblasFixOffset) ? 0 : (offsetc == CblasColOffset ? 1:2);
        int_gemm_funcs arg = {
            small_kernels[(transb == CblasTrans) + 2 * (transa == CblasTrans) + beta_offset * 4],
            gemm_kernels,
            pack_a_funs[transa == CblasTrans],
            pack_b_funs[transb == CblasTrans],
            beta_funcs[beta_offset + opt_offset * 6],
            post_op_kernels[alpha == 1],
            compute_idx[transa == CblasTrans],
            compute_idx[transb == CblasTrans],
            move_oc[beta_offset],
        };
        (gemm_impl_8bit(&arg, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc, 262144));
    } else if (Layout == CblasRowMajor) {
        int beta_offset = (offsetc == CblasFixOffset) ? 3 : (offsetc == CblasColOffset ? 4 : 5);
        int beta_offset_small = (offsetc == CblasFixOffset) ? 0 : (offsetc == CblasColOffset ? 2 : 1);
        int_gemm_funcs arg = {
            small_kernels[(transa == CblasTrans) + 2 * (transb == CblasTrans) + beta_offset_small * 4],
            gemm_kernels,
            pack_a_funs[transb == CblasTrans],
            pack_b_funs[transa == CblasTrans],
            beta_funcs[beta_offset + opt_offset * 6],
            post_op_kernels[alpha == 1],
            compute_idx[transb == CblasTrans],
            compute_idx[transa == CblasTrans],
            move_oc[beta_offset_small]
        };
        (gemm_impl_8bit(&arg, n, m, k, alpha, b, ldb, ob, a, lda, oa, beta, c, ldc, oc, 262144));
    }
    else {
        printf("Incorrect layout");
        return;
    }
}

#ifdef __cplusplus
}
#endif
