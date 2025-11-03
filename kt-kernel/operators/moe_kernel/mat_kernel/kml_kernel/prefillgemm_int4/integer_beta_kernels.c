#include <stdint.h>
#include "integer_gemm_kernels.h"
#include "helping_macros.h"
#include "beta_macros.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
// Column major C, Fixed C_offset
void BETA_SUFF(beta_cf_s8)(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc) {
    DTYPE oc_val = (DTYPE)*oc;
    DTYPE* c_typed_ptr = (DTYPE*) c_ptr;
    DTYPE beta_val = (DTYPE)beta;
    for (size_t ni = 0; ni < n_block_size; ++ni) {
        for (size_t mi = 0; mi < m_block_size; ++mi) {
            c_typed_ptr[ldc * ni + mi] = beta_val * ((DTYPE)c_ptr[ldc * ni + mi]) + oc_val;
        }
    }
}

// Column major C, Column major C_offset
void BETA_SUFF(beta_cc_s8)(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc) {
    DTYPE* c_typed_ptr = (DTYPE*) c_ptr;
    DTYPE beta_val = (DTYPE)beta;
    for (size_t ni = 0; ni < n_block_size; ++ni) {
        for (size_t mi = 0; mi < m_block_size; ++mi) {
            c_typed_ptr[ldc * ni + mi] = beta_val * ((DTYPE)c_ptr[ldc * ni + mi]) + ((DTYPE)oc[mi]);
        }
    }
}

// Column major C, Row major C_offset
void BETA_SUFF(beta_cr_s8)(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc) {
    DTYPE* c_typed_ptr = (DTYPE*) c_ptr;
    DTYPE beta_val = (DTYPE)beta;
    for (size_t ni = 0; ni < n_block_size; ++ni) {
        for (size_t mi = 0; mi < m_block_size; ++mi) {
            c_typed_ptr[ldc * ni + mi] = beta_val * ((DTYPE)c_ptr[ldc * ni + mi]) + ((DTYPE)oc[ni]);
        }
    }
}

// Row major C, Fixed C_offset
// for row-major we actually swap m and n values so we reswap it here again
void BETA_SUFF(beta_rf_s8)(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc) {
    DTYPE oc_val = (DTYPE)*oc;
    DTYPE* c_typed_ptr = (DTYPE*) c_ptr;
    DTYPE beta_val = (DTYPE)beta;
    for (size_t mi = 0; mi < n_block_size; ++mi) {
        for (size_t ni = 0; ni < m_block_size; ++ni) {
            c_typed_ptr[ldc * mi + ni] = beta_val * ((DTYPE)c_ptr[ldc * mi + ni]) + oc_val;
        }
    }
}

// Row major C, Column major C_offset
// for row-major we actually swap m and n values so we reswap it here again
void BETA_SUFF(beta_rc_s8)(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc) {
    DTYPE* c_typed_ptr = (DTYPE*) c_ptr;
    DTYPE beta_val = (DTYPE)beta;
    for (size_t mi = 0; mi < n_block_size; ++mi) {
        for (size_t ni = 0; ni < m_block_size; ++ni) {
            c_typed_ptr[ldc * mi + ni] = beta_val * ((DTYPE)c_ptr[ldc * mi + ni]) + ((DTYPE)oc[mi]);
        }
    }
}

// Row major C, Row major C_offset
// for row-major we actually swap m and n values so we reswap it here again
void BETA_SUFF(beta_rr_s8)(int32_t* c_ptr, const int32_t* oc, float beta, size_t m_block_size, size_t n_block_size, size_t ldc) {
    DTYPE* c_typed_ptr = (DTYPE*) c_ptr;
    DTYPE beta_val = (DTYPE)beta;
    for (size_t mi = 0; mi < n_block_size; ++mi) {
        for (size_t ni = 0; ni < m_block_size; ++ni) {
            c_typed_ptr[ldc * mi + ni] = beta_val * ((DTYPE)c_ptr[ldc * mi + ni]) + ((DTYPE)oc[ni]);
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif /* __cplusplus */
