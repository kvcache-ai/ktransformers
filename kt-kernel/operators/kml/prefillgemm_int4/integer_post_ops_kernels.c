#include <stdint.h>
#include <math.h>
#include "integer_gemm_kernels.h"
#include "helping_macros.h"
#include "beta_macros.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

// Fixed OC
void BETA_SUFF(post_ops)(float alpha, const int32_t* bufferC, int32_t* current_c_ptr, size_t m, size_t n_block, size_t ldc) {
    float* current_c_float_ptr = (float*) current_c_ptr;
    double double_alpha = (double) alpha;
    for (size_t n_idx = 0; n_idx < n_block; ++n_idx) {
        for (size_t m_idx = 0; m_idx < m; ++m_idx) {
            current_c_ptr[m_idx + n_idx * ldc] = round(((double)(current_c_float_ptr[m_idx + n_idx * ldc]))
                                                    + double_alpha * ((double) bufferC[m_idx + n_idx * LDC(m, ldc)]));
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif /* __cplusplus */