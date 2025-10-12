#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "integer_gemm_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

// here we will use BLASINT8, because we care only about type size, not signness (no math operations required)
void gemm_impl_8bit(int_gemm_funcs* arg, size_t m, size_t n, size_t k, float alpha,
                    const void* a, size_t lda, const BLASINT8 oa,
                    const void* b, size_t ldb, const BLASINT8 ob,
                    float beta, int32_t* c, size_t ldc, const int32_t* oc, size_t small_switch) {

    void (*small_kernel)(const size_t, const size_t, const size_t, const float,
                         const void *, const size_t, const BLASINT8,
                         const void *, const size_t, const BLASINT8,
                         const float, int32_t *, const size_t, const int32_t *) = arg->small_kernel;

    if (m * n * k < small_switch) { // experimentally measured constant
        small_kernel(m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc);
    } else {
        // Corner cases optimizations
        if ((alpha == 1.0f) && (beta == 0.0f || beta == 1.0f)) {
            gemm_driver_opt(arg, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc);
        } else {
            gemm_driver(arg, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc);
        }
    }
}

#ifdef __cplusplus
}
#endif /* __cplusplus */