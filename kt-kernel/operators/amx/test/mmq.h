#ifndef MMQ_H
#define MMQ_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_amx_init(void);

bool ggml_compute_forward_mul_mat_use_amx(struct ggml_tensor* dst);

void ggml_mul_mat_amx(struct ggml_tensor* dst, int nth, int ith, void* wdata, int wsize);

/**
 * @param m
 * @param n
 * @param k
 * @param a
 * @param a_type
 * @param b
 * @param b_type
 * @param c
 * @param c_type
 * @param ldc c stride in elements
 * @param ith
 * @param nth
 * @param wdata auxillary data area
 * @param wsize size of auxillary data size
 */

void mat_mul_amx(int m, int n, int k, const void* a, int a_type, const void* b, int b_type, void* c, int c_type,
                 int ldc, int ith, int nth, void* wdata, int wsize);

#ifdef __cplusplus
}
#endif

#endif  // MMQ_H
