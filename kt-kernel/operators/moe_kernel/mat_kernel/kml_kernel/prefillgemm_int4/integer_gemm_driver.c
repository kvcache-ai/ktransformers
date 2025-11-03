#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "integer_gemm_kernels.h"
#include "beta_macros.h"

#define OLD_N_SIZE 8
#define PACKED_LD_STEP(n_step, k_step, ldb) (n_step * (ldb / 2) + (k_step / 2) * OLD_N_SIZE)

void BETA_SUFF(gemm_driver)(int_gemm_funcs* arg, size_t m, size_t n, size_t k, float alpha,
    const void* a, size_t lda, const BLASINT8 oa,
    const void* b, size_t ldb, const BLASINT8 ob,
    float beta, int32_t* c, size_t ldc, const int32_t* oc) {

        void (**gemm_kernels)(const void*, const void*, int32_t*, size_t, int64_t, int64_t) = arg->gemm_kernels;
        void (*pack_a_fun)(void*, const void*, size_t, size_t, size_t, const BLASINT8) = arg->pack_a_fun;
        void (*pack_b_fun)(void*, const void*, size_t, size_t, size_t, const BLASINT8) = arg->pack_b_fun;
        void (*beta_func)(int32_t*, const int32_t*, float, size_t, size_t, size_t) = arg->beta_func;
        size_t (*a_indexing)(size_t m, size_t n, size_t ld) = arg->a_indexing;
        size_t (*b_indexing)(size_t m, size_t n, size_t ld) = arg->b_indexing;

#ifndef BETA_OPT
        void (*post_ops_func)(float, const int32_t*, int32_t*, size_t, size_t, size_t) = arg->post_ops_func;
#endif // BETA_OPT

        const BLASINT8* a_typed = (const BLASINT8*) a;
        const BLASINT8* b_typed = (const BLASINT8*) b;

        BLASINT8* bufferA = (BLASINT8*) aligned_alloc(ALIGNMENT, sizeof(BLASINT8) * K_BLOCK * M_BLOCK);
        BLASINT8* bufferB = (BLASINT8*) aligned_alloc(ALIGNMENT, sizeof(BLASINT8) * K_BLOCK * N_BLOCK);

    // Tmp buffer is not needed when (alpha = 1 and beta = 0/1)
#ifdef BETA_OPT
        int32_t* bufferC = c;
#else
        int32_t* bufferC = (int32_t*) aligned_alloc(ALIGNMENT, sizeof(int32_t) * m * N_BLOCK);
#endif

        if (!bufferA || !bufferB || !bufferC) {
            free(bufferA);
            free(bufferB);
            free(bufferC);
            printf("Integer GEMM unsuccessful allocation");
            return;
        }
        // printf("pack b beta: %f\n",beta);
        beta_func(c, oc, beta, m, n, ldc);

        for (size_t n_block = 0; n_block < n; n_block += N_BLOCK) {
            size_t n_block_size = n - n_block;
            if (n_block_size > N_BLOCK) {
                n_block_size = N_BLOCK;
            }
        
#ifndef BETA_OPT
        // fill bufferC w/ zeros
        for (size_t tmp_idx = 0; tmp_idx < (m * N_BLOCK); ++tmp_idx) {
            bufferC[tmp_idx] = 0;
        }
#endif // BETA_OPT

        if (alpha != 0.0f){
            for (size_t k_block = 0; k_block < k; k_block += K_BLOCK){
                size_t k_block_size = k - k_block;
                if (k_block_size > K_BLOCK) {
                    k_block_size = K_BLOCK;
                }
                size_t k_block_size_up = (k_block_size + KERNEL_K_STEP - 1) / KERNEL_K_STEP * KERNEL_K_STEP;

                const BLASINT8* curr_b_ptr = b_typed + b_indexing(k_block, n_block, ldb);

                pack_b_fun(bufferB, curr_b_ptr, n_block_size, k_block_size, ldb, ob);
                for (size_t m_block = 0; m_block < m; m_block += M_BLOCK) {
                    size_t m_block_size = m - m_block;
                    if (m_block_size > M_BLOCK) {
                        m_block_size = M_BLOCK;
                    }
                    const BLASINT8* curr_a_ptr = a_typed + PACKED_LD_STEP(m_block, k_block, lda);
                    pack_a_fun(bufferA, curr_a_ptr, m_block_size, k_block_size, lda, oa);
                    // loop over bufferB, taking parts which fit into L1
                    for (size_t n_sub_block = 0; n_sub_block < n_block_size; n_sub_block += KERNEL_N_STEP) {
                        size_t n_sub_block_size = n_block_size - n_sub_block;
                        if (n_sub_block_size > KERNEL_N_STEP) {
                            n_sub_block_size = KERNEL_N_STEP;
                        }
                        BLASINT8* current_bufferB_ptr = bufferB + n_sub_block * k_block_size_up;
                        // loop over bufferA, taking parts which fit into L1
                        for (size_t m_sub_block = 0; m_sub_block < m_block_size; m_sub_block += KERNEL_M_STEP) {
                            size_t m_sub_block_size = m_block_size - m_sub_block;
                            if (m_sub_block_size > KERNEL_M_STEP) {
                                m_sub_block_size = KERNEL_M_STEP;
                            }
                            BLASINT8* current_bufferA_ptr = bufferA + m_sub_block * k_block_size_up;
#ifdef BETA_OPT
                            int32_t* current_bufferC_ptr = bufferC + n_block * ldc + n_sub_block * LDC(m, ldc) + m_sub_block + m_block;
#else
                            int32_t* current_bufferC_ptr = bufferC + n_sub_block * LDC(m, ldc) + m_sub_block + m_block;
#endif
                            // call kernel which performs loop over k_block_size
                            gemm_kernels[(n_sub_block_size - 1) + (m_sub_block_size - 1) * KERNEL_N_STEP](current_bufferA_ptr, current_bufferB_ptr, current_bufferC_ptr,
                            LDC(m, ldc), k_block_size_up, COMP_SV_LEN);
                        }
                    }
                }
            }
        }
        
#ifndef BETA_OPT
        // copy C data from bufferC multiplying by alpha and adding initial C data (scaled by beta)
        int32_t* current_c_ptr = c + n_block * ldc; // col major
        post_ops_func(alpha, bufferC, current_c_ptr, LDC(m, ldc), n_block_size, ldc);
#endif
    }

    free(bufferA);
    free(bufferB);

#ifndef BETA_OPT
    free(bufferC);
#endif
    }