#include <stdint.h>
#include "integer_gemm_kernels.h"
#include "helping_macros.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if defined(TRANSA)
    // row major
    #define INDEXING_A(row_idx, col_idx, lda) ((col_idx) * (lda) + row_idx)
    #define ADD_A_SUFF(name) ADD_PACK_A_T_SUFF(name)
#elif defined(NOTRANSA)
    // col major
    #define INDEXING_A(row_idx, col_idx, lda) ((row_idx) * (lda) + col_idx)
    #define ADD_A_SUFF(name) ADD_PACK_A_N_SUFF(name)
#else
    #error "Neither TRANSA or NOTRANSA is defined"
#endif

#define OLD_N_SIZE 8
#define NEW_N_SIZE 4

void ADD_A_SUFF(pack_a)(void* bufferA, const void* curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda, const BLASINT8 oa) {
    LHS_INT_TYPE* bufferA_typed = (LHS_INT_TYPE*) bufferA;
    LHS_INT_TYPE* curr_a_ptr_typed = (LHS_INT_TYPE*) curr_a_ptr;

    // printf("m_block_size:%lu ,k_block_size: %lu\n", m_block_size, k_block_size);

    for(size_t old_split_n = 0; old_split_n < (m_block_size / OLD_N_SIZE); old_split_n++) {
        for(size_t split_k = 0; split_k < (k_block_size / KERNEL_K_STEP); split_k++) {
            for(size_t old_idx_n = 0; old_idx_n < OLD_N_SIZE; old_idx_n++) {
                for(size_t idx_k = 0; idx_k < KERNEL_K_STEP; idx_k++) {
                    size_t n_idx = old_split_n * OLD_N_SIZE + old_idx_n;
                    size_t new_split_n = n_idx / NEW_N_SIZE;
                    size_t new_idx_n   = n_idx % NEW_N_SIZE;

                    size_t old_buff_idx = 
                        old_split_n * OLD_N_SIZE * lda +
                        split_k * OLD_N_SIZE * KERNEL_K_STEP +
                        old_idx_n * KERNEL_K_STEP +
                        idx_k;
                    size_t new_buff_idx = 
                        new_split_n * NEW_N_SIZE * k_block_size +
                        split_k * NEW_N_SIZE * KERNEL_K_STEP +
                        new_idx_n * KERNEL_K_STEP +
                        idx_k;
                    bufferA_typed[new_buff_idx] = curr_a_ptr_typed[old_buff_idx];
                }
            }
        }
    }

    // for(size_t n_idx = 0; n_idx < m_block_size; n_idx++) {
    //     for(size_t k_idx = 0; k_idx < k_block_size; k_idx++) {
    //         size_t old_split_n = n_idx / OLD_N_SIZE;
    //         size_t old_idx_n   = n_idx % OLD_N_SIZE;
    //         size_t new_split_n = n_idx / NEW_N_SIZE;
    //         size_t new_idx_n   = n_idx % NEW_N_SIZE;
    //         size_t split_k     = k_idx / KERNEL_K_STEP;
    //         size_t idx_k       = k_idx % KERNEL_K_STEP;

    //         size_t old_buff_idx = 
    //             old_split_n * OLD_N_SIZE * lda +
    //             split_k * OLD_N_SIZE * KERNEL_K_STEP +
    //             old_idx_n * KERNEL_K_STEP +
    //             idx_k;
    //         size_t new_buff_idx = 
    //             new_split_n * NEW_N_SIZE * k_block_size +
    //             split_k * NEW_N_SIZE * KERNEL_K_STEP +
    //             new_idx_n * KERNEL_K_STEP +
    //             idx_k;
    //         bufferA_typed[new_buff_idx] = curr_a_ptr_typed[old_buff_idx] + oa;
    //     }
    // }

    // size_t k_block_size_up = (k_block_size + KERNEL_K_STEP - 1) / KERNEL_K_STEP * KERNEL_K_STEP;
    // size_t k_portions = k_block_size / KERNEL_K_STEP;
    // size_t k_resid = k_block_size - KERNEL_K_STEP * k_portions;

    // size_t m_portions = m_block_size / KERNEL_M_STEP;
    // size_t m_resid = m_block_size - KERNEL_M_STEP * m_portions;

    // for (size_t im4 = 0; im4 < m_portions; ++im4) {
    //     for (size_t ik16 = 0; ik16 < k_portions; ++ik16) {
    //         for (size_t im = 0; im < KERNEL_M_STEP; ++im) {
    //             for (size_t ik = 0; ik < KERNEL_K_STEP; ++ik) {
    //                 bufferA_typed[ik + im * KERNEL_K_STEP + KERNEL_K_STEP * KERNEL_M_STEP * ik16 + k_block_size_up * KERNEL_M_STEP * im4] = 
    //                     curr_a_ptr_typed[INDEXING_A((ik16 * KERNEL_K_STEP + ik), (im4 * KERNEL_M_STEP + im), lda)] + oa;
    //             }
    //         }
    //     }
    //     if (k_resid) {
    //         for (size_t im = 0; im < KERNEL_M_STEP; ++im) {
    //             for (size_t ik = 0; ik < k_resid; ++ik) {
    //                 bufferA_typed[ik + im * KERNEL_K_STEP + KERNEL_K_STEP * KERNEL_M_STEP * k_portions + k_block_size_up * KERNEL_M_STEP * im4] = 
    //                     curr_a_ptr_typed[INDEXING_A((k_portions * KERNEL_K_STEP + ik), (im4 * KERNEL_M_STEP + im), lda)] + oa;
    //             }
    //         }
    //         for (size_t im = 0; im < KERNEL_M_STEP; ++im) {
    //             for (size_t ik = k_resid; ik < KERNEL_K_STEP; ++ik) {
    //                 bufferA_typed[ik + im * KERNEL_K_STEP + KERNEL_K_STEP * KERNEL_M_STEP * k_portions + k_block_size_up * KERNEL_M_STEP * im4] = 0;
    //             }
    //         }
    //     }
    // }
    // if (m_resid) {
    //     for (size_t ik16 = 0; ik16 < k_portions; ++ik16) {
    //         for (size_t im = 0; im < m_resid; ++im) {
    //             for (size_t ik = 0; ik < KERNEL_K_STEP; ++ik) {
    //                 bufferA_typed[ik + im * KERNEL_K_STEP + KERNEL_K_STEP * m_resid * ik16 + k_block_size_up * KERNEL_M_STEP * m_portions] = 
    //                     curr_a_ptr_typed[INDEXING_A((ik16 * KERNEL_K_STEP + ik), (m_portions * KERNEL_M_STEP + im), lda)] + oa;
    //             }
    //         }
    //     }
    //     if (k_resid) {
    //         for (size_t im = 0; im < m_resid; ++im) {
    //             for (size_t ik = 0; ik < k_resid; ++ik) {
    //                 bufferA_typed[ik + im * KERNEL_K_STEP + KERNEL_K_STEP * m_resid * k_portions + k_block_size_up * KERNEL_M_STEP * m_portions] = 
    //                     curr_a_ptr_typed[INDEXING_A((k_portions * KERNEL_K_STEP + ik), (m_portions * KERNEL_M_STEP + im), lda)] + oa;
    //             }
    //         }
    //         for (size_t im = 0; im < m_resid; ++im) {
    //             for (size_t ik = k_resid; ik < KERNEL_K_STEP; ++ik) {
    //                 bufferA_typed[ik + im * KERNEL_K_STEP + KERNEL_K_STEP * m_resid * k_portions + k_block_size_up * KERNEL_M_STEP * m_portions] = 0;
    //             }
    //         }
    //     }
    // }
}

#ifdef __cplusplus
} // extern "C"
#endif /* __cplusplus */
