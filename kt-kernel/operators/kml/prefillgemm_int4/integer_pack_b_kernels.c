#include <stdint.h>
#include "integer_gemm_kernels.h"
#include "helping_macros.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if defined(TRANSB)
    // row major
    #define INDEXING_B(row_idx, col_idx, ldb) ((col_idx) * (ldb) + row_idx)
    #define ADD_B_SUFF(name) ADD_PACK_B_T_SUFF(name)
#elif defined(NOTRANSB)
    // col major
    #define INDEXING_B(row_idx, col_idx, ldb) ((row_idx) * (ldb) + col_idx)
    #define ADD_B_SUFF(name) ADD_PACK_B_N_SUFF(name)
#else
    #error "Neither TRANSB or NOTRANSB is defined."
#endif
    
void ADD_B_SUFF(pack_b)(void* bufferB, const void* curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb, const BLASINT8 ob) {
    RHS_INT_TYPE* bufferB_typed = (RHS_INT_TYPE*) bufferB;
    RHS_INT_TYPE* curr_b_ptr_typed = (RHS_INT_TYPE*) curr_b_ptr;
    size_t k_block_size_up = (k_block_size + KERNEL_K_STEP - 1) / KERNEL_K_STEP * KERNEL_K_STEP;
    size_t k_portions = k_block_size / KERNEL_K_STEP;
    size_t k_resid = k_block_size - KERNEL_K_STEP * k_portions;

    size_t n_portions = n_block_size / KERNEL_N_STEP;
    size_t n_resid = n_block_size - KERNEL_N_STEP * n_portions;

    for (size_t in4 = 0; in4 < n_portions; ++in4) {
        for (size_t in = 0; in < KERNEL_N_STEP; ++in) {
            for (size_t ik16 = 0; ik16 < k_block_size / KERNEL_K_STEP; ++ik16) {
                for (size_t ik = 0; ik < KERNEL_K_STEP; ++ik) {
                    bufferB_typed[ik + KERNEL_K_STEP * in + KERNEL_K_STEP * KERNEL_N_STEP * ik16 + k_block_size_up * KERNEL_N_STEP * in4] = curr_b_ptr_typed[INDEXING_B((KERNEL_N_STEP * in4 + in), (KERNEL_K_STEP * ik16 + ik), ldb)] + ob;
                }
            }

            if (k_resid) {
                for (size_t ik = 0; ik < k_resid; ++ik) {
                    bufferB_typed[ik + KERNEL_K_STEP * in + KERNEL_K_STEP * KERNEL_N_STEP * k_portions + k_block_size_up * KERNEL_N_STEP * in4] = curr_b_ptr_typed[INDEXING_B((KERNEL_N_STEP * in4 + in), (KERNEL_K_STEP * k_portions + ik), ldb)] + ob;
                }
                for (size_t ik = k_resid; ik < KERNEL_K_STEP; ++ik) {
                    bufferB_typed[ik + KERNEL_K_STEP * in + KERNEL_K_STEP * KERNEL_N_STEP * k_portions + k_block_size_up * KERNEL_N_STEP * in4] = 0;
                }
            }
        }
    }
    if (n_resid) {
        for (size_t in = 0; in < n_resid; ++in) {
            for (size_t ik16 = 0; ik16 < k_block_size / KERNEL_K_STEP; ++ik16) {
                for (size_t ik = 0; ik < KERNEL_K_STEP; ++ik) {
                    bufferB_typed[ik + KERNEL_K_STEP * in + KERNEL_K_STEP * n_resid * ik16 + k_block_size_up * KERNEL_N_STEP * n_portions] = curr_b_ptr_typed[INDEXING_B((KERNEL_N_STEP * n_portions + in), (KERNEL_K_STEP * ik16 + ik), ldb)] + ob;
                }
            }
            if (k_resid) {
                for (size_t ik = 0; ik < k_resid; ++ik) {
                    bufferB_typed[ik + KERNEL_K_STEP * in + KERNEL_K_STEP * n_resid * k_portions + k_block_size_up * KERNEL_N_STEP * n_portions] = curr_b_ptr_typed[INDEXING_B((KERNEL_N_STEP * n_portions + in), (KERNEL_K_STEP * k_portions + ik), ldb)] + ob;
                }
                for (size_t ik = k_resid; ik < KERNEL_K_STEP; ++ik) {
                    bufferB_typed[ik + KERNEL_K_STEP * in + KERNEL_K_STEP * n_resid * k_portions + k_block_size_up * KERNEL_N_STEP * n_portions] = 0;
                }
            }
        }
    }
    
    // printf("n_block_size:%lu ,k_block_size: %lu\n", n_block_size, k_block_size);

    // for(size_t n_idx = 0; n_idx < n_block_size; n_idx++) {
    //     for(size_t k_idx = 0; k_idx < k_block_size; k_idx++) {
    //         size_t old_split_n = n_idx / OLD_N_SIZE;
    //         size_t old_idx_n   = n_idx % OLD_N_SIZE;
    //         size_t new_split_n = n_idx / NEW_N_SIZE;
    //         size_t new_idx_n   = n_idx % NEW_N_SIZE;
    //         size_t split_k     = k_idx / KERNEL_K_STEP;
    //         size_t idx_k       = k_idx % KERNEL_K_STEP;

    //         size_t old_buff_idx = 
    //             old_split_n * OLD_N_SIZE * ldb +
    //             split_k * OLD_N_SIZE * KERNEL_K_STEP +
    //             old_idx_n * KERNEL_K_STEP +
    //             idx_k;
    //         size_t new_buff_idx = 
    //             new_split_n * NEW_N_SIZE * k_block_size +
    //             split_k * NEW_N_SIZE * KERNEL_K_STEP +
    //             new_idx_n * KERNEL_K_STEP +
    //             idx_k;
    //         bufferB_typed[new_buff_idx] = curr_b_ptr_typed[old_buff_idx] + ob;
    //     }
    // }

}
        
#ifdef __cplusplus
} // extern "C"
#endif /* __cplusplus */
