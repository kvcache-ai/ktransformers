// RISC-V rv64 iqk_mul_mat stub.
// Copyright(c) 2026 by KVCache.AI, All Rights Reserved.
//
// Missing-file matrix reference: iqk_mul_mat_amd_rv64.cpp
// (sibling: iqk_mul_mat_amd_avx2.cpp)
//
// iqk_mul_mat.inc only provides AVX2 / ARM NEON kernels; the RVV port falls
// back to the generic tinyBLAS path via sgemm.cpp, so the iqk fast path
// simply reports "unsupported" here (same contract as the *_unsupported
// variants in tinyblas_cpu_unsupported.cpp).

#if defined(__riscv) && __riscv_xlen == 64
#include "sgemm.h"

bool iqk_mul_mat(long, long, long, int, const void*, long, int, const void*, long, float*, long, int, int) {
    return false;
}

bool iqk_mul_mat_moe(long, long, long, int, int, const void*, const void*, float*, long, long, const void*, int, int) {
    return false;
}
#endif  // __riscv
