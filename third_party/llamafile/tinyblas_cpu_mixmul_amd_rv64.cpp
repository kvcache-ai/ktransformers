// RISC-V rv64 tinyBLAS mixmul backend.
// Copyright(c) 2026 by KVCache.AI, All Rights Reserved.
//
// Mirrors tinyblas_cpu_mixmul_arm80.cpp: instantiates the shared
// tinyblas_cpu_mixmul.inc template with the generic llamafile_mixmul symbol
// name, compiled for rv64 with the RVV backend selected inside
// tinyblas_cpu.h (KT_RVV).
//
// Missing-file matrix reference: tinyblas_cpu_mixmul_amd_rv64.cpp
// (siblings: tinyblas_cpu_mixmul_amd_avx.cpp / _avx2.cpp / _avx512f.cpp)

#if defined(__riscv) && __riscv_xlen == 64 && defined(KT_RVV)
#define llamafile_mixmul llamafile_mixmul_rv64
#include "tinyblas_cpu_mixmul.inc"

/**
 * Returns number of shared memory bytes llamafile_mixmul() needs.
 */
size_t llamafile_mixmul_needs(const ggml_tensor* weights, const ggml_tensor* thought, const ggml_tensor* plan) {
    ggml_compute_params params{};
    params.wsize = 0x7ffff000;
    params.wdata = (void*)0x1000;
    MixMul mm{&params, weights, thought, plan, 0};
    if (mm.allocate_shared_memory())
        return mm.get_allocated_bytes();
    else
        return 0;
}

#endif  // __riscv
