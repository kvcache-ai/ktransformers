// RISC-V rv64 tinyBLAS sgemm backend.
// Copyright(c) 2026 by KVCache.AI, All Rights Reserved.
//
// Mirrors tinyblas_cpu_sgemm_arm82.cpp: instantiates the shared
// tinyblas_cpu_sgemm.inc template with the generic llamafile_sgemm /
// llamafile_mixmul symbol names, compiled for rv64 with the RVV backend
// selected inside tinyblas_cpu.h (KT_RVV).
//
// Missing-file matrix reference: tinyblas_cpu_sgemm_amd_rv64.cpp
// (siblings: tinyblas_cpu_sgemm_amd_avx.cpp / _avx2.cpp / _avx512f.cpp)

#if defined(__riscv) && __riscv_xlen == 64 && defined(KT_RVV)
#define llamafile_sgemm llamafile_sgemm_rv64
#include "tinyblas_cpu_sgemm.inc"
#endif  // __riscv
