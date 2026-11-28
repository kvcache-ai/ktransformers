// RISC-V RVV 1.0 vector math helpers for tinyblas / llamafile port.
// Copyright(c) 2026 by KVCache.AI, All Rights Reserved.
//
// Provides tinyBLAS-compatible primitives (load/store/add/sub/mul/madd/hsum)
// parameterized over RVV vector types so that tinyblas_cpu.h can treat RVV
// like SSE/AVX/NEON. Requires -march=rv64gcv (RVV 1.0) and GCC >= 13
// (gcc-toolset-14 on openEuler 24.03) or LLVM >= 15.
//
// VLEN on supported hardware is 128 bits => vfloat32m1_t holds 4 lanes,
// matching the SSE (4-wide) tiling path in tinyBLAS.

#pragma once

#include <riscv_vector.h>
#include <stdint.h>

#if defined(__riscv) && defined(__riscv_v_intrinsic)

namespace rvv {

// ------------------------------------------------------------------ //
// float32m1 (4 lanes at VLEN=128) ops used by tinyBLAS<...,float,...>
// ------------------------------------------------------------------ //

inline vfloat32m1_t vzero_f32m1() { return __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1()); }

inline vfloat32m1_t vload_f32(const float* p, size_t vl) { return __riscv_vle32_v_f32m1(p, vl); }

inline void vstore_f32(float* p, vfloat32m1_t v, size_t vl) { __riscv_vse32_v_f32m1(p, v, vl); }

inline vfloat32m1_t vadd_f32(vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
    return __riscv_vfadd_vv_f32m1(a, b, vl);
}

inline vfloat32m1_t vsub_f32(vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
    return __riscv_vfsub_vv_f32m1(a, b, vl);
}

inline vfloat32m1_t vmul_f32(vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
    return __riscv_vfmul_vv_f32m1(a, b, vl);
}

// Fused multiply-accumulate: returns a*b + c
inline vfloat32m1_t vfmacc_f32(vfloat32m1_t c, vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
    return __riscv_vfmacc_vv_f32m1(c, a, b, vl);
}

// Horizontal sum of a 4-lane (VLEN=128) f32m1 vector.
inline float vhsum_f32m1(vfloat32m1_t v) {
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t red = __riscv_vfredusum_vs_f32m1_f32m1(v, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
    return __riscv_vfmv_f_s_f32m1_f32(red);
}

// ------------------------------------------------------------------ //
// bf16 -> fp32 widening load (ggml_bf16_t is uint16_t storage)
// ------------------------------------------------------------------ //

inline vfloat32m1_t vload_bf16(const uint16_t* p, size_t vl) {
    vuint16m1_t u = __riscv_vle16_v_u16m1(p, vl);
    vuint32m2_t w = __riscv_vwaddu_vx_u32m2(u, 0, vl);
    w = __riscv_vsll_vx_u32m2(w, 16, vl);
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vget_v_u32m2_u32m1(w, 0));
}

inline vfloat32m1_t vmul_bf16_f32(vfloat32m1_t a_f32, const uint16_t* b, size_t vl) {
    return __riscv_vfmul_vv_f32m1(a_f32, vload_bf16(b, vl), vl);
}

}  // namespace rvv

#endif  // __riscv_v_intrinsic
