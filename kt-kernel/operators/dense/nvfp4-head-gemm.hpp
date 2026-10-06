// Dense NVFP4 GEMM for the LM head.
//
// WHY THIS EXISTS
// ---------------
// kt-kernel ships no dense 4-bit GEMM: every 4-bit kernel in 0.7.1 is MoE-shaped
// (`operators/*/mxfp4-moe.hpp` and friends), so a checkpoint that quantizes its
// `lm_head` -- which nvidia's Qwen3.6 NVFP4 release does, declaring it
// W4A16_NVFP4, group_size 16 -- loads correctly but cannot be executed. The
// logits projection is a plain (1, K) @ (K, N) matmul with no expert indexing,
// so it needs a plain kernel rather than an expert-scheduled one.
//
// CONTRACT (pinned from the stored tensors, cross-checked against
// operators/avx2/mxfp4-moe.hpp, and validated in numpy before being written here)
//
//   weight          uint8   (N, K/2)    two E2M1 nibbles per byte, low first
//   weight_scale    fp8e4m3 (N, K/16)   one scale per group of 16 along K
//   weight_scale_2  float   scalar      per-tensor global
//   input_scale     float   scalar      activation global
//
//   W[n,k] = E2M1(nibble) * weight_scale[n, k/16] * weight_scale_2
//   logits = x @ W^T
//
// The E2M1 codebook is the same 16-entry table the MXFP4 MoE kernel uses, so
// this file reuses that representation rather than inventing one:
//   0, .5, 1, 1.5, 2, 3, 4, 6  and the same values negated.
//
// A numpy implementation of exactly this math was validated against the real
// checkpoint before this file was written: blocked and unblocked forms agree to
// 0.0, and the resulting logits match an independently dequantized bf16 head to
// 0.23% with the same argmax. Keep this kernel's output comparable to that.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace kt::nvfp4_head {

// E2M1 (FP4) -> bf16 bit patterns, identical to the MXFP4 MoE kernel's tables.
// Keep these in sync with operators/avx2/mxfp4-moe.hpp:fp4_bf16_{lo,hi}; they
// were verified against the codebook derived from the checkpoint's own bytes.
alignas(16) static constexpr uint8_t kFp4Bf16Lo[16] = {
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
alignas(16) static constexpr uint8_t kFp4Bf16Hi[16] = {
    0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
    0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};

// Scalar nibble -> float, used for tails and as the definition of record.
static inline float fp4_to_float(uint8_t nib) {
  const uint32_t bf16 = static_cast<uint32_t>(kFp4Bf16Lo[nib]) |
                        (static_cast<uint32_t>(kFp4Bf16Hi[nib]) << 8);
  const uint32_t fp32 = bf16 << 16;
  float out;
  std::memcpy(&out, &fp32, sizeof(float));
  return out;
}

// fp8 e4m3 -> float. The checkpoint stores group scales as F8_E4M3, and the
// decode is inlined here so this header stays dependency-free: subnormals are
// man * 2^-9 (e4m3 has a 3-bit mantissa and bias 7), and the all-ones mantissa
// saturates as e4m3fn does.
static inline float fp4_scale_to_float(uint8_t bits) {
  const uint32_t sign = (bits >> 7) & 0x1u;
  const uint32_t exp = (bits >> 3) & 0xFu;
  const uint32_t man = bits & 0x7u;
  if (exp == 0) {
    const float v = static_cast<float>(man) * std::ldexp(1.0f, -9);
    return sign ? -v : v;
  }
  if (exp == 0xF && man == 0x7u) {
    return sign ? -448.0f : 448.0f;
  }
  const float v = std::ldexp(1.0f + static_cast<float>(man) / 8.0f,
                             static_cast<int>(exp) - 7);
  return sign ? -v : v;
}

// Dequantize one full row of packed weights into dst (length k). This is the
// definition of record for the kernel: the vectorized path must agree with it.
//
// `scale` holds one FLOAT per group, not encoded fp8 bytes: the caller converts
// the checkpoint's e4m3 scales once at load time. Decoding e4m3 here instead
// would mean re-deriving a value the framework already has, and getting the bit
// layout wrong silently yields plausible-but-wrong scales (an early version of
// this file read float32 input as if it were e4m3 and row norms came out wrong).
static inline void dequant_row_scalar(const uint8_t* packed, const float* scale,
                                      int k, float scale2, float* dst) {
  for (int kk = 0; kk < k; ++kk) {
    const uint8_t byte = packed[kk >> 1];
    const uint8_t nib = (kk & 1) ? static_cast<uint8_t>(byte >> 4)
                                 : static_cast<uint8_t>(byte & 0x0F);
    const float g = scale[kk >> 4];
    dst[kk] = fp4_to_float(nib) * g * scale2;
  }
}

// Dense GEMM: out[m, n] = sum_k x[m, k] * W[n, k]
//
// W is supplied in its packed form, so the inner loop dequantizes on the fly
// exactly as the MoE kernel does. Blocking over N keeps the working set inside
// cache, which is the same reason the MoE kernel uses N_BLOCK/K_BLOCK.
struct DenseNVFP4HeadConfig {
  int m = 1;
  int n = 0;
  int k = 0;
  int block_n = 4096;
  const uint8_t* weight = nullptr;
  const float* weight_scale = nullptr;
  float weight_scale_2 = 1.0f;
};

static inline void dense_nvfp4_head_gemm(const DenseNVFP4HeadConfig& cfg,
                                         const float* x, int ldx, float* out,
                                         int ldo) {
  const int n = cfg.n;
  const int k = cfg.k;
  const int group = 16;
  const int khalf = k / 2;
  const int kgroups = k / group;

  std::vector<float> w_row(static_cast<size_t>(k));

  for (int nb = 0; nb < n; nb += cfg.block_n) {
    const int nend = std::min(nb + cfg.block_n, n);
    for (int nn = nb; nn < nend; ++nn) {
      const uint8_t* packed = cfg.weight + static_cast<size_t>(nn) * khalf;
      const float* scale = cfg.weight_scale + static_cast<size_t>(nn) * kgroups;
      dequant_row_scalar(packed, scale, k, cfg.weight_scale_2, w_row.data());

      for (int mm = 0; mm < cfg.m; ++mm) {
        const float* xr = x + static_cast<size_t>(mm) * ldx;
        float acc = 0.0f;
#if defined(__AVX2__)
        __m256 vacc = _mm256_setzero_ps();
        int kk = 0;
        for (; kk + 8 <= k; kk += 8) {
          const __m256 xv = _mm256_loadu_ps(xr + kk);
          const __m256 wv = _mm256_loadu_ps(w_row.data() + kk);
          vacc = _mm256_fmadd_ps(xv, wv, vacc);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, vacc);
        for (int t = 0; t < 8; ++t) acc += tmp[t];
        for (; kk < k; ++kk) acc += xr[kk] * w_row[kk];
#else
        for (int kk = 0; kk < k; ++kk) acc += xr[kk] * w_row[kk];
#endif
        out[static_cast<size_t>(mm) * ldo + nn] = acc;
      }
    }
  }
}

}  // namespace kt::nvfp4_head
