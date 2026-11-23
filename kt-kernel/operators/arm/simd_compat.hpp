// SPDX-License-Identifier: Apache-2.0
#pragma once

#if !defined(__aarch64__)
#error "The ARM SFT compatibility layer requires AArch64"
#endif
#if !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error "The ARM SFT compatibility layer requires SVE and BF16"
#endif

#define SIMDE_ENABLE_NATIVE_ALIASES
#include <arm_sve.h>
#include <asm/hwcap.h>
#include <simde/x86/avx512.h>
#include <simde/x86/f16c.h>
#include <simde/x86/fma.h>
#include <sys/auxv.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifndef _MM_FROUND_NO_EXC
#define _MM_FROUND_NO_EXC 8
#endif

namespace kt::arm {

inline void require_sve_bf16() {
  if (!(getauxval(AT_HWCAP) & HWCAP_SVE) || !(getauxval(AT_HWCAP2) & HWCAP2_SVEBF16)) {
    throw std::runtime_error("ARM SFT requires runtime SVE BF16 support");
  }
}

inline uint16_t to_bf16_rne(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000u) == 0) return static_cast<uint16_t>((bits >> 16) & 0x8000u);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline svfloat32_t flush_fp32(svfloat32_t value) {
  const auto bits = svreinterpret_u32_f32(value);
  const auto exponent = svand_n_u32_x(svptrue_b32(), bits, 0x7f800000u);
  return svreinterpret_f32_u32(
      svsel_u32(svcmpeq_n_u32(svptrue_b32(), exponent, 0), svand_n_u32_x(svptrue_b32(), bits, 0x80000000u), bits));
}

inline svbfloat16_t flush_bf16(svbfloat16_t value) {
  const auto bits = svreinterpret_u16_bf16(value);
  const auto exponent = svand_n_u16_x(svptrue_b16(), bits, 0x7f80u);
  return svreinterpret_bf16_u16(
      svsel_u16(svcmpeq_n_u16(svptrue_b16(), exponent, 0), svand_n_u16_x(svptrue_b16(), bits, 0x8000u), bits));
}

// VDPBF16PS accumulates the high product before the low product. BFDOT has
// different rounding semantics; widening BFMLAL preserves the two FMA steps.
inline __m512 dpbf16(__m512 source, __m512bh a, __m512bh b) {
  auto result = simde__m512_to_private(source);
  const auto ap = simde__m512bh_to_private(a);
  const auto bp = simde__m512bh_to_private(b);
  const auto* aptr = reinterpret_cast<const uint16_t*>(&ap);
  const auto* bptr = reinterpret_cast<const uint16_t*>(&bp);
  auto* rptr = reinterpret_cast<float*>(&result);
  const uint64_t lanes = svcntw();
  for (uint64_t i = 0; i < 16; i += lanes) {
    const auto pg = svwhilelt_b32(i, uint64_t(16));
    const auto hp = svwhilelt_b16(i * 2, uint64_t(32));
    const auto av = flush_bf16(svreinterpret_bf16_u16(svld1_u16(hp, aptr + i * 2)));
    const auto bv = flush_bf16(svreinterpret_bf16_u16(svld1_u16(hp, bptr + i * 2)));
    auto acc = flush_fp32(svld1_f32(pg, rptr + i));
    acc = flush_fp32(svbfmlalt_f32(acc, av, bv));
    acc = flush_fp32(svbfmlalb_f32(acc, av, bv));
    svst1_f32(pg, rptr + i, acc);
  }
  return simde__m512_from_private(result);
}

// A GEMM tile contains only BF16 dot accumulation. Use the thread's FZ/RNE
// controls for that scope, then restore both control and exception status.
class ScopedBF16FZ {
 public:
  ScopedBF16FZ() {
    asm volatile("mrs %0, fpcr" : "=r"(control_));
    asm volatile("mrs %0, fpsr" : "=r"(status_));
    constexpr uint64_t trap_mask = (uint64_t(0x1f) << 8) | (uint64_t(1) << 15);
    const uint64_t control = (control_ & ~((uint64_t(3) << 22) | trap_mask)) | (uint64_t(1) << 24);
    asm volatile("msr fpcr, %0" : : "r"(control) : "memory");
  }
  ~ScopedBF16FZ() {
    asm volatile("msr fpsr, %0" : : "r"(status_) : "memory");
    asm volatile("msr fpcr, %0" : : "r"(control_) : "memory");
  }
  ScopedBF16FZ(const ScopedBF16FZ&) = delete;
  ScopedBF16FZ& operator=(const ScopedBF16FZ&) = delete;

 private:
  uint64_t control_;
  uint64_t status_;
};

inline __m256bh cvt16(__m512 source) {
  auto input = simde__m512_to_private(source);
  simde__m256bh_private result;
  for (int i = 0; i < 16; ++i) result.u16[i] = to_bf16_rne(input.f32[i]);
  return simde__m256bh_from_private(result);
}

inline __m512bh cvt32(__m512 high, __m512 low) {
  auto hi = simde__m512_to_private(high);
  auto lo = simde__m512_to_private(low);
  simde__m512bh_private result;
  for (int i = 0; i < 16; ++i) {
    result.u16[i] = to_bf16_rne(lo.f32[i]);
    result.u16[16 + i] = to_bf16_rne(hi.f32[i]);
  }
  return simde__m512bh_from_private(result);
}

inline __m128bh cvt8(__m256 source) {
  auto input = simde__m256_to_private(source);
  simde__m128bh_private result;
  for (int i = 0; i < 8; ++i) result.u16[i] = to_bf16_rne(input.f32[i]);
  return simde__m128bh_from_private(result);
}

inline float reduce_add(__m512 source) {
  auto input = simde__m512_to_private(source);
  for (int gap = 8; gap != 0; gap /= 2) {
    for (int i = 0; i < gap; ++i) input.f32[i] = input.f32[i] + input.f32[i + gap];
  }
  return input.f32[0];
}

inline __m512i cvt_unsigned(__m512 source, int rounding = 4) {
  auto input = simde__m512_to_private(source);
  simde__m512i_private result;
  for (int i = 0; i < 16; ++i) {
    const double value = input.f32[i];
    double rounded;
    switch (rounding & 7) {
      case 1:
        rounded = std::floor(value);
        break;
      case 2:
        rounded = std::ceil(value);
        break;
      case 3:
        rounded = std::trunc(value);
        break;
      default:
        rounded = std::nearbyint(value);
        break;
    }
    result.u32[i] =
        std::isfinite(rounded) && rounded >= 0 && rounded <= 4294967295.0 ? static_cast<uint32_t>(rounded) : UINT32_MAX;
  }
  return simde__m512i_from_private(result);
}

inline __m128i narrow_unsigned8(__m512i source) {
  auto input = simde__m512i_to_private(source);
  simde__m128i_private result;
  for (int i = 0; i < 16; ++i) result.u8[i] = static_cast<uint8_t>(std::min(input.u32[i], uint32_t(255)));
  return simde__m128i_from_private(result);
}

inline __m512i gather32(__m512i indices, const void* base, int scale) {
  const auto input = simde__m512i_to_private(indices);
  simde__m512i_private result;
  auto* bytes = static_cast<const char*>(base);
  for (int i = 0; i < 16; ++i) {
    int32_t value;
    std::memcpy(&value, bytes + static_cast<int64_t>(input.i32[i]) * scale, sizeof(value));
    result.i32[i] = value;
  }
  return simde__m512i_from_private(result);
}

[[noreturn]] inline void unsupported_amx_tile() {
  throw std::logic_error("AMX tile execution is unavailable in the ARM SFT backend");
}
}  // namespace kt::arm

#undef _mm512_dpbf16_ps
#define _mm512_dpbf16_ps(source, a, b) kt::arm::dpbf16((source), (a), (b))
#define _mm512_cvtneps_pbh(source) kt::arm::cvt16((source))
#define _mm512_cvtne2ps_pbh(high, low) kt::arm::cvt32((high), (low))
#define _mm256_cvtneps_pbh(source) kt::arm::cvt8((source))
#undef _mm512_reduce_add_ps
#define _mm512_reduce_add_ps(source) kt::arm::reduce_add((source))
#define _mm512_cvtps_epu32(source) kt::arm::cvt_unsigned((source))
#define _mm512_cvt_roundps_epu32(source, rounding) kt::arm::cvt_unsigned((source), (rounding))
#define _mm512_cvtusepi32_epi8(source) kt::arm::narrow_unsigned8((source))
#define _mm512_i32gather_epi32(indices, base, scale) kt::arm::gather32((indices), (base), (scale))
#define _mm512_stream_si512(destination, source) simde_mm512_storeu_si512((destination), (source))

// These names occur in parsed, unselected AMX helpers. Accidental tile dispatch
// must fail explicitly; the ARM backend uses only the SIMD branch.
#define _tile_dpbssd(...) kt::arm::unsupported_amx_tile()
#define _tile_dpbsud(...) kt::arm::unsupported_amx_tile()
#define _tile_dpbusd(...) kt::arm::unsupported_amx_tile()
#define _tile_dpbuud(...) kt::arm::unsupported_amx_tile()
#define _tile_dpbf16ps(...) kt::arm::unsupported_amx_tile()
#define _tile_loadd(...) kt::arm::unsupported_amx_tile()
#define _tile_stored(...) kt::arm::unsupported_amx_tile()
#define _tile_zero(...) kt::arm::unsupported_amx_tile()
#define _tile_loadconfig(...) kt::arm::unsupported_amx_tile()
#define _tile_release(...) kt::arm::unsupported_amx_tile()
