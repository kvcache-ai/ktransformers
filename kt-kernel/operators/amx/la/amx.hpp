#ifndef AMX_HPP
#define AMX_HPP
#include <emmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <tmmintrin.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <stdexcept>

#include "llama.cpp/ggml-quants.h"

// Include the split AMX headers
#include "amx_config.hpp"
#include "amx_kernels.hpp"

namespace amx {

static inline __m512 exp_avx512(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(1.44269504089f);
  const __m512 c1 = _mm512_set1_ps(0.69314718056f);

  __m512 y = _mm512_mul_ps(x, log2e);
  __m512i int_part = _mm512_cvtps_epi32(y);
  __m512 frac_part = _mm512_sub_ps(y, _mm512_cvtepi32_ps(int_part));

  const __m512 poly_1 = _mm512_set1_ps(0.9999999995f);
  const __m512 poly_2 = _mm512_set1_ps(0.6931471805f);
  const __m512 poly_3 = _mm512_set1_ps(0.2402265069f);
  const __m512 poly_4 = _mm512_set1_ps(0.0555041087f);
  const __m512 poly_5 = _mm512_set1_ps(0.0096181291f);
  const __m512 poly_6 = _mm512_set1_ps(0.0013333558f);

  __m512 frac_exp = _mm512_fmadd_ps(
      frac_part, poly_6,
      _mm512_fmadd_ps(frac_part, poly_5,
                      _mm512_fmadd_ps(frac_part, poly_4,
                                      _mm512_fmadd_ps(frac_part, poly_3, _mm512_fmadd_ps(frac_part, poly_2, poly_1)))));

  __m512 two_pow_i = _mm512_scalef_ps(_mm512_set1_ps(1.0f), _mm512_cvtepi32_ps(int_part));
  return _mm512_mul_ps(two_pow_i, frac_exp);
}

static inline __m512 act_fn(__m512 gate_val, __m512 up_val) {
  __m512 neg_gate_val = _mm512_sub_ps(_mm512_setzero_ps(), gate_val);
  // Clamp neg_gate_val to avoid exp overflow (exp(88) overflows for float32)
  const __m512 max_exp_input = _mm512_set1_ps(88.0f);
  neg_gate_val = _mm512_min_ps(neg_gate_val, max_exp_input);
  __m512 exp_neg_gate = exp_avx512(neg_gate_val);
  __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_gate);
  __m512 act_val = _mm512_div_ps(gate_val, denom);

  return _mm512_mul_ps(act_val, up_val);
}

#define AMX_DISPATCH_QTYPES(QA, QB, ...)                                 \
  [&] {                                                                  \
    switch (QB) {                                                        \
      case GGML_TYPE_Q8_0: {                                             \
        using qb = block_q8_0;                                           \
        switch (QA) {                                                    \
          case GGML_TYPE_Q4_0: {                                         \
            using qa = block_q4_0;                                       \
            return __VA_ARGS__();                                        \
          }                                                              \
          case GGML_TYPE_Q8_0: {                                         \
            using qa = block_q8_0;                                       \
            return __VA_ARGS__();                                        \
          }                                                              \
          default:                                                       \
            throw std::runtime_error("Unsupported quantized data type"); \
        }                                                                \
      }                                                                  \
      case GGML_TYPE_Q8_K: {                                             \
        using qb = block_q8_K;                                           \
        switch (QA) {                                                    \
          case GGML_TYPE_Q4_K: {                                         \
            using qa = block_q4_K;                                       \
            return __VA_ARGS__();                                        \
          }                                                              \
          default:                                                       \
            throw std::runtime_error("Unsupported quantized data type"); \
        }                                                                \
      }                                                                  \
      case GGML_TYPE_BF16: {                                             \
        using qb = ggml_bf16_t;                                          \
        switch (QA) {                                                    \
          case GGML_TYPE_BF16: {                                         \
            using qa = ggml_bf16_t;                                      \
            return __VA_ARGS__();                                        \
          }                                                              \
          default:                                                       \
            throw std::runtime_error("Unsupported quantized data type"); \
        }                                                                \
      }                                                                  \
      default:                                                           \
        throw std::runtime_error("Unsupported quantized data type");     \
    }                                                                    \
  }()

inline void gemm(int m, int n, int k, const void* a, size_t lda, int type_a, const void* b, size_t ldb, int type_b,
                 void* c, size_t ldc, int type_c, int ith, int nth) {
  assert(reinterpret_cast<intptr_t>(c) % 64 == 0);
  assert(ldc % 64 == 0);
  assert(type_c == GGML_TYPE_F32);
  float* cs = (float*)c;
  AMX_DISPATCH_QTYPES(type_a, type_b, [&]() { mat_mul(m, n, k, (qa*)a, lda, (qb*)b, ldb, cs, ldc, ith, nth); });
}

inline void init_tile(int type_a, int type_b, int type_c) {
#ifdef HAVE_AMX
  enable_amx();
  assert(type_c == GGML_TYPE_F32);
  AMX_DISPATCH_QTYPES(type_a, type_b, []() { return GemmKernel<qa, qb, float>::type::config(); });
#endif
}

inline int recommended_nth(int m, int n, int k, int type_a, int type_b, int type_c) {
  assert(type_c == GGML_TYPE_F32);
  return AMX_DISPATCH_QTYPES(type_a, type_b, [&]() { return GemmKernel<qa, qb, float>::type::recommended_nth(m); });
}

}  // namespace amx

#endif  // AMX_HPP