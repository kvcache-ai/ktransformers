#ifndef AMX_QUANTIZATION_HPP
#define AMX_QUANTIZATION_HPP
#include <algorithm>
#include <cmath>

#include "amx_config.hpp"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "utils.hpp"

namespace amx {

struct blocks_aligned_q4_0_ref {
  static constexpr int block_size = 64;
  static constexpr double bytes_per_element = double(sizeof(ggml_half) + double(block_size) / 2) / block_size;

  ggml_half* d;
  uint8_t* qs;

  blocks_aligned_q4_0_ref offset(size_t blck_cnt) const {
    blocks_aligned_q4_0_ref re;
    re.d = &d[blck_cnt];
    re.qs = &qs[blck_cnt * block_size / 2];
    return re;
  }

  static size_t expected_data_size(int64_t k) {
    assert(k % block_size == 0);
    return (sizeof(ggml_half) + block_size / 2) * (k / block_size);
  }

  uint8_t* get_qs(int block_idx) { return offset_pointer(qs, block_idx * (block_size / 2)); }

  static blocks_aligned_q4_0_ref quantize(const float* RESTRICT x, void* RESTRICT data, int64_t k) {
    assert(reinterpret_cast<intptr_t>(data) % 64 == 0);

    blocks_aligned_q4_0_ref re;
    re.qs = reinterpret_cast<uint8_t*>(data);
    re.d = reinterpret_cast<ggml_half*>(offset_pointer(re.qs, k / 2));

    static const int qk = block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
      float amax = 0.0f;  // absolute max
      float max = 0.0f;

      for (int j = 0; j < qk; j++) {
        const float v = x[i * qk + j];
        if (amax < fabsf(v)) {
          amax = fabsf(v);
          max = v;
        }
      }

      const float d = max / -8;
      const float id = d ? 1.0f / d : 0.0f;

      re.d[i] = GGML_FP32_TO_FP16(d);

      for (int j = 0; j < qk / 2; ++j) {
        const float x0 = x[i * qk + 0 + j] * id;
        const float x1 = x[i * qk + qk / 2 + j] * id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        re.get_qs(i)[j] = xi0;
        re.get_qs(i)[j] |= xi1 << 4;
      }
    }
    return re;
  }

  void dequantize(float* y, int64_t k) {
    static const int qk = block_size;
    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
      const float d = GGML_FP16_TO_FP32(this->d[i]);

      for (int j = 0; j < qk / 2; ++j) {
        const int x0 = (get_qs(i)[j] & 0x0F) - 8;
        const int x1 = (get_qs(i)[j] >> 4) - 8;

        y[i * qk + j + 0] = x0 * d;
        y[i * qk + j + qk / 2] = x1 * d;
      }
    }
  }
};

struct blocks_aligned_q8_0_ref {
  static constexpr int block_size = 64;
  static constexpr double bytes_per_element = double(sizeof(ggml_half) + block_size) / block_size;

  ggml_half* d;
  int8_t* qs;

  blocks_aligned_q8_0_ref offset(size_t blck_cnt) const {
    blocks_aligned_q8_0_ref re;
    re.d = &d[blck_cnt];
    re.qs = &qs[blck_cnt * block_size];
    return re;
  }

  static size_t expected_data_size(int64_t k) {
    assert(k % block_size == 0);
    return (sizeof(ggml_half) + block_size) * (k / block_size);
  }
  int8_t* get_qs(int block_idx) { return offset_pointer(qs, block_idx * block_size); }

  static blocks_aligned_q8_0_ref quantize(const float* RESTRICT x, void* RESTRICT data, int64_t k) {
    assert(k % block_size == 0);
    assert(reinterpret_cast<intptr_t>(data) % 64 == 0);

    blocks_aligned_q8_0_ref re;
    re.qs = reinterpret_cast<int8_t*>(data);
    re.d = reinterpret_cast<ggml_half*>(offset_pointer(re.qs, k));
    const int nb = k / block_size;

    for (int i = 0; i < nb; i++) {
      float amax = 0.0f;  // absolute max

      for (int j = 0; j < block_size; j++) {
        const float v = x[i * block_size + j];
        amax = MAX(amax, fabsf(v));
      }

      const float d = amax / ((1 << 7) - 1);
      const float id = d ? 1.0f / d : 0.0f;

      re.d[i] = GGML_FP32_TO_FP16(d);

      for (int j = 0; j < block_size; ++j) {
        const float x0 = x[i * block_size + j] * id;
        re.get_qs(i)[j] = roundf(x0);
      }
    }
    return re;
  }

  void dequantize(float* y, int64_t k) {
    static const int qk = block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
      const float d = GGML_FP16_TO_FP32(this->d[i]);

      for (int j = 0; j < qk; ++j) {
        y[i * qk + j] = get_qs(i)[j] * d;
      }
    }
  }
};

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

template <typename Block>
struct Dequantizer {};

const __m256i MASK256_LO = _mm256_set1_epi8(0x0f);
const __m256i MASK256_4HI = _mm256_set1_epi8(0xf0);
const __m256i MASK256_8 = _mm256_set1_epi8(8);

const __m512i MASK512_LO = _mm512_set1_epi8(0x0f);
const __m512i MASK512_4HI = _mm512_set1_epi8(0xf0);
const __m512i MASK512_8 = _mm512_set1_epi8(8);

inline __m256i dequant4x32(const uint8_t* qs) {
  const __m128i aux128 = _mm_loadu_si128((const __m128i*)qs);
  return _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128), MASK256_LO);
}

inline __m256i unaligned_copy8x32(const int8_t* qs) { return _mm256_loadu_si256((const __m256i*)qs); }

inline __m512i copy8x64(const int8_t* qs) { return _mm512_load_si512((const __m512i*)qs); }

inline __m256i lo4bit(const uint8_t* qs) {
  return _mm256_and_si256(_mm256_loadu_si256((const __m256i*)qs), MASK256_LO);
}
inline __m256i hi4bit(const uint8_t* qs) {
  return _mm256_srli_epi16(_mm256_and_si256(_mm256_loadu_si256((const __m256i*)qs), MASK256_4HI), 4);
}

inline __m128i make_q4K_scale_and_min(const uint8_t* scales8) {
  __m128i re;
  uint32_t* aux32 = (uint32_t*)&re;
  const uint16_t* scales = (const uint16_t*)scales8;
  const uint32_t a0 = scales[0] | (scales[1] << 16);
  const uint32_t a1 = scales[2] | (scales[3] << 16);
  const uint32_t a2 = scales[4] | (scales[5] << 16);
  aux32[3] = ((a2 >> 4) & 0x0f0f0f0f) | ((a1 >> 2) & 0x30303030);
  aux32[1] = ((a2 >> 0) & 0x0f0f0f0f) | ((a0 >> 2) & 0x30303030);
  aux32[2] = a1 & 0x3f3f3f3f;
  aux32[0] = a0 & 0x3f3f3f3f;
  // aux32[1:0] is scale
  // aux32[3:2] is min
  return re;
}

inline __m256i merge_q8K_bsum(block_q8_K* b) {
  return _mm256_madd_epi16(_mm256_loadu_si256((__m256i*)b->bsums), _mm256_set1_epi16(1));
}

inline __m512i _mm512_dpbssd_epi32(__m512i src, __m512i a, __m512i b) {
  // 提取高低256-bit部分
  __m256i a_lo = _mm512_extracti64x4_epi64(a, 0);
  __m256i a_hi = _mm512_extracti64x4_epi64(a, 1);
  __m256i b_lo = _mm512_extracti64x4_epi64(b, 0);
  __m256i b_hi = _mm512_extracti64x4_epi64(b, 1);

  // 根据a的符号调整b的符号
  b_lo = _mm256_sign_epi8(b_lo, a_lo);
  b_hi = _mm256_sign_epi8(b_hi, a_hi);

  // 将修改后的b重新组合
  b = _mm512_inserti64x4(b, b_lo, 0);
  b = _mm512_inserti64x4(b, b_hi, 1);

  // 取绝对值
  a = _mm512_abs_epi8(a);

  // 进行dot-product计算
  return _mm512_dpbusd_epi32(src, a, b);
}

}  // namespace amx

#endif  // AMX_QUANTIZATION_HPP