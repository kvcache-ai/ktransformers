#ifndef AMX_BF16_DWEIGHT_HPP
#define AMX_BF16_DWEIGHT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "amx_kernels.hpp"
#include "amx_raw_kernels.hpp"

namespace amx {

struct BF16DWeightTimings {
  uint64_t pack_a_ns = 0;
  uint64_t pack_a_calls = 0;
  uint64_t pack_b_ns = 0;
  uint64_t pack_b_calls = 0;
  uint64_t panel_input_ns = 0;
  uint64_t panel_input_calls = 0;
  uint64_t panel_grad_output_ns = 0;
  uint64_t panel_grad_output_calls = 0;
  uint64_t kernel_gate_up_ns = 0;
  uint64_t kernel_gate_up_calls = 0;
  uint64_t kernel_down_ns = 0;
  uint64_t kernel_down_calls = 0;
  uint64_t store_ns = 0;
  uint64_t store_calls = 0;

  void reset() { *this = {}; }
};

inline BF16DWeightTimings& bf16_dweight_timings() {
  static thread_local BF16DWeightTimings timings;
  return timings;
}

class BF16DWeightScratch {
 public:
  using Kernel = GemmKernel224BF16;
  static constexpr int M_STEP = Kernel::M_STEP;
  static constexpr int N_STEP = Kernel::N_STEP;

  BF16DWeightScratch() = default;
  BF16DWeightScratch(const BF16DWeightScratch&) = delete;
  BF16DWeightScratch& operator=(const BF16DWeightScratch&) = delete;

  ~BF16DWeightScratch() {
    std::free(a0_);
    std::free(a1_);
    std::free(b_);
  }

  void ensure(int padded_k) {
    if (padded_k <= capacity_k_) return;
    const size_t a_elements = static_cast<size_t>(M_STEP) * padded_k;
    const size_t b_elements = static_cast<size_t>(N_STEP) * padded_k;
    resize(a0_, a_elements);
    resize(a1_, a_elements);
    resize(b_, b_elements);
    capacity_k_ = padded_k;
  }

  ggml_bf16_t* a0() { return a0_; }
  ggml_bf16_t* a1() { return a1_; }
  ggml_bf16_t* b() { return b_; }
  float* c0() { return c0_; }
  float* c1() { return c1_; }

 private:
  static void resize(ggml_bf16_t*& buffer, size_t elements) {
    void* replacement = nullptr;
    if (posix_memalign(&replacement, 64, elements * sizeof(ggml_bf16_t)) != 0 || replacement == nullptr) {
      throw std::runtime_error("failed to allocate BF16 dWeight scratch");
    }
    std::free(buffer);
    buffer = static_cast<ggml_bf16_t*>(replacement);
  }

  int capacity_k_ = 0;
  ggml_bf16_t* a0_ = nullptr;
  ggml_bf16_t* a1_ = nullptr;
  ggml_bf16_t* b_ = nullptr;
  alignas(64) float c0_[M_STEP * N_STEP];
  alignas(64) float c1_[M_STEP * N_STEP];
};

inline BF16DWeightScratch& bf16_dweight_scratch() {
  static thread_local BF16DWeightScratch scratch;
  return scratch;
}

class BF16DWeightKernel {
 public:
  using Kernel = GemmKernel224BF16;
  using BufferA = Kernel::BufferA;
  using BufferB = Kernel::BufferB;
  static constexpr int M_STEP = Kernel::M_STEP;
  static constexpr int N_STEP = Kernel::N_STEP;
  static constexpr int K_STEP = Kernel::K_STEP;

  static int padded_k(int routes) { return std::max(K_STEP, (routes + K_STEP - 1) / K_STEP * K_STEP); }

  static void configure_worker() { Kernel::config(); }

  static void pack_a_transposed(BufferA& destination, const ggml_bf16_t* source, int source_stride, int source_column,
                                int row_count, int routes, int destination_row = 0) {
    assert(destination_row >= 0 && destination_row + row_count <= destination.max_m);
    assert(destination_row % M_STEP == 0 && row_count <= M_STEP);
    const int k = destination.k;
    for (int k_begin = 0; k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* tile = destination.get_submat(destination.max_m, k, destination_row, k_begin);
      std::memset(tile, 0, M_STEP * K_STEP * sizeof(ggml_bf16_t));
      const int valid_k = std::min(K_STEP, routes - k_begin);
      if (valid_k <= 0) continue;
      for (int row = 0; row < row_count; ++row) {
        for (int kk = 0; kk < valid_k; ++kk) {
          tile[row * K_STEP + kk] = source[static_cast<size_t>(k_begin + kk) * source_stride + source_column + row];
        }
      }
    }
  }

  static void pack_b_transposed(BufferB& destination, const ggml_bf16_t* source, int source_stride, int source_column,
                                int row_count, int routes, int destination_row = 0) {
    assert(destination_row >= 0 && destination_row + row_count <= destination.n);
    assert(destination_row % N_STEP == 0 && row_count <= N_STEP);
    const int k = destination.k;
    for (int k_begin = 0; k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* tile = destination.get_submat(destination.n, k, destination_row, k_begin);
      std::memset(tile, 0, N_STEP * K_STEP * sizeof(ggml_bf16_t));
      const int valid_k = std::min(K_STEP, routes - k_begin);
      if (valid_k > 0) {
        for (int row = 0; row < row_count; ++row) {
          for (int kk = 0; kk < valid_k; ++kk) {
            tile[row * K_STEP + kk] = source[static_cast<size_t>(k_begin + kk) * source_stride + source_column + row];
          }
        }
      }
      transpose_16x16_32bit(reinterpret_cast<__m512i*>(tile));
      transpose_16x16_32bit(reinterpret_cast<__m512i*>(tile + Kernel::TILE_N * K_STEP));
    }
  }

 private:
  template <int ROWS>
  static inline __attribute__((always_inline)) void multiply_avx_rows(
      int padded_k, float* destination, BufferA& a, BufferB& b, int m_begin, int n_begin, int row_offset) {
    static_assert(ROWS > 0 && ROWS <= M_STEP);
    __m512 accum_lo[ROWS];
    __m512 accum_hi[ROWS];

#pragma GCC unroll 12
    for (int row = 0; row < ROWS; ++row) {
      accum_lo[row] = _mm512_setzero_ps();
      accum_hi[row] = _mm512_setzero_ps();
    }

    for (int k_begin = 0; k_begin < padded_k; k_begin += K_STEP) {
      const auto* a_pairs = reinterpret_cast<const int32_t*>(a.get_submat(a.max_m, padded_k, m_begin, k_begin));
      const auto* b_vectors = reinterpret_cast<const __m512bh*>(b.get_submat(b.n, padded_k, n_begin, k_begin));

      for (int k_pair = 0; k_pair < K_STEP / 2; ++k_pair) {
        const __m512bh b_lo = b_vectors[k_pair];
        const __m512bh b_hi = b_vectors[Kernel::TILE_N + k_pair];
#pragma GCC unroll 12
        for (int row = 0; row < ROWS; ++row) {
          const __m512bh a_pair =
              reinterpret_cast<__m512bh>(_mm512_set1_epi32(a_pairs[(row_offset + row) * (K_STEP / 2) + k_pair]));
          accum_lo[row] = _mm512_dpbf16_ps(accum_lo[row], a_pair, b_lo);
          accum_hi[row] = _mm512_dpbf16_ps(accum_hi[row], a_pair, b_hi);
        }
      }
    }

#pragma GCC unroll 12
    for (int row = 0; row < ROWS; ++row) {
      _mm512_store_ps(destination + (row_offset + row) * N_STEP, accum_lo[row]);
      _mm512_store_ps(destination + (row_offset + row) * N_STEP + Kernel::TILE_N, accum_hi[row]);
    }
  }

 public:

  static void multiply(int padded_k, float* destination, BufferA& a, BufferB& b, int m_begin = 0, int n_begin = 0) {
    assert(m_begin >= 0 && m_begin + M_STEP <= a.max_m);
    assert(n_begin >= 0 && n_begin + N_STEP <= b.n);
    assert(m_begin % M_STEP == 0 && n_begin % N_STEP == 0);
    if constexpr (AMX_AVAILABLE) {
      for (int k_block_begin = 0; k_block_begin < padded_k; k_block_begin += Kernel::K_BLOCK) {
        Kernel::amx_kernel(a.max_m, b.n, padded_k, m_begin, n_begin, k_block_begin, destination, &a, &b);
      }
    } else {
      // Keep the complete 32x32 FP32 output tile in registers. The generic forward
      // kernel spans all 32 rows at once and spills its 64 accumulators on AVX512.
      multiply_avx_rows<12>(padded_k, destination, a, b, m_begin, n_begin, 0);
      multiply_avx_rows<12>(padded_k, destination, a, b, m_begin, n_begin, 12);
      multiply_avx_rows<8>(padded_k, destination, a, b, m_begin, n_begin, 24);
    }
  }

  static void store_bf16(const float* source, ggml_bf16_t* destination, int destination_stride, int row_count,
                         int column_count, bool accumulate = false, float scale = 1.0f) {
    const __m512 scale_vector = _mm512_set1_ps(scale);
    for (int row = 0; row < row_count; ++row) {
      const float* src_row = source + row * N_STEP;
      ggml_bf16_t* dst_row = destination + static_cast<size_t>(row) * destination_stride;
      if (column_count == N_STEP) {
        __m512 lo = _mm512_loadu_ps(src_row);
        __m512 hi = _mm512_loadu_ps(src_row + 16);
        lo = _mm512_mul_ps(lo, scale_vector);
        hi = _mm512_mul_ps(hi, scale_vector);
        if (accumulate) {
          __m512 old_lo;
          __m512 old_hi;
          avx512_32xbf16_to_32xfp32(reinterpret_cast<__m512i*>(dst_row), &old_lo, &old_hi);
          lo = _mm512_add_ps(lo, old_lo);
          hi = _mm512_add_ps(hi, old_hi);
        }
        avx512_32xfp32_to_32xbf16(&lo, &hi, reinterpret_cast<__m512i*>(dst_row));
      } else {
        for (int column = 0; column < column_count; ++column) {
          float value = src_row[column] * scale;
          if (accumulate) value += GGML_BF16_TO_FP32(dst_row[column]);
          dst_row[column] = GGML_FP32_TO_BF16(value);
        }
      }
    }
  }
};

}  // namespace amx

#endif  // AMX_BF16_DWEIGHT_HPP
