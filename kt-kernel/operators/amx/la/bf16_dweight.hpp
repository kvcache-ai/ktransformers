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
                                int row_count, int routes) {
    const int k = destination.k;
    for (int k_begin = 0; k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* tile = destination.get_submat(M_STEP, k, 0, k_begin);
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
                                int row_count, int routes) {
    const int k = destination.k;
    for (int k_begin = 0; k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* tile = destination.get_submat(N_STEP, k, 0, k_begin);
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

  static void multiply(int padded_k, float* destination, BufferA& a, BufferB& b) {
    for (int k_block_begin = 0; k_block_begin < padded_k; k_block_begin += Kernel::K_BLOCK) {
      if constexpr (AMX_AVAILABLE) {
        Kernel::amx_kernel(M_STEP, N_STEP, padded_k, 0, 0, k_block_begin, destination, &a, &b);
      } else {
        Kernel::avx_kernel_4(M_STEP, N_STEP, padded_k, 0, 0, k_block_begin, destination, &a, &b);
      }
    }
  }

  static void store_bf16(const float* source, ggml_bf16_t* destination, int destination_stride, int row_count,
                         int column_count) {
    for (int row = 0; row < row_count; ++row) {
      const float* src_row = source + row * N_STEP;
      ggml_bf16_t* dst_row = destination + static_cast<size_t>(row) * destination_stride;
      if (column_count == N_STEP) {
        __m512 lo = _mm512_loadu_ps(src_row);
        __m512 hi = _mm512_loadu_ps(src_row + 16);
        avx512_32xfp32_to_32xbf16(&lo, &hi, reinterpret_cast<__m512i*>(dst_row));
      } else {
        for (int column = 0; column < column_count; ++column) {
          dst_row[column] = GGML_FP32_TO_BF16(src_row[column]);
        }
      }
    }
  }
};

}  // namespace amx

#endif  // AMX_BF16_DWEIGHT_HPP
