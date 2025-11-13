#ifndef CPUINFER_OPERATOR_KERNEL_LA_HPP
#define CPUINFER_OPERATOR_KERNEL_LA_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "../api/common.h"
#include "../mat_kernel/batch_gemm_api.hpp"
#include "llama.cpp/ggml.h"
static const size_t MAX_Nth_B = 1024, MAX_N_B = 1024, MAX_K_B = 10240;
namespace moe_kernel {
template <typename T>
T *offset_pointer(T *ptr, size_t byte_offset) {
  return reinterpret_cast<T *>(reinterpret_cast<char *>(ptr) + byte_offset);
}

inline float bf16_to_fp32(ggml_bf16_t src) {
  // 将 bfloat16 的 16 位移到 float32 的高 16 位，低 16 位填充 0
  uint16_t *src_16 = reinterpret_cast<uint16_t *>(&src);
  uint32_t packed = (uint32_t)*src_16 << 16;

  // 使用 union 将 uint32 解释为 float
  union {
    uint32_t u;
    float f;
  } converter;

  converter.u = packed;
  return converter.f;
}

inline float fp16_to_fp32(ggml_fp16_t src) { return ggml_fp16_to_fp32(src); }

template <typename K>
struct BufferAImpl {
  int8_t *a;
  float *d;
  int max_m, k;
  bool if_pack = false;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  // K_BLOCK is runtime-configurable via kernel tiling; expose as function to avoid constexpr requirements
  static inline int K_BLOCK() { return K::K_BLOCK; }
  static constexpr int PACK_SIZE_M = K::PACK_SIZE_M;
  static constexpr int PACK_SIZE_K = K::PACK_SIZE_K;

  static size_t required_size(int max_m, int k) { return sizeof(int8_t) * max_m * k + sizeof(float) * max_m; }

  BufferAImpl(int max_m, int k, void *ptr, bool if_pack = false) : max_m(max_m), k(k), if_pack(if_pack) {
    set_data(ptr);
  }

  BufferAImpl(int max_m, int k, bool if_pack = false) : max_m(max_m), k(k), if_pack(if_pack) {
    if (max_m % M_STEP != 0 || k % K_STEP != 0) {
      throw std::runtime_error("max_m and k must be multiples of M_STEP and K_STEP respectively");
    }
  }

  void set_data(void *ptr) {
    a = reinterpret_cast<int8_t *>(ptr);
    d = reinterpret_cast<float *>(a + max_m * k);
  }

  size_t required_size() const { return sizeof(int8_t) * max_m * k + sizeof(float) * max_m; }

  BufferAImpl<K> offset_row(size_t row_begin, size_t row_block) {
    auto buffera = BufferAImpl<K>(row_block, k, a + row_begin * k, if_pack);
    buffera.d = d + row_begin;
    return buffera;
  }

  // 将输入的 A 矩阵转换成 int8_t 的形式，
  // 这里的 A 矩阵是 m * k 的矩阵，存储在 a 中, 是行主序的 (row major)
  void from_mat(int m, ggml_bf16_t *src, int ith, int mth) {
    // printf("in A from_mat, m = %d, ith = %d, nth = %d\n", m, ith, nth);
    auto [m_start, m_end] = K::split_range_m(m, ith, mth);
    int m_block_begin = m_start;
    int m_block_size = m_end - m_block_begin;
    if (m_block_size < 0) {
      throw std::runtime_error("m_block_size is negative, this should not happen");
    }
    for (int m_begin = 0; m_begin < m_block_size; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m_block_size; i++) {
        float amax = 0;
        // TODO: 后续用 SVE 来加速
        for (int j = 0; j < k; j++) {
          // 先把 src 转换成 float
          float f = bf16_to_fp32(src[(m_block_begin + m_begin + i) * k + j]);
          f = f < 0 ? -f : f;
          if (f > amax) {
            amax = f;
          }
        }
        d[m_block_begin + m_begin + i] = amax / ((1 << 7) - 1);
        // TODO: 后续用 SVE 来加速
        // 通过这个 amax 来量化这一行
        for (int j = 0; j < k; j++) {
          // 先把 src 转换成 float
          float f = bf16_to_fp32(src[(m_block_begin + m_begin + i) * k + j]);
          if (if_pack) {
            throw std::runtime_error("Packing is deprecated in this function");
            size_t split_m = (m_begin + i) / PACK_SIZE_M;
            size_t m_idx = (m_begin + i) % PACK_SIZE_M;
            size_t split_k = j / PACK_SIZE_K;
            size_t k_idx = j % PACK_SIZE_K;
            size_t buff_idx = m_block_begin * k + split_m * PACK_SIZE_M * k + split_k * PACK_SIZE_K * PACK_SIZE_M +
                              m_idx * PACK_SIZE_K + k_idx;
            a[buff_idx] = static_cast<int8_t>(std::round(f / d[m_block_begin + m_begin + i]));
          } else {
            // 这里的 amax 是当前行的最大值
            a[(m_block_begin + m_begin + i) * k + j] =
                static_cast<int8_t>(std::round(f / d[m_block_begin + m_begin + i]));
          }
        }
      }
    }
  }

  void from_mat(int m, ggml_fp16_t *src, int ith, int mth) {
    // printf("in A from_mat, m = %d, ith = %d, nth = %d\n", m, ith, nth);
    auto [m_start, m_end] = K::split_range_m(m, ith, mth);
    int m_block_begin = m_start;
    int m_block_size = m_end - m_block_begin;
    if (m_block_size < 0) {
      throw std::runtime_error("m_block_size is negative, this should not happen");
    }
    for (int m_begin = 0; m_begin < m_block_size; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m_block_size; i++) {
        float amax = 0;
        // TODO: 后续用 SVE 来加速
        for (int j = 0; j < k; j++) {
          // 先把 src 转换成 float
          float f = fp16_to_fp32(src[(m_block_begin + m_begin + i) * k + j]);
          f = f < 0 ? -f : f;
          if (f > amax) {
            amax = f;
          }
        }
        d[m_block_begin + m_begin + i] = amax / ((1 << 7) - 1);
        // TODO: 后续用 SVE 来加速
        // 通过这个 amax 来量化这一行
        for (int j = 0; j < k; j++) {
          // 先把 src 转换成 float
          float f = fp16_to_fp32(src[(m_block_begin + m_begin + i) * k + j]);
          if (if_pack) {
            throw std::runtime_error("Packing is deprecated in this function");
            size_t split_m = (m_begin + i) / PACK_SIZE_M;
            size_t m_idx = (m_begin + i) % PACK_SIZE_M;
            size_t split_k = j / PACK_SIZE_K;
            size_t k_idx = j % PACK_SIZE_K;
            size_t buff_idx = m_block_begin * k + split_m * PACK_SIZE_M * k + split_k * PACK_SIZE_K * PACK_SIZE_M +
                              m_idx * PACK_SIZE_K + k_idx;
            a[buff_idx] = static_cast<int8_t>(std::round(f / d[m_block_begin + m_begin + i]));
          } else {
            // 这里的 amax 是当前行的最大值
            a[(m_block_begin + m_begin + i) * k + j] =
                static_cast<int8_t>(std::round(f / d[m_block_begin + m_begin + i]));
          }
        }
      }
    }
  }

  // 这里是针对 gate_output 作为 fp32 的形式，量化到 int8_t 的形式
  // 这里的 A 矩阵是 m * n (intermediate_size) 的矩阵，存储在 a 中, 是行主序的 (row major)
  void from_mat(int m, float *src, int ith, int mth) {
    assert(m <= max_m);
    // assert(ith == 0 && nth == 1);
    auto [m_start, m_end] = K::split_range_m(m, ith, mth);
    int m_block_begin = m_start;
    int m_block_size = m_end - m_block_begin;
    for (int m_begin = 0; m_begin < m_block_size; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m_block_size; i++) {
        float amax = 0;
        // TODO: 后续用 SVE 来加速
        for (int j = 0; j < k; j++) {
          // 先把 src 转换成 float
          float f = src[(m_block_begin + m_begin + i) * k + j];
          f = f < 0 ? -f : f;
          if (f > amax) {
            amax = f;
          }
        }
        d[m_block_begin + m_begin + i] = amax / ((1 << 7) - 1);
        // TODO: 后续用 SVE 来加速
        // 通过这个 amax 来量化这一行
        for (int j = 0; j < k; j++) {
          // 先把 src 转换成 float
          float f = src[(m_block_begin + m_begin + i) * k + j];
          if (if_pack) {
            throw std::runtime_error("Packing is deprecated in this function");
            size_t split_m = (m_begin + i) / PACK_SIZE_M;
            size_t m_idx = (m_begin + i) % PACK_SIZE_M;
            size_t split_k = j / PACK_SIZE_K;
            size_t k_idx = j % PACK_SIZE_K;
            size_t buff_idx = m_block_begin * k + split_m * PACK_SIZE_M * k + split_k * PACK_SIZE_K * PACK_SIZE_M +
                              m_idx * PACK_SIZE_K + k_idx;
            a[buff_idx] = static_cast<int8_t>(std::round(f / d[m_block_begin + m_begin + i]));
          } else {
            // 这里的 amax 是当前行的最大值
            a[(m_block_begin + m_begin + i) * k + j] =
                static_cast<int8_t>(std::round(f / d[m_block_begin + m_begin + i]));
          }
        }
      }
    }
  }

  void from_mat(int m, float *src) {
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
        float amax = 0;
        // TODO: 后续用 SVE 来加速
        for (int j = 0; j < k; j++) {
          // 先把 src 转换成 float
          float f = src[(m_begin + i) * k + j];
          f = f < 0 ? -f : f;
          if (f > amax) {
            amax = f;
          }
        }
        d[m_begin + i] = amax / ((1 << 7) - 1);
        // TODO: 后续用 SVE 来加速
        // 通过这个 amax 来量化这一行
        for (int j = 0; j < k; j++) {
          // 先把 src 转换成 float
          float f = src[(m_begin + i) * k + j];
          // 这里的 amax 是当前行的最大值
          a[(m_begin + i) * k + j] = static_cast<int8_t>(std::round(f / d[m_begin + i]));
        }
      }
    }
  }

  // 反量化
  void to_mat(int m, float *dst, int ith, int mth) {
    auto [m_start, m_end] = K::split_range_m(m, ith, mth);
    int m_block_begin = m_start;
    int m_block_size = m_end - m_block_begin;
    for (int m_begin = 0; m_begin < m_block_size; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m_block_size; i++) {
        for (int j = 0; j < k; j++) {
          float f = static_cast<float>(a[(m_block_begin + m_begin + i) * k + j]);
          f *= d[m_block_begin + m_begin + i];
          dst[(m_block_begin + m_begin + i) * k + j] = f;
        }
      }
    }
  }

  float *get_scale(int m, int m_begin) { return d + m_begin; }
};

template <typename K>
struct BufferCImpl {
  int32_t *c;
  int max_m, n;
  bool if_row_major;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int N_STEP = K::N_STEP;
  // N_BLOCK is runtime-configurable via kernel tiling; expose as function to avoid constexpr requirements
  static inline int N_BLOCK() { return K::N_BLOCK; }

  static size_t required_size(int max_m, int n) { return sizeof(int32_t) * max_m * n; }

  BufferCImpl(int max_m, int n, void *ptr, bool if_row_major = false) : max_m(max_m), n(n), if_row_major(if_row_major) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    c = reinterpret_cast<int *>(ptr);
  }

  BufferCImpl(int max_m, int n, bool if_row_major = false) : max_m(max_m), n(n), if_row_major(if_row_major) {}

  void set_data(void *ptr) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    c = reinterpret_cast<int32_t *>(ptr);
  }
  size_t required_size() const { return sizeof(int32_t) * max_m * n; }

  // void to_mat(int m, float **dst, int ith, int nth) {
  //   *dst = c + ith * N_BLOCK;
  // }
};

struct GemmKernelInt8 {
  using dt = int8_t;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 1;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  // static const int M_STEP = TILE_M * 2;
  // static const int N_STEP = TILE_N * 2;
  // static const int K_STEP = TILE_K;
  static const int M_STEP = 1;
  static const int N_STEP = 1;
  static const int K_STEP = 1;

  // static inline const int N_BLOCK = 1024;
  // Make tiling params runtime-configurable (modifiable via Python bindings)
  static inline int N_BLOCK_UP_GATE = 32;
  static inline int N_BLOCK_DOWN = 64;
  static inline int N_BLOCK_UP_GATE_PREFI = 32;
  static inline int N_BLOCK_DOWN_PREFI = 64;
  static inline int N_BLOCK = 64;
  static inline int M_BLOCK = 320;
  // static inline const int N_BLOCK = 32;
  static inline int K_BLOCK = 7168;

  // Setter/getter for runtime tiling configuration
  static void set_tiling(int n_block_up_gate, int n_block_down, int n_block, int m_block, int k_block,
                         int n_block_up_gate_prefi, int n_block_down_prefi) {
    N_BLOCK_UP_GATE = n_block_up_gate;
    N_BLOCK_DOWN = n_block_down;
    N_BLOCK = n_block;
    M_BLOCK = m_block;
    K_BLOCK = k_block;
    N_BLOCK_UP_GATE_PREFI = n_block_up_gate_prefi;
    N_BLOCK_DOWN_PREFI = n_block_down_prefi;
  }
  static std::tuple<int, int, int, int, int, int, int> get_tiling() {
    return std::make_tuple(N_BLOCK_UP_GATE, N_BLOCK_DOWN, N_BLOCK, M_BLOCK, K_BLOCK, N_BLOCK_UP_GATE_PREFI,
                           N_BLOCK_DOWN_PREFI);
  }

  static inline const int PACK_SIZE_N = 8;
  static inline const int PACK_SIZE_M = 8;
  static inline const int PACK_SIZE_K = 32;

  static std::string name() { return "INT8"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }
  // type_: d for decode, p for prefill
  static int recommended_nth_down(int n, char type_ = 'd') {
    if (type_ == 'p') {
      if (n % N_BLOCK_DOWN_PREFI != 0) {
        throw std::invalid_argument("n must be multiple of N_BLOCK_DOWN_PREFI in prefill");
      }
      return n / N_BLOCK_DOWN_PREFI;
    } else {
      if (n % N_BLOCK_DOWN != 0) {
        throw std::invalid_argument("n must be multiple of N_BLOCK_DOWN in decode");
      }
      return n / N_BLOCK_DOWN;
    }
  }

  static int recommended_nth_up_gate(int n, char type_ = 'd') {
    if (type_ == 'p') {
      if (n % N_BLOCK_UP_GATE_PREFI != 0) {
        throw std::invalid_argument("n must be multiple of N_BLOCK_UP_GATE_PREFI in prefill");
      }
      return n / N_BLOCK_UP_GATE_PREFI;
    } else {
      if (n % N_BLOCK_UP_GATE != 0) {
        throw std::invalid_argument("n must be multiple of N_BLOCK_UP_GATE in decode");
      }
      return n / N_BLOCK_UP_GATE;
    }
  }

  static int recommended_mth(int m) { return (m + M_BLOCK - 1) / M_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth, int block_size = N_BLOCK) {
    int n_start = block_size * ith;
    int n_end = std::min(n, block_size * (ith + 1));
    return {n_start, n_end};
  }

  static std::pair<int, int> split_range_m(int m, int ith, int mth = 0) {
    int m_start = M_BLOCK * ith;
    int m_end = std::min(m, M_BLOCK * (ith + 1));
    return {m_start, m_end};
  }

  static std::pair<int, int> split_range_n_block(int n, int ith, int nth, int block) {
    int n_start = block * ith;
    int n_end = std::min(n, block * (ith + 1));
    return {n_start, n_end};
  }

  using BufferA = BufferAImpl<GemmKernelInt8>;
  using BufferC = BufferCImpl<GemmKernelInt8>;

  struct BufferB {
    int8_t *b;
    std::vector<int8_t *> b_pack;  // b_pack[i] -> the ith block (the ith packed matrix of B)
    size_t reorder_B_size;
    size_t nth_B;       // number of blocks of B
    size_t block_size;  // size of each block of B
    float *d;
    int n, k;
    static constexpr bool SCALE = true;
    bool if_pack = false;
    // n for normal, u for up_gate, d for down
    static size_t required_size(int n, int k, bool if_pack = false, char mat_type = 'n', bool plain = true) {
      int nth, n_block;
      if (if_pack && !plain) {
        switch (mat_type) {
          case 'n':
            nth = recommended_nth(n);
            n_block = N_BLOCK;
            break;
          case 'u':
            nth = recommended_nth_up_gate(n);
            n_block = N_BLOCK_UP_GATE;
            break;
          case 'd':
            nth = recommended_nth_down(n);
            n_block = N_BLOCK_DOWN;
            break;
          default:
            throw std::invalid_argument("Invalid mat_type");
        }
        size_t reorder_B_size = get_reorder_B_size(KernelCblasRowMajor, KernelCblasNoTrans, k, n_block);
        return sizeof(int8_t) * nth * reorder_B_size + sizeof(float) * n;
      } else {
        return sizeof(int8_t) * n * k + sizeof(float) * n;
      }
    }
    BufferB(int n, int k, bool if_pack = false, char mat_type = 'n', bool plain = true) : n(n), k(k), if_pack(if_pack) {
      int nth, n_block;
      if (if_pack && !plain) {
        switch (mat_type) {
          case 'n':
            nth = recommended_nth(n);
            n_block = N_BLOCK;
            break;
          case 'u':
            nth = recommended_nth_up_gate(n);
            n_block = N_BLOCK_UP_GATE;
            break;
          case 'd':
            nth = recommended_nth_down(n);
            n_block = N_BLOCK_DOWN;
            break;
          default:
            throw std::invalid_argument("Invalid mat_type");
        }
        reorder_B_size = get_reorder_B_size(KernelCblasRowMajor, KernelCblasNoTrans, k, n_block);
        nth_B = nth;
        block_size = n_block;
        b_pack.resize(nth);
      }
      if (n % N_STEP != 0 || k % K_STEP != 0) {
        throw std::runtime_error("n and k must be multiples of N_STEP and K_STEP respectively");
      }
    }
    BufferB(int n, int k, void *ptr, bool if_pack = false, char mat_type = 'n', bool plain = true)
        : BufferB(n, k, if_pack, mat_type, plain) {
      set_data(ptr, plain);
      // printf("mat_type:%c,nth_B:%zu,b_pack_ptr[0]:%p,d_ptr:%p,ptr:%p\n", mat_type, nth_B, b_pack[0], d, ptr);
    }
    void set_data(void *ptr, bool plain = true) {
      if (if_pack && !plain) {
        for (size_t i = 0; i < nth_B; i++) {
          b_pack[i] = reinterpret_cast<int8_t *>(ptr) + i * reorder_B_size;
        }
        d = reinterpret_cast<float *>((int8_t *)ptr + nth_B * reorder_B_size);
      } else {
        b = reinterpret_cast<int8_t *>(ptr);
        d = reinterpret_cast<float *>(b + n * k);
      }
    }
    size_t required_size() const { return sizeof(int8_t) * n * k + sizeof(float) * n; }
    BufferB offset_col(size_t col_begin, size_t col_block) {
      auto bufferb = BufferB(col_block, k, b + col_begin * k, if_pack);
      bufferb.d = d + col_begin;
      return bufferb;
    }
    // B 矩阵是 K * N 的矩阵，存储在 b 中, 是列主序的 (column major)
    void from_mat(ggml_bf16_t *src, int ith, int nth, int n_new = -1, bool if_pack = false,
                  bool plain = true) {  // CHECK: nth has no usage
      if (n_new > 0) {
        n = n_new;  // 如果 n_new 大于 0，则使用 n_new
      }
      // 这里将 src 转换成 int8_t 的形式，按照k 维度量化  (也就是按列量化)
      int8_t *b_t = nullptr;
      if ((if_pack || this->if_pack) && !plain) {
        b_t = (int8_t *)malloc(sizeof(int8_t) * n * k);
      }
      auto [n_start, n_end] = split_range_n(n, ith, nth, block_size);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP && n_begin + i < n_block_size; i++) {
          float amax = 0;
          // TODO: 后续用 SVE 来加速
          for (int j = 0; j < k; j++) {
            // 先把 src 转换成 float
            float f = bf16_to_fp32(src[(n_block_begin + n_begin + i) * k + j]);
            f = f < 0 ? -f : f;
            if (f > amax) {
              amax = f;
            }
          }
          d[n_block_begin + n_begin + i] = amax / ((1 << 7) - 1);
          // TODO: 后续用 SVE 来加速
          // 通过这个 amax 来量化这一列
          for (int j = 0; j < k; j++) {
            // 先把 src 转换成 float
            float f = bf16_to_fp32(src[(n_block_begin + n_begin + i) * k + j]);
            if ((if_pack || this->if_pack) && plain) {
              size_t split_n = (n_begin + i) / PACK_SIZE_N;
              size_t n_idx = (n_begin + i) % PACK_SIZE_N;
              size_t split_k = j / PACK_SIZE_K;
              size_t k_idx = j % PACK_SIZE_K;

              size_t buff_idx = n_block_begin * k + split_n * PACK_SIZE_N * k + split_k * PACK_SIZE_N * PACK_SIZE_K +
                                n_idx * PACK_SIZE_K + k_idx;
              b[buff_idx] = static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
            } else if ((if_pack || this->if_pack) && !plain) {
              // 这里的 amax 是当前列的最大值
              b_t[(n_begin + i) * k + j] = static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
            } else {
              b[(n_block_begin + n_begin + i) * k + j] =
                  static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
            }
          }
        }
      }
      if ((if_pack || this->if_pack) && !plain) {
        // 在这里调用 AMD 的reorder函数
        reorder_B_gemm(KernelCblasColMajor, KernelCblasNoTrans, k, n_block_size, k, b_t, b_pack[ith]);
        free(b_t);
      }
    }

    void from_mat(float *src, int ith, int nth, int n_new = -1, bool if_pack = false) {  // CHECK: nth has no usage
      if (n_new > 0) {
        n = n_new;  // 如果 n_new 大于 0，则使用 n_new
      }
      // 这里将 src 转换成 int8_t 的形式，按照k 维度量化  (也就是按列量化)
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      // printf("n_start = %d, n_end = %d, n = %d\n", n_start, n_end, n);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      float average = 0;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP && n_begin + i < n_block_size; i++) {
          float amax = 0;
          // TODO: 后续用 SVE 来加速
          for (int j = 0; j < k; j++) {
            // 先把 src 转换成 float
            float f = src[(n_block_begin + n_begin + i) * k + j];
            f = f < 0 ? -f : f;
            average += f;
            if (f > amax) {
              amax = f;
            }
          }
          average /= k;
          d[n_block_begin + n_begin + i] = amax / ((1 << 7) - 1);
          // printf("amax: %f,average: %f\n", amax, average);
          // TODO: 后续用 SVE 来加速
          // 通过这个 amax 来量化这一列
          for (int j = 0; j < k; j++) {
            // 先把 src 转换成 float
            float f = src[(n_block_begin + n_begin + i) * k + j];
            // 这里的 amax 是当前列的最大值
            if (if_pack || this->if_pack) {
              size_t split_n = (n_begin + i) / PACK_SIZE_N;
              size_t n_idx = (n_begin + i) % PACK_SIZE_N;
              size_t split_k = j / PACK_SIZE_K;
              size_t k_idx = j % PACK_SIZE_K;

              size_t buff_idx = n_block_begin * k + split_n * PACK_SIZE_N * k + split_k * PACK_SIZE_N * PACK_SIZE_K +
                                n_idx * PACK_SIZE_K + k_idx;
              b[buff_idx] = static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
            } else {
              b[(n_block_begin + n_begin + i) * k + j] =
                  static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
            }
          }
        }
      }
    }

    void from_mat_row_major(float *src, int ld, int ith, int nth, int n_new = -1) {  // CHECK: nth has no usage
      if (n_new > 0) {
        n = n_new;  // 如果 n_new 大于 0，则使用 n_new
      }
      // 这里将 src 转换成 int8_t 的形式，按照k 维度量化  (也就是按列量化),但是 src 是行主序的
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP && n_begin + i < n_block_size; i++) {
          float amax = 0;
          for (int j = 0; j < k; j++) {
            float f = src[j * ld + (n_block_begin + n_begin + i)];
            f = f < 0 ? -f : f;
            if (f > amax) {
              amax = f;
            }
          }
          d[n_block_begin + n_begin + i] = amax / ((1 << 7) - 1);
          for (int j = 0; j < k; j++) {
            float f = src[j * ld + (n_block_begin + n_begin + i)];
            // 这里的 amax 是当前列的最大值
            b[(n_block_begin + n_begin + i) * k + j] =
                static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
          }
        }
      }
    }

    // 将内容解量化为 float
    void to_mat(float *dst, int ith, int nth, int n_new = -1) {
      if (n_new > 0) {
        n = n_new;  // 如果 n_new 大于 0，则使用 n_new
      }
      // 这里将 b 转换成 float 的形式，按照k 维度解量化
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP && n_begin + i < n_block_size; i++) {
          // 通过这个 amax 来解量化这一列
          for (int j = 0; j < k; j++) {
            // 先把 b 转换成 float
            int8_t b_val = b[(n_block_begin + n_begin + i) * k + j];
            float d_val = d[n_block_begin + n_begin + i];
            dst[(n_block_begin + n_begin + i) * k + j] = b_val * d_val;
          }
        }
      }
    }

    float *get_scale(int n, int n_begin) { return d + n_begin; }
  };
  /* 将 buffer A 转为 buffer B, [m,k](row major) -> [k,n](column major) (n = m)
    而量化部分没变化，直接 buffer A 的 d = buffer B 的 d，校验 m 和 n 以及 k是否相等，才能转换
  */
  static void convert_buffer_a_to_buffer_b(BufferA *ba, BufferB *bb) {
    if (bb->n != ba->max_m || bb->k != ba->k || bb->if_pack != ba->if_pack) {
      throw std::runtime_error(
          "BufferA and BufferB dimensions do not match for conversion, or they are not the same pack.");
    }
    bb->b = ba->a;
    bb->d = ba->d;
  }

  static void convert_buffer_b_to_buffer_a(BufferB *bb, BufferA *ba) {
    if (ba->max_m != bb->n || ba->k != bb->k || ba->if_pack != bb->if_pack) {
      throw std::runtime_error(
          "BufferB and BufferA dimensions do not match for conversion, or they are not the same pack.");
    }
    ba->a = bb->b;
    ba->d = bb->d;
  }
  // 改变当前 C 的 view
  static void change_view(BufferC *c_src, BufferC *c_dst) {
    if (c_src->max_m != c_dst->n || c_src->n != c_dst->max_m || c_src->if_row_major == c_dst->if_row_major) {
      throw std::runtime_error("C buffer size mismatch or they are the same major");
    }
    c_dst->c = c_src->c;
  }
  // 此函数作用是，对 int32结果的 c 矩阵应用 A和 B 矩阵的scale（反量化）
  // 这里的 c 矩阵是 m * n 的矩阵，存储在 c 中, 是行主序的 (row major)
  // A 矩阵是 m * k 的矩阵，按照行量化，其 scale 是 d 是 m 维度的，对应每一行的量化系数
  // B 矩阵是 k * n 的矩阵，按照列量化，其 scale 是 d 是 n 维度的，对应每一列的量化系数
  // C 的第 i 行第 j 列的缩放值就是 A 的第 i 行的缩放值 * B 的第 j 列的缩放值
  static void apply_scale(int m, int n, float *c, BufferA *ba, BufferB *bb, BufferC *bc) {
    // TODO: 后续用 SVE 来加速
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
        float *scale_a = ba->get_scale(m, m_begin + i);
        for (int n_begin = 0; n_begin < n; n_begin += N_STEP) {
          for (int j = 0; j < N_STEP && n_begin + j < n; j++) {
            float *scale_b = bb->get_scale(n, n_begin + j);
            c[(m_begin + i) * n + (n_begin + j)] = (*scale_a) * (*scale_b) * bc->c[(m_begin + i) * n + (n_begin + j)];
          }
        }
      }
    }
  }

  // 对第二个维度分块的 apply scale
  static void apply_scale(int m, int n, float *c, BufferA *ba, BufferB *bb, BufferC *bc, int ith, int nth, int block,
                          int jth = -1) {
    // printf("use split apply scale\n");
    auto [n_start, n_end] = split_range_n_block(n, ith, nth, block);
    int m_start = 0, m_end = m;
    if (jth != -1) {
      auto tmp = split_range_m(m, jth);
      m_start = tmp.first;
      m_end = tmp.second;
    }
    // TODO: 后续用 SVE 来加速
    for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
        float *scale_a = ba->get_scale(m, m_begin + i);
        for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
          for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
            float *scale_b = bb->get_scale(n, n_begin + j);
            c[(m_begin + i) * n + (n_begin + j)] = (*scale_a) * (*scale_b) * bc->c[(m_begin + i) * n + (n_begin + j)];
          }
        }
      }
    }
  }

  // 两个维度均有分块的 apply scale
  // C 矩阵区分是 row major 还是 column major
  static void apply_scale(float *c, int ldc, BufferA *ba, BufferB *bb, BufferC *bc, int m_start, int m_end, int n_start,
                          int n_end, bool if_row_major = true, long long c_row_idx_offset = 0,
                          long long c_col_idx_offset = 0) {
    if (if_row_major) {
      for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
          float *scale_a = ba->get_scale(m_end, m_begin + i);
          for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
            for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
              float *scale_b = bb->get_scale(n_end, n_begin + j);
              c[(m_begin + i + c_row_idx_offset) * ldc + (n_begin + j + c_col_idx_offset)] =
                  (*scale_a) * (*scale_b) *
                  bc->c[(m_begin + i + c_row_idx_offset) * ldc + (n_begin + j + c_col_idx_offset)];
            }
          }
        }
      }
    } else {
      for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
        for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
          float *scale_b = bb->get_scale(n_end, n_begin + j);
          for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
            for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
              float *scale_a = ba->get_scale(m_end, m_begin + i);
              c[(n_begin + j + c_col_idx_offset) * ldc + (m_begin + i + c_row_idx_offset)] =
                  (*scale_a) * (*scale_b) *
                  bc->c[(n_begin + j + c_col_idx_offset) * ldc + (m_begin + i + c_row_idx_offset)];
            }
          }
        }
      }
    }
  }

  // 两个维度均有分块的 apply scale
  // C 矩阵区分是 row major 还是 column major
  static void apply_scale(float *c, int ldc, BufferA *ba, BufferB *bb, int32_t *bc, int m_start, int m_end, int n_start,
                          int n_end, bool if_row_major = true, long long c_row_idx_offset = 0,
                          long long c_col_idx_offset = 0) {
    if (if_row_major) {
      for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
          float *scale_a = ba->get_scale(m_end, m_begin + i);
          for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
            for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
              float *scale_b = bb->get_scale(n_end, n_begin + j);
              c[(m_begin + i + c_row_idx_offset) * ldc + (n_begin + j + c_col_idx_offset)] =
                  (*scale_a) * (*scale_b) *
                  bc[(m_begin + i + c_row_idx_offset) * ldc + (n_begin + j + c_col_idx_offset)];
            }
          }
        }
      }
    } else {
      for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
        for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
          float *scale_b = bb->get_scale(n_end, n_begin + j);
          for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
            for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
              float *scale_a = ba->get_scale(m_end, m_begin + i);
              c[(n_begin + j + c_col_idx_offset) * ldc + (m_begin + i + c_row_idx_offset)] =
                  (*scale_a) * (*scale_b) *
                  bc[(n_begin + j + c_col_idx_offset) * ldc + (m_begin + i + c_row_idx_offset)];
            }
          }
        }
      }
    }
  }
};

struct GemmKernelInt4 {
  using dt = int4_2_t;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 0.5;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  // static const int M_STEP = TILE_M * 2;
  // static const int N_STEP = TILE_N * 2;
  // static const int K_STEP = TILE_K;
  static const int M_STEP = 1;
  static const int N_STEP = 1;
  static const int K_STEP = 1;

  // static inline const int N_BLOCK = 1024;
  // Make tiling params runtime-configurable (modifiable via Python bindings)
  static inline int N_BLOCK_UP_GATE = 256;
  static inline int N_BLOCK_DOWN = 1024;
  static inline int N_BLOCK_UP_GATE_PREFI = 256;
  static inline int N_BLOCK_DOWN_PREFI = 1024;
  static inline int N_BLOCK = 64;
  static inline int M_BLOCK = 320;
  // static inline const int N_BLOCK = 32;
  static inline int K_BLOCK = 7168;

  // Setter/getter for runtime tiling configuration
  static void set_tiling(int n_block_up_gate, int n_block_down, int n_block, int m_block, int k_block,
                         int n_block_up_gate_prefi, int n_block_down_prefi) {
    N_BLOCK_UP_GATE = n_block_up_gate;
    N_BLOCK_DOWN = n_block_down;
    N_BLOCK = n_block;
    M_BLOCK = m_block;
    K_BLOCK = k_block;
    N_BLOCK_UP_GATE_PREFI = n_block_up_gate_prefi;
    N_BLOCK_DOWN_PREFI = n_block_down_prefi;
  }
  static std::tuple<int, int, int, int, int, int, int> get_tiling() {
    return std::make_tuple(N_BLOCK_UP_GATE, N_BLOCK_DOWN, N_BLOCK, M_BLOCK, K_BLOCK, N_BLOCK_UP_GATE_PREFI,
                           N_BLOCK_DOWN_PREFI);
  }

  static inline const int PACK_SIZE_N = 8;
  static inline const int PACK_SIZE_K = 32;
  static inline const int PACK_SIZE_M = 8;

  static std::string name() { return "INT4"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static int recommended_nth_down(int n, char type_ = 'd') {
    if (type_ == 'p') {
      if (n % N_BLOCK_DOWN_PREFI != 0) {
        throw std::invalid_argument("n must be multiple of N_BLOCK_DOWN_PREFI in prefill");
      }
      return n / N_BLOCK_DOWN_PREFI;
    } else {
      if (n % N_BLOCK_DOWN != 0) {
        throw std::invalid_argument("n must be multiple of N_BLOCK_DOWN in decode");
      }
      return n / N_BLOCK_DOWN;
    }
  }
  static int recommended_mth(int m) { return (m + M_BLOCK - 1) / M_BLOCK; }

  static int recommended_nth_up_gate(int n, char type_ = 'd') {
    if (type_ == 'p') {
      if (n % N_BLOCK_UP_GATE_PREFI != 0) {
        throw std::invalid_argument("n must be multiple of N_BLOCK_UP_GATE_PREFI in prefill");
      }
      return n / N_BLOCK_UP_GATE_PREFI;
    } else {
      if (n % N_BLOCK_UP_GATE != 0) {
        throw std::invalid_argument("n must be multiple of N_BLOCK_UP_GATE in decode");
      }
      return n / N_BLOCK_UP_GATE;
    }
  }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }
  static std::pair<int, int> split_range_m(int m, int ith, int mth = 0) {
    int n_start = M_BLOCK * ith;
    int n_end = std::min(m, M_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static std::pair<int, int> split_range_n_block(int n, int ith, int nth, int block) {
    int n_start = block * ith;
    int n_end = std::min(n, block * (ith + 1));
    return {n_start, n_end};
  }

  using BufferA = BufferAImpl<GemmKernelInt4>;
  using BufferC = BufferCImpl<GemmKernelInt4>;

  struct BufferB {
    dt *b;
    float *d;
    int n, k;
    std::vector<int8_t *> b_pack;  // b_pack[i] -> the ith block (the ith packed matrix of B)
    static constexpr bool SCALE = true;
    bool if_pack = false;

    // static size_t required_size(int n, int k) { return sizeof(int8_t) * n * k / 2 + sizeof(float) * n; }
    static size_t required_size(int n, int k, bool if_pack = false, char mat_type = 'n', bool plain = true) {
      int nth, n_block;
      if (if_pack && !plain) {
        switch (mat_type) {
          case 'n':
            nth = recommended_nth(n);
            n_block = N_BLOCK;
            break;
          case 'u':
            nth = recommended_nth_up_gate(n);
            n_block = N_BLOCK_UP_GATE;
            break;
          case 'd':
            nth = recommended_nth_down(n);
            n_block = N_BLOCK_DOWN;
            break;
          default:
            throw std::invalid_argument("Invalid mat_type");
        }
        size_t reorder_B_size = get_reorder_B_size(KernelCblasRowMajor, KernelCblasNoTrans, k, n_block);
        return sizeof(int8_t) * nth * reorder_B_size + sizeof(float) * n;
      } else {
        return sizeof(int8_t) * n * k / 2 + sizeof(float) * n;
      }
    }

    // BufferB(int n, int k, void *ptr, bool if_pack = false) : n(n), k(k), if_pack(if_pack) {
    //   b = reinterpret_cast<dt *>(ptr);
    //   d = reinterpret_cast<float *>(moe_kernel::offset_pointer(b, n * k / 2));
    // }
    BufferB(int n, int k, bool if_pack = false, char mat_type = 'n', bool plain = true) : n(n), k(k), if_pack(if_pack) {
      if (n % N_STEP != 0 || k % K_STEP != 0) {
        throw std::runtime_error("n and k must be multiples of N_STEP and K_STEP respectively");
      }
    }
    BufferB(int n, int k, void *ptr, bool if_pack = false, char mat_type = 'n', bool plain = true)
        : BufferB(n, k, if_pack, mat_type, plain) {
      set_data(ptr, plain);
    }
    void set_data(void *ptr, bool plain = true) {
      b = reinterpret_cast<dt *>(ptr);
      d = reinterpret_cast<float *>(moe_kernel::offset_pointer(b, n * k / 2));
    }
    size_t required_size() const { return sizeof(int8_t) * n * k / 2 + sizeof(float) * n; }
    BufferB offset_col(size_t col_begin, size_t col_block) {
      auto bufferb = BufferB(col_block, k, moe_kernel::offset_pointer(b, (col_begin * k) / 2), if_pack);
      bufferb.d = d + col_begin;
      return bufferb;
    }
    // B 矩阵是 K * N 的矩阵，存储在 b 中, 是列主序的 (column major)
    void from_mat(ggml_bf16_t *src, int ith, int nth, int n_new = -1, bool if_pack = false,
                  bool plain = true) {  // CHECK: nth has no usage
      if (!if_pack && !this->if_pack) throw std::runtime_error("from mat for buffer should be packed");
      if (n_new > 0) {
        n = n_new;  // 如果 n_new 大于 0，则使用 n_new
      }
      // 这里将 src 转换成 int8_t 的形式，按照k 维度量化  (也就是按列量化)
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP && n_begin + i < n_block_size; i++) {
          float amax = 0;
          // TODO: 后续用 SVE 来加速
          for (int j = 0; j < k; j++) {
            // 先把 src 转换成 float
            float f = bf16_to_fp32(src[(n_block_begin + n_begin + i) * k + j]);
            f = f < 0 ? -f : f;
            if (f > amax) {
              amax = f;
            }
          }
          d[n_block_begin + n_begin + i] = amax / 112.0;
          // TODO: 后续用 SVE 来加速
          for (int k_start = 0; k_start < k; k_start += (PACK_SIZE_K * 2)) {
            for (int j = 0; j < PACK_SIZE_K; j++) {
              size_t split_n = (n_begin + i) / PACK_SIZE_N;
              size_t n_idx = (n_begin + i) % PACK_SIZE_N;
              size_t split_k = k_start / (PACK_SIZE_K * 2);
              size_t k_idx = j;

              size_t buff_idx = n_block_begin * k / 2 + split_n * PACK_SIZE_N * k / 2 +
                                split_k * PACK_SIZE_N * PACK_SIZE_K + n_idx * PACK_SIZE_K + k_idx;

              float f0 = bf16_to_fp32(src[(n_block_begin + n_begin + i) * k + k_start + j]);
              float f1 = bf16_to_fp32(src[(n_block_begin + n_begin + i) * k + k_start + j + PACK_SIZE_K]);
              // static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
              int8_t b0 = static_cast<int8_t>(std::round((f0 / (d[n_block_begin + n_begin + i] * 16.0))) * 16);
              int8_t b1 = static_cast<int8_t>(std::round((f1 / (d[n_block_begin + n_begin + i] * 16.0))) * 16);
              int8_t b01 = (b0 & 0xF0) | ((b1 >> 4) & 0x0F);
              // int8_t b01 = ((b0 << 4) & 0xF0) | ((b1)&0x0F);

              b[buff_idx] = b01;
            }
          }
        }
      }
    }

    void from_mat(float *src, int ith, int nth, int n_new = -1, bool if_pack = false) {  // CHECK: nth has no usage
      if (!if_pack && !this->if_pack) throw std::runtime_error("from mat for buffer should be packed");
      if (n_new > 0) {
        n = n_new;  // 如果 n_new 大于 0，则使用 n_new
      }
      // 这里将 src 转换成 int8_t 的形式，按照k 维度量化  (也就是按列量化)
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      // DEBUG: 查看 average 值
      float average = 0;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP && n_begin + i < n_block_size; i++) {
          float amax = 0;
          // TODO: 后续用 SVE 来加速
          for (int j = 0; j < k; j++) {
            // 先把 src 转换成 float
            float f = src[(n_block_begin + n_begin + i) * k + j];
            f = f < 0 ? -f : f;
            average += f;
            if (f > amax) {
              amax = f;
            }
          }
          average /= k;
          d[n_block_begin + n_begin + i] = amax / 112.0;
          // printf("amax: %f,average: %f\n", amax, average);
          // TODO: 后续用 SVE 来加速
          // 通过这个 amax 来量化这一列
          for (int k_start = 0; k_start < k; k_start += (PACK_SIZE_K * 2)) {
            for (int j = 0; j < PACK_SIZE_K; j++) {
              size_t split_n = (n_begin + i) / PACK_SIZE_N;
              size_t n_idx = (n_begin + i) % PACK_SIZE_N;
              size_t split_k = k_start / (PACK_SIZE_K * 2);
              size_t k_idx = j;

              size_t buff_idx = n_block_begin * k / 2 + split_n * PACK_SIZE_N * k / 2 +
                                split_k * PACK_SIZE_N * PACK_SIZE_K + n_idx * PACK_SIZE_K + k_idx;

              float f0 = (src[(n_block_begin + n_begin + i) * k + k_start + j]);
              float f1 = (src[(n_block_begin + n_begin + i) * k + k_start + j + PACK_SIZE_K]);
              // static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
              int8_t b0 = static_cast<int8_t>(std::round((f0 / (d[n_block_begin + n_begin + i] * 16.0))) * 16);
              int8_t b1 = static_cast<int8_t>(std::round((f1 / (d[n_block_begin + n_begin + i] * 16.0))) * 16);
              int8_t b01 = (b0 & 0xF0) | ((b1 >> 4) & 0x0F);
              // int8_t b01 = ((b0 << 4) & 0xF0) | ((b1)&0x0F);
              // if (n_begin == 0 && i == 0 && k_start == 0 && j <= 10) {
              //   printf("b0: %d, b1: %d, b01: %d,f0: %f, f1: %f, scale: %f\n", b0, b1, b01, f0, f1,
              //          d[n_block_begin + n_begin + i]);
              // }

              b[buff_idx] = b01;
            }
          }
        }
      }
      // printf("from_mat done, n: %d, k: %d, if_pack: %d\n", n, k, if_pack);
    }

    float *get_scale(int n, int n_begin) { return d + n_begin; }
  };
  /* 将 buffer A 转为 buffer B, [m,k](row major) -> [k,n](column major) (n = m)
    而量化部分没变化，直接 buffer A 的 d = buffer B 的 d，校验 m 和 n 以及 k是否相等，才能转换
  */
  static void convert_buffer_a_to_buffer_b(BufferA *ba, BufferB *bb) {
    if (bb->n != ba->max_m || bb->k != ba->k || bb->if_pack != ba->if_pack) {
      throw std::runtime_error(
          "BufferA and BufferB dimensions do not match for conversion, or they are not the same pack.");
    }
    throw std::runtime_error("int4 not support convert");
    // bb->b = ba->a;
    // bb->d = ba->d;
  }

  static void convert_buffer_b_to_buffer_a(BufferB *bb, BufferA *ba) {
    if (ba->max_m != bb->n || ba->k != bb->k || ba->if_pack != bb->if_pack) {
      throw std::runtime_error(
          "BufferB and BufferA dimensions do not match for conversion, or they are not the same pack.");
    }
    throw std::runtime_error("int4 not support convert");

    // ba->a = bb->b;
    // ba->d = bb->d;
  }
  // 改变当前 C 的 view
  static void change_view(BufferC *c_src, BufferC *c_dst) {
    if (c_src->max_m != c_dst->n || c_src->n != c_dst->max_m || c_src->if_row_major == c_dst->if_row_major) {
      throw std::runtime_error("C buffer size mismatch or they are the same major");
    }
    throw std::runtime_error("int4 not support convert");

    // c_dst->c = c_src->c;
  }
  // 此函数作用是，对 int32结果的 c 矩阵应用 A和 B 矩阵的scale（反量化）
  // 这里的 c 矩阵是 m * n 的矩阵，存储在 c 中, 是行主序的 (row major)
  // A 矩阵是 m * k 的矩阵，按照行量化，其 scale 是 d 是 m 维度的，对应每一行的量化系数
  // B 矩阵是 k * n 的矩阵，按照列量化，其 scale 是 d 是 n 维度的，对应每一列的量化系数
  // C 的第 i 行第 j 列的缩放值就是 A 的第 i 行的缩放值 * B 的第 j 列的缩放值
  static void apply_scale(int m, int n, float *c, BufferA *ba, BufferB *bb, BufferC *bc) {
    // TODO: 后续用 SVE 来加速
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
        float *scale_a = ba->get_scale(m, m_begin + i);
        for (int n_begin = 0; n_begin < n; n_begin += N_STEP) {
          for (int j = 0; j < N_STEP && n_begin + j < n; j++) {
            float *scale_b = bb->get_scale(n, n_begin + j);
            c[(m_begin + i) * n + (n_begin + j)] = (*scale_a) * (*scale_b) * bc->c[(m_begin + i) * n + (n_begin + j)];
          }
        }
      }
    }
  }
  // 对第二个维度分块的 apply scale
  static void apply_scale(int m, int n, float *c, BufferA *ba, BufferB *bb, BufferC *bc, int ith, int nth, int block, int jth = -1) {
    // printf("use split apply scale\n");
    auto [n_start, n_end] = split_range_n_block(n, ith, nth, block);
    int m_start = 0, m_end = m;
    if (jth != -1) {
      auto tmp = split_range_m(m, jth);
      m_start = tmp.first;
      m_end = tmp.second;
    }
    // TODO: 后续用 SVE 来加速
    for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
        float *scale_a = ba->get_scale(m, m_begin + i);
        for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
          for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
            float *scale_b = bb->get_scale(n, n_begin + j);
            c[(m_begin + i) * n + (n_begin + j)] = (*scale_a) * (*scale_b) * bc->c[(m_begin + i) * n + (n_begin + j)];
          }
        }
      }
    }
  }
  // 两个维度均有分块的 apply scale
  // C 矩阵区分是 row major 还是 column major
  static void apply_scale(float *c, int ldc, BufferA *ba, BufferB *bb, BufferC *bc, int m_start, int m_end, int n_start,
                          int n_end, bool if_row_major = true, long long c_row_idx_offset = 0,
                          long long c_col_idx_offset = 0) {
    if (if_row_major) {
      for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
          float *scale_a = ba->get_scale(m_end, m_begin + i);
          for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
            for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
              float *scale_b = bb->get_scale(n_end, n_begin + j);
              c[(m_begin + i + c_row_idx_offset) * ldc + (n_begin + j + c_col_idx_offset)] =
                  (*scale_a) * (*scale_b) *
                  bc->c[(m_begin + i + c_row_idx_offset) * ldc + (n_begin + j + c_col_idx_offset)];
            }
          }
        }
      }
    } else {
      for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
        for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
          float *scale_b = bb->get_scale(n_end, n_begin + j);
          for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
            for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
              float *scale_a = ba->get_scale(m_end, m_begin + i);
              c[(n_begin + j + c_col_idx_offset) * ldc + (m_begin + i + c_row_idx_offset)] =
                  (*scale_a) * (*scale_b) *
                  bc->c[(n_begin + j + c_col_idx_offset) * ldc + (m_begin + i + c_row_idx_offset)];
            }
          }
        }
      }
    }
  }

  // 两个维度均有分块的 apply scale
  // C 矩阵区分是 row major 还是 column major
  static void apply_scale(float *c, int ldc, BufferA *ba, BufferB *bb, int32_t *bc, int m_start, int m_end, int n_start,
                          int n_end, bool if_row_major = true, long long c_row_idx_offset = 0,
                          long long c_col_idx_offset = 0) {
    if (if_row_major) {
      for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
          float *scale_a = ba->get_scale(m_end, m_begin + i);
          for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
            for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
              float *scale_b = bb->get_scale(n_end, n_begin + j);
              c[(m_begin + i + c_row_idx_offset) * ldc + (n_begin + j + c_col_idx_offset)] =
                  (*scale_a) * (*scale_b) *
                  bc[(m_begin + i + c_row_idx_offset) * ldc + (n_begin + j + c_col_idx_offset)];
            }
          }
        }
      }
    } else {
      for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
        for (int j = 0; j < N_STEP && n_begin + j < n_end; j++) {
          float *scale_b = bb->get_scale(n_end, n_begin + j);
          for (int m_begin = m_start; m_begin < m_end; m_begin += M_STEP) {
            for (int i = 0; i < M_STEP && m_begin + i < m_end; i++) {
              float *scale_a = ba->get_scale(m_end, m_begin + i);
              c[(n_begin + j + c_col_idx_offset) * ldc + (m_begin + i + c_row_idx_offset)] =
                  (*scale_a) * (*scale_b) *
                  bc[(n_begin + j + c_col_idx_offset) * ldc + (m_begin + i + c_row_idx_offset)];
            }
          }
        }
      }
    }
  }
};

}  // namespace moe_kernel

#endif