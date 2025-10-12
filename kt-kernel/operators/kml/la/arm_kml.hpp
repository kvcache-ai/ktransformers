#ifndef CPUINFER_OPERATOR_KML_LA_HPP
#define CPUINFER_OPERATOR_KML_LA_HPP

#include "batch_gemm_api.hpp"
// #include <boost/serialization/strong_typedef.hpp>

// #include "../../common.hpp"
#include <arm_sve.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "kblas.h"
#include "llama.cpp/ggml.h"
#include "utils.hpp"

// BOOST_STRONG_TYPEDEF(int8_t, int4_2_t);
#define STRONG_TYPEDEF(T, D)                                   \
  struct D {                                                   \
    T t;                                                       \
    explicit D(const T &v) : t(v) {}                           \
    D() = default;                                             \
    D(const D &) = default;                                    \
    D &operator=(const D &) = default;                         \
    D &operator=(const T &rhs) {                               \
      t = rhs;                                                 \
      return *this;                                            \
    }                                                          \
    operator const T &() const { return t; }                   \
    operator T &() { return t; }                               \
    bool operator==(const D &rhs) const { return t == rhs.t; } \
    bool operator!=(const D &rhs) const { return t != rhs.t; } \
    bool operator<(const D &rhs) const { return t < rhs.t; }   \
  };
STRONG_TYPEDEF(int8_t, int4_2_t);

namespace arm_kml {
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
  static constexpr int K_BLOCK = K::K_BLOCK;
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

  void from_mat(int m, float16_t *src, int ith, int mth) {
    assert(m <= max_m);
    assert(ith == 0 && mth == 1);
    if (!(ith == 0 && mth == 1)) {
      throw std::runtime_error("m must be a multiple of M_STEP");
    }
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
  static constexpr int N_BLOCK = K::N_BLOCK;

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

// struct MLAGemmKernelInt8 {
//   using dt = int8_t;
//   using output_t = int32_t;
//   struct BufferA {
//     int8_t *a;
//     float *d;
//     int max_m, k;

//     BufferA(int max_m, int k, void *ptr) : max_m(max_m), k(k) {
//       assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
//       assert(max_m % GemmKernelInt8::M_STEP == 0);
//       assert(k % GemmKernelInt8::K_STEP == 0);
//       a = reinterpret_cast<int8_t *>(ptr);
//       d = reinterpret_cast<float *>(a + max_m * k);
//     }

//     void from_mat(int m, float16_t *src, int ith, int nth) {}
//     float *get_scale(int m, int m_begin) { return d + m_begin; }
//   };
// };

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
  static inline const int N_BLOCK_UP_GATE = 256;
  static inline const int N_BLOCK_DOWN = 1024;
  static inline const int N_BLOCK = 64;
  static inline const int M_BLOCK = 64;
  // static inline const int N_BLOCK = 32;
  static inline const int K_BLOCK = 7168;

  static inline const int PACK_SIZE_N = 8;
  static inline const int PACK_SIZE_M = 8;
  static inline const int PACK_SIZE_K = 32;

  static std::string name() { return "INT8"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static int recommended_nth_down(int n) {
    assert(n % N_BLOCK == 0);
    return n / N_BLOCK_DOWN;
  }

  static int recommended_nth_up_gate(int n) {
    assert(n % N_BLOCK_UP_GATE == 0);
    return n / N_BLOCK_UP_GATE;
  }

  static int recommended_mth(int m) { return (m + M_BLOCK - 1) / M_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static std::pair<int, int> split_range_m(int m, int ith, int mth) {
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
    float *d;
    int n, k;
    static constexpr bool SCALE = true;
    bool if_pack = false;

    static size_t required_size(int n, int k) { return sizeof(int8_t) * n * k + sizeof(float) * n; }

    BufferB(int n, int k, void *ptr, bool if_pack = false) : n(n), k(k), if_pack(if_pack) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      b = reinterpret_cast<int8_t *>(ptr);
      d = reinterpret_cast<float *>(b + n * k);
    }
    BufferB(int n, int k, bool if_pack = false) : n(n), k(k), if_pack(if_pack) {
      if (n % N_STEP != 0 || k % K_STEP != 0) {
        throw std::runtime_error("n and k must be multiples of N_STEP and K_STEP respectively");
      }
    }
    void set_data(void *ptr) {
      b = reinterpret_cast<int8_t *>(ptr);
      d = reinterpret_cast<float *>(b + n * k);
    }
    size_t required_size() const { return sizeof(int8_t) * n * k + sizeof(float) * n; }
    BufferB offset_col(size_t col_begin, size_t col_block) {
      auto bufferb = BufferB(col_block, k, b + col_begin * k, if_pack);
      bufferb.d = d + col_begin;
      return bufferb;
    }
    // B 矩阵是 K * N 的矩阵，存储在 b 中, 是列主序的 (column major)
    void from_mat(ggml_bf16_t *src, int ith, int nth, int n_new = -1,
                  bool if_pack = false) {  // CHECK: nth has no usage
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
          d[n_block_begin + n_begin + i] = amax / ((1 << 7) - 1);
          // TODO: 后续用 SVE 来加速
          // 通过这个 amax 来量化这一列
          for (int j = 0; j < k; j++) {
            // 先把 src 转换成 float
            float f = bf16_to_fp32(src[(n_block_begin + n_begin + i) * k + j]);
            if (if_pack || this->if_pack) {
              size_t split_n = (n_begin + i) / PACK_SIZE_N;
              size_t n_idx = (n_begin + i) % PACK_SIZE_N;
              size_t split_k = j / PACK_SIZE_K;
              size_t k_idx = j % PACK_SIZE_K;

              size_t buff_idx = n_block_begin * k + split_n * PACK_SIZE_N * k + split_k * PACK_SIZE_N * PACK_SIZE_K +
                                n_idx * PACK_SIZE_K + k_idx;
              b[buff_idx] = static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
            } else {
              // 这里的 amax 是当前列的最大值
              b[(n_block_begin + n_begin + i) * k + j] =
                  static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
            }
          }
        }
      }
    }

    void from_mat(float16_t *src, int ith, int nth, int n_new = -1, bool if_pack = false) {  // CHECK: nth has no usage
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
            float f = src[(n_block_begin + n_begin + i) * k + j];
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
            float f = src[(n_block_begin + n_begin + i) * k + j];
            if (if_pack || this->if_pack) {
              size_t split_n = (n_begin + i) / PACK_SIZE_N;
              size_t n_idx = (n_begin + i) % PACK_SIZE_N;
              size_t split_k = j / PACK_SIZE_K;
              size_t k_idx = j % PACK_SIZE_K;

              size_t buff_idx = n_block_begin * k + split_n * PACK_SIZE_N * k + split_k * PACK_SIZE_N * PACK_SIZE_K +
                                n_idx * PACK_SIZE_K + k_idx;
              b[buff_idx] = static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
            } else {
              // 这里的 amax 是当前列的最大值
              b[(n_block_begin + n_begin + i) * k + j] =
                  static_cast<int8_t>(std::round(f / d[n_block_begin + n_begin + i]));
            }
          }
        }
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
  static void apply_scale(int m, int n, float *c, BufferA *ba, BufferB *bb, BufferC *bc, int ith, int nth, int block) {
    // printf("use split apply scale\n");
    auto [n_start, n_end] = split_range_n_block(n, ith, nth, block);
    // TODO: 后续用 SVE 来加速
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
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
  static inline const int N_BLOCK_UP_GATE = 256;
  static inline const int N_BLOCK_DOWN = 1024;
  static inline const int N_BLOCK = 64;
  static inline const int M_BLOCK = 64;
  // static inline const int N_BLOCK = 32;
  static inline const int K_BLOCK = 7168;

  static inline const int PACK_SIZE_N = 8;
  static inline const int PACK_SIZE_K = 32;
  static inline const int PACK_SIZE_M = 8;

  static std::string name() { return "INT4"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static int recommended_nth_down(int n) {
    assert(n % N_BLOCK == 0);
    return n / N_BLOCK_DOWN;
  }
  static int recommended_mth(int m) { return (m + M_BLOCK - 1) / M_BLOCK; }

  static int recommended_nth_up_gate(int n) {
    assert(n % N_BLOCK_UP_GATE == 0);
    return n / N_BLOCK_UP_GATE;
  }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }
  static std::pair<int, int> split_range_m(int m, int ith, int mth) {
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
    static constexpr bool SCALE = true;
    bool if_pack = false;

    static size_t required_size(int n, int k) { return sizeof(int8_t) * n * k / 2 + sizeof(float) * n; }

    BufferB(int n, int k, void *ptr, bool if_pack = false) : n(n), k(k), if_pack(if_pack) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % K_STEP == 0);
      b = reinterpret_cast<dt *>(ptr);
      d = reinterpret_cast<float *>(arm_kml::offset_pointer(b, n * k / 2));
    }

    BufferB(int n, int k, bool if_pack = false) : n(n), k(k), if_pack(if_pack) {
      // assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % K_STEP == 0);
    }
    void set_data(void *ptr) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      b = reinterpret_cast<dt *>(ptr);
      d = reinterpret_cast<float *>(arm_kml::offset_pointer(b, n * k / 2));
    }
    size_t required_size() const { return sizeof(int8_t) * n * k / 2 + sizeof(float) * n; }
    BufferB offset_col(size_t col_begin, size_t col_block) {
      auto bufferb = BufferB(col_block, k, arm_kml::offset_pointer(b, (col_begin * k) / 2), if_pack);
      bufferb.d = d + col_begin;
      return bufferb;
    }
    // B 矩阵是 K * N 的矩阵，存储在 b 中, 是列主序的 (column major)
    void from_mat(ggml_bf16_t *src, int ith, int nth, int n_new = -1,
                  bool if_pack = false) {  // CHECK: nth has no usage
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

    void from_mat(float16_t *src, int ith, int nth, int n_new = -1, bool if_pack = false) {  // CHECK: nth has no usage
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
            float f = src[(n_block_begin + n_begin + i) * k + j];
            f = f < 0 ? -f : f;
            if (f > amax) {
              amax = f;
            }
          }
          d[n_block_begin + n_begin + i] = amax / 112.0;
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
  static void apply_scale(int m, int n, float *c, BufferA *ba, BufferB *bb, BufferC *bc, int ith, int nth, int block) {
    // printf("use split apply scale\n");
    auto [n_start, n_end] = split_range_n_block(n, ith, nth, block);
    // TODO: 后续用 SVE 来加速
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
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

inline CBLAS_TRANSPOSE flip_trans(CBLAS_TRANSPOSE trans) {
  if (trans == CblasNoTrans) {
    return CblasTrans;
  } else if (trans == CblasTrans) {
    return CblasNoTrans;
  } else {
    throw std::runtime_error("Unsupported transpose");
  }
}

template <typename F>
struct MatRef {
  F *data = nullptr;
  size_t R, C, ld;
  CBLAS_ORDER order;
  CBLAS_TRANSPOSE trans;
  bool if_pack = false;
  static inline const int PACK_SIZE_N = 8;
  static inline const int PACK_SIZE_M = 8;
  static inline const int PACK_SIZE_K = 32;

  MatRef() {}
  MatRef(F *data, size_t R, size_t C, size_t ld, CBLAS_ORDER order, CBLAS_TRANSPOSE trans = CblasNoTrans,
         bool if_pack = false)
      : data(data), R(R), C(C), ld(ld), order(order), trans(trans), if_pack(if_pack) {}

  MatRef t() {
    MatRef re = *this;
    std::swap(re.R, re.C);
    CBLAS_ORDER new_order = (order == CblasRowMajor) ? CblasColMajor : CblasRowMajor;
    re.order = new_order;
    return re;
  }

  CBLAS_TRANSPOSE trans_from(CBLAS_ORDER order) {
    if (order == this->order) {
      return trans;
    } else {
      return flip_trans(trans);
    }
  }

  MatRef offset_block(size_t row, size_t col, size_t r_block, size_t c_block) {
    if (trans == CblasTrans) {
      std::swap(row, col);
      std::swap(r_block, c_block);
    }
    int devide_elements_size = 1;
    if constexpr (std::is_same_v<F, int4_2_t>) devide_elements_size = 2;
    // printf("devide_elements_size : %d\n", devide_elements_size);
    if (order == CblasRowMajor) {
      if (if_pack) {
        // if (devide_elements_size == 2)
        // printf("data:%p,after: %p,offset: %d\n", data, data + (row * ld + col * PACK_SIZE_M) /
        // devide_elements_size,
        //        (row * ld + col * PACK_SIZE_M) / devide_elements_size);
        return MatRef(data + (row * ld + col * PACK_SIZE_M) / devide_elements_size, r_block, c_block, ld, order,
                      CblasNoTrans, if_pack);
      } else {
        return MatRef(data + (row * ld + col) / devide_elements_size, r_block, c_block, ld, order, CblasNoTrans,
                      if_pack);
      }
    } else if (order == CblasColMajor) {
      if (if_pack) {
        // if (devide_elements_size == 2)
        // printf("data:%p,after: %p,offset: %d\n", data, data + (col * ld + row * PACK_SIZE_N) /
        // devide_elements_size,
        //        (col * ld + row * PACK_SIZE_N) / devide_elements_size);
        return MatRef(data + (col * ld + row * PACK_SIZE_N) / devide_elements_size, r_block, c_block, ld, order,
                      CblasNoTrans, if_pack);
      } else {
        return MatRef(data + (col * ld + row) / devide_elements_size, r_block, c_block, ld, order, CblasNoTrans,
                      if_pack);
      }
    } else {
      throw std::runtime_error("Unsupported order");
    }
  }

  inline MatRef trans_view() {
    if (order == CblasRowMajor) {
      return MatRef(data, C, R, ld, CblasColMajor, trans, if_pack);
    } else {
      return MatRef(data, C, R, ld, CblasRowMajor, trans, if_pack);
    }
  }

  MatRef offset_row(size_t row_begin, size_t row_block) { return offset_block(row_begin, 0, row_block, C); }

  MatRef offset_col(size_t col_begin, size_t col_block) { return offset_block(0, col_begin, R, col_block); }

  F &at(size_t row, size_t col) {
    if (trans == CblasTrans) {
      throw std::runtime_error("Unsupported trans");
    }

    if (order == CblasRowMajor) {
      return data[row * ld + col];
    } else if (order == CblasColMajor) {
      return data[col * ld + row];
    } else {
      throw std::runtime_error("Unsupported order");
    }
  }
};

template <typename A, typename B, typename C>
static void mul_mat(MatRef<A> a, MatRef<B> b, MatRef<C> c, C alpha, C beta) {
  assert(a.C == b.R);
  assert(a.R == c.R);
  assert(b.C == c.C);
  // assert(a.order == b.order);
  // assert(a.order == c.order);
  assert(c.trans == CblasNoTrans);
  BLASINT8 oa = 0, ob = 0;
  int32_t oc = 0;

  if constexpr (std::is_same_v<A, float> && std::is_same_v<B, float> && std::is_same_v<C, float>) {
    cblas_sgemm(c.order, a.trans_from(c.order), b.trans_from(c.order), c.R, c.C, a.C, alpha, a.data, a.ld, b.data, b.ld,
                beta, c.data, c.ld);

  } else if constexpr (std::is_same_v<A, float16_t> && std::is_same_v<B, float16_t> && std::is_same_v<C, float16_t>) {
    cblas_hgemm(c.order, a.trans_from(c.order), b.trans_from(c.order), c.R, c.C, a.C, alpha, a.data, a.ld, b.data, b.ld,
                beta, c.data, c.ld);
  } else if constexpr (std::is_same_v<A, float16_t> && std::is_same_v<B, float16_t> && std::is_same_v<C, float>) {
    cblas_shgemm(c.order, a.trans_from(c.order), b.trans_from(c.order), c.R, c.C, a.C, alpha, a.data, a.ld, b.data,
                 b.ld, beta, c.data, c.ld);
  } else if constexpr (std::is_same_v<A, bfloat16_t> && std::is_same_v<B, bfloat16_t> &&
                       std::is_same_v<C, bfloat16_t>) {
    cblas_bgemm(c.order, a.trans_from(c.order), b.trans_from(c.order), c.R, c.C, a.C, alpha, a.data, a.ld, b.data, b.ld,
                beta, c.data, c.ld);
  } else if constexpr (std::is_same_v<A, int8_t> && std::is_same_v<B, int8_t> && std::is_same_v<C, int32_t>) {
    if (b.if_pack) {
      prefill_cblas_gemm_s8s8s32(c.order, a.trans_from(c.order), b.trans_from(c.order), CblasFixOffset, c.R, c.C, a.C,
                                 alpha, a.data, a.ld, oa, b.data, b.ld, ob, beta, c.data, c.ld, &oc);
    } else {
      cblas_gemm_s8s8s32(c.order, a.trans_from(c.order), b.trans_from(c.order), CblasFixOffset, c.R, c.C, a.C, alpha,
                         a.data, a.ld, oa, b.data, b.ld, ob, beta, c.data, c.ld, &oc);
    }

  } else if constexpr (std::is_same_v<A, int8_t> && std::is_same_v<B, int4_2_t> && std::is_same_v<C, int32_t>) {
    // throw std::runtime_error("INT4 does not support cblas_gemm_s8s8s32, please use decode_cblas_gemm_s8s8s32");
    if (b.if_pack) {
      prefill_int4_cblas_gemm_s8s8s32(c.order, a.trans_from(c.order), b.trans_from(c.order), CblasFixOffset, c.R, c.C,
                                      a.C, alpha, a.data, a.ld, oa, b.data, b.ld, ob, beta, c.data, c.ld, &oc);
    } else {
      throw std::runtime_error(
          "INT4 does not support cblas_gemm_s8s8s32 for unpack, please use decode_cblas_gemm_s8s8s32");
    }

  } else {
    throw std::runtime_error("Unsupported type");
  }
}

template <typename A, typename B, typename C>
static void decode_mul_mat(MatRef<A> a, MatRef<B> b, MatRef<C> c, C alpha, C beta) {
  assert(a.C == b.R);
  assert(a.R == c.R);
  assert(b.C == c.C);
  // assert(a.order == b.order);
  // assert(a.order == c.order);
  assert(c.trans == CblasNoTrans);
  BLASINT incX = 1, incY = 1;
  BLASINT8 oa = 0, ob = 0;
  int32_t oc = 0;
  if constexpr (std::is_same_v<A, float> && std::is_same_v<B, float> && std::is_same_v<C, float>) {
    cblas_sgemv(a.order, a.trans, a.R, a.C, alpha, a.data, a.ld, b.data, incX, beta, c.data, incY);
  } else if constexpr (std::is_same_v<A, int8_t> && std::is_same_v<B, int8_t> && std::is_same_v<C, int32_t>) {
    // printf("debug: c.order: %d, a.order: %d, b.order: %d,c.R: %zu, c.C: %zu, a.C: %zu, alpha: %d, a.ld: %ld, oa: %d,
    // b.ld: %ld, ob: %d, beta: %d, c.ld: %ld, oc: %d\n",
    //        c.order, a.order, b.order, c.R, c.C, a.C, alpha, a.ld, oa, b.ld, ob, beta, c.ld, oc);
    if (b.if_pack)
      decode_cblas_gemm_s8s8s32(c.order, a.trans_from(c.order), b.trans_from(c.order), CblasFixOffset, c.R, c.C, a.C,
                                alpha, a.data, a.ld, oa, b.data, b.ld, ob, beta, c.data, c.ld, &oc);
    else
      throw std::runtime_error("Unsupported type");

  } else if constexpr (std::is_same_v<A, int8_t> && std::is_same_v<B, int4_2_t> && std::is_same_v<C, int32_t>) {
    decode_int4_cblas_gemm_s8s8s32(c.order, a.trans_from(c.order), b.trans_from(c.order), CblasFixOffset, c.R, c.C, a.C,
                                   alpha, a.data, a.ld, oa, b.data, b.ld, ob, beta, c.data, c.ld, &oc);
  } else {
    throw std::runtime_error("Unsupported type");
  }
}

template <typename A, typename B, typename C>
static void mul_mat(MatRef<A> a, MatRef<B> b, MatRef<C> c) {
  mul_mat(a, b, c, static_cast<C>(1.0), static_cast<C>(1.0));
}

template <typename A, typename B, typename C>
static void mul_mat_clearc(MatRef<A> a, MatRef<B> b, MatRef<C> c) {
  mul_mat(a, b, c, static_cast<C>(1.0), static_cast<C>(0.0));
}

template <typename A, typename B, typename C>
static void decode_mul_mat_clearc(MatRef<A> a, MatRef<B> b, MatRef<C> c) {
  decode_mul_mat(a, b, c, static_cast<C>(1.0), static_cast<C>(0.0));
}

template <typename A, typename B, typename C>
static void decode_mul_mat(MatRef<A> a, MatRef<B> b, MatRef<C> c) {
  decode_mul_mat(a, b, c, static_cast<C>(1.0), static_cast<C>(1.0));
}

}  // namespace arm_kml

#endif