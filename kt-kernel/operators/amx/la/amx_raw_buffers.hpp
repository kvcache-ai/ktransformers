#ifndef AMX_RAW_BUFFERS_HPP
#define AMX_RAW_BUFFERS_HPP

/**
 * @file amx_raw_buffers.hpp
 * @brief Raw data format buffer management (FP8, BF16, etc.)
 *
 * 本文件实现原精度格式的缓冲区管理，用于 DeepSeek V3.2 等原精度推理。
 *
 * 缓冲区类型：
 * - BufferAFP8Impl: 输入激活缓冲区，支持动态 FP8 量化
 * - BufferBFP8Impl: 权重缓冲区，FP8 格式 + 128x128 块缩放
 * - BufferBFP8BlockImpl: 优化的块量化权重缓冲区
 *
 * 内存布局：
 * - FP8 数据：1 字节/元素
 * - Scale：4 字节/块（BufferB 每 128x128 块一个，BufferA 每 128 行一个）
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "amx_config.hpp"
#include "amx_raw_utils.hpp"
#include "amx_utils.hpp"
#include "llama.cpp/ggml-impl.h"
#include "pack.hpp"
#include "utils.hpp"

namespace amx {

// ============================================================================
// BufferAFP8Impl: FP8 激活缓冲区（支持动态量化）
// ============================================================================
/* 物理布局(按 bf16 元素数)：
 * 逻辑矩阵 A 为 (m, k) 行主序，m pad 到 max_m(=m_block_size，M_STEP 的倍数)。
 * 存储顺序：
 *   k_block(K_BLOCK 列) → m_block(M_STEP 行) → k_step(K_STEP 列) → (M_STEP×K_STEP) 行主序 tile。
 * 因此可视为 5D：
 *   a[k_blocks][m_blocks][k_steps][M_STEP][K_STEP]，
 *   k_blocks = ceil(k / K_BLOCK)，m_blocks = max_m / M_STEP，
 *   k_steps = K_BLOCK / K_STEP（最后一个 k_block 可能更小）。
 * get_submat(m_begin, k_begin) 返回连续的 (M_STEP×K_STEP) tile。
*/
template <typename K>
struct BufferABF16Impl {
  ggml_bf16_t* a;
  int max_m, k;
  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;

  static size_t required_size(int max_m, int k) { return sizeof(ggml_bf16_t) * max_m * k; }

  BufferABF16Impl(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(k % K_STEP == 0);
    a = reinterpret_cast<ggml_bf16_t*>(ptr);
  }

  void set_data(void* new_ptr) { a = reinterpret_cast<ggml_bf16_t*>(new_ptr); }

  void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
    assert(m <= max_m);
    assert(ith == 0 && nth == 1);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            __m512i* s = (__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin);
            __m512i* d =
                (__m512i*)(a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP);
            avx512_copy_32xbf16(s, d);
          }
        }
      }
    }
  }

  ggml_bf16_t* get_submat(int m, int k, int m_begin, int k_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
  }
};

// ============================================================================
// BufferB
// ============================================================================

/**
 * @brief BF16 BufferB
 * 物理布局(按 bf16 元素数)：
 * 逻辑矩阵 B 为 (n, k) 行主序（用于 NT GEMM），n 按 N_BLOCK 分块。
 * 存储顺序：
 *   n_block(N_BLOCK 行) → k_block(K_BLOCK 列) → n_step(N_STEP 行) → k_step(K_STEP 列)
 *   → (N_STEP×K_STEP) tile；每个 tile 内部再对两个 16×16 子块做 transpose，
 *   以匹配 AMX BTile 的 VNNI 布局（TILE_K/VNNI_BLK × TILE_N*VNNI_BLK）。
 * 因此可视为 6D：
 *   b[n_blocks][k_blocks][n_steps][k_steps][N_STEP][K_STEP]，
 *   n_blocks = ceil(n / N_BLOCK)，k_blocks = ceil(k / K_BLOCK)，
 *   n_steps = N_BLOCK / N_STEP，k_steps = K_BLOCK / K_STEP（尾块可能更小）。
 * get_submat(n_begin, k_begin) 返回连续的 (N_STEP×K_STEP) tile 起始地址。
 * @tparam K Kernel 类型
 */

template <typename K>
struct BufferBBF16Impl {
  ggml_bf16_t* b;
  int n, k;
  static constexpr bool SCALE = false;
  static constexpr int N_STEP = K::N_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static constexpr int TILE_N = K::TILE_N;
  static size_t required_size(int n, int k) { return sizeof(ggml_bf16_t) * n * k; }

  BufferBBF16Impl(int n, int k, void* ptr) : n(n), k(k) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % K_STEP == 0);
    b = reinterpret_cast<ggml_bf16_t*>(ptr);
  }
  void set_data(void* new_ptr) { b = reinterpret_cast<ggml_bf16_t*>(new_ptr); }

  void from_mat(ggml_bf16_t* src, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < N_STEP; i++) {
            __m512i* s = (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin);
            __m512i* d = (__m512i*)(b + n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                    k_begin * N_STEP + i * K_STEP);
            avx512_copy_32xbf16(s, d);
          }
          transpose_16x16_32bit((__m512i*)(b + n_block_begin * k + k_block_begin * n_block_size +
                                           n_begin * k_block_size + k_begin * N_STEP));
          transpose_16x16_32bit((__m512i*)(b + n_block_begin * k + k_block_begin * n_block_size +
                                           n_begin * k_block_size + k_begin * N_STEP + TILE_N * K_STEP));
        }
      }
    }
  }
  ggml_bf16_t* get_submat(int n, int k, int n_begin, int k_begin) {
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    n_begin -= n_block_begin;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return b + n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP;
  }
};

/**
 * @brief FP8 权重缓冲区
 *
 * 存储 FP8 格式的权重矩阵，每个 128x128 块有一个缩放因子。
 * 这与 DeepSeek V3.2 的原精度格式匹配。
 *
 * @tparam K Kernel 类型
 */
template <typename K>
struct BufferBFP8Impl {
  uint8_t* b;     // FP8 weight
  ggml_bf16_t* d; // scale_inv [n / k_group_size, k / k_group_size]
  int n, k, k_group_size; // k_group_size = 128 in DeepSeek

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static constexpr bool SCALE = true;

  /**
    * @brief 计算所需内存大小
    */
  static size_t required_size(int n, int k, int k_group_size) {
    int n_blocks_n = (n + k_group_size - 1) / k_group_size;
    int n_blocks_k = (k + k_group_size - 1) / k_group_size;
    return sizeof(uint8_t) * n * k +
          sizeof(ggml_bf16_t) * n_blocks_n * n_blocks_k;
  }

  /**
    * @brief 构造函数
    */
  BufferBFP8Impl(int n, int k, int k_group_size, void* ptr) : n(n), k(k), k_group_size(k_group_size) {
    set_data(ptr);
  }

  void set_data(void* ptr) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    b = reinterpret_cast<uint8_t*>(ptr);
    d = reinterpret_cast<ggml_bf16_t*>(b + (size_t)n * k);
  }

  static constexpr int mat_offset[8] = {0, 2, 4, 6, 1, 3, 5, 7}; // fp8 matrix offset for reordering
  /**
    * @brief 从原始 FP8 权重加载（已经是量化格式）
    *
    * @param b_src FP8 权重源数据 (n-major, n×k)
    * @param d_src BF16 scale_inv 源数据 (n-major, ceil(n/128)×ceil(k/128))
    */
  void from_mat(const uint8_t* b_src, const float* d_src, int n_src, int k_src, int ith, int nth) {
    // correct !
    assert(b != nullptr && d != nullptr);
    assert(n_src == n && k_src == k);
    assert(N_STEP == 32 && K_STEP == 32); // from mat block copy assumes this

    // Copy scales (per 128x128 block). Each thread copies its own n-block range.
    const int n_blocks_k = (k + k_group_size - 1) / k_group_size;
    if (d_src != nullptr) {
      auto [n_start, n_end] = K::split_range_n(n, ith, nth);
      int bn_start = n_start / k_group_size;
      int bn_end = (n_end + k_group_size - 1) / k_group_size;
      memcpy(d + bn_start * n_blocks_k,
             d_src + bn_start * n_blocks_k,
             sizeof(ggml_bf16_t) * (bn_end - bn_start) * n_blocks_k);
    }

    // Reorder FP8 weights into KT block-major layout (same panel->tile order as BF16 BufferB).
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      int n_step_size = std::min(N_STEP, n_block_size - n_begin);
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          int k_step_size = std::min(K_STEP, k_block_size - k_begin);
          // [k_step_size, n_step_size] block copy
          const uint8_t* block_b_src = b_src + (size_t)(n_block_begin + n_begin) * k + k_block_begin + k_begin;
          uint64_t* block_b_dst = reinterpret_cast<uint64_t*>(
              b + (size_t)n_block_begin * k +
              (size_t)k_block_begin * n_block_size +
              (size_t)n_begin * k_block_size +
              (size_t)k_begin * N_STEP);
          for (int i = 0; i < 8; i++) {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(block_b_src + (size_t)i * k * 4);
            for (int j = 0; j < 16; j ++) {
              uint64_t val = (((uint64_t)s[j])) | 
                             (((uint64_t)s[j + (k / 2) * 1]) << 16) |
                             (((uint64_t)s[j + (k / 2) * 2]) << 32) |
                             (((uint64_t)s[j + (k / 2) * 3]) << 48);
              block_b_dst[8 * j + mat_offset[i]] = val;
            }
          }
        }
      }
    }
  }

  /**
    * @brief get scale_inv
    */
  ggml_bf16_t* get_scale(int n, int n_begin, int k, int k_begin) {
    int n_blocks_k = (k + k_group_size - 1) / k_group_size;
    int bn = n_begin / k_group_size;
    int bk = k_begin / k_group_size;
    return d + bn * n_blocks_k + bk;
  }

  /**
    * @brief 获取子矩阵指针
    */
  uint8_t* get_submat(int n, int k, int n_begin, int k_begin) {
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    n_begin -= n_block_begin;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return b + (size_t)n_block_begin * k +
           (size_t)k_block_begin * n_block_size +
           (size_t)n_begin * k_block_size +
           (size_t)k_begin * N_STEP;
  }

    /**
     * @brief write_buffer
     *
     * @param dst BF16 输出缓冲区
     * @param n_begin N 起始索引
     * @param k_begin K 起始索引
     */
    void to_mat(uint8_t* b_dst, ggml_bf16_t* d_dst, int ith, int nth) const {
      //TODO: not implemented now, the inverse of from_mat
    }
};

// ============================================================================
// BufferCFP8Impl: FP32 输出缓冲区
// ============================================================================

/**
 * @brief FP32 输出缓冲区
 *
 * 存储 FP32 格式的累加器，支持转换为 BF16 输出
 *
 * @tparam K Kernel 类型
 */
template <typename K>
struct BufferCFP32Impl {
  float* c;
  int max_m, n;
  static constexpr int M_STEP = K::M_STEP;
  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  // 物理布局(按 float 元素数)：
  // 逻辑矩阵 C 为 (max_m, n) 行主序，max_m 为 M_STEP 的倍数，
  // n 按 N_BLOCK 分块。
  // 存储顺序：
  //   n_block(N_BLOCK 列) → m_block(M_STEP 行) → n_step(N_STEP 列) → (M_STEP×N_STEP) 行主序 tile。
  // 因此可视为 5D：
  //   c[n_blocks][m_blocks][n_steps][M_STEP][N_STEP]，
  //   n_blocks = ceil(n / N_BLOCK)，m_blocks = max_m / M_STEP，
  //   n_steps = N_BLOCK / N_STEP（尾块可能更小）。
  // get_submat(m_begin, n_begin) 返回连续的 (M_STEP×N_STEP) tile 起始地址。

  static size_t required_size(int max_m, int n) { return sizeof(float) * max_m * n; }

  BufferCFP32Impl(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    c = reinterpret_cast<float*>(ptr);
  }

  void set_data(void* new_ptr) { c = reinterpret_cast<float*>(new_ptr); }

  void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
    assert(m <= max_m);
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          __m512* x0 =
              (__m512*)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP);
          __m512* x1 = (__m512*)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP +
                                  i * N_STEP + 16);
          avx512_32xfp32_to_32xbf16(x0, x1, (__m512i*)(dst + (m_begin + i) * n + n_block_begin + n_begin));
        }
      }
    }
  }

  float* get_submat(int m, int n, int m_begin, int n_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    n_begin -= n_block_begin;
    return c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP;
  }
};

template <typename K>
struct BufferCFP32ReduceImpl {
  float* c;
  float* reduce_buf;
  int max_m, n;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;

  static size_t required_size(int max_m, int n) { return sizeof(float) * (size_t)max_m * n * 2; }

  BufferCFP32ReduceImpl(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    set_data(ptr);
  }

  void set_data(void* ptr) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    c = reinterpret_cast<float*>(ptr);
    reduce_buf = c + (size_t)max_m * n;
  }

  void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
    assert(m <= max_m);
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          __m512* x0 =
              (__m512*)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP);
          __m512* x1 = (__m512*)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP +
                                  i * N_STEP + 16);
          avx512_32xfp32_to_32xbf16(x0, x1, (__m512i*)(dst + (m_begin + i) * n + n_block_begin + n_begin));
        }
      }
    }
  }

  float* get_submat(int m, int n, int m_begin, int n_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    n_begin -= n_block_begin;
    return c + (size_t)m_block_size * n_block_begin + (size_t)m_begin * n_block_size + (size_t)n_begin * M_STEP;
  }

  float* get_reduce_submat(int m, int n, int m_begin, int n_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    n_begin -= n_block_begin;
    return reduce_buf + (size_t)m_block_size * n_block_begin + (size_t)m_begin * n_block_size + (size_t)n_begin * M_STEP;
  }
};

}  // namespace amx

#endif  // AMX_RAW_BUFFERS_HPP
