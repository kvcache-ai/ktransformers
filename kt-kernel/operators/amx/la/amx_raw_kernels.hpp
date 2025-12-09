#ifndef AMX_RAW_KERNELS_HPP
#define AMX_RAW_KERNELS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>

#include "amx_config.hpp"
#include "amx_raw_utils.hpp"
#include "amx_utils.hpp"
#include "llama.cpp/ggml-impl.h"

namespace amx {

// FP8 (e4m3) AMX kernel that mirrors the GemmKernel224BF interface.
struct GemmKernel224FP8 {
  using fp8_t = uint8_t;
  using dt = ggml_bf16_t;
  using output_t = float;

  static constexpr double ELEMENT_SIZE = 1.0;
  static const int TILE_M = 16;
  static const int TILE_K = 32;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 2;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int BLOCK_SIZE = fp8::FP8_BLOCK_SIZE;  // 128 x 128 block quantization
  static inline const int N_BLOCK = BLOCK_SIZE;
  static inline const int K_BLOCK = BLOCK_SIZE;

  static std::string name() { return "FP8"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {
#ifdef HAVE_AMX
    enable_amx();
    TileConfig tile_config;

    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(dt));
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));
    for (int i = 4; i < 8; i++) tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
#endif
  }

  static void load_a(dt* a, size_t lda) {
#ifdef HAVE_AMX
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
#else
    (void)a;
    (void)lda;
#endif
  }

  static void load_b(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    _tile_loadd(2, b, ldb);
    _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
#else
    (void)b;
    (void)ldb;
#endif
  }

  static void clean_c() {
#ifdef HAVE_AMX
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
#endif
  }

  static void load_c(output_t* c, size_t ldc) {
#ifdef HAVE_AMX
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
#else
    (void)c;
    (void)ldc;
#endif
  }

  static void store_c(output_t* c, size_t ldc) {
#ifdef HAVE_AMX
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
#else
    (void)c;
    (void)ldc;
#endif
  }

  static void run_tile() {
#ifdef HAVE_AMX
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
#endif
  }

 private:
  static constexpr size_t align64(size_t v) { return (v + 63) & ~size_t(63); }

 public:
  struct BufferA {
    fp8_t* a_fp8;
    dt* a_bf16;
    float* scales;
    int max_m, k;
    int blocks_m, blocks_k;

    static constexpr int M_STEP = GemmKernel224FP8::M_STEP;

    static size_t required_size(int max_m, int k) {
      int blocks_m = (max_m + BLOCK_SIZE - 1) / BLOCK_SIZE;
      int blocks_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
      size_t fp8_bytes = sizeof(fp8_t) * max_m * k;
      size_t scale_bytes = sizeof(float) * blocks_m * blocks_k;
      size_t bf16_bytes = sizeof(dt) * max_m * k;
      size_t scales_offset = align64(fp8_bytes);
      size_t bf16_offset = align64(scales_offset + scale_bytes);
      return bf16_offset + bf16_bytes;
    }

    // Overload for moe_base.hpp compatibility (group_size is ignored for FP8)
    static size_t required_size(int max_m, int k, int /*group_size*/) {
      return required_size(max_m, k);
    }

    BufferA(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_m % M_STEP == 0);
      assert(k % K_STEP == 0);
      blocks_m = (max_m + BLOCK_SIZE - 1) / BLOCK_SIZE;
      blocks_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
      set_data(ptr);
    }

    // Overload for moe_base.hpp compatibility (group_size is ignored for FP8)
    BufferA(int max_m, int k, int /*group_size*/, void* ptr) : BufferA(max_m, k, ptr) {}

    void set_data(void* new_ptr) {
      size_t fp8_bytes = sizeof(fp8_t) * max_m * k;
      size_t scale_bytes = sizeof(float) * blocks_m * blocks_k;
      size_t scales_offset = align64(fp8_bytes);
      size_t bf16_offset = align64(scales_offset + scale_bytes);
      a_fp8 = reinterpret_cast<fp8_t*>(new_ptr);
      scales = reinterpret_cast<float*>(offset_pointer(a_fp8, scales_offset));
      a_bf16 = reinterpret_cast<dt*>(offset_pointer(a_fp8, bf16_offset));
    }

    void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
      assert(m <= max_m);
      assert(ith == 0 && nth == 1);

      // Compute block-wise scales and quantize to FP8
      for (int bm = 0; bm < blocks_m; bm++) {
        int row_start = bm * BLOCK_SIZE;
        int rows = std::min(BLOCK_SIZE, m - row_start);
        if (rows <= 0) break;
        for (int bk = 0; bk < blocks_k; bk++) {
          int col_start = bk * BLOCK_SIZE;
          int cols = std::min(BLOCK_SIZE, k - col_start);
          if (cols <= 0) break;

          float amax = 0.0f;
          for (int i = 0; i < rows; i++) {
            const ggml_bf16_t* src_row = src + (row_start + i) * k + col_start;
            for (int j = 0; j < cols; j++) {
              float v = GGML_BF16_TO_FP32(src_row[j]);
              amax = std::max(amax, std::fabs(v));
            }
          }
          float scale =
              amax > 0.0f ? std::pow(2.0f, std::ceil(std::log2(amax / fp8::FP8_E4M3_MAX))) : 1.0f;
          scales[bm * blocks_k + bk] = scale;
          float inv_scale = scale ? 1.0f / scale : 0.0f;

          for (int i = 0; i < rows; i++) {
            const ggml_bf16_t* src_row = src + (row_start + i) * k + col_start;
            fp8_t* dst_row = a_fp8 + (row_start + i) * k + col_start;
            for (int j = 0; j < cols; j++) {
              dst_row[j] = fp8::fp32_to_fp8_e4m3(GGML_BF16_TO_FP32(src_row[j]) * inv_scale);
            }
          }
        }
      }

      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            int block_m = m_begin / BLOCK_SIZE;
            int block_k = (k_block_begin + k_begin) / BLOCK_SIZE;
            float scale = scales[block_m * blocks_k + block_k];
            dt* dst = a_bf16 + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
            for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
              const fp8_t* src_fp8 = a_fp8 + (m_begin + i) * k + k_block_begin + k_begin;
              fp8::fp8x32_to_bf16x32(src_fp8, dst + i * K_STEP, scale);
            }
          }
        }
      }
    }

    dt* get_submat(int m, int k_dim, int m_begin, int k_begin) {
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k_dim - k_block_begin);
      return a_bf16 + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
    }
  };

  struct BufferB {
    fp8_t* b_fp8;
    dt* b_bf16;
    float* scales;
    int n, k;
    int blocks_n, blocks_k;
    static constexpr bool SCALE = true;

    static size_t required_size(int n, int k) {
      int blocks_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
      int blocks_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
      size_t fp8_bytes = sizeof(fp8_t) * n * k;
      size_t scale_bytes = sizeof(float) * blocks_n * blocks_k;
      size_t bf16_bytes = sizeof(dt) * n * k;
      size_t scales_offset = align64(fp8_bytes);
      size_t bf16_offset = align64(scales_offset + scale_bytes);
      return bf16_offset + bf16_bytes;
    }

    BufferB(int n, int k, void* ptr) : n(n), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % K_STEP == 0);
      blocks_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
      blocks_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
      set_data(ptr);
    }

    void set_data(void* new_ptr) {
      size_t fp8_bytes = sizeof(fp8_t) * n * k;
      size_t scale_bytes = sizeof(float) * blocks_n * blocks_k;
      size_t scales_offset = align64(fp8_bytes);
      size_t bf16_offset = align64(scales_offset + scale_bytes);
      b_fp8 = reinterpret_cast<fp8_t*>(new_ptr);
      scales = reinterpret_cast<float*>(offset_pointer(b_fp8, scales_offset));
      b_bf16 = reinterpret_cast<dt*>(offset_pointer(b_fp8, bf16_offset));
    }

    float get_block_scale(int n_idx, int k_idx) const {
      int block_n = n_idx / BLOCK_SIZE;
      int block_k = k_idx / BLOCK_SIZE;
      return scales[block_n * blocks_k + block_k];
    }

    void from_mat(ggml_bf16_t* src, int ith, int nth) {
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;

      for (int nb = n_start; nb < n_end; nb += BLOCK_SIZE) {
        int rows = std::min(BLOCK_SIZE, n - nb);
        for (int kb = 0; kb < k; kb += BLOCK_SIZE) {
          int cols = std::min(BLOCK_SIZE, k - kb);
          float amax = 0.0f;
          for (int i = 0; i < rows; i++) {
            const ggml_bf16_t* src_row = src + (nb + i) * k + kb;
            for (int j = 0; j < cols; j++) {
              float v = GGML_BF16_TO_FP32(src_row[j]);
              amax = std::max(amax, std::fabs(v));
            }
          }
          float scale =
              amax > 0.0f ? std::pow(2.0f, std::ceil(std::log2(amax / fp8::FP8_E4M3_MAX))) : 1.0f;
          scales[(nb / BLOCK_SIZE) * blocks_k + kb / BLOCK_SIZE] = scale;
          float inv_scale = scale ? 1.0f / scale : 0.0f;

          for (int i = 0; i < rows; i++) {
            const ggml_bf16_t* src_row = src + (nb + i) * k + kb;
            fp8_t* dst_row = b_fp8 + (nb + i) * k + kb;
            for (int j = 0; j < cols; j++) {
              dst_row[j] = fp8::fp32_to_fp8_e4m3(GGML_BF16_TO_FP32(src_row[j]) * inv_scale);
            }
          }
        }
      }

      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            dt* dst = b_bf16 + n_block_begin * k + k_block_begin * n_block_size +
                      n_begin * k_block_size + k_begin * N_STEP;
            for (int i = 0; i < N_STEP && n_begin + i < n_block_size; i++) {
              int global_n = n_block_begin + n_begin + i;
              float scale = scales[(global_n / BLOCK_SIZE) * blocks_k +
                                   (k_block_begin + k_begin) / BLOCK_SIZE];
              const fp8_t* src_fp8 = b_fp8 + global_n * k + k_block_begin + k_begin;
              fp8::fp8x32_to_bf16x32(src_fp8, dst + i * K_STEP, scale);
            }
            transpose_16x16_32bit((__m512i*)dst);
            transpose_16x16_32bit((__m512i*)(dst + TILE_N * K_STEP));
          }
        }
      }
    }

    dt* get_submat(int n_dim, int k_dim, int n_begin, int k_begin) {
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      n_begin -= n_block_begin;
      int n_block_size = std::min(N_BLOCK, n_dim - n_block_begin);
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k_dim - k_block_begin);
      return b_bf16 + n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
             k_begin * N_STEP;
    }
  };

  struct BufferC {
    float* c;
    int max_m, n;

    static constexpr int M_STEP = GemmKernel224FP8::M_STEP;

    static size_t required_size(int max_m, int n) { return sizeof(float) * max_m * n; }

    BufferC(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_m % M_STEP == 0);
      assert(n % N_STEP == 0);
      c = reinterpret_cast<float*>(ptr);
    }

    void set_data(void* new_ptr) { c = reinterpret_cast<float*>(new_ptr); }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      assert(m <= max_m);
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            const float* x0 = c + m_block_size * n_block_begin + m_begin * n_block_size +
                              n_begin * M_STEP + i * N_STEP;
            const float* x1 = x0 + 16;
            __m512 f0 = _mm512_load_ps(x0);
            __m512 f1 = _mm512_load_ps(x1);
            __m512i packed = _mm512_cvtne2ps_pbh(f1, f0);
            _mm512_store_si512(
                (__m512i*)(dst + (m_begin + i) * n + n_block_begin + n_begin), packed);
          }
        }
      }
    }

    float* get_submat(int m_dim, int n_dim, int m_begin, int n_begin) {
      int m_block_size = (m_dim + M_STEP - 1) / M_STEP * M_STEP;
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      int n_block_size = std::min(N_BLOCK, n_dim - n_block_begin);
      n_begin -= n_block_begin;
      return c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP;
    }
  };
};

// ============================================================================
// FP8 GEMM convenience functions (mat_mul_fp8 / vec_mul_fp8)
// ============================================================================

/**
 * @brief FP8 matrix multiplication using AMX
 *
 * Performs: C = A @ B^T where A is BF16 input, B is FP8 weights with block scales
 * FP8->BF16 conversion is done on-the-fly during BufferA/BufferB preparation.
 *
 * @param m Number of rows in A (batch size)
 * @param n Number of rows in B / columns in output (output features)
 * @param k Number of columns in A / B (input features)
 * @param ba Prepared BufferA containing tiled BF16 activations
 * @param bb Prepared BufferB containing FP8 weights converted to tiled BF16
 * @param bc Output BufferC for float32 accumulation
 * @param ith Thread index (0-based)
 * @param nth Total number of threads
 */
inline void mat_mul_fp8(int m, int n, int k,
                        std::shared_ptr<GemmKernel224FP8::BufferA> ba,
                        std::shared_ptr<GemmKernel224FP8::BufferB> bb,
                        std::shared_ptr<GemmKernel224FP8::BufferC> bc,
                        int ith, int nth) {
  using K = GemmKernel224FP8;
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);

        if (k_block_begin == 0) {
          K::clean_c();
        } else {
          K::load_c(c, K::N_STEP * sizeof(float));
        }

        for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k;
             k_begin += K::K_STEP) {
          K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin),
                    K::K_STEP * sizeof(ggml_bf16_t));
          K::load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin),
                    K::K_STEP * sizeof(ggml_bf16_t));
          K::run_tile();
        }

        K::store_c(c, K::N_STEP * sizeof(float));
      }
    }
  }
}

/**
 * @brief FP8 vector multiplication (same as mat_mul_fp8 for small batch)
 *
 * For FP8, vec_mul uses the same implementation as mat_mul since
 * the AMX tile operations are efficient for both cases.
 */
inline void vec_mul_fp8(int m, int n, int k,
                        std::shared_ptr<GemmKernel224FP8::BufferA> ba,
                        std::shared_ptr<GemmKernel224FP8::BufferB> bb,
                        std::shared_ptr<GemmKernel224FP8::BufferC> bc,
                        int ith, int nth) {
  mat_mul_fp8(m, n, k, ba, bb, bc, ith, nth);
}

/**
 * @brief FP8 matrix multiplication with raw pointers (for external BufferB)
 *
 * This variant accepts raw BufferB pointer, useful when BufferB is
 * managed externally (e.g., in MoE with shared weight buffers).
 */
inline void mat_mul_fp8(int m, int n, int k,
                        std::shared_ptr<GemmKernel224FP8::BufferA> ba,
                        GemmKernel224FP8::BufferB* bb,
                        std::shared_ptr<GemmKernel224FP8::BufferC> bc,
                        int ith, int nth) {
  using K = GemmKernel224FP8;
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);

        if (k_block_begin == 0) {
          K::clean_c();
        } else {
          K::load_c(c, K::N_STEP * sizeof(float));
        }

        for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k;
             k_begin += K::K_STEP) {
          K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin),
                    K::K_STEP * sizeof(ggml_bf16_t));
          K::load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin),
                    K::K_STEP * sizeof(ggml_bf16_t));
          K::run_tile();
        }

        K::store_c(c, K::N_STEP * sizeof(float));
      }
    }
  }
}

inline void vec_mul_fp8(int m, int n, int k,
                        std::shared_ptr<GemmKernel224FP8::BufferA> ba,
                        GemmKernel224FP8::BufferB* bb,
                        std::shared_ptr<GemmKernel224FP8::BufferC> bc,
                        int ith, int nth) {
  mat_mul_fp8(m, n, k, ba, bb, bc, ith, nth);
}

/**
 * @brief FP8 matrix multiplication with all raw pointers
 *
 * For use when all buffers are managed externally.
 */
inline void mat_mul_fp8(int m, int n, int k,
                        GemmKernel224FP8::BufferA* ba,
                        GemmKernel224FP8::BufferB* bb,
                        GemmKernel224FP8::BufferC* bc,
                        int ith, int nth) {
  using K = GemmKernel224FP8;
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);

        if (k_block_begin == 0) {
          K::clean_c();
        } else {
          K::load_c(c, K::N_STEP * sizeof(float));
        }

        for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k;
             k_begin += K::K_STEP) {
          K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin),
                    K::K_STEP * sizeof(ggml_bf16_t));
          K::load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin),
                    K::K_STEP * sizeof(ggml_bf16_t));
          K::run_tile();
        }

        K::store_c(c, K::N_STEP * sizeof(float));
      }
    }
  }
}

inline void vec_mul_fp8(int m, int n, int k,
                        GemmKernel224FP8::BufferA* ba,
                        GemmKernel224FP8::BufferB* bb,
                        GemmKernel224FP8::BufferC* bc,
                        int ith, int nth) {
  mat_mul_fp8(m, n, k, ba, bb, bc, ith, nth);
}

}  // namespace amx

#endif  // AMX_RAW_KERNELS_HPP
