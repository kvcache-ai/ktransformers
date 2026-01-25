/**
 * @file test_lora_kernel.cpp
 * @brief Unit test for LoRA AVX512 kernel - correctness and performance
 *
 * Build:
 *   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512bf16 -std=c++17 \
 *       test_lora_kernel.cpp -o test_lora_kernel -lpthread
 *
 * Run:
 *   ./test_lora_kernel
 */

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

// BF16 type (use uint16_t as storage)
using ggml_bf16_t = uint16_t;

// BF16 <-> FP32 conversion
inline float bf16_to_fp32(ggml_bf16_t x) {
  uint32_t tmp = static_cast<uint32_t>(x) << 16;
  float result;
  memcpy(&result, &tmp, sizeof(float));
  return result;
}

inline ggml_bf16_t fp32_to_bf16(float x) {
  uint32_t tmp;
  memcpy(&tmp, &x, sizeof(float));
  return static_cast<ggml_bf16_t>(tmp >> 16);
}

#define GGML_BF16_TO_FP32(x) bf16_to_fp32(x)
#define GGML_FP32_TO_BF16(x) fp32_to_bf16(x)

// AVX512 helper: convert 32 BF16 to 2x16 FP32
inline void avx512_32xbf16_to_32xfp32(__m512i* src, __m512* dst0, __m512* dst1) {
  __m256i lo = _mm512_extracti64x4_epi64(*src, 0);
  __m256i hi = _mm512_extracti64x4_epi64(*src, 1);
  *dst0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16));
  *dst1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16));
}

// ============================================================================
// AMX support
// ============================================================================
#ifdef __AMX_TILE__
#define AMX_AVAILABLE 1
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

static bool amx_initialized = false;

bool init_amx() {
  if (amx_initialized) return true;

  unsigned long bitmask = 0;
  if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask) != 0) {
    return false;
  }

  if (!(bitmask & (1 << XFEATURE_XTILEDATA))) {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
      return false;
    }
  }

  amx_initialized = true;
  return true;
}

// AMX tile configuration
struct TileConfig {
  uint8_t palette_id = 1;
  uint8_t start_row = 0;
  uint8_t reserved[14] = {0};
  uint16_t colsb[16] = {0};
  uint8_t rows[16] = {0};

  void set_row_col(int tile, int rows_, int colsb_) {
    rows[tile] = rows_;
    colsb[tile] = colsb_;
  }

  void set_config() { _tile_loadconfig(this); }
};

// Configure AMX for BF16 matmul: A[16,32] x B[16,32]^T -> C[16,16]
// A tile: M=16 rows, K=32 BF16 cols -> 16 rows x 64 bytes
// B tile (VNNI): K/2=16 rows, N*2=32 BF16 cols -> 16 rows x 64 bytes
// C tile: M=16 rows, N=16 FP32 cols -> 16 rows x 64 bytes
void configure_amx_bf16() {
  TileConfig cfg;
  // Tile 0: A matrix [16 rows, 64 bytes per row]
  cfg.set_row_col(0, 16, 64);
  // Tile 1: B matrix in VNNI format [16 rows, 64 bytes per row]
  cfg.set_row_col(1, 16, 64);
  // Tile 2: C matrix [16 rows, 64 bytes per row]
  cfg.set_row_col(2, 16, 64);
  cfg.set_config();
}
#else
#define AMX_AVAILABLE 0
bool init_amx() { return false; }
void configure_amx_bf16() {}
#endif

// ============================================================================
// AMX implementation
// For LoRA: input[T,K] x lora_a^T[K,R] -> output[T,R]
//
// _tile_dpbf16ps computes: C[m,n] += sum_k(A[m,k] * B[n,k])
// where A is [M, K] in row-major and B is [N, K] in VNNI format.
//
// VNNI format for BF16:
//   Original B: [N, K] where B[n, k] is at row n, col k
//   VNNI packed: [K/2, N, 2] - pairs of K values for each N interleaved
//   Memory layout: for k_pair in 0..K/2, for n in 0..N: store B[n, 2*k_pair], B[n, 2*k_pair+1]
//
// Tile dimensions:
//   A tile: 16 rows x 64 bytes = 16 rows x 32 BF16 cols (M=16, K=32)
//   B tile: 16 rows x 64 bytes = 16 k_pairs x 32 BF16 = 16 k_pairs x (16 N * 2) (K/2=16, N=16)
//   C tile: 16 rows x 64 bytes = 16 rows x 16 FP32 cols (M=16, N=16)
// ============================================================================
#if AMX_AVAILABLE
void lora_matmul_amx(const ggml_bf16_t* input,   // [num_tokens, k_dim]
                     const ggml_bf16_t* lora_a,  // [rank, k_dim]
                     float* output,              // [num_tokens, rank]
                     int num_tokens, int k_dim, int rank) {
  // AMX tile sizes for BF16
  constexpr int TILE_M = 16;  // rows of A, rows of C
  constexpr int TILE_K = 32;  // K dimension (must be multiple of 2 for BF16 VNNI)
  constexpr int TILE_N = 16;  // cols of C, "rows" of B in logical sense

  // Temporary buffers for tile packing (aligned)
  alignas(64) ggml_bf16_t tile_a[TILE_M * TILE_K];            // A tile: [16, 32] row-major
  alignas(64) ggml_bf16_t tile_b[(TILE_K / 2) * TILE_N * 2];  // B tile: [16, 32] in VNNI format
  alignas(64) float tile_c[TILE_M * TILE_N];                  // C tile: [16, 16]

  // Process tokens in blocks of TILE_M
  for (int t_begin = 0; t_begin < num_tokens; t_begin += TILE_M) {
    int t_end = std::min(t_begin + TILE_M, num_tokens);
    int t_count = t_end - t_begin;

    // Process ranks in blocks of TILE_N
    for (int r_begin = 0; r_begin < rank; r_begin += TILE_N) {
      int r_end = std::min(r_begin + TILE_N, rank);
      int r_count = r_end - r_begin;

      // Zero the C tile
      _tile_zero(2);

      // Accumulate over K dimension
      for (int k_begin = 0; k_begin < k_dim; k_begin += TILE_K) {
        int k_end = std::min(k_begin + TILE_K, k_dim);
        int k_count = k_end - k_begin;

        // Pack A tile: input[t_begin:t_end, k_begin:k_end] -> tile_a[16, 32]
        // Simple row-major layout: A[m, k] at index m * TILE_K + k
        memset(tile_a, 0, sizeof(tile_a));
        for (int ti = 0; ti < t_count; ti++) {
          for (int ki = 0; ki < k_count; ki++) {
            tile_a[ti * TILE_K + ki] = input[(t_begin + ti) * k_dim + k_begin + ki];
          }
        }

        // Pack B tile in VNNI format: lora_a[r_begin:r_end, k_begin:k_end]
        // VNNI format: for each k_pair (0, 2, 4, ...), store all N values for k and k+1 interleaved
        // Layout: tile_b[k_pair * N * 2 + n * 2 + (k % 2)] = lora_a[r, k]
        memset(tile_b, 0, sizeof(tile_b));
        for (int ri = 0; ri < r_count; ri++) {
          for (int ki = 0; ki < k_count; ki++) {
            int k_pair = ki / 2;
            int k_off = ki % 2;
            // VNNI index: k_pair row, then N*2 elements per row, then n*2 + k_off
            tile_b[k_pair * (TILE_N * 2) + ri * 2 + k_off] = lora_a[(r_begin + ri) * k_dim + k_begin + ki];
          }
        }

        // Load tiles and compute
        // A: stride = 64 bytes (32 BF16)
        // B: stride = 64 bytes (16*2 BF16 = 32 BF16)
        _tile_loadd(0, tile_a, TILE_K * sizeof(ggml_bf16_t));
        _tile_loadd(1, tile_b, TILE_N * 2 * sizeof(ggml_bf16_t));
        _tile_dpbf16ps(2, 0, 1);
      }

      // Store C tile
      _tile_stored(2, tile_c, TILE_N * sizeof(float));

      // Copy valid results to output
      for (int ti = 0; ti < t_count; ti++) {
        for (int ri = 0; ri < r_count; ri++) {
          output[(t_begin + ti) * rank + r_begin + ri] = tile_c[ti * TILE_N + ri];
        }
      }
    }
  }
}
#else
void lora_matmul_amx(const ggml_bf16_t* input, const ggml_bf16_t* lora_a, float* output, int num_tokens, int k_dim,
                     int rank) {
  // Fallback to reference when AMX not available
  for (int t = 0; t < num_tokens; t++) {
    for (int r = 0; r < rank; r++) {
      float sum = 0.0f;
      for (int k = 0; k < k_dim; k++) {
        sum += bf16_to_fp32(input[t * k_dim + k]) * bf16_to_fp32(lora_a[r * k_dim + k]);
      }
      output[t * rank + r] = sum;
    }
  }
}
#endif

// ============================================================================
// Reference implementation (naive scalar)
// ============================================================================
void lora_matmul_reference(const ggml_bf16_t* input,   // [num_tokens, k_dim]
                           const ggml_bf16_t* lora_a,  // [rank, k_dim]
                           float* output,              // [num_tokens, rank]
                           int num_tokens, int k_dim, int rank) {
  for (int t = 0; t < num_tokens; t++) {
    for (int r = 0; r < rank; r++) {
      float sum = 0.0f;
      for (int k = 0; k < k_dim; k++) {
        sum += GGML_BF16_TO_FP32(input[t * k_dim + k]) * GGML_BF16_TO_FP32(lora_a[r * k_dim + k]);
      }
      output[t * rank + r] = sum;
    }
  }
}

// ============================================================================
// Old AVX512 implementation (reduce every chunk - BAD)
// ============================================================================
void lora_matmul_avx512_old(const ggml_bf16_t* input, const ggml_bf16_t* lora_a, float* output, int num_tokens,
                            int k_dim, int rank) {
  for (int t = 0; t < num_tokens; t++) {
    const ggml_bf16_t* inp_row = input + t * k_dim;

    for (int r = 0; r < rank; r++) {
      const ggml_bf16_t* w_row = lora_a + r * k_dim;
      float sum = 0.0f;

      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512 inp0, inp1, w0, w1;
        avx512_32xbf16_to_32xfp32((__m512i*)(inp_row + k), &inp0, &inp1);
        avx512_32xbf16_to_32xfp32((__m512i*)(w_row + k), &w0, &w1);
        // BAD: reduce every chunk
        sum += _mm512_reduce_add_ps(_mm512_mul_ps(inp0, w0));
        sum += _mm512_reduce_add_ps(_mm512_mul_ps(inp1, w1));
      }
      for (; k < k_dim; k++) {
        sum += GGML_BF16_TO_FP32(inp_row[k]) * GGML_BF16_TO_FP32(w_row[k]);
      }
      output[t * rank + r] = sum;
    }
  }
}

// ============================================================================
// New AVX512 implementation (8-rank parallel, deferred reduce - GOOD)
// ============================================================================
void lora_matmul_avx512_new(const ggml_bf16_t* input, const ggml_bf16_t* lora_a, float* output, int num_tokens,
                            int k_dim, int rank) {
  constexpr int RANK_BLOCK = 8;

  for (int t = 0; t < num_tokens; t++) {
    const ggml_bf16_t* inp_row = input + t * k_dim;
    float* out_row = output + t * rank;

    int r = 0;
    // Process 8 ranks at a time
    for (; r + RANK_BLOCK <= rank; r += RANK_BLOCK) {
      // 16 accumulators: 2 per rank (for inp0/inp1 halves)
      __m512 acc0_0 = _mm512_setzero_ps(), acc1_0 = _mm512_setzero_ps();
      __m512 acc0_1 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps();
      __m512 acc0_2 = _mm512_setzero_ps(), acc1_2 = _mm512_setzero_ps();
      __m512 acc0_3 = _mm512_setzero_ps(), acc1_3 = _mm512_setzero_ps();
      __m512 acc0_4 = _mm512_setzero_ps(), acc1_4 = _mm512_setzero_ps();
      __m512 acc0_5 = _mm512_setzero_ps(), acc1_5 = _mm512_setzero_ps();
      __m512 acc0_6 = _mm512_setzero_ps(), acc1_6 = _mm512_setzero_ps();
      __m512 acc0_7 = _mm512_setzero_ps(), acc1_7 = _mm512_setzero_ps();

      const ggml_bf16_t* w0 = lora_a + (r + 0) * k_dim;
      const ggml_bf16_t* w1 = lora_a + (r + 1) * k_dim;
      const ggml_bf16_t* w2 = lora_a + (r + 2) * k_dim;
      const ggml_bf16_t* w3 = lora_a + (r + 3) * k_dim;
      const ggml_bf16_t* w4 = lora_a + (r + 4) * k_dim;
      const ggml_bf16_t* w5 = lora_a + (r + 5) * k_dim;
      const ggml_bf16_t* w6 = lora_a + (r + 6) * k_dim;
      const ggml_bf16_t* w7 = lora_a + (r + 7) * k_dim;

      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512 inp0, inp1;
        avx512_32xbf16_to_32xfp32((__m512i*)(inp_row + k), &inp0, &inp1);

        __m512 wv0, wv1;
        avx512_32xbf16_to_32xfp32((__m512i*)(w0 + k), &wv0, &wv1);
        acc0_0 = _mm512_fmadd_ps(inp0, wv0, acc0_0);
        acc1_0 = _mm512_fmadd_ps(inp1, wv1, acc1_0);

        avx512_32xbf16_to_32xfp32((__m512i*)(w1 + k), &wv0, &wv1);
        acc0_1 = _mm512_fmadd_ps(inp0, wv0, acc0_1);
        acc1_1 = _mm512_fmadd_ps(inp1, wv1, acc1_1);

        avx512_32xbf16_to_32xfp32((__m512i*)(w2 + k), &wv0, &wv1);
        acc0_2 = _mm512_fmadd_ps(inp0, wv0, acc0_2);
        acc1_2 = _mm512_fmadd_ps(inp1, wv1, acc1_2);

        avx512_32xbf16_to_32xfp32((__m512i*)(w3 + k), &wv0, &wv1);
        acc0_3 = _mm512_fmadd_ps(inp0, wv0, acc0_3);
        acc1_3 = _mm512_fmadd_ps(inp1, wv1, acc1_3);

        avx512_32xbf16_to_32xfp32((__m512i*)(w4 + k), &wv0, &wv1);
        acc0_4 = _mm512_fmadd_ps(inp0, wv0, acc0_4);
        acc1_4 = _mm512_fmadd_ps(inp1, wv1, acc1_4);

        avx512_32xbf16_to_32xfp32((__m512i*)(w5 + k), &wv0, &wv1);
        acc0_5 = _mm512_fmadd_ps(inp0, wv0, acc0_5);
        acc1_5 = _mm512_fmadd_ps(inp1, wv1, acc1_5);

        avx512_32xbf16_to_32xfp32((__m512i*)(w6 + k), &wv0, &wv1);
        acc0_6 = _mm512_fmadd_ps(inp0, wv0, acc0_6);
        acc1_6 = _mm512_fmadd_ps(inp1, wv1, acc1_6);

        avx512_32xbf16_to_32xfp32((__m512i*)(w7 + k), &wv0, &wv1);
        acc0_7 = _mm512_fmadd_ps(inp0, wv0, acc0_7);
        acc1_7 = _mm512_fmadd_ps(inp1, wv1, acc1_7);
      }

      // Final reduce (only once per rank block)
      out_row[r + 0] = _mm512_reduce_add_ps(acc0_0) + _mm512_reduce_add_ps(acc1_0);
      out_row[r + 1] = _mm512_reduce_add_ps(acc0_1) + _mm512_reduce_add_ps(acc1_1);
      out_row[r + 2] = _mm512_reduce_add_ps(acc0_2) + _mm512_reduce_add_ps(acc1_2);
      out_row[r + 3] = _mm512_reduce_add_ps(acc0_3) + _mm512_reduce_add_ps(acc1_3);
      out_row[r + 4] = _mm512_reduce_add_ps(acc0_4) + _mm512_reduce_add_ps(acc1_4);
      out_row[r + 5] = _mm512_reduce_add_ps(acc0_5) + _mm512_reduce_add_ps(acc1_5);
      out_row[r + 6] = _mm512_reduce_add_ps(acc0_6) + _mm512_reduce_add_ps(acc1_6);
      out_row[r + 7] = _mm512_reduce_add_ps(acc0_7) + _mm512_reduce_add_ps(acc1_7);

      // Scalar tail for k_dim
      for (int rr = 0; rr < RANK_BLOCK; rr++) {
        float tail_sum = 0.0f;
        for (int kk = k; kk < k_dim; kk++) {
          tail_sum += GGML_BF16_TO_FP32(inp_row[kk]) * GGML_BF16_TO_FP32(lora_a[(r + rr) * k_dim + kk]);
        }
        out_row[r + rr] += tail_sum;
      }
    }

    // Remainder ranks (< 8)
    for (; r < rank; r++) {
      const ggml_bf16_t* w_row = lora_a + r * k_dim;
      __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512 inp0, inp1, wv0, wv1;
        avx512_32xbf16_to_32xfp32((__m512i*)(inp_row + k), &inp0, &inp1);
        avx512_32xbf16_to_32xfp32((__m512i*)(w_row + k), &wv0, &wv1);
        acc0 = _mm512_fmadd_ps(inp0, wv0, acc0);
        acc1 = _mm512_fmadd_ps(inp1, wv1, acc1);
      }
      float sum = _mm512_reduce_add_ps(acc0) + _mm512_reduce_add_ps(acc1);
      for (; k < k_dim; k++) {
        sum += GGML_BF16_TO_FP32(inp_row[k]) * GGML_BF16_TO_FP32(w_row[k]);
      }
      out_row[r] = sum;
    }
  }
}

// ============================================================================
// Optimized AVX512 with native BF16 dot product + 12-rank parallel + prefetch
// ============================================================================
void lora_matmul_avx512_opt(const ggml_bf16_t* input, const ggml_bf16_t* lora_a, float* output, int num_tokens,
                            int k_dim, int rank) {
  constexpr int RANK_BLOCK = 12;  // 12 ranks = 24 accumulators, fits in 32 ZMM regs

  for (int t = 0; t < num_tokens; t++) {
    const ggml_bf16_t* inp_row = input + t * k_dim;
    float* out_row = output + t * rank;

    int r = 0;
    // Process 12 ranks at a time
    for (; r + RANK_BLOCK <= rank; r += RANK_BLOCK) {
      // 12 accumulators using native BF16 dpbf16
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();
      __m512 acc4 = _mm512_setzero_ps();
      __m512 acc5 = _mm512_setzero_ps();
      __m512 acc6 = _mm512_setzero_ps();
      __m512 acc7 = _mm512_setzero_ps();
      __m512 acc8 = _mm512_setzero_ps();
      __m512 acc9 = _mm512_setzero_ps();
      __m512 acc10 = _mm512_setzero_ps();
      __m512 acc11 = _mm512_setzero_ps();

      const ggml_bf16_t* w0 = lora_a + (r + 0) * k_dim;
      const ggml_bf16_t* w1 = lora_a + (r + 1) * k_dim;
      const ggml_bf16_t* w2 = lora_a + (r + 2) * k_dim;
      const ggml_bf16_t* w3 = lora_a + (r + 3) * k_dim;
      const ggml_bf16_t* w4 = lora_a + (r + 4) * k_dim;
      const ggml_bf16_t* w5 = lora_a + (r + 5) * k_dim;
      const ggml_bf16_t* w6 = lora_a + (r + 6) * k_dim;
      const ggml_bf16_t* w7 = lora_a + (r + 7) * k_dim;
      const ggml_bf16_t* w8 = lora_a + (r + 8) * k_dim;
      const ggml_bf16_t* w9 = lora_a + (r + 9) * k_dim;
      const ggml_bf16_t* w10 = lora_a + (r + 10) * k_dim;
      const ggml_bf16_t* w11 = lora_a + (r + 11) * k_dim;

      int k = 0;
      // Main loop with prefetch - process 64 BF16 (2x32) per iteration
      for (; k + 64 <= k_dim; k += 64) {
        // Prefetch next cache lines (64 bytes = 32 BF16)
        _mm_prefetch((const char*)(inp_row + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w0 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w1 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w2 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w3 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w4 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w5 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w6 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w7 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w8 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w9 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w10 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(w11 + k + 128), _MM_HINT_T0);

        // First 32 BF16
        __m512bh inp_bf16_0 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row + k));
        acc0 = _mm512_dpbf16_ps(acc0, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w0 + k)));
        acc1 = _mm512_dpbf16_ps(acc1, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w1 + k)));
        acc2 = _mm512_dpbf16_ps(acc2, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w2 + k)));
        acc3 = _mm512_dpbf16_ps(acc3, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w3 + k)));
        acc4 = _mm512_dpbf16_ps(acc4, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w4 + k)));
        acc5 = _mm512_dpbf16_ps(acc5, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w5 + k)));
        acc6 = _mm512_dpbf16_ps(acc6, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w6 + k)));
        acc7 = _mm512_dpbf16_ps(acc7, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w7 + k)));
        acc8 = _mm512_dpbf16_ps(acc8, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w8 + k)));
        acc9 = _mm512_dpbf16_ps(acc9, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w9 + k)));
        acc10 = _mm512_dpbf16_ps(acc10, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w10 + k)));
        acc11 = _mm512_dpbf16_ps(acc11, inp_bf16_0, (__m512bh)_mm512_loadu_si512((__m512i*)(w11 + k)));

        // Second 32 BF16
        __m512bh inp_bf16_1 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row + k + 32));
        acc0 = _mm512_dpbf16_ps(acc0, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w0 + k + 32)));
        acc1 = _mm512_dpbf16_ps(acc1, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w1 + k + 32)));
        acc2 = _mm512_dpbf16_ps(acc2, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w2 + k + 32)));
        acc3 = _mm512_dpbf16_ps(acc3, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w3 + k + 32)));
        acc4 = _mm512_dpbf16_ps(acc4, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w4 + k + 32)));
        acc5 = _mm512_dpbf16_ps(acc5, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w5 + k + 32)));
        acc6 = _mm512_dpbf16_ps(acc6, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w6 + k + 32)));
        acc7 = _mm512_dpbf16_ps(acc7, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w7 + k + 32)));
        acc8 = _mm512_dpbf16_ps(acc8, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w8 + k + 32)));
        acc9 = _mm512_dpbf16_ps(acc9, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w9 + k + 32)));
        acc10 = _mm512_dpbf16_ps(acc10, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w10 + k + 32)));
        acc11 = _mm512_dpbf16_ps(acc11, inp_bf16_1, (__m512bh)_mm512_loadu_si512((__m512i*)(w11 + k + 32)));
      }

      // Handle remaining 32-element blocks
      for (; k + 32 <= k_dim; k += 32) {
        __m512bh inp_bf16 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row + k));
        acc0 = _mm512_dpbf16_ps(acc0, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w0 + k)));
        acc1 = _mm512_dpbf16_ps(acc1, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w1 + k)));
        acc2 = _mm512_dpbf16_ps(acc2, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w2 + k)));
        acc3 = _mm512_dpbf16_ps(acc3, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w3 + k)));
        acc4 = _mm512_dpbf16_ps(acc4, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w4 + k)));
        acc5 = _mm512_dpbf16_ps(acc5, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w5 + k)));
        acc6 = _mm512_dpbf16_ps(acc6, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w6 + k)));
        acc7 = _mm512_dpbf16_ps(acc7, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w7 + k)));
        acc8 = _mm512_dpbf16_ps(acc8, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w8 + k)));
        acc9 = _mm512_dpbf16_ps(acc9, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w9 + k)));
        acc10 = _mm512_dpbf16_ps(acc10, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w10 + k)));
        acc11 = _mm512_dpbf16_ps(acc11, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w11 + k)));
      }

      // Final horizontal reduce
      out_row[r + 0] = _mm512_reduce_add_ps(acc0);
      out_row[r + 1] = _mm512_reduce_add_ps(acc1);
      out_row[r + 2] = _mm512_reduce_add_ps(acc2);
      out_row[r + 3] = _mm512_reduce_add_ps(acc3);
      out_row[r + 4] = _mm512_reduce_add_ps(acc4);
      out_row[r + 5] = _mm512_reduce_add_ps(acc5);
      out_row[r + 6] = _mm512_reduce_add_ps(acc6);
      out_row[r + 7] = _mm512_reduce_add_ps(acc7);
      out_row[r + 8] = _mm512_reduce_add_ps(acc8);
      out_row[r + 9] = _mm512_reduce_add_ps(acc9);
      out_row[r + 10] = _mm512_reduce_add_ps(acc10);
      out_row[r + 11] = _mm512_reduce_add_ps(acc11);

      // Scalar tail
      for (int rr = 0; rr < RANK_BLOCK; rr++) {
        float tail_sum = 0.0f;
        for (int kk = k; kk < k_dim; kk++) {
          tail_sum += GGML_BF16_TO_FP32(inp_row[kk]) * GGML_BF16_TO_FP32(lora_a[(r + rr) * k_dim + kk]);
        }
        out_row[r + rr] += tail_sum;
      }
    }

    // Process remaining ranks with 8-rank kernel
    for (; r + 8 <= rank; r += 8) {
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();
      __m512 acc4 = _mm512_setzero_ps();
      __m512 acc5 = _mm512_setzero_ps();
      __m512 acc6 = _mm512_setzero_ps();
      __m512 acc7 = _mm512_setzero_ps();

      const ggml_bf16_t* w0 = lora_a + (r + 0) * k_dim;
      const ggml_bf16_t* w1 = lora_a + (r + 1) * k_dim;
      const ggml_bf16_t* w2 = lora_a + (r + 2) * k_dim;
      const ggml_bf16_t* w3 = lora_a + (r + 3) * k_dim;
      const ggml_bf16_t* w4 = lora_a + (r + 4) * k_dim;
      const ggml_bf16_t* w5 = lora_a + (r + 5) * k_dim;
      const ggml_bf16_t* w6 = lora_a + (r + 6) * k_dim;
      const ggml_bf16_t* w7 = lora_a + (r + 7) * k_dim;

      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512bh inp_bf16 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row + k));
        acc0 = _mm512_dpbf16_ps(acc0, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w0 + k)));
        acc1 = _mm512_dpbf16_ps(acc1, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w1 + k)));
        acc2 = _mm512_dpbf16_ps(acc2, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w2 + k)));
        acc3 = _mm512_dpbf16_ps(acc3, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w3 + k)));
        acc4 = _mm512_dpbf16_ps(acc4, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w4 + k)));
        acc5 = _mm512_dpbf16_ps(acc5, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w5 + k)));
        acc6 = _mm512_dpbf16_ps(acc6, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w6 + k)));
        acc7 = _mm512_dpbf16_ps(acc7, inp_bf16, (__m512bh)_mm512_loadu_si512((__m512i*)(w7 + k)));
      }

      out_row[r + 0] = _mm512_reduce_add_ps(acc0);
      out_row[r + 1] = _mm512_reduce_add_ps(acc1);
      out_row[r + 2] = _mm512_reduce_add_ps(acc2);
      out_row[r + 3] = _mm512_reduce_add_ps(acc3);
      out_row[r + 4] = _mm512_reduce_add_ps(acc4);
      out_row[r + 5] = _mm512_reduce_add_ps(acc5);
      out_row[r + 6] = _mm512_reduce_add_ps(acc6);
      out_row[r + 7] = _mm512_reduce_add_ps(acc7);

      for (int rr = 0; rr < 8; rr++) {
        float tail_sum = 0.0f;
        for (int kk = k; kk < k_dim; kk++) {
          tail_sum += GGML_BF16_TO_FP32(inp_row[kk]) * GGML_BF16_TO_FP32(lora_a[(r + rr) * k_dim + kk]);
        }
        out_row[r + rr] += tail_sum;
      }
    }

    // Remainder ranks (< 8)
    for (; r < rank; r++) {
      const ggml_bf16_t* w_row = lora_a + r * k_dim;
      __m512 acc = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512bh inp_bf16 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row + k));
        __m512bh w_bf16 = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k));
        acc = _mm512_dpbf16_ps(acc, inp_bf16, w_bf16);
      }
      float sum = _mm512_reduce_add_ps(acc);
      for (; k < k_dim; k++) {
        sum += GGML_BF16_TO_FP32(inp_row[k]) * GGML_BF16_TO_FP32(w_row[k]);
      }
      out_row[r] = sum;
    }
  }
}

// ============================================================================
// Optimized AVX512 v2: Process 2 tokens x 8 ranks to maximize weight reuse
// ============================================================================
void lora_matmul_avx512_opt2(const ggml_bf16_t* input, const ggml_bf16_t* lora_a, float* output, int num_tokens,
                             int k_dim, int rank) {
  constexpr int RANK_BLOCK = 8;
  constexpr int TOKEN_BLOCK = 2;

  int t = 0;
  // Process 2 tokens at a time
  for (; t + TOKEN_BLOCK <= num_tokens; t += TOKEN_BLOCK) {
    const ggml_bf16_t* inp_row0 = input + t * k_dim;
    const ggml_bf16_t* inp_row1 = input + (t + 1) * k_dim;
    float* out_row0 = output + t * rank;
    float* out_row1 = output + (t + 1) * rank;

    int r = 0;
    // Process 8 ranks at a time, 2 tokens
    for (; r + RANK_BLOCK <= rank; r += RANK_BLOCK) {
      // 16 accumulators: 8 ranks x 2 tokens
      __m512 acc_t0_r0 = _mm512_setzero_ps();
      __m512 acc_t0_r1 = _mm512_setzero_ps();
      __m512 acc_t0_r2 = _mm512_setzero_ps();
      __m512 acc_t0_r3 = _mm512_setzero_ps();
      __m512 acc_t0_r4 = _mm512_setzero_ps();
      __m512 acc_t0_r5 = _mm512_setzero_ps();
      __m512 acc_t0_r6 = _mm512_setzero_ps();
      __m512 acc_t0_r7 = _mm512_setzero_ps();
      __m512 acc_t1_r0 = _mm512_setzero_ps();
      __m512 acc_t1_r1 = _mm512_setzero_ps();
      __m512 acc_t1_r2 = _mm512_setzero_ps();
      __m512 acc_t1_r3 = _mm512_setzero_ps();
      __m512 acc_t1_r4 = _mm512_setzero_ps();
      __m512 acc_t1_r5 = _mm512_setzero_ps();
      __m512 acc_t1_r6 = _mm512_setzero_ps();
      __m512 acc_t1_r7 = _mm512_setzero_ps();

      const ggml_bf16_t* w0 = lora_a + (r + 0) * k_dim;
      const ggml_bf16_t* w1 = lora_a + (r + 1) * k_dim;
      const ggml_bf16_t* w2 = lora_a + (r + 2) * k_dim;
      const ggml_bf16_t* w3 = lora_a + (r + 3) * k_dim;
      const ggml_bf16_t* w4 = lora_a + (r + 4) * k_dim;
      const ggml_bf16_t* w5 = lora_a + (r + 5) * k_dim;
      const ggml_bf16_t* w6 = lora_a + (r + 6) * k_dim;
      const ggml_bf16_t* w7 = lora_a + (r + 7) * k_dim;

      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        // Load weights once, reuse for both tokens
        __m512bh w_bf16_0 = (__m512bh)_mm512_loadu_si512((__m512i*)(w0 + k));
        __m512bh w_bf16_1 = (__m512bh)_mm512_loadu_si512((__m512i*)(w1 + k));
        __m512bh w_bf16_2 = (__m512bh)_mm512_loadu_si512((__m512i*)(w2 + k));
        __m512bh w_bf16_3 = (__m512bh)_mm512_loadu_si512((__m512i*)(w3 + k));
        __m512bh w_bf16_4 = (__m512bh)_mm512_loadu_si512((__m512i*)(w4 + k));
        __m512bh w_bf16_5 = (__m512bh)_mm512_loadu_si512((__m512i*)(w5 + k));
        __m512bh w_bf16_6 = (__m512bh)_mm512_loadu_si512((__m512i*)(w6 + k));
        __m512bh w_bf16_7 = (__m512bh)_mm512_loadu_si512((__m512i*)(w7 + k));

        // Token 0
        __m512bh inp_bf16_t0 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row0 + k));
        acc_t0_r0 = _mm512_dpbf16_ps(acc_t0_r0, inp_bf16_t0, w_bf16_0);
        acc_t0_r1 = _mm512_dpbf16_ps(acc_t0_r1, inp_bf16_t0, w_bf16_1);
        acc_t0_r2 = _mm512_dpbf16_ps(acc_t0_r2, inp_bf16_t0, w_bf16_2);
        acc_t0_r3 = _mm512_dpbf16_ps(acc_t0_r3, inp_bf16_t0, w_bf16_3);
        acc_t0_r4 = _mm512_dpbf16_ps(acc_t0_r4, inp_bf16_t0, w_bf16_4);
        acc_t0_r5 = _mm512_dpbf16_ps(acc_t0_r5, inp_bf16_t0, w_bf16_5);
        acc_t0_r6 = _mm512_dpbf16_ps(acc_t0_r6, inp_bf16_t0, w_bf16_6);
        acc_t0_r7 = _mm512_dpbf16_ps(acc_t0_r7, inp_bf16_t0, w_bf16_7);

        // Token 1
        __m512bh inp_bf16_t1 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row1 + k));
        acc_t1_r0 = _mm512_dpbf16_ps(acc_t1_r0, inp_bf16_t1, w_bf16_0);
        acc_t1_r1 = _mm512_dpbf16_ps(acc_t1_r1, inp_bf16_t1, w_bf16_1);
        acc_t1_r2 = _mm512_dpbf16_ps(acc_t1_r2, inp_bf16_t1, w_bf16_2);
        acc_t1_r3 = _mm512_dpbf16_ps(acc_t1_r3, inp_bf16_t1, w_bf16_3);
        acc_t1_r4 = _mm512_dpbf16_ps(acc_t1_r4, inp_bf16_t1, w_bf16_4);
        acc_t1_r5 = _mm512_dpbf16_ps(acc_t1_r5, inp_bf16_t1, w_bf16_5);
        acc_t1_r6 = _mm512_dpbf16_ps(acc_t1_r6, inp_bf16_t1, w_bf16_6);
        acc_t1_r7 = _mm512_dpbf16_ps(acc_t1_r7, inp_bf16_t1, w_bf16_7);
      }

      // Reduce and store
      out_row0[r + 0] = _mm512_reduce_add_ps(acc_t0_r0);
      out_row0[r + 1] = _mm512_reduce_add_ps(acc_t0_r1);
      out_row0[r + 2] = _mm512_reduce_add_ps(acc_t0_r2);
      out_row0[r + 3] = _mm512_reduce_add_ps(acc_t0_r3);
      out_row0[r + 4] = _mm512_reduce_add_ps(acc_t0_r4);
      out_row0[r + 5] = _mm512_reduce_add_ps(acc_t0_r5);
      out_row0[r + 6] = _mm512_reduce_add_ps(acc_t0_r6);
      out_row0[r + 7] = _mm512_reduce_add_ps(acc_t0_r7);
      out_row1[r + 0] = _mm512_reduce_add_ps(acc_t1_r0);
      out_row1[r + 1] = _mm512_reduce_add_ps(acc_t1_r1);
      out_row1[r + 2] = _mm512_reduce_add_ps(acc_t1_r2);
      out_row1[r + 3] = _mm512_reduce_add_ps(acc_t1_r3);
      out_row1[r + 4] = _mm512_reduce_add_ps(acc_t1_r4);
      out_row1[r + 5] = _mm512_reduce_add_ps(acc_t1_r5);
      out_row1[r + 6] = _mm512_reduce_add_ps(acc_t1_r6);
      out_row1[r + 7] = _mm512_reduce_add_ps(acc_t1_r7);

      // Scalar tail
      for (int rr = 0; rr < RANK_BLOCK; rr++) {
        float tail_sum0 = 0.0f, tail_sum1 = 0.0f;
        for (int kk = k; kk < k_dim; kk++) {
          float w = GGML_BF16_TO_FP32(lora_a[(r + rr) * k_dim + kk]);
          tail_sum0 += GGML_BF16_TO_FP32(inp_row0[kk]) * w;
          tail_sum1 += GGML_BF16_TO_FP32(inp_row1[kk]) * w;
        }
        out_row0[r + rr] += tail_sum0;
        out_row1[r + rr] += tail_sum1;
      }
    }

    // Remainder ranks for both tokens
    for (; r < rank; r++) {
      const ggml_bf16_t* w_row = lora_a + r * k_dim;
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512bh w_bf16 = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k));
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row0 + k)), w_bf16);
        acc1 = _mm512_dpbf16_ps(acc1, (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row1 + k)), w_bf16);
      }
      float sum0 = _mm512_reduce_add_ps(acc0);
      float sum1 = _mm512_reduce_add_ps(acc1);
      for (; k < k_dim; k++) {
        float w = GGML_BF16_TO_FP32(w_row[k]);
        sum0 += GGML_BF16_TO_FP32(inp_row0[k]) * w;
        sum1 += GGML_BF16_TO_FP32(inp_row1[k]) * w;
      }
      out_row0[r] = sum0;
      out_row1[r] = sum1;
    }
  }

  // Handle remaining single token
  for (; t < num_tokens; t++) {
    const ggml_bf16_t* inp_row = input + t * k_dim;
    float* out_row = output + t * rank;

    for (int r = 0; r < rank; r++) {
      const ggml_bf16_t* w_row = lora_a + r * k_dim;
      __m512 acc = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512bh inp_bf16 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row + k));
        __m512bh w_bf16 = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k));
        acc = _mm512_dpbf16_ps(acc, inp_bf16, w_bf16);
      }
      float sum = _mm512_reduce_add_ps(acc);
      for (; k < k_dim; k++) {
        sum += GGML_BF16_TO_FP32(inp_row[k]) * GGML_BF16_TO_FP32(w_row[k]);
      }
      out_row[r] = sum;
    }
  }
}

// ============================================================================
// Optimized AVX512 v3: T_BLOCK=4 x R_BLOCK=4 for better arithmetic intensity
//
// Arithmetic intensity analysis:
//   Per k=32 iteration, we load:
//   - 4 weight vectors (4 ranks × 64 bytes = 256 bytes)
//   - 4 input vectors (4 tokens × 64 bytes = 256 bytes)
//   Total: 512 bytes
//   FLOPs: 4 tokens × 4 ranks × 32 elements × 2 = 1024 FLOPs
//   Intensity: 1024 / 512 = 2.0 FLOP/byte
//
// Compare to opt2 (T=2, R=8):
//   - 8 weight vectors = 512 bytes
//   - 2 input vectors = 128 bytes
//   Total: 640 bytes, FLOPs: 1024
//   Intensity: 1024 / 640 = 1.6 FLOP/byte
// ============================================================================
void lora_matmul_avx512_opt3(const ggml_bf16_t* input, const ggml_bf16_t* lora_a, float* output, int num_tokens,
                             int k_dim, int rank) {
  constexpr int T_BLOCK = 4;
  constexpr int R_BLOCK = 4;

  int t = 0;
  // Process 4 tokens at a time
  for (; t + T_BLOCK <= num_tokens; t += T_BLOCK) {
    const ggml_bf16_t* inp0 = input + (t + 0) * k_dim;
    const ggml_bf16_t* inp1 = input + (t + 1) * k_dim;
    const ggml_bf16_t* inp2 = input + (t + 2) * k_dim;
    const ggml_bf16_t* inp3 = input + (t + 3) * k_dim;
    float* out0 = output + (t + 0) * rank;
    float* out1 = output + (t + 1) * rank;
    float* out2 = output + (t + 2) * rank;
    float* out3 = output + (t + 3) * rank;

    int r = 0;
    // Process 4 ranks at a time
    for (; r + R_BLOCK <= rank; r += R_BLOCK) {
      // 16 accumulators: 4 tokens × 4 ranks
      __m512 acc_t0_r0 = _mm512_setzero_ps(), acc_t0_r1 = _mm512_setzero_ps();
      __m512 acc_t0_r2 = _mm512_setzero_ps(), acc_t0_r3 = _mm512_setzero_ps();
      __m512 acc_t1_r0 = _mm512_setzero_ps(), acc_t1_r1 = _mm512_setzero_ps();
      __m512 acc_t1_r2 = _mm512_setzero_ps(), acc_t1_r3 = _mm512_setzero_ps();
      __m512 acc_t2_r0 = _mm512_setzero_ps(), acc_t2_r1 = _mm512_setzero_ps();
      __m512 acc_t2_r2 = _mm512_setzero_ps(), acc_t2_r3 = _mm512_setzero_ps();
      __m512 acc_t3_r0 = _mm512_setzero_ps(), acc_t3_r1 = _mm512_setzero_ps();
      __m512 acc_t3_r2 = _mm512_setzero_ps(), acc_t3_r3 = _mm512_setzero_ps();

      const ggml_bf16_t* w0 = lora_a + (r + 0) * k_dim;
      const ggml_bf16_t* w1 = lora_a + (r + 1) * k_dim;
      const ggml_bf16_t* w2 = lora_a + (r + 2) * k_dim;
      const ggml_bf16_t* w3 = lora_a + (r + 3) * k_dim;

      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        // Load weights once (4 cache lines)
        __m512bh wv0 = (__m512bh)_mm512_loadu_si512((__m512i*)(w0 + k));
        __m512bh wv1 = (__m512bh)_mm512_loadu_si512((__m512i*)(w1 + k));
        __m512bh wv2 = (__m512bh)_mm512_loadu_si512((__m512i*)(w2 + k));
        __m512bh wv3 = (__m512bh)_mm512_loadu_si512((__m512i*)(w3 + k));

        // Load inputs (4 cache lines) and compute
        __m512bh iv0 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp0 + k));
        acc_t0_r0 = _mm512_dpbf16_ps(acc_t0_r0, iv0, wv0);
        acc_t0_r1 = _mm512_dpbf16_ps(acc_t0_r1, iv0, wv1);
        acc_t0_r2 = _mm512_dpbf16_ps(acc_t0_r2, iv0, wv2);
        acc_t0_r3 = _mm512_dpbf16_ps(acc_t0_r3, iv0, wv3);

        __m512bh iv1 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp1 + k));
        acc_t1_r0 = _mm512_dpbf16_ps(acc_t1_r0, iv1, wv0);
        acc_t1_r1 = _mm512_dpbf16_ps(acc_t1_r1, iv1, wv1);
        acc_t1_r2 = _mm512_dpbf16_ps(acc_t1_r2, iv1, wv2);
        acc_t1_r3 = _mm512_dpbf16_ps(acc_t1_r3, iv1, wv3);

        __m512bh iv2 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp2 + k));
        acc_t2_r0 = _mm512_dpbf16_ps(acc_t2_r0, iv2, wv0);
        acc_t2_r1 = _mm512_dpbf16_ps(acc_t2_r1, iv2, wv1);
        acc_t2_r2 = _mm512_dpbf16_ps(acc_t2_r2, iv2, wv2);
        acc_t2_r3 = _mm512_dpbf16_ps(acc_t2_r3, iv2, wv3);

        __m512bh iv3 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp3 + k));
        acc_t3_r0 = _mm512_dpbf16_ps(acc_t3_r0, iv3, wv0);
        acc_t3_r1 = _mm512_dpbf16_ps(acc_t3_r1, iv3, wv1);
        acc_t3_r2 = _mm512_dpbf16_ps(acc_t3_r2, iv3, wv2);
        acc_t3_r3 = _mm512_dpbf16_ps(acc_t3_r3, iv3, wv3);
      }

      // Reduce and store
      out0[r + 0] = _mm512_reduce_add_ps(acc_t0_r0);
      out0[r + 1] = _mm512_reduce_add_ps(acc_t0_r1);
      out0[r + 2] = _mm512_reduce_add_ps(acc_t0_r2);
      out0[r + 3] = _mm512_reduce_add_ps(acc_t0_r3);
      out1[r + 0] = _mm512_reduce_add_ps(acc_t1_r0);
      out1[r + 1] = _mm512_reduce_add_ps(acc_t1_r1);
      out1[r + 2] = _mm512_reduce_add_ps(acc_t1_r2);
      out1[r + 3] = _mm512_reduce_add_ps(acc_t1_r3);
      out2[r + 0] = _mm512_reduce_add_ps(acc_t2_r0);
      out2[r + 1] = _mm512_reduce_add_ps(acc_t2_r1);
      out2[r + 2] = _mm512_reduce_add_ps(acc_t2_r2);
      out2[r + 3] = _mm512_reduce_add_ps(acc_t2_r3);
      out3[r + 0] = _mm512_reduce_add_ps(acc_t3_r0);
      out3[r + 1] = _mm512_reduce_add_ps(acc_t3_r1);
      out3[r + 2] = _mm512_reduce_add_ps(acc_t3_r2);
      out3[r + 3] = _mm512_reduce_add_ps(acc_t3_r3);

      // Scalar tail for k
      for (int rr = 0; rr < R_BLOCK; rr++) {
        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int kk = k; kk < k_dim; kk++) {
          float w = GGML_BF16_TO_FP32(lora_a[(r + rr) * k_dim + kk]);
          sum0 += GGML_BF16_TO_FP32(inp0[kk]) * w;
          sum1 += GGML_BF16_TO_FP32(inp1[kk]) * w;
          sum2 += GGML_BF16_TO_FP32(inp2[kk]) * w;
          sum3 += GGML_BF16_TO_FP32(inp3[kk]) * w;
        }
        out0[r + rr] += sum0;
        out1[r + rr] += sum1;
        out2[r + rr] += sum2;
        out3[r + rr] += sum3;
      }
    }

    // Remainder ranks
    for (; r < rank; r++) {
      const ggml_bf16_t* w_row = lora_a + r * k_dim;
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512bh wv = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k));
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512((__m512i*)(inp0 + k)), wv);
        acc1 = _mm512_dpbf16_ps(acc1, (__m512bh)_mm512_loadu_si512((__m512i*)(inp1 + k)), wv);
        acc2 = _mm512_dpbf16_ps(acc2, (__m512bh)_mm512_loadu_si512((__m512i*)(inp2 + k)), wv);
        acc3 = _mm512_dpbf16_ps(acc3, (__m512bh)_mm512_loadu_si512((__m512i*)(inp3 + k)), wv);
      }
      float sum0 = _mm512_reduce_add_ps(acc0);
      float sum1 = _mm512_reduce_add_ps(acc1);
      float sum2 = _mm512_reduce_add_ps(acc2);
      float sum3 = _mm512_reduce_add_ps(acc3);
      for (; k < k_dim; k++) {
        float w = GGML_BF16_TO_FP32(w_row[k]);
        sum0 += GGML_BF16_TO_FP32(inp0[k]) * w;
        sum1 += GGML_BF16_TO_FP32(inp1[k]) * w;
        sum2 += GGML_BF16_TO_FP32(inp2[k]) * w;
        sum3 += GGML_BF16_TO_FP32(inp3[k]) * w;
      }
      out0[r] = sum0;
      out1[r] = sum1;
      out2[r] = sum2;
      out3[r] = sum3;
    }
  }

  // Handle remaining tokens with 2-token kernel
  for (; t + 2 <= num_tokens; t += 2) {
    const ggml_bf16_t* inp0 = input + t * k_dim;
    const ggml_bf16_t* inp1 = input + (t + 1) * k_dim;
    float* out0 = output + t * rank;
    float* out1 = output + (t + 1) * rank;

    for (int r = 0; r < rank; r++) {
      const ggml_bf16_t* w_row = lora_a + r * k_dim;
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512bh wv = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k));
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512((__m512i*)(inp0 + k)), wv);
        acc1 = _mm512_dpbf16_ps(acc1, (__m512bh)_mm512_loadu_si512((__m512i*)(inp1 + k)), wv);
      }
      float sum0 = _mm512_reduce_add_ps(acc0);
      float sum1 = _mm512_reduce_add_ps(acc1);
      for (; k < k_dim; k++) {
        float w = GGML_BF16_TO_FP32(w_row[k]);
        sum0 += GGML_BF16_TO_FP32(inp0[k]) * w;
        sum1 += GGML_BF16_TO_FP32(inp1[k]) * w;
      }
      out0[r] = sum0;
      out1[r] = sum1;
    }
  }

  // Handle remaining single token
  for (; t < num_tokens; t++) {
    const ggml_bf16_t* inp_row = input + t * k_dim;
    float* out_row = output + t * rank;

    for (int r = 0; r < rank; r++) {
      const ggml_bf16_t* w_row = lora_a + r * k_dim;
      __m512 acc = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        acc = _mm512_dpbf16_ps(acc, (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row + k)),
                               (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k)));
      }
      float sum = _mm512_reduce_add_ps(acc);
      for (; k < k_dim; k++) {
        sum += GGML_BF16_TO_FP32(inp_row[k]) * GGML_BF16_TO_FP32(w_row[k]);
      }
      out_row[r] = sum;
    }
  }
}

// ============================================================================
// Test utilities
// ============================================================================
void fill_random_bf16(ggml_bf16_t* data, size_t count, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < count; i++) {
    data[i] = GGML_FP32_TO_BF16(dist(rng));
  }
}

bool check_correctness(const float* ref, const float* test, size_t count, float rtol = 5e-3f, float atol = 1e-4f) {
  float max_diff = 0.0f;
  float max_rdiff = 0.0f;
  size_t max_diff_idx = 0;

  for (size_t i = 0; i < count; i++) {
    float diff = std::abs(ref[i] - test[i]);
    float rdiff = diff / (std::abs(ref[i]) + 1e-8f);
    if (diff > max_diff) {
      max_diff = diff;
      max_diff_idx = i;
    }
    if (rdiff > max_rdiff) {
      max_rdiff = rdiff;
    }
    if (diff > atol && rdiff > rtol) {
      printf("  MISMATCH at index %zu: ref=%.6f, test=%.6f, diff=%.6e, rdiff=%.6e\n", i, ref[i], test[i], diff, rdiff);
      return false;
    }
  }
  printf("  max_diff=%.6e, max_rdiff=%.6e at index %zu\n", max_diff, max_rdiff, max_diff_idx);
  return true;
}

double benchmark(std::function<void()> fn, int warmup = 3, int iterations = 10) {
  // Warmup
  for (int i = 0; i < warmup; i++) {
    fn();
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    fn();
  }
  auto end = std::chrono::high_resolution_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  return total_ms / iterations;
}

// ============================================================================
// Main test
// ============================================================================
int main(int argc, char** argv) {
  // Initialize AMX if available
  bool has_amx = init_amx();
  if (has_amx) {
    configure_amx_bf16();
    printf("AMX initialized successfully\n");
  } else {
    printf("AMX not available on this system\n");
  }

  // Test configurations
  struct TestConfig {
    int num_tokens;
    int k_dim;
    int rank;
    const char* name;
  };

  std::vector<TestConfig> configs = {
      // Typical DeepSeek LoRA configs
      {1, 7168, 8, "decode: T=1, K=7168, R=8"},
      {1, 7168, 16, "decode: T=1, K=7168, R=16"},
      {1, 7168, 32, "decode: T=1, K=7168, R=32"},
      {32, 7168, 8, "prefill: T=32, K=7168, R=8"},
      {32, 7168, 16, "prefill: T=32, K=7168, R=16"},
      {32, 7168, 32, "prefill: T=32, K=7168, R=32"},
      {128, 7168, 16, "prefill: T=128, K=7168, R=16"},
      {256, 7168, 16, "prefill: T=256, K=7168, R=16"},
      {512, 7168, 16, "prefill: T=512, K=7168, R=16"},
      {1024, 7168, 16, "prefill: T=1024, K=7168, R=16"},
      // intermediate_size cases
      {32, 18432, 16, "down: T=32, K=18432, R=16"},
      {128, 18432, 16, "down: T=128, K=18432, R=16"},
      {512, 18432, 16, "down: T=512, K=18432, R=16"},
      {1024, 18432, 16, "down: T=1024, K=18432, R=16"},
  };

  std::mt19937 rng(42);

  printf("=== LoRA Kernel Unit Test ===\n\n");

  for (const auto& cfg : configs) {
    printf("----------------------------------------\n");
    printf("Config: %s\n", cfg.name);
    printf("  num_tokens=%d, k_dim=%d, rank=%d\n", cfg.num_tokens, cfg.k_dim, cfg.rank);

    // Allocate aligned memory (64-byte alignment for AVX512)
    size_t input_size = (size_t)cfg.num_tokens * cfg.k_dim;
    size_t weight_size = (size_t)cfg.rank * cfg.k_dim;
    size_t output_size = (size_t)cfg.num_tokens * cfg.rank;

    // Pad sizes to 32 elements (64 bytes) for alignment
    size_t input_padded = ((input_size + 31) / 32) * 32;
    size_t weight_padded = ((weight_size + 31) / 32) * 32;

    ggml_bf16_t* input = (ggml_bf16_t*)aligned_alloc(64, input_padded * sizeof(ggml_bf16_t));
    ggml_bf16_t* lora_a = (ggml_bf16_t*)aligned_alloc(64, weight_padded * sizeof(ggml_bf16_t));
    std::vector<float> output_ref(output_size);
    std::vector<float> output_old(output_size);
    std::vector<float> output_new(output_size);
    std::vector<float> output_opt(output_size);
    std::vector<float> output_opt2(output_size);
    std::vector<float> output_opt3(output_size);
    std::vector<float> output_amx(output_size);

    memset(input, 0, input_padded * sizeof(ggml_bf16_t));
    memset(lora_a, 0, weight_padded * sizeof(ggml_bf16_t));

    // Fill random data
    fill_random_bf16(input, input_size, rng);
    fill_random_bf16(lora_a, weight_size, rng);

    // Compute reference
    lora_matmul_reference(input, lora_a, output_ref.data(), cfg.num_tokens, cfg.k_dim, cfg.rank);

    // Compute old AVX512
    lora_matmul_avx512_old(input, lora_a, output_old.data(), cfg.num_tokens, cfg.k_dim, cfg.rank);

    // Compute new AVX512
    lora_matmul_avx512_new(input, lora_a, output_new.data(), cfg.num_tokens, cfg.k_dim, cfg.rank);

    // Check correctness
    printf("\nCorrectness:\n");
    printf("  Old AVX512 vs Reference: ");
    bool old_ok = check_correctness(output_ref.data(), output_old.data(), output_size);
    printf("  %s\n", old_ok ? "PASS" : "FAIL");

    printf("  New AVX512 vs Reference: ");
    bool new_ok = check_correctness(output_ref.data(), output_new.data(), output_size);
    printf("  %s\n", new_ok ? "PASS" : "FAIL");

    // Optimized AVX512 correctness check
    lora_matmul_avx512_opt(input, lora_a, output_opt.data(), cfg.num_tokens, cfg.k_dim, cfg.rank);
    printf("  Opt AVX512 vs Reference: ");
    bool opt_ok = check_correctness(output_ref.data(), output_opt.data(), output_size);
    printf("  %s\n", opt_ok ? "PASS" : "FAIL");

    // Optimized AVX512 v2 (2-token batching) correctness check
    lora_matmul_avx512_opt2(input, lora_a, output_opt2.data(), cfg.num_tokens, cfg.k_dim, cfg.rank);
    printf("  Opt2 AVX512 vs Reference: ");
    bool opt2_ok = check_correctness(output_ref.data(), output_opt2.data(), output_size);
    printf("  %s\n", opt2_ok ? "PASS" : "FAIL");

    // Optimized AVX512 v3 (4-token x 4-rank blocking) correctness check
    lora_matmul_avx512_opt3(input, lora_a, output_opt3.data(), cfg.num_tokens, cfg.k_dim, cfg.rank);
    printf("  Opt3 AVX512 vs Reference: ");
    bool opt3_ok = check_correctness(output_ref.data(), output_opt3.data(), output_size);
    printf("  %s\n", opt3_ok ? "PASS" : "FAIL");

    // AMX correctness check
    if (has_amx) {
      lora_matmul_amx(input, lora_a, output_amx.data(), cfg.num_tokens, cfg.k_dim, cfg.rank);
      printf("  AMX vs Reference: ");
      bool amx_ok = check_correctness(output_ref.data(), output_amx.data(), output_size);
      printf("  %s\n", amx_ok ? "PASS" : "FAIL");
    }

    // Benchmark
    printf("\nPerformance:\n");

    double ref_ms = benchmark(
        [&]() { lora_matmul_reference(input, lora_a, output_ref.data(), cfg.num_tokens, cfg.k_dim, cfg.rank); });

    double old_ms = benchmark(
        [&]() { lora_matmul_avx512_old(input, lora_a, output_old.data(), cfg.num_tokens, cfg.k_dim, cfg.rank); });

    double new_ms = benchmark(
        [&]() { lora_matmul_avx512_new(input, lora_a, output_new.data(), cfg.num_tokens, cfg.k_dim, cfg.rank); });

    double opt_ms = benchmark(
        [&]() { lora_matmul_avx512_opt(input, lora_a, output_opt.data(), cfg.num_tokens, cfg.k_dim, cfg.rank); });

    double opt2_ms = benchmark(
        [&]() { lora_matmul_avx512_opt2(input, lora_a, output_opt2.data(), cfg.num_tokens, cfg.k_dim, cfg.rank); });

    double opt3_ms = benchmark(
        [&]() { lora_matmul_avx512_opt3(input, lora_a, output_opt3.data(), cfg.num_tokens, cfg.k_dim, cfg.rank); });

    double amx_ms = 0.0;
    if (has_amx) {
      amx_ms =
          benchmark([&]() { lora_matmul_amx(input, lora_a, output_amx.data(), cfg.num_tokens, cfg.k_dim, cfg.rank); });
    }

    // Calculate GFLOPS
    double flops = 2.0 * cfg.num_tokens * cfg.k_dim * cfg.rank;
    double ref_gflops = flops / (ref_ms * 1e6);
    double old_gflops = flops / (old_ms * 1e6);
    double new_gflops = flops / (new_ms * 1e6);
    double opt_gflops = flops / (opt_ms * 1e6);
    double opt2_gflops = flops / (opt2_ms * 1e6);
    double opt3_gflops = flops / (opt3_ms * 1e6);
    double amx_gflops = has_amx ? flops / (amx_ms * 1e6) : 0.0;

    printf("  Reference:   %.3f ms (%.2f GFLOPS)\n", ref_ms, ref_gflops);
    printf("  Old AVX512:  %.3f ms (%.2f GFLOPS) - %.2fx vs ref\n", old_ms, old_gflops, ref_ms / old_ms);
    printf("  New AVX512:  %.3f ms (%.2f GFLOPS) - %.2fx vs ref, %.2fx vs old\n", new_ms, new_gflops, ref_ms / new_ms,
           old_ms / new_ms);
    printf("  Opt3 AVX512: %.3f ms (%.2f GFLOPS) - %.2fx vs ref, %.2fx vs new\n", opt3_ms, opt3_gflops,
           ref_ms / opt3_ms, new_ms / opt3_ms);
    if (has_amx) {
      printf("  AMX:         %.3f ms (%.2f GFLOPS) - %.2fx vs ref, %.2fx vs opt3\n", amx_ms, amx_gflops,
             ref_ms / amx_ms, opt3_ms / amx_ms);
    }

    // Free aligned memory
    free(input);
    free(lora_a);

    printf("\n");
  }

  printf("=== Test Complete ===\n");
  return 0;
}
