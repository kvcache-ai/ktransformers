#ifndef AMX_KERNELS_HPP
#define AMX_KERNELS_HPP
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "amx_buffers.hpp"
#include "amx_config.hpp"
#include "amx_quantization.hpp"
#include "amx_utils.hpp"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llamafile/sgemm.h"
#include "utils.hpp"

namespace amx {

// Compile-time detection: true when AMX intrinsics are available
#if defined(__AMX__) || defined(__AMXINT8__) || defined(__AMXBF16__) || defined(__AMX_TILE__) || defined(HAVE_AMX)
inline constexpr bool AMX_AVAILABLE = true;
#ifndef HAVE_AMX
#define HAVE_AMX
#endif
#else
inline constexpr bool AMX_AVAILABLE = false;
#endif

/*
We use 1-3-3
 C = A x B


A is a row major matrix of size M x K, usually an Linear Layer weight matrix
B is a col major vector of size K x N, usually an input vector, N is usually
quite small

   B
 A C
 A C
 A C

  TMM 0-2: A
  TMM 3: B
  TMM 4-6: C

   3
 0 4
 1 5
 2 6
*/

template <class, class>
struct dpb133 {
  static void run();
};

template <>
inline void dpb133<int8_t, int8_t>::run() {
  _tile_dpbssd(4, 0, 3);
  _tile_dpbssd(5, 1, 3);
  _tile_dpbssd(6, 2, 3);
}

template <>
inline void dpb133<int8_t, uint8_t>::run() {
  _tile_dpbsud(4, 0, 3);
  _tile_dpbsud(5, 1, 3);
  _tile_dpbsud(6, 2, 3);
}

template <>
inline void dpb133<uint8_t, int8_t>::run() {
  _tile_dpbusd(4, 0, 3);
  _tile_dpbusd(5, 1, 3);
  _tile_dpbusd(6, 2, 3);
}

template <>
inline void dpb133<uint8_t, uint8_t>::run() {
  _tile_dpbuud(4, 0, 3);
  _tile_dpbuud(5, 1, 3);
  _tile_dpbuud(6, 2, 3);
}

template <int TILE_K = 32>
struct GemmKernel133 {
  static const int TILE_M = 16;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;
  static const int OUTPUT_T_SIZE = 4;

  static const int M_STEP = TILE_M * 3;
  static const int N_STEP = TILE_N;
  static const int K_STEP = TILE_K;

  static int recommended_nth(int m) { return (m + M_STEP - 1) / M_STEP; }

  static void config() {
#ifdef HAVE_AMX
    TileConfig tile_config;

    for (int i = 0; i < 3; i++) tile_config.set_row_col(i, TILE_M, TILE_K);

    tile_config.set_row_col(3, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK);

    for (int i = 4; i < 7; i++) tile_config.set_row_col(i, TILE_M, TILE_N * OUTPUT_T_SIZE);

    tile_config.set_config();
#endif
  }

  template <typename TA, typename TB, typename TC>
  static void run_full_tile(const TA* a, size_t lda, const TB* b, size_t ldb, TC* c, size_t ldc) {
#ifdef HAVE_AMX
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
    _tile_loadd(2, offset_pointer(a, lda * TILE_M * 2), lda);

    _tile_loadd(3, b, ldb);

    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, ldc * TILE_N), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_N * 2), ldc);

    dpb133<TA, TB>::run();

    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, ldc * TILE_N), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_N * 2), ldc);
#endif
  }

  template <typename TA, typename TB, typename TC>
  static void run_full_tile_zero(const TA* a, size_t lda, const TB* b, size_t ldb, TC* c, size_t ldc) {
#ifdef HAVE_AMX
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
    _tile_loadd(2, offset_pointer(a, lda * TILE_M * 2), lda);

    _tile_loadd(3, b, ldb);

    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);

    dpb133<TA, TB>::run();

    // debug_tiles(7);

    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, ldc * TILE_N), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_N * 2), ldc);
#endif
  }

  static void convert_full_tile_b_to_vnni_inplace(void* b) { transpose_16x8_32bit((__m256i*)b); }

  template <typename TA>
  struct ATile {
    TA v[3 * TILE_M * TILE_K];
    void partial_load(TA* a, int m, int k, size_t lda) {
      // memset(v, 0, sizeof(TA) * 3 * TILE_M * TILE_K);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
          v[i * TILE_K + j] = a[i * lda + j];
        }
      }
    }

    void partial_load_quant(block_q4_0* a, int m, int k, size_t lda) {
      assert(k == 32);
      // memset(v, 0, sizeof(TA) * 3 * TILE_M * TILE_K);
      __m256i* vv = (__m256i*)v;
      for (int i = 0; i < m; i++) {
        vv[i] = dequant4x32(offset_pointer(a, lda * i)->qs);
        vv[i] = _mm256_sub_epi8(vv[i], _mm256_set1_epi8(8));
      }
    }

    void partial_load_quant(block_q8_0* a, int m, int k, size_t lda) {
      assert(k == 32);
      // memset(v, 0, sizeof(TA) * 3 * TILE_M * TILE_K);
      __m256i* vv = (__m256i*)v;
      for (int i = 0; i < m; i++) {
        vv[i] = unaligned_copy8x32(offset_pointer(a, lda * i)->qs);
      }
    }

    template <typename QA>
    void partial_load_quant(TA* a, int m, size_t lda) {
      // memset(v, 0, sizeof(TA) * 3 * TILE_M * TILE_K);
      if constexpr (std::is_same_v<QA, blocks_aligned_q8_0_ref>) {
        __m512i* vv = (__m512i*)v;
        for (int i = 0; i < m; i++) {
          vv[i] = copy8x64(offset_pointer(a, lda * i));
        }
      } else if constexpr (std::is_same_v<QA, blocks_aligned_q4_0_ref>) {
        assert(0);
      } else {
        assert(0);
      }
    }

    void partial_load_quant(block_q4_K* a, int m, int inner_block_idx, size_t lda) {
      // memset(v, 0, sizeof(TA) * 3 * TILE_M * TILE_K);
      __m256i* vv = (__m256i*)v;

      size_t qs_offset = inner_block_idx / 2 * 32;
      for (int i = 0; i < m; i++) {
        block_q4_K* spa = offset_pointer_row_major(a, i, 0, lda);
        if (inner_block_idx % 2 == 0) {
          vv[i] = lo4bit(spa->qs + qs_offset);
        } else {
          vv[i] = hi4bit(spa->qs + qs_offset);
        }
      }
    }

    void partial_load_quant(blocks_aligned_q8_0_ref a, int m, int k, int blck_stride) {
      // memset(v, 0, sizeof(TA) * 3 * TILE_M * TILE_K);
      __m512i* vv = (__m512i*)v;
      for (int i = 0; i < m; i++) {
        vv[i] = copy8x64(a.offset(blck_stride * i).qs);
      }
    }
  };

  template <typename TB>
  struct alignas(64) BTile {
    TB v[TILE_N * TILE_K];
    __m512 scale = {};

    void partial_load(TB* b, int n, int k, size_t ldb) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
          v[i * TILE_K + j] = b[i * ldb + j];
        }
      }
      transpose_16x8_32bit((__m256i*)v);
    }

    void partial_load_quant(block_q8_0* b, int n, int k, size_t ldb) {
      assert(k == 32);
      memset(v, 0, sizeof(TB) * TILE_K * TILE_N);
      __m256i* vv = (__m256i*)v;
      float* bss = reinterpret_cast<float*>(&scale);
      for (int i = 0; i < n; i++) {
        vv[i] = unaligned_copy8x32(offset_pointer(b, ldb * i)->qs);
        float sb = GGML_FP16_TO_FP32(offset_pointer_col_major(b, 0, i, ldb)->d);
        bss[i] = sb;
      }

      transpose_16x8_32bit(vv);
    }

    void partial_load_quant(blocks_aligned_q8_0_ref b, int n, int k, int blck_stride) {
      assert(k == 64);
      memset(v, 0, sizeof(TB) * TILE_K * TILE_N);
      __m512i* vv = (__m512i*)v;
      float* vs = reinterpret_cast<float*>(&scale);
      for (int i = 0; i < n; i++) {
        auto ref = b.offset(blck_stride * i);
        vv[i] = copy8x64(ref.qs);
        float sb = GGML_FP16_TO_FP32(*ref.d);
        vs[i] = sb;
      }
      transpose_16x16_32bit(vv);
    }

    void load_from(TB* b, size_t ldb) {
      __m256i* vb = (__m256i*)b;
      __m256i* vo = (__m256i*)v;
      for (int i = 0; i < 16; i++) {
        vo[i] = *offset_pointer(vb, ldb * i);
      }
      transpose_16x8_32bit(vo);
    }

    template <typename TA, typename TC>
    void run_full_ac(TA* a, size_t lda, TC* c, size_t ldc) {
      run_full_tile(a, lda, v, TILE_N * VNNI_BLK, c, ldc);
    }
  };

  template <typename TB>
  struct alignas(64) BTileSum {
    TB v[TILE_N * TILE_K];
    __m512 scale = {};
    __m512 sum = {};
    void partial_load_quant(block_q8_K* b, int n, int inner_block_idx, size_t ldb) {
      memset(v, 0, TILE_K * TILE_N);
      __m256i* vv = (__m256i*)v;
      float* scale_s = reinterpret_cast<float*>(&scale);
      float* sum_s = reinterpret_cast<float*>(&sum);
      for (int i = 0; i < n; i++) {
        block_q8_K* spb = offset_pointer_col_major(b, 0, i, ldb);
        vv[i] = unaligned_copy8x32(spb->qs + inner_block_idx * 32);
        scale_s[i] = spb->d;
        sum_s[i] =
            spb->bsums[inner_block_idx * 2] + spb->bsums[inner_block_idx * 2 + 1];  // TODO: may this will be slow
        // printf("scale[%d] = %f, sum_s[%d] = %f\n", i, scale_s[i], i,
        // sum_s[i]);
      }
      transpose_16x8_32bit(vv);
    }
  };
  template <typename TC>
  struct alignas(64) CTile {
    static_assert(sizeof(TC) == 4);
    TC v[3 * TILE_M * TILE_N] = {};

    void partial_load(TC* c, int m, int n, size_t ldc) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          v[i * TILE_N + j] = offset_pointer(c, ldc * i)[j];
        }
      }
    }

    void partial_store(TC* c, int m, int n, size_t ldc) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          offset_pointer(c, ldc * i)[j] = v[i * TILE_N + j];
        }
      }
    }

    void to_fp32() {
      __m512i* vv = (__m512i*)v;
      __m512* vf = (__m512*)v;
      for (int i = 0; i < 3 * TILE_M; i++) {
        vf[i] = _mm512_cvtepi32_ps(vv[i]);
      }
    }
  };

  template <typename TA, typename TB, typename TC>
  struct PartialTiles {
    ATile<TA> ta;
    BTile<TB> tb;
    CTile<TC> tc;
    void partial_run(int m, int n, int k, TA* a, size_t lda, TB* b, size_t ldb, TC* c, size_t ldc) {
      ta.partial_load(a, m, k, lda);
      tb.partial_load(b, n, k, ldb);
      tc.partial_load(c, m, n, ldc);
      run_full_tile(ta.v, TILE_K, tb.v, TILE_N * VNNI_BLK, tc.v, TILE_N * OUTPUT_T_SIZE);
      tc.partial_store(c, m, n, ldc);
    }

    template <typename QA>
    void partial_run_quant(int m, int n, int k, QA* a, size_t lda, block_q8_0* b, size_t ldb, float* c, size_t ldc) {
      assert(QK4_0 == 32);
      assert(QK8_0 == 32);

      ta.partial_load_quant(a, m, k, lda);
      tb.partial_load_quant(b, n, k, ldb);

      run_full_tile_zero(ta.v, TILE_K, tb.v, TILE_N * VNNI_BLK, tc.v, TILE_N * OUTPUT_T_SIZE);

      __m512i* cs = (__m512i*)tc.v;
      for (int i = 0; i < m; i++) {
        __m512 as = _mm512_set1_ps(GGML_FP16_TO_FP32(offset_pointer_row_major(a, i, 0, lda)->d));
        __m512* now = reinterpret_cast<__m512*>(offset_pointer_row_major(c, i, 0, ldc));
        *now = _mm512_fmadd_ps(_mm512_mul_ps(as, tb.scale), _mm512_cvtepi32_ps(cs[i]), *now);
      }
    }

    template <typename QA>
    void partial_run_quant_ac(int m, int n, int k, QA* a, size_t lda, float* c, size_t ldc) {
      assert(QK4_0 == 32);
      assert(QK8_0 == 32);

      ta.partial_load_quant(a, m, k, lda);

      run_full_tile_zero(ta.v, TILE_K, tb.v, TILE_N * VNNI_BLK, tc.v, TILE_N * OUTPUT_T_SIZE);

      __m512i* cs = (__m512i*)tc.v;
      for (int i = 0; i < m; i++) {
        __m512 as = _mm512_set1_ps(GGML_FP16_TO_FP32(offset_pointer_row_major(a, i, 0, lda)->d));
        __m512* now = reinterpret_cast<__m512*>(offset_pointer_row_major(c, i, 0, ldc));
        *now = _mm512_fmadd_ps(_mm512_mul_ps(as, tb.scale), _mm512_cvtepi32_ps(cs[i]), *now);
      }
    }

    template <typename AQA>
    void partial_run_quant_ac(int m, int n, int k, AQA a, int a_blck_stride, float* c, size_t ldc) {
      assert(AQA::block_size == 64);

      ta.partial_load_quant(a, m, k, a_blck_stride);

      run_full_tile_zero(ta.v, TILE_K, tb.v, TILE_N * VNNI_BLK, tc.v, TILE_N * OUTPUT_T_SIZE);

      __m512i* cs = (__m512i*)tc.v;
      for (int i = 0; i < m; i++) {
        __m512 as = _mm512_set1_ps(GGML_FP16_TO_FP32(*a.offset(i * a_blck_stride).d));
        // printf("%f\n", GGML_FP16_TO_FP32(*a.offset(i * a_blck_stride).d));
        __m512* now = reinterpret_cast<__m512*>(offset_pointer_row_major(c, i, 0, ldc));
        *now = _mm512_fmadd_ps(_mm512_mul_ps(as, tb.scale), _mm512_cvtepi32_ps(cs[i]), *now);
      }
    }
  };

  template <typename TA, typename TB, typename TC>
  struct PartialTilesSum {
    ATile<TA> ta;
    BTileSum<TB> tb;
    CTile<TC> tc;

    void partial_run_quant_ac(int m, int n, int inner_block_idx, block_q4_K* a, size_t lda, float* c, size_t ldc,
                              float a_scale, float a_min) {
      ta.partial_load_quant(a, m, inner_block_idx, lda);

      run_full_tile_zero(ta.v, TILE_K, tb.v, TILE_N * VNNI_BLK, tc.v, TILE_N * OUTPUT_T_SIZE);

      __m512i* cs = (__m512i*)tc.v;
      for (int i = 0; i < m; i++) {
        __m512* now = reinterpret_cast<__m512*>(offset_pointer_row_major(c, i, 0, ldc));
        *now = _mm512_fmadd_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(cs[i]), _mm512_set1_ps(a_scale)),
                                             _mm512_mul_ps(tb.sum, _mm512_set1_ps(a_min))),
                               tb.scale, *now);
        // C += Bscale * (Ascale * dp - Amin * Bsum)
      }
    }
  };
};

struct GemmKernel133BF {
  using dt = ggml_bf16_t;
  using output_t = float;
  static const int TILE_M = 16;
  static const int TILE_K = 32;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 2;

  static const int M_STEP = TILE_M * 3;
  static const int N_STEP = TILE_N;
  static const int K_STEP = TILE_K;

  static int recommended_nth(int m) { return (m + M_STEP - 1) / M_STEP; }
  static void config() {
#ifdef HAVE_AMX
    enable_amx();
    TileConfig tile_config;

    // size is 16 x 32
    for (int i = 0; i < 3; i++) tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(dt));

    // size is 8 x 64
    tile_config.set_row_col(3, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));

    // size is 16 x 64
    for (int i = 4; i < 7; i++) tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
#endif
  }

  static void run_full_tile(const dt* a, size_t lda, const dt* b, size_t ldb, output_t* c, size_t ldc) {
#ifdef HAVE_AMX
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
    _tile_loadd(2, offset_pointer(a, lda * TILE_M * 2), lda);

    _tile_loadd(3, b, ldb);

    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, ldc * TILE_N), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_N * 2), ldc);

    _tile_dpbf16ps(4, 0, 3);
    _tile_dpbf16ps(5, 1, 3);
    _tile_dpbf16ps(6, 2, 3);

    // debug_tiles(7);

    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, ldc * TILE_N), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_N * 2), ldc);
#endif
  }

  struct ATile {
    dt v[3 * TILE_M * TILE_K];

    void partial_load(dt* a, int m, int k, size_t lda) {
      assert(k == TILE_K);
      __m512* vv = (__m512*)v;
      __m512* va = (__m512*)a;
      for (int i = 0; i < m; i++) {
        vv[i] = *offset_pointer_row_major(va, i, 0, lda);
      }
    }
  };

  struct alignas(64) BTile {
    dt v[TILE_N * TILE_K];

    void full_load(dt* b, size_t ldb) { partial_load(b, TILE_N, TILE_K, ldb); }

    void partial_load(dt* b, int n, int k, size_t ldb) {
      __m512* vv = (__m512*)v;
      __m512* vb = (__m512*)b;
      for (int i = 0; i < n; i++) {
        vv[i] = *offset_pointer_col_major(vb, 0, i, ldb);
      }
      transpose_16x16_32bit((__m512i*)v);
    }

    template <typename TA, typename TC>
    void run_full_ac(TA* a, size_t lda, TC* c, size_t ldc) {
      run_full_tile(a, lda, v, TILE_N * VNNI_BLK * sizeof(dt), c, ldc);
    }
  };

  struct alignas(64) CTile {
    output_t v[3 * TILE_M * TILE_N];
    // c must be 64 aligned, ldc must be 64 aligned
    void partial_load(float* c, int m, int n, size_t ldc) {
      assert(n <= TILE_N);
      __m512* vv = (__m512*)v;
      __m512* vc = (__m512*)c;
      for (int i = 0; i < m; i++) {
        vv[i] = *offset_pointer_row_major(vc, i, 0, ldc);
      }
    }

    void partial_store(float* c, int m, int n, size_t ldc) {
      assert(n <= TILE_N);
      __m512* vv = (__m512*)v;
      __m512* vc = (__m512*)c;
      for (int i = 0; i < m; i++) {
        *offset_pointer_row_major(vc, i, 0, ldc) = vv[i];
      }
    }
  };

  struct PartialTiles {
    ATile ta;
    BTile tb;
    CTile tc;
    void partial_run(int m, int n, int k, dt* a, size_t lda, dt* b, size_t ldb, output_t* c, size_t ldc) {
      ta.partial_load(a, m, k, lda);
      tb.partial_load(b, n, k, ldb);
      tc.partial_load(c, m, n, ldc);
      run_full_tile(ta.v, TILE_K * sizeof(dt), tb.v, TILE_N * VNNI_BLK * sizeof(dt), tc.v, TILE_N * sizeof(output_t));
      tc.partial_store(c, m, n, ldc);
    }
  };
};

template <typename T1, typename T2>
constexpr T2 convert_to(const T1& value) {
  if constexpr (std::is_same<T1, T2>::value) {
    return value;
  } else if constexpr (std::is_same<T1, ggml_bf16_t>::value && std::is_same<T2, float>::value) {
    return GGML_BF16_TO_FP32(value);
  } else if constexpr (std::is_same<T1, float>::value && std::is_same<T2, ggml_bf16_t>::value) {
    return GGML_FP32_TO_BF16(value);
  }
}

struct GemmKernel224BF {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 2;
  static const int TILE_M = 16;
  static const int TILE_K = 32;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 2;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 1792;
  static std::string name() { return "BF16"; }

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

    // size is 16 x 32
    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(dt));

    // size is 16 x 32
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));

    // size is 16 x 16
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

  struct BufferA {
    ggml_bf16_t* a;
    int max_m, k;

    static size_t required_size(int max_m, int k) { return sizeof(ggml_bf16_t) * max_m * k; }

    BufferA(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
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

  struct BufferB {
    ggml_bf16_t* b;
    int n, k;
    static constexpr bool SCALE = false;

    static size_t required_size(int n, int k) { return sizeof(ggml_bf16_t) * n * k; }

    BufferB(int n, int k, void* ptr) : n(n), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % K_STEP == 0);
      b = reinterpret_cast<ggml_bf16_t*>(ptr);
    }

    void set_data(void* new_ptr) { b = reinterpret_cast<ggml_bf16_t*>(new_ptr); }

    void from_mat(ggml_bf16_t* src, int ith, int nth) {
      auto [n_start, n_end] = split_range_n(n, ith, nth);
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

  struct BufferC {
    float* c;
    int max_m, n;

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
};

struct GemmKernel224Int8 {
  using dt = int8_t;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 1;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  // static inline const int N_BLOCK = 256;
  static inline const int N_BLOCK = 64;
  // static inline const int N_BLOCK = 32;
  static inline const int K_BLOCK = 3584;
  static std::string name() { return "INT8"; }

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

    // size is 16 x 64
    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(dt));

    // size is 16 x 64
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));

    // size is 16 x 16
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
    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
#endif
  }

  using BufferA = BufferAImpl<GemmKernel224Int8>;
  using BufferC = BufferCImpl<GemmKernel224Int8>;

  struct BufferB {
    int8_t* b;
    float* d;
    int n, k;
    static constexpr bool SCALE = true;

    static size_t required_size(int n, int k) { return sizeof(int8_t) * n * k + sizeof(float) * n; }

    BufferB(int n, int k, void* ptr) : n(n), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % K_STEP == 0);
      if (n % N_STEP || k % K_STEP) {
        printf("n: %d, k: %d, N_STEP: %d, K_STEP: %d\n", n, k, N_STEP, K_STEP);
        throw std::runtime_error("BufferB: n and k must be multiples of N_STEP and K_STEP");
      }
      b = reinterpret_cast<int8_t*>(ptr);
      d = reinterpret_cast<float*>(b + n * k);
    }

    void from_mat(ggml_bf16_t* src, int ith, int nth) {  // CHECK: nth has no usage
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP; i++) {
          float amax = 0.0f;
          for (int j = 0; j < k; j += 32) {
            __m512 f0, f1;
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + j), &f0, &f1);
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
          }
          d[n_block_begin + n_begin + i] = amax / ((1 << 7) - 1);
        }
      }
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            for (int i = 0; i < N_STEP; i++) {
              __m512 id = _mm512_set1_ps(d[n_block_begin + n_begin + i] ? 1.0f / d[n_block_begin + n_begin + i] : 0.0f);
              int8_t* dst = b + n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                            k_begin * N_STEP + i * K_STEP;
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin),
                                        &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
              __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
              __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
              __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
              __m128i s0 = _mm512_cvtsepi32_epi8(i0);
              __m128i s1 = _mm512_cvtsepi32_epi8(i1);
              __m128i s2 = _mm512_cvtsepi32_epi8(i2);
              __m128i s3 = _mm512_cvtsepi32_epi8(i3);
              _mm_store_si128((__m128i*)dst, s0);
              _mm_store_si128((__m128i*)(dst + 16), s1);
              _mm_store_si128((__m128i*)(dst + 32), s2);
              _mm_store_si128((__m128i*)(dst + 48), s3);
            }
            transpose_16x16_32bit((__m512i*)(b + n_block_begin * k + k_block_begin * n_block_size +
                                             n_begin * k_block_size + k_begin * N_STEP));
            transpose_16x16_32bit((__m512i*)(b + n_block_begin * k + k_block_begin * n_block_size +
                                             n_begin * k_block_size + k_begin * N_STEP + TILE_N * K_STEP));
          }
        }
      }
    }

    int8_t* get_submat(int n, int k, int n_begin, int k_begin) {
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      n_begin -= n_block_begin;
      int n_block_size = std::min(N_BLOCK, n - n_block_begin);
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      return b + n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP;
    }

    float* get_scale(int n, int n_begin) { return d + n_begin; }
  };

  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    using K = GemmKernel224Int8;
    if (k_block_begin == 0) {
      K::clean_c();
    } else {
      K::load_c((int32_t*)c, K::N_STEP * sizeof(int32_t));
    }
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::K_STEP) {
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), K::K_STEP * sizeof(int8_t));
      K::load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::K_STEP * sizeof(int8_t));
      K::run_tile();
    }
    K::store_c((int32_t*)c, K::N_STEP * sizeof(int32_t));
  }
  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    __m512i* c512 = (__m512i*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_si512();
        c512[m_i * 2 + 1] = _mm512_setzero_si512();
      }
    }

    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      static_assert(K_STEP * sizeof(int8_t) == sizeof(__m512i));
      static_assert(N_STEP / TILE_N == 2, "Must be lke this");

      int32_t* a32 = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma = _mm512_set1_epi32(a32[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(c512[m_i * 2 + n_i], ma, b512[n_i * 16 + k_i]);
          }
        }
      }
    }
  }

  static void apply_scale(int m, int n, int m_begin, int n_begin, float* c, BufferA* ba, BufferB* bb) {
    using K = GemmKernel224Int8;
    int to = m - m_begin;
    if (m - m_begin > K::M_STEP) {
      to = K::M_STEP;
    }
    for (int i = 0; i < to; i++) {
      __m512 as = _mm512_set1_ps(*ba->get_scale(m, m_begin + i));
      __m512 bs = _mm512_load_ps(bb->get_scale(n, n_begin));
      __m512i now = _mm512_load_si512((__m512i*)(c + i * K::N_STEP));
      __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      _mm512_store_ps((__m512*)(c + i * K::N_STEP), result);
      bs = _mm512_load_ps(bb->get_scale(n, n_begin) + K::TILE_N);
      now = _mm512_load_si512((__m512i*)(c + i * K::N_STEP + K::TILE_N));
      result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      _mm512_store_ps((__m512*)(c + i * K::N_STEP + K::TILE_N), result);
    }
  }
};

struct GemmKernel224Int4 {
  using dt = void;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 0.5;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  // static inline const int N_BLOCK = 256;
  static inline const int N_BLOCK = 128;
  // static inline const int N_BLOCK = 64;
  // static inline const int K_BLOCK = 7168;
  static inline const int K_BLOCK = 3584;
  // static inline const int K_BLOCK = 2560;

  static std::string name() { return "INT4"; }

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

    // size is 16 x 64
    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K);

    // size is 16 x 64
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK);

    // size is 16 x 16
    for (int i = 4; i < 8; i++) tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
#endif
  }

  alignas(64) static constexpr uint8_t hi_mask_arr[64] = {
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};

  alignas(64) static constexpr uint8_t lo_mask_arr[64] = {
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};

  alignas(64) static constexpr uint8_t sign_mask_arr[64] = {
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
  };

  static __m512i hi_mask() { return *((__m512i*)(&hi_mask_arr[0])); }
  static __m128i hi_mask_128() { return *((__m128i*)(&hi_mask_arr[0])); }
  static __m512i lo_mask() { return *((__m512i*)(&lo_mask_arr[0])); }
  static __m128i lo_mask_128() { return *((__m128i*)(&lo_mask_arr[0])); }
  static __m128i si_mask_128() { return *((__m128i*)(&sign_mask_arr[0])); }

  static void load_b_hi(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i)));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N))));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
#else
    (void)b;
    (void)ldb;
#endif
  }

  static void load_b_lo(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_slli_epi32(_mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_slli_epi32(
          _mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N)))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
#else
    (void)b;
    (void)ldb;
#endif
  }

  static void load_a(dt* a, size_t lda) {
#ifdef HAVE_AMX
    _tile_stream_loadd(0, a, lda);
    _tile_stream_loadd(1, offset_pointer(a, lda * TILE_M), lda);
#else
    (void)a;
    (void)lda;
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
    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
#endif
  }

  using BufferA = BufferAImpl<GemmKernel224Int4>;
  using BufferB = BufferBInt4Impl<GemmKernel224Int4>;
  using BufferC = BufferCImpl<GemmKernel224Int4>;

  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    using K = GemmKernel224Int4;
    __m512i* c512 = (__m512i*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_si512();
        c512[m_i * 2 + 1] = _mm512_setzero_si512();
      }
    }

    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      int32_t* a32_lo = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      int32_t* a32_hi = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP);
      __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma_lo = _mm512_set1_epi32(a32_lo[m_i * 16 + k_i]);
          __m512i ma_hi = _mm512_set1_epi32(a32_hi[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            __m512i b512_lo = _mm512_slli_epi32(_mm512_and_si512(K::lo_mask(), b512[n_i * 16 + k_i]), 4);
            c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(c512[m_i * 2 + n_i], ma_lo, b512_lo);
            __m512i b512_hi = _mm512_and_si512(K::hi_mask(), b512[n_i * 16 + k_i]);
            c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(c512[m_i * 2 + n_i], ma_hi, b512_hi);
          }
        }
      }
    }
  }
  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    using K = GemmKernel224Int4;
    if (k_block_begin == 0) {
      K::clean_c();
    } else {
      // printf("load from c int4\n");
      K::load_c((int32_t*)c, K::N_STEP * sizeof(int32_t));
    }
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), K::K_STEP * sizeof(int8_t));
      K::load_b_lo(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::BufferB::B_K_STEP / 2);
      K::run_tile();
      // DEBUG
      // if(m_begin == 0 && n_begin == 0 && k_begin==0){
      //   int8_t *ba_ptr = ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      //   int8_t *bb_ptr = (int8_t *)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      //   printf("k_begin:%d,k_block_begin:%d\n",k_begin,k_block_begin);
      //   for(int j=0;j<4096;j++){
      //     printf("a[%d]: %d ", j, ba_ptr[j]);
      //   }
      //   printf("\n");
      //   for(int j=0;j<4096;j++){
      //     printf("b[%d]: %d ", j, bb_ptr[j]);
      //   }
      //   printf("\n");
      // }

      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP), K::K_STEP * sizeof(int8_t));
      K::load_b_hi(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::BufferB::B_K_STEP / 2);
      K::run_tile();
    }

    // debug_tiles_224();
    K::store_c((int32_t*)c, K::N_STEP * sizeof(int32_t));
    // DEBUG c 的值,第一行的前 30 列
    // printf("\nint4, m_begin:%d,n_begin:%d,k_block_begin:%d\n",m_begin,n_begin,k_block_begin);
    // for(int j=0;j<30;j++){
    //   printf("c[%d]: %d ", j, ((int32_t *)c)[j]);
    // }
    // printf("\n");
  }

  static void apply_scale(int m, int n, int m_begin, int n_begin, float* c, BufferA* ba, BufferB* bb) {
    using K = GemmKernel224Int4;
    int to = m - m_begin;
    if (m - m_begin > K::M_STEP) {
      to = K::M_STEP;
    }
    for (int i = 0; i < to; i++) {
      __m512 as = _mm512_set1_ps(*ba->get_scale(m, m_begin + i));
      __m512 bs = _mm512_load_ps(bb->get_scale(n, n_begin));
      __m512i now = _mm512_load_epi32((__m512i*)(c + i * K::N_STEP));
      __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      // if(i==0){
      //   printf("\nnormal\n");
      //   printf("m_begin:%d,n_begin:%d\n", m_begin, n_begin);
      //   // 打印 result 结果，16 个 float 数值
      //   for(int j = 0; j < 16; j++) {
      //     float val = *((float *) &result + j);
      //     int32_t now_val = *((int32_t *) &now + j);
      //     printf("result[%d]: %f,now:%d ", j, val, now_val);
      //   }
      //   printf("\n");
      // }
      _mm512_store_ps((__m512*)(c + i * K::N_STEP), result);
      bs = _mm512_load_ps(bb->get_scale(n, n_begin) + K::TILE_N);
      now = _mm512_load_si512((__m512i*)(c + i * K::N_STEP + K::TILE_N));
      result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      // if(i==0){
      //   printf("\nnormal\n");
      //   printf("m_begin:%d,n_begin:%d\n", m_begin, n_begin);
      //   // 打印 result 结果，16 个 float 数值
      //   for(int j = 0; j < 16; j++) {
      //     float val = *((float *) &result + j);
      //     int32_t now_val = *((int32_t *) &now + j);
      //     printf("result[%d]: %f,now:%d ", j+16, val, now_val);
      //   }
      //   printf("\n");
      // }
      _mm512_store_ps((__m512*)(c + i * K::N_STEP + K::TILE_N), result);
    }
  }
};

struct GemmKernel224Int4_1 {
  using dt = void;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 0.5;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  // static inline const int K_BLOCK = 7168;
  static inline const int K_BLOCK = 3584;
  // static inline const int K_BLOCK = 2560;
  static std::string name() { return "INT4_1"; }

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

    // size is 16 x 64
    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K);

    // size is 16 x 64
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK);

    // size is 16 x 16
    for (int i = 4; i < 8; i++) tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
#endif
  }

  alignas(64) static constexpr uint8_t hi_mask_arr[64] = {
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};

  alignas(64) static constexpr uint8_t lo_mask_arr[64] = {
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};

  alignas(64) static constexpr uint8_t sign_mask_arr[64] = {
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
  };

  static __m512i hi_mask() { return *((__m512i*)(&hi_mask_arr[0])); }
  static __m128i hi_mask_128() { return *((__m128i*)(&hi_mask_arr[0])); }
  static __m512i lo_mask() { return *((__m512i*)(&lo_mask_arr[0])); }
  static __m128i lo_mask_128() { return *((__m128i*)(&lo_mask_arr[0])); }
  static __m128i si_mask_128() { return *((__m128i*)(&sign_mask_arr[0])); }

  static void load_b_hi(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i)));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N))));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
#else
    (void)b;
    (void)ldb;
#endif
  }

  static void load_b_lo(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_slli_epi32(_mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_slli_epi32(
          _mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N)))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
#else
    (void)b;
    (void)ldb;
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

  // static void load_b(dt* b, size_t ldb) {
  //   _tile_loadd(2, b, ldb);
  //   _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  // }

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
    _tile_dpbsud(4, 0, 2);
    _tile_dpbsud(5, 0, 3);
    _tile_dpbsud(6, 1, 2);
    _tile_dpbsud(7, 1, 3);
#endif
  }

  using BufferA = BufferAWithSumImpl<GemmKernel224Int4_1>;

  using BufferB = BufferBInt4WithZeroImpl<GemmKernel224Int4_1>;

  using BufferC = BufferCImpl<GemmKernel224Int4_1>;

  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    using K = GemmKernel224Int4_1;
    __m512i* c512 = (__m512i*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_si512();
        c512[m_i * 2 + 1] = _mm512_setzero_si512();
      }
    }
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      int32_t* a32_lo = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      int32_t* a32_hi = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP);
      __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma_lo = _mm512_set1_epi32(a32_lo[m_i * 16 + k_i]);
          __m512i ma_hi = _mm512_set1_epi32(a32_hi[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            __m512i b512_lo = _mm512_slli_epi32(_mm512_and_si512(K::lo_mask(), b512[n_i * 16 + k_i]), 4);
            c512[m_i * 2 + n_i] = _mm512_dpbusd_epi32(c512[m_i * 2 + n_i], b512_lo, ma_lo);
            __m512i b512_hi = _mm512_and_si512(K::hi_mask(), b512[n_i * 16 + k_i]);
            c512[m_i * 2 + n_i] = _mm512_dpbusd_epi32(c512[m_i * 2 + n_i], b512_hi, ma_hi);
          }
        }
      }
    }
  }
  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    using K = GemmKernel224Int4_1;
    if (k_block_begin == 0) {
      K::clean_c();
    } else {
      K::load_c((int32_t*)c, K::N_STEP * sizeof(int32_t));
    }
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      // printf("offset a %ld\n", pointer_offset(ba->get_submat(m, k, m_begin, k_block_begin + k_begin),
      // ba->a)); printf("offset b %ld\n", pointer_offset(bb->get_submat(n, k, n_begin, k_block_begin +
      // k_begin), bb->b));
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), K::K_STEP * sizeof(int8_t));
      K::load_b_lo(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::BufferB::B_K_STEP / 2);
      K::run_tile();
      // DEBUG
      // if(m_begin == 0 && n_begin == 0 && k_begin==0){
      //   int8_t *ba_ptr = ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      //   int8_t *bb_ptr = (int8_t *)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      //   printf("k_begin:%d,k_block_begin:%d\n",k_begin,k_block_begin);
      //   for(int j=0;j<2048;j++){
      //     printf("a[%d]: %d ", j, ba_ptr[j]);
      //   }
      //   printf("\n");
      //   for(int j=0;j<2048;j++){
      //     printf("b[%d]: %d ", j, bb_ptr[j]);
      //   }
      //   printf("\n");
      // }
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP), K::K_STEP * sizeof(int8_t));
      K::load_b_hi(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::BufferB::B_K_STEP / 2);
      K::run_tile();
    }

    // debug_tiles_224();
    K::store_c((int32_t*)c, K::N_STEP * sizeof(int32_t));
    // DEBUG c 的值,第一行的前 30 列
    // printf("\nint4_1, m_begin:%d,n_begin:%d,k_block_begin:%d\n",m_begin,n_begin,k_block_begin);
    // for(int j=0;j<30;j++){
    //   printf("c[%d]: %d ", j, ((int32_t *)c)[j]);
    // }
    // printf("\n");
  }

  static void apply_scale(int m, int n, int m_begin, int n_begin, float* c, BufferA* ba, BufferB* bb) {
    using K = GemmKernel224Int4_1;
    int to = m - m_begin;
    if (m - m_begin > K::M_STEP) {
      to = K::M_STEP;
    }
    for (int i = 0; i < to; i++) {
      __m512 as = _mm512_set1_ps(*ba->get_scale(m, m_begin + i));
      __m512 asum = _mm512_set1_ps(*ba->get_sum(m, m_begin + i));

      __m512 bs = _mm512_load_ps(bb->get_scale(n, n_begin));
      __m512 b_mins = _mm512_load_ps(bb->get_min(n, n_begin));
      __m512i now = _mm512_load_epi32((__m512i*)(c + i * K::N_STEP));
      __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      result = _mm512_add_ps(result, _mm512_mul_ps(asum, b_mins));
      _mm512_store_ps((__m512*)(c + i * K::N_STEP), result);

      bs = _mm512_load_ps(bb->get_scale(n, n_begin) + K::TILE_N);
      b_mins = _mm512_load_ps(bb->get_min(n, n_begin) + K::TILE_N);
      now = _mm512_load_si512((__m512i*)(c + i * K::N_STEP + K::TILE_N));
      result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      result = _mm512_add_ps(result, _mm512_mul_ps(asum, b_mins));
      _mm512_store_ps((__m512*)(c + i * K::N_STEP + K::TILE_N), result);
    }
  }
};

template <typename TA, typename TB, typename TC>
void mat_mul_single(int m, int n, int k, TA* a, size_t lda, TB* b, size_t ldb, TC* c, size_t ldc);
template <>
inline void mat_mul_single(int m, int n, int k, int8_t* a, size_t lda, int8_t* b, size_t ldb, int32_t* c, size_t ldc) {
  using Kernel = GemmKernel133<32>;
  for (int m_begin = 0; m_begin < m; m_begin += GemmKernel133<32>::M_STEP) {
    int m_end = std::min(m_begin + GemmKernel133<32>::M_STEP, m);
    for (int n_begin = 0; n_begin < n; n_begin += GemmKernel133<32>::N_STEP) {
      int n_end = std::min(n_begin + GemmKernel133<32>::N_STEP, n);
      for (int k_begin = 0; k_begin < k; k_begin += GemmKernel133<32>::K_STEP) {
        int k_end = std::min(k_begin + GemmKernel133<32>::K_STEP, k);
        int8_t* as = offset_pointer_row_major(a, m_begin, k_begin, lda);
        int8_t* bs = offset_pointer_col_major(b, k_begin, n_begin, ldb);
        int32_t* cs = offset_pointer_row_major(c, m_begin, n_begin, ldc);
        GemmKernel133<32>::BTile<int8_t> tb;
        if (n_end - n_begin == GemmKernel133<32>::N_STEP && k_end - k_begin == GemmKernel133<32>::K_STEP) {
          tb.load_from(bs, ldb);
        } else {
          tb.partial_load(bs, n_end - n_begin, k_end - k_begin, ldb);
        }
        if (m_end - m_begin == GemmKernel133<32>::M_STEP && k_end - k_begin == GemmKernel133<32>::K_STEP) {
          // printf("sub mat mul, full tile: (%d,%d),(%d,%d),(%d,%d)\n",
          // m_begin, m_end, n_begin, n_end, k_begin, k_end);
          tb.run_full_ac(as, lda, cs, ldc);
        } else {
          // printf("sub mat mul, partial tile: (%d,%d),(%d,%d),(%d,%d)\n",
          // m_begin, m_end, n_begin, n_end, k_begin, k_end);
          GemmKernel133<32>::PartialTiles<int8_t, int8_t, int32_t> p;
          p.partial_run(m_end - m_begin, n_end - n_begin, k_end - k_begin, as, lda, bs, ldb, cs, ldc);
        }
      }
    }
  }
}

template <>
inline void mat_mul_single(int m, int n, int k, ggml_bf16_t* a, size_t lda, ggml_bf16_t* b, size_t ldb, float* c,
                           size_t ldc) {
  // // GemmKernel133BF::config();

  // for (int m_begin = 0; m_begin < m; m_begin += GemmKernel133BF::M_STEP) {
  //   int m_end = std::min(m_begin + GemmKernel133BF::M_STEP, m);
  //   for (int n_begin = 0; n_begin < n; n_begin += GemmKernel133BF::N_STEP) {
  //     int n_end = std::min(n_begin + GemmKernel133BF::N_STEP, n);

  //     for (int k_begin = 0; k_begin < k; k_begin += GemmKernel133BF::K_STEP)
  //     {
  //       int k_end = std::min(k_begin + GemmKernel133BF::K_STEP, k);

  //       ggml_bf16_t* as = offset_pointer_row_major(a, m_begin, k_begin, lda);
  //       ggml_bf16_t* bs = offset_pointer_col_major(b, k_begin, n_begin, ldb);
  //       GemmKernel133BF::BTile tb;
  //       if (n_end - n_begin == GemmKernel133BF::N_STEP && k_end - k_begin ==
  //       GemmKernel133BF::K_STEP) {
  //         tb.full_load(bs, ldb);
  //       } else {
  //         tb.partial_load(bs, n_end - n_begin, k_end - k_begin, ldb);
  //       }
  //       float* cs = offset_pointer_row_major(c, m_begin, n_begin, ldc);

  //       if (m_end - m_begin == GemmKernel133<32>::M_STEP && k_end - k_begin
  //       == GemmKernel133<32>::K_STEP) {
  //         // printf("sub mat mul, full tile: (%d,%d),(%d,%d),(%d,%d)\n",
  //         m_begin, m_end, n_begin, n_end, k_begin,
  //         // k_end);
  //         tb.run_full_ac(as, lda, cs, ldc);
  //       } else {
  //         // printf("sub mat mul, partial tile: (%d,%d),(%d,%d),(%d,%d)\n",
  //         m_begin, m_end, n_begin, n_end, k_begin,
  //         // k_end);
  //         GemmKernel133BF::PartialTiles p;
  //         p.partial_run(m_end - m_begin, n_end - n_begin, k_end - k_begin,
  //         as, lda, bs, ldb, cs, ldc);
  //       }
  //     }
  //   }
  // }
}

template <typename QA>
void mat_mul_single(int m, int n, int k, QA* a, size_t lda, block_q8_0* b, size_t ldb, float* c, size_t ldc) {
  // amx::init();
  assert(QK8_0 == 32);
  assert(QK4_0 == 32);
  assert(GemmKernel133<32>::K_STEP == 32);
  // assert(reinterpret_cast<intptr_t>(c) % 64 == 0);
  assert(ldc % 64 == 0);

  // GemmKernal133::config();
  for (int n_begin = 0; n_begin < n; n_begin += GemmKernel133<32>::N_STEP) {
    int n_end = std::min(n_begin + GemmKernel133<32>::N_STEP, n);

    for (int k_begin = 0; k_begin < k; k_begin += GemmKernel133<32>::K_STEP) {
      int k_end = std::min(k_begin + GemmKernel133<32>::K_STEP, k);
      int kb = k_begin / GemmKernel133<32>::K_STEP;
      block_q8_0* bs = offset_pointer_col_major(b, kb, n_begin, ldb);
      GemmKernel133<32>::PartialTiles<int8_t, int8_t, int32_t> p;
      p.tb.partial_load_quant(bs, n_end - n_begin, k_end - k_begin, ldb);
      for (int m_begin = 0; m_begin < m; m_begin += GemmKernel133<32>::M_STEP) {
        int m_end = std::min(m_begin + GemmKernel133<32>::M_STEP, m);
        QA* as = offset_pointer_row_major(a, m_begin, kb, lda);

        float* cs = offset_pointer_row_major(c, m_begin, n_begin, ldc);
        // printf("sub mat mul: (%d,%d),(%d,%d),(%d,%d) %ld %ld\n", m_begin,
        // m_end, n_begin, n_end, k_begin, k_end,as-a,bs-b);

        // p.partial_run_quant(m_end - m_begin, n_end - n_begin, k_end -
        // k_begin, as, lda, bs, ldb, cs, ldc);
        p.partial_run_quant_ac(m_end - m_begin, n_end - n_begin, k_end - k_begin, as, lda, cs, ldc);
      }
    }
  }
}

inline void mat_mul_single(int m, int n, int k, block_q4_K* a, size_t lda, block_q8_K* b, size_t ldb, float* c,
                           size_t ldc) {
  assert(QK_K == 256);
  assert(k % QK_K == 0);
  assert(QK_K % GemmKernel133<32>::K_STEP == 0);
  assert(GemmKernel133<32>::K_STEP == 32);
  assert(ldc % 64 == 0);

  for (int m_begin = 0; m_begin < m; m_begin += GemmKernel133<32>::M_STEP) {
    int m_end = std::min(m_begin + GemmKernel133<32>::M_STEP, m);
    for (int n_begin = 0; n_begin < n; n_begin += GemmKernel133<32>::N_STEP) {
      int n_end = std::min(n_begin + GemmKernel133<32>::N_STEP, n);
      float* cs = offset_pointer_row_major(c, m_begin, n_begin, ldc);
      for (int k_bigstart = 0; k_bigstart < k; k_bigstart += QK_K) {
        int k_bigend = k_bigstart + QK_K;
        int super_block_index = k_bigstart / QK_K;

        block_q8_K* super_bs = offset_pointer_col_major(b, super_block_index, n_begin, ldb);

        block_q4_K* super_as = offset_pointer_row_major(a, m_begin, super_block_index, lda);
        float super_scale = GGML_FP16_TO_FP32(super_as->d);
        float super_min = GGML_FP16_TO_FP32(super_as->dmin);
        __m512 a_sm = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(make_q4K_scale_and_min(super_as->scales))),
            _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(super_scale)), _mm256_set1_ps(super_min), 1));
        float* a_scale = reinterpret_cast<float*>(&a_sm);
        float* a_min = a_scale + 8;

        for (int inner_idx = 0; inner_idx < 256 / 32; inner_idx++) {
          amx::GemmKernel133<32>::PartialTilesSum<uint8_t, int8_t, float> t;
          // printf("sub mat mul: (%d,%d),(%d,%d),(%d,%d) %d\n", m_begin, m_end,
          // n_begin, n_end, k_bigstart,
          //        k_bigend,inner_idx);
          t.tb.partial_load_quant(super_bs, n_end - n_begin, inner_idx, ldb);
          t.partial_run_quant_ac(m_end - m_begin, n_end - n_begin, inner_idx, super_as, lda, cs, ldc,
                                 a_scale[inner_idx], a_min[inner_idx]);
        }
      }
    }
  }
}

inline void mat_mul_single(int m, int n, int k, blocks_aligned_q8_0_ref a, int a_blck_stride, blocks_aligned_q8_0_ref b,
                           int b_blck_stride, float* c, size_t ldc) {
  using Kernel = GemmKernel133<64>;
  using TA = uint8_t;
  using TB = int8_t;

  for (int m_begin = 0; m_begin < m; m_begin += Kernel::M_STEP) {
    int m_end = std::min(m_begin + Kernel::M_STEP, m);
    for (int n_begin = 0; n_begin < n; n_begin += Kernel::N_STEP) {
      int n_end = std::min(n_begin + Kernel::N_STEP, n);
      for (int k_begin = 0; k_begin < k; k_begin += Kernel::K_STEP) {
        int k_end = std::min(k_begin + Kernel::K_STEP, k);

        int k_block = k_begin / Kernel::K_STEP;

        auto as = a.offset(m_begin * a_blck_stride + k_block);
        auto bs = b.offset(n_begin * b_blck_stride + k_block);
        auto cs = offset_pointer_row_major(c, m_begin, n_begin, ldc);

        // printf("sub mat mul: (%d,%d),(%d,%d),(%d,%d) %ld %ld\n", m_begin,
        // m_end, n_begin, n_end, k_begin, k_end,as.d-a.d,bs.d-b.d);

        Kernel::PartialTiles<TA, TB, int32_t> t;
        t.tb.partial_load_quant(bs, n_end - n_begin, k_end - k_begin, b_blck_stride);
        t.partial_run_quant_ac(m_end - m_begin, n_end - n_begin, k_end - k_begin, as, a_blck_stride, cs, ldc);
      }
    }
  }
}

inline void merge_mat(int d0, int d1, float* a, float* b, size_t ld) {
  __m512* va = (__m512*)a;
  __m512* vb = (__m512*)b;

  size_t d1v = (d1 + 15) / 16;

  for (int i = 0; i < d0; i++) {
    auto ta = offset_pointer_row_major(va, i, 0, ld);
    auto tb = offset_pointer_row_major(vb, i, 0, ld);
    for (int j = 0; j < d1v; j++) {
      ta[j] = _mm512_add_ps(ta[j], tb[j]);
    }
  }
}

inline void merge_mats(int d0, int d1, int cnt, float** data, size_t ld) {
  for (int i = 0; i < cnt; i++) {
    assert((intptr_t)data[i] % 64 == 0);
    assert(ld % 64 == 0);
  }

  while (cnt > 1) {
    int new_cnt = (cnt + 1) / 2;
    for (int i = 0; i < new_cnt; i++) {
      int j = new_cnt + i;
      if (j < cnt) {
        // printf("merge %d %d\n", i, j);
        merge_mat(d0, d1, data[i], data[j], ld);
      }
    }
    cnt = new_cnt;
  }
}

template <typename TA, typename TB, typename TC>
struct GemmKernel {
  static_assert(sizeof(TA) == -1, "No associated type defined for this type.");
  using type = GemmKernel224BF;
};

template <typename TB>
struct GemmKernel<uint8_t, TB, float> {
  using type = GemmKernel133<32>;
};

template <typename TB>
struct GemmKernel<int8_t, TB, float> {
  using type = GemmKernel133<32>;
};

template <>
struct GemmKernel<block_q4_0, block_q8_0, float> {
  using type = GemmKernel133<32>;
};

template <>
struct GemmKernel<block_q8_0, block_q8_0, float> {
  using type = GemmKernel133<32>;
};

template <>
struct GemmKernel<block_q4_K, block_q8_K, float> {
  using type = GemmKernel133<32>;
};

template <>
struct GemmKernel<ggml_bf16_t, ggml_bf16_t, float> {
  // using type = GemmKernel133BF;
  using type = GemmKernel224BF;
};

// template <typename TA, typename TB, typename TC>
// void mat_mul(int m, int n, int k, TA* a, size_t lda, TB* b, size_t ldb, TC*
// c, size_t ldc, int ith, int nth) {
//   using K = typename GemmKernel<TA, TB, TC>::type;

//   int m_partition_count = (m + K::M_STEP - 1) / K::M_STEP;
//   int partition_count_per_thread = (m_partition_count + nth - 1) / nth;
//   int partition_start = ith * partition_count_per_thread;
//   int partition_end = std::min(partition_start + partition_count_per_thread,
//   m_partition_count); int m_start = partition_start * K::M_STEP; int m_end =
//   std::min(m, partition_end * K::M_STEP);

//   mat_mul_single(m_end - m_start, n, k, offset_pointer(a, m_start * lda),
//   lda, b, ldb, offset_pointer(c, m_start * ldc),
//                  ldc);
// }

template <typename TA, typename TB, typename TC>
void mat_mul(int m, int n, int k, TA* a, size_t lda, TB* b, size_t ldb, TC* c, size_t ldc, int ith, int nth) {
  using K = typename GemmKernel<TA, TB, TC>::type;

  int n_partition_count = (n + K::N_STEP - 1) / K::N_STEP;
  int partition_count_per_thread = (n_partition_count + nth - 1) / nth;
  int partition_start = ith * partition_count_per_thread;
  int partition_end = std::min(partition_start + partition_count_per_thread, n_partition_count);
  int n_start = partition_start * K::N_STEP;
  int n_end = std::min(n, partition_end * K::N_STEP);

  mat_mul_single(m, n_end - n_start, k, a, lda, offset_pointer_col_major(b, 0, n_start, ldb), ldb,
                 offset_pointer_row_major(c, 0, n_start, ldc), ldc);
}

inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernel224BF::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF::BufferB> bb, std::shared_ptr<GemmKernel224BF::BufferC> bc, int ith,
                    int nth) {
  using K = GemmKernel224BF;
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  // printf("n_start %d n_end %d\n", n_start, n_end);
  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);
        // if (m - m_begin == 1) {
        if (false) {
          // if(k_block_begin==0&&m_begin==0&&n_begin==n_start)
          // printf("AVX");
          __m512* c512 = (__m512*)c;
          if (k_block_begin == 0) {
            for (int m_i = 0; m_i < m; m_i++) {
              c512[m_i * 2] = _mm512_setzero_ps();
              c512[m_i * 2 + 1] = _mm512_setzero_ps();
            }
          }

          for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::K_STEP) {
            int32_t* a32 = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
            __m512bh* b512 = (__m512bh*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
            for (int m_i = 0; m_i < m; m_i++) {
              for (int k_i = 0; k_i < 16; k_i++) {
                __m512bh ma = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
                for (int n_i = 0; n_i < 2; n_i++) {
                  c512[m_i * 2 + n_i] = _mm512_dpbf16_ps(c512[m_i * 2 + n_i], ma, b512[n_i * 16 + k_i]);
                }
              }
            }
          }

        } else {
          if (k_block_begin == 0) {
            K::clean_c();
          } else {
            K::load_c(c, K::N_STEP * sizeof(float));
          }
          for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::K_STEP) {
            K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), K::K_STEP * sizeof(ggml_bf16_t));
            K::load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::K_STEP * sizeof(ggml_bf16_t));
            K::run_tile();
          }
          K::store_c(c, K::N_STEP * sizeof(float));
        }
      }
    }
  }
}

inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernel224BF::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF::BufferB> bb, std::shared_ptr<GemmKernel224BF::BufferC> bc, int ith,
                    int nth) {
  mat_mul(m, n, k, ba, bb, bc, ith, nth);
}

template <typename K, bool amx_or_avx = true>
void integer_mat_mul(int m, int n, int k, typename K::BufferA* ba, typename K::BufferB* bb, typename K::BufferC* bc,
                     int ith, int nth) {
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);
        if constexpr (amx_or_avx && AMX_AVAILABLE) {
          K::amx_kernel(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
        } else {
          K::avx_kernel(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
        }

        if (k_block_begin + K::K_BLOCK >= k) {
          K::apply_scale(m, n, m_begin, n_begin, c, ba, bb);
        }
      }
    }
  }
}

inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernel224Int8::BufferA> ba,
                    std::shared_ptr<GemmKernel224Int8::BufferB> bb, std::shared_ptr<GemmKernel224Int8::BufferC> bc,
                    int ith, int nth) {
  integer_mat_mul<GemmKernel224Int8, false>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernel224Int8::BufferA> ba,
                    std::shared_ptr<GemmKernel224Int8::BufferB> bb, std::shared_ptr<GemmKernel224Int8::BufferC> bc,
                    int ith, int nth) {
  integer_mat_mul<GemmKernel224Int8, true>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernel224Int4::BufferA> ba,
                    std::shared_ptr<GemmKernel224Int4::BufferB> bb, std::shared_ptr<GemmKernel224Int4::BufferC> bc,
                    int ith, int nth) {
  integer_mat_mul<GemmKernel224Int4, false>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernel224Int4::BufferA> ba,
                    std::shared_ptr<GemmKernel224Int4::BufferB> bb, std::shared_ptr<GemmKernel224Int4::BufferC> bc,
                    int ith, int nth) {
  integer_mat_mul<GemmKernel224Int4, true>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernel224Int4_1::BufferA> ba,
                    std::shared_ptr<GemmKernel224Int4_1::BufferB> bb, std::shared_ptr<GemmKernel224Int4_1::BufferC> bc,
                    int ith, int nth) {
  integer_mat_mul<GemmKernel224Int4_1, false>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernel224Int4_1::BufferA> ba,
                    std::shared_ptr<GemmKernel224Int4_1::BufferB> bb, std::shared_ptr<GemmKernel224Int4_1::BufferC> bc,
                    int ith, int nth) {
  integer_mat_mul<GemmKernel224Int4_1, true>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul(int m, int n, int k, blocks_aligned_q8_0_ref aref, int a_blck_stride, blocks_aligned_q8_0_ref bref,
                    int b_blck_stride, float* c, size_t ldc, int ith, int nth) {
  using K = GemmKernel133<64>;

  int m_partition_count = (m + K::M_STEP - 1) / K::M_STEP;
  int partition_count_per_thread = (m_partition_count + nth - 1) / nth;
  int partition_start = ith * partition_count_per_thread;
  int partition_end = std::min(partition_start + partition_count_per_thread, m_partition_count);
  int m_start = partition_start * K::M_STEP;
  int m_end = std::min(m, partition_end * K::M_STEP);

  mat_mul_single(m_end - m_start, n, k, aref.offset(m_start * a_blck_stride), a_blck_stride, bref, b_blck_stride,
                 offset_pointer(c, m_start * ldc), ldc);
}

// K-group quantization kernel with intermediate int32 accumulation
struct GemmKernel224Int4KGroup {
  using dt = void;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 0.5;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;
  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;
  static inline const int N_BLOCK = 256;
  // K_BLOCK should match k_group_size for proper scaling
  static inline const int K_BLOCK = 7168;  // Will be overridden by k_group_size

  static std::string name() { return "INT4_KGROUP"; }
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
    // size is 16 x 64
    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K);
    // size is 16 x 64
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK);
    // size is 16 x 16
    for (int i = 4; i < 8; i++) tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));
    tile_config.set_config();
#endif
  }

  alignas(64) static constexpr uint8_t hi_mask_arr[64] = {
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};

  alignas(64) static constexpr uint8_t lo_mask_arr[64] = {
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};

  static __m512i hi_mask() { return *((__m512i*)(&hi_mask_arr[0])); }
  static __m512i lo_mask() { return *((__m512i*)(&lo_mask_arr[0])); }

  static void clean_c() {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
  }

  static void load_c(output_t* c, size_t ldc) {
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void store_c(output_t* c, size_t ldc) {
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void load_a(dt* a, size_t lda) {
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
  }

  static void load_b_lo(dt* b, size_t ldb) {
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      // __m512i temp = _mm512_and_si512(lo_mask(), *static_cast<__m512i *>(offset_pointer(b, ldb * i)));
      // db[i] = _mm512_slli_epi32(temp, 4);
      db[i] = _mm512_slli_epi32(_mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      // __m512i temp = _mm512_and_si512(lo_mask(), *static_cast<__m512i *>(offset_pointer(b, ldb * (i + TILE_N))));
      // db[i] = _mm512_slli_epi32(temp, 4);
      db[i] = _mm512_slli_epi32(
          _mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N)))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
  }

  static void load_b_hi(dt* b, size_t ldb) {
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i)));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N))));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
  }

  static void run_tile() {
#ifdef HAVE_AMX
    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
#endif
  }

  using BufferA = BufferAKGroupImpl<GemmKernel224Int4KGroup>;
  using BufferB = BufferBKGroupImpl<GemmKernel224Int4KGroup>;
  using BufferC = BufferCReduceImpl<GemmKernel224Int4KGroup>;

  // K-group aware AVX kernel - processes a single B_K_STEP chunk
  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, int32_t* int_c, BufferA* ba,
                         BufferB* bb, int k_group_size) {
    using K = GemmKernel224Int4KGroup;
    __m512i* c512 = (__m512i*)int_c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    // Initialize int_c to zero at the start of k_group
    if (k_block_begin % k_group_size == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_si512();
        c512[m_i * 2 + 1] = _mm512_setzero_si512();
      }
    }
    int k_offset = k_block_begin % K::BufferB::B_K_STEP;
    if (k_offset == 0) {
      int32_t* a32_lo = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin);
      __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin);
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma_lo = _mm512_set1_epi32(a32_lo[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            __m512i b512_lo = _mm512_slli_epi32(_mm512_and_si512(K::lo_mask(), b512[n_i * 16 + k_i]), 4);
            c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(c512[m_i * 2 + n_i], ma_lo, b512_lo);
          }
        }
      }
    } else {
      int32_t* a32_hi = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin);
      __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin - K::K_STEP);
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma_hi = _mm512_set1_epi32(a32_hi[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            __m512i b512_hi = _mm512_and_si512(K::hi_mask(), b512[n_i * 16 + k_i]);
            c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(c512[m_i * 2 + n_i], ma_hi, b512_hi);
          }
        }
      }
    }
  }

  // K-group aware AMX kernel - processes a single K_STEP chunk (lo or hi nibble)
  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, int32_t* int_c, BufferA* ba,
                         BufferB* bb, int k_group_size) {
    using K = GemmKernel224Int4KGroup;
    // Initialize or load int_c at start of k_group
    if (k_block_begin % k_group_size == 0) {
      K::clean_c();
    } else {
      K::load_c(int_c, K::N_STEP * sizeof(int32_t));
    }

    // Determine if we're processing lo or hi nibble based on position within B_K_STEP
    int k_offset = k_block_begin % K::BufferB::B_K_STEP;
    if (k_offset == 0) {
      // Process lo nibble
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin), K::K_STEP * sizeof(int8_t));
      K::load_b_lo(bb->get_submat(n, k, n_begin, k_block_begin), K::BufferB::B_K_STEP / 2);
      K::run_tile();
    } else {
      // Process hi nibble (k_offset == K_STEP)
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin), K::K_STEP * sizeof(int8_t));
      K::load_b_hi(bb->get_submat(n, k, n_begin, k_block_begin - K::K_STEP), K::BufferB::B_K_STEP / 2);
      K::run_tile();
    }

    K::store_c(int_c, K::N_STEP * sizeof(int32_t));
  }

  // K-group aware scale application
  static void apply_scale_kgroup(int m, int n, int m_begin, int n_begin, int k_begin, float* c, int32_t* int_c,
                                 BufferA* ba, BufferB* bb, int k, int k_group_size) {
    using K = GemmKernel224Int4KGroup;
    int to = m - m_begin;
    if (m - m_begin > K::M_STEP) {
      to = K::M_STEP;
    }

    for (int i = 0; i < to; i++) {
      // Get scale for this k_group
      __m512 as = _mm512_set1_ps(*ba->get_scale(m, m_begin + i, k, k_begin));
      __m512 bs = _mm512_load_ps(bb->get_scale(n, n_begin, k, k_begin));
      __m512i now = _mm512_load_epi32((__m512i*)(int_c + i * K::N_STEP));
      __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      // Load existing float value from c and add
      __m512 existing = _mm512_load_ps((__m512*)(c + i * K::N_STEP));
      result = _mm512_add_ps(existing, result);
      _mm512_store_ps((__m512*)(c + i * K::N_STEP), result);

      // Second half
      bs = _mm512_load_ps(bb->get_scale(n, n_begin, k, k_begin) + K::TILE_N);
      now = _mm512_load_si512((__m512i*)(int_c + i * K::N_STEP + K::TILE_N));
      result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      existing = _mm512_load_ps((__m512*)(c + i * K::N_STEP + K::TILE_N));
      result = _mm512_add_ps(existing, result);
      _mm512_store_ps((__m512*)(c + i * K::N_STEP + K::TILE_N), result);
    }
  }
};
struct GemmKernel224Int4_1KGroup {
  using dt = void;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 0.5;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  // static inline const int K_BLOCK = 7168;
  static inline const int K_BLOCK = 3584;
  // static inline const int K_BLOCK = 2560;
  static std::string name() { return "INT4_1K"; }

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

    // size is 16 x 64
    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K);

    // size is 16 x 64
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK);

    // size is 16 x 16
    for (int i = 4; i < 8; i++) tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
#endif
  }

  alignas(64) static constexpr uint8_t hi_mask_arr[64] = {
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};

  alignas(64) static constexpr uint8_t lo_mask_arr[64] = {
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};

  alignas(64) static constexpr uint8_t sign_mask_arr[64] = {
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
  };

  static __m512i hi_mask() { return *((__m512i*)(&hi_mask_arr[0])); }
  static __m128i hi_mask_128() { return *((__m128i*)(&hi_mask_arr[0])); }
  static __m512i lo_mask() { return *((__m512i*)(&lo_mask_arr[0])); }
  static __m128i lo_mask_128() { return *((__m128i*)(&lo_mask_arr[0])); }
  static __m128i si_mask_128() { return *((__m128i*)(&sign_mask_arr[0])); }

  static void load_b_hi(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i)));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N))));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
#else
    (void)b;
    (void)ldb;
#endif
  }

  static void load_b_lo(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_slli_epi32(_mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_slli_epi32(
          _mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N)))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
#else
    (void)b;
    (void)ldb;
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

  // static void load_b(dt* b, size_t ldb) {
  //   _tile_loadd(2, b, ldb);
  //   _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  // }

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
    _tile_dpbsud(4, 0, 2);
    _tile_dpbsud(5, 0, 3);
    _tile_dpbsud(6, 1, 2);
    _tile_dpbsud(7, 1, 3);
#endif
  }

  using BufferA = BufferAWithSumKGroupImpl<GemmKernel224Int4_1KGroup>;

  using BufferB = BufferBInt4WithZeroKGroupImpl<GemmKernel224Int4_1KGroup>;

  using BufferC = BufferCReduceImpl<GemmKernel224Int4_1KGroup>;

  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, int32_t* int_c, BufferA* ba,
                         BufferB* bb, int k_group_size) {
    using K = GemmKernel224Int4_1KGroup;
    __m512i* c512 = (__m512i*)int_c;
    int m_block_end = std::min(m - m_begin, M_STEP);
    if (k_block_begin % k_group_size == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_si512();
        c512[m_i * 2 + 1] = _mm512_setzero_si512();
      }
    }
    int k_offset = k_block_begin % K::BufferB::B_K_STEP;
    if (k_offset == 0) {
      int32_t* a32_lo = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin);
      __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin);
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma_lo = _mm512_set1_epi32(a32_lo[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            __m512i b512_lo = _mm512_slli_epi32(_mm512_and_si512(K::lo_mask(), b512[n_i * 16 + k_i]), 4);
            c512[m_i * 2 + n_i] = _mm512_dpbusd_epi32(c512[m_i * 2 + n_i], b512_lo, ma_lo);
          }
        }
      }
    } else {
      int32_t* a32_hi = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin);
      __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin - K::K_STEP);
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma_hi = _mm512_set1_epi32(a32_hi[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            __m512i b512_hi = _mm512_and_si512(K::hi_mask(), b512[n_i * 16 + k_i]);
            c512[m_i * 2 + n_i] = _mm512_dpbusd_epi32(c512[m_i * 2 + n_i], b512_hi, ma_hi);
          }
        }
      }
    }
  }
  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, int32_t* int_c, BufferA* ba,
                         BufferB* bb, int k_group_size) {
    using K = GemmKernel224Int4_1KGroup;
    if (k_block_begin % k_group_size == 0) {
      K::clean_c();
    } else {
      K::load_c(int_c, K::N_STEP * sizeof(int32_t));
    }

    // Determine if we're processing lo or hi nibble based on position within B_K_STEP
    int k_offset = k_block_begin % K::BufferB::B_K_STEP;
    if (k_offset == 0) {
      // Process lo nibble
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin), K::K_STEP * sizeof(int8_t));
      K::load_b_lo(bb->get_submat(n, k, n_begin, k_block_begin), K::BufferB::B_K_STEP / 2);
      K::run_tile();
    } else {
      // Process hi nibble (k_offset == K_STEP)
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin), K::K_STEP * sizeof(int8_t));
      K::load_b_hi(bb->get_submat(n, k, n_begin, k_block_begin - K::K_STEP), K::BufferB::B_K_STEP / 2);
      K::run_tile();
    }

    K::store_c(int_c, K::N_STEP * sizeof(int32_t));
  }

  static void apply_scale_kgroup(int m, int n, int m_begin, int n_begin, int k_begin, float* c, int32_t* int_c,
                                 BufferA* ba, BufferB* bb, int k, int k_group_size) {
    using K = GemmKernel224Int4_1KGroup;
    int to = m - m_begin;
    if (m - m_begin > K::M_STEP) {
      to = K::M_STEP;
    }
    for (int i = 0; i < to; i++) {
      __m512 as = _mm512_set1_ps(*ba->get_scale(m, m_begin + i, k, k_begin));
      __m512 asum = _mm512_set1_ps(*ba->get_sum(m, m_begin + i, k, k_begin));

      __m512 bs = _mm512_load_ps(bb->get_scale(n, n_begin, k, k_begin));
      __m512 b_mins = _mm512_load_ps(bb->get_min(n, n_begin, k, k_begin));
      __m512i now = _mm512_load_epi32((__m512i*)(int_c + i * K::N_STEP));
      __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      result = _mm512_add_ps(result, _mm512_mul_ps(asum, b_mins));
      __m512 existing = _mm512_load_ps((__m512*)(c + i * K::N_STEP));
      result = _mm512_add_ps(result, existing);
      _mm512_store_ps((__m512*)(c + i * K::N_STEP), result);

      bs = _mm512_load_ps(bb->get_scale(n, n_begin, k, k_begin) + K::TILE_N);
      b_mins = _mm512_load_ps(bb->get_min(n, n_begin, k, k_begin) + K::TILE_N);
      now = _mm512_load_si512((__m512i*)(int_c + i * K::N_STEP + K::TILE_N));
      result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      result = _mm512_add_ps(result, _mm512_mul_ps(asum, b_mins));
      existing = _mm512_load_ps((__m512*)(c + i * K::N_STEP + K::TILE_N));
      result = _mm512_add_ps(result, existing);
      _mm512_store_ps((__m512*)(c + i * K::N_STEP + K::TILE_N), result);
    }
  }
};

struct GemmKernel224Int4_1_LowKGroup {
  using dt = void;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 0.5;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  // static inline const int K_BLOCK = 7168;
  static inline const int K_BLOCK = 3584;
  // static inline const int K_BLOCK = 2560;
  static std::string name() { return "INT4_1K"; }

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

    // size is 16 x 64
    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K);

    // size is 16 x 64
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK);

    // size is 16 x 16
    for (int i = 4; i < 8; i++) tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
#endif
  }

  alignas(64) static constexpr uint8_t hi_mask_arr[64] = {
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};

  alignas(64) static constexpr uint8_t lo_mask_arr[64] = {
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};

  alignas(64) static constexpr uint8_t sign_mask_arr[64] = {
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
  };

  static __m512i hi_mask() { return *((__m512i*)(&hi_mask_arr[0])); }
  static __m128i hi_mask_128() { return *((__m128i*)(&hi_mask_arr[0])); }
  static __m512i lo_mask() { return *((__m512i*)(&lo_mask_arr[0])); }
  static __m128i lo_mask_128() { return *((__m128i*)(&lo_mask_arr[0])); }
  static __m128i si_mask_128() { return *((__m128i*)(&sign_mask_arr[0])); }

  static void load_b_hi(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_srli_epi32(_mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_srli_epi32(
          _mm512_and_si512(hi_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N)))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
#else
    (void)b;
    (void)ldb;
#endif
  }

  static void load_b_lo(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i* db = reinterpret_cast<__m512i*>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * i)));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(lo_mask(), *static_cast<__m512i*>(offset_pointer(b, ldb * (i + TILE_N))));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
#else
    (void)b;
    (void)ldb;
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

  // static void load_b(dt* b, size_t ldb) {
  //   _tile_loadd(2, b, ldb);
  //   _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  // }

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
    _tile_dpbsud(4, 0, 2);
    _tile_dpbsud(5, 0, 3);
    _tile_dpbsud(6, 1, 2);
    _tile_dpbsud(7, 1, 3);
#endif
  }

  using BufferA = BufferAWithSumKGroupImpl<GemmKernel224Int4_1_LowKGroup>;

  using BufferB = BufferBInt4WithZeroLowKGroupImpl<GemmKernel224Int4_1_LowKGroup>;

  using BufferC = BufferCReduceImpl<GemmKernel224Int4_1_LowKGroup>;

  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, int32_t* int_c, BufferA* ba,
                         BufferB* bb, int k_group_size) {
    using K = GemmKernel224Int4_1_LowKGroup;
    __m512i* c512 = (__m512i*)int_c;
    int m_block_end = std::min(m - m_begin, M_STEP);
    if (k_block_begin % k_group_size == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_si512();
        c512[m_i * 2 + 1] = _mm512_setzero_si512();
      }
    }
    int k_offset = k_block_begin % K::BufferB::B_K_STEP;
    if (k_offset == 0) {
      int32_t* a32_lo = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin);
      __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin);
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma_lo = _mm512_set1_epi32(a32_lo[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            __m512i b512_lo = _mm512_and_si512(K::lo_mask(), b512[n_i * 16 + k_i]);
            c512[m_i * 2 + n_i] = _mm512_dpbusd_epi32(c512[m_i * 2 + n_i], b512_lo, ma_lo);
          }
        }
      }
    } else {
      int32_t* a32_hi = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin);
      __m512i* b512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin - K::K_STEP);
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma_hi = _mm512_set1_epi32(a32_hi[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            __m512i b512_hi = _mm512_srli_epi32(_mm512_and_si512(K::hi_mask(), b512[n_i * 16 + k_i]), 4);
            c512[m_i * 2 + n_i] = _mm512_dpbusd_epi32(c512[m_i * 2 + n_i], b512_hi, ma_hi);
          }
        }
      }
    }
  }
  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, int32_t* int_c, BufferA* ba,
                         BufferB* bb, int k_group_size) {
    using K = GemmKernel224Int4_1_LowKGroup;
    if (k_block_begin % k_group_size == 0) {
      K::clean_c();
    } else {
      K::load_c(int_c, K::N_STEP * sizeof(int32_t));
    }

    // Determine if we're processing lo or hi nibble based on position within B_K_STEP
    int k_offset = k_block_begin % K::BufferB::B_K_STEP;
    if (k_offset == 0) {
      // Process lo nibble
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin), K::K_STEP * sizeof(int8_t));
      K::load_b_lo(bb->get_submat(n, k, n_begin, k_block_begin), K::BufferB::B_K_STEP / 2);
      K::run_tile();
    } else {
      // Process hi nibble (k_offset == K_STEP)
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin), K::K_STEP * sizeof(int8_t));
      K::load_b_hi(bb->get_submat(n, k, n_begin, k_block_begin - K::K_STEP), K::BufferB::B_K_STEP / 2);
      K::run_tile();
    }

    K::store_c(int_c, K::N_STEP * sizeof(int32_t));
  }

  static void apply_scale_kgroup(int m, int n, int m_begin, int n_begin, int k_begin, float* c, int32_t* int_c,
                                 BufferA* ba, BufferB* bb, int k, int k_group_size) {
    using K = GemmKernel224Int4_1_LowKGroup;
    int to = m - m_begin;
    if (m - m_begin > K::M_STEP) {
      to = K::M_STEP;
    }
    for (int i = 0; i < to; i++) {
      __m512 as = _mm512_set1_ps(*ba->get_scale(m, m_begin + i, k, k_begin));
      __m512 asum = _mm512_set1_ps(*ba->get_sum(m, m_begin + i, k, k_begin));

      __m512 bs = _mm512_load_ps(bb->get_scale(n, n_begin, k, k_begin));
      __m512 b_mins = _mm512_load_ps(bb->get_min(n, n_begin, k, k_begin));
      __m512i now = _mm512_load_epi32((__m512i*)(int_c + i * K::N_STEP));
      __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      result = _mm512_add_ps(result, _mm512_mul_ps(asum, b_mins));
      __m512 existing = _mm512_load_ps((__m512*)(c + i * K::N_STEP));
      result = _mm512_add_ps(result, existing);
      _mm512_store_ps((__m512*)(c + i * K::N_STEP), result);

      bs = _mm512_load_ps(bb->get_scale(n, n_begin, k, k_begin) + K::TILE_N);
      b_mins = _mm512_load_ps(bb->get_min(n, n_begin, k, k_begin) + K::TILE_N);
      now = _mm512_load_si512((__m512i*)(int_c + i * K::N_STEP + K::TILE_N));
      result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      result = _mm512_add_ps(result, _mm512_mul_ps(asum, b_mins));
      existing = _mm512_load_ps((__m512*)(c + i * K::N_STEP + K::TILE_N));
      result = _mm512_add_ps(result, existing);
      _mm512_store_ps((__m512*)(c + i * K::N_STEP + K::TILE_N), result);
    }
  }
};

// K2 Signed Int4 K-group quantization kernel (AVX only, no AMX)
// For K2 MoE - signed int4 range: [-8, 7]
struct GemmKernel224Int4SmallKGroup {
  using dt = uint8_t;  // packed int4 type
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 0.5;
  static const int VNNI_BLK = 4;
  
  static const int M_STEP = 1;
  static const int N_STEP = 32;
  static const int K_STEP = 32;

  static inline const int N_BLOCK = 256;
  // K_BLOCK should match k_group_size for proper scaling
  static inline const int K_BLOCK = 7168;  // Will be overridden by k_group_size

  static std::string name() { return "K2_INT4_KGROUP"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }
  static void config() {}

  alignas(64) static constexpr uint8_t hi_mask_arr[32] = {
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0
    };

  alignas(64) static constexpr uint8_t lo_mask_arr[32] = {
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F
    };
  
  alignas(64) static constexpr uint8_t sign_xor_arr[32] = {
      0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
      0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88
    };
  static __m256i hi_mask() { return *((__m256i*)(&hi_mask_arr[0])); }
  static __m256i lo_mask() { return *((__m256i*)(&lo_mask_arr[0])); }
  static __m256i sign_xor_mask() { return *((__m256i*)(&sign_xor_arr[0])); }

  using BufferA = BufferASmallKGroupImpl<GemmKernel224Int4SmallKGroup>;
  using BufferB = BufferBInt4KGroupImpl<GemmKernel224Int4SmallKGroup>;  // Use new signed int4 buffer
  using BufferC = BufferCReduceImpl<GemmKernel224Int4SmallKGroup>;

  // K-group aware AVX kernel for signed int4
  static inline __m512i compressed_int4_to_int8_avx512(__m256i b256) {
    b256 = _mm256_xor_si256(b256, sign_xor_mask());
    __m256i b_hi = _mm256_and_si256(b256, hi_mask());
    __m256i b_lo = _mm256_slli_epi16(_mm256_andnot_si256(hi_mask(), b256), 4);

    __m256i unpack_lo = _mm256_unpacklo_epi8(b_lo, b_hi);
    __m256i unpack_hi = _mm256_unpackhi_epi8(b_lo, b_hi);
    __m512i result = _mm512_inserti64x4(_mm512_castsi256_si512(unpack_lo), unpack_hi, 1);
    const __m512i lane_shuffle = _mm512_set_epi64(7, 6, 3, 2, 5, 4, 1, 0);
    return _mm512_permutexvar_epi64(lane_shuffle, result);
  }
  static inline void integer_mat_vec_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB *bb, BufferC* bc, int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    for (int m_begin = 0; m_begin < m; m_begin ++) {
      float* c = bc->get_submat(m, n, m_begin, n_start);
      __m512i* a512 = (__m512i*)ba->get_submat(m, k, m_begin, 0);
      
      for (int n_block_begin = n_start; n_block_begin < n_end; n_block_begin ++) {
        __m256i* b256 = (__m256i*)bb->get_submat(n, k, n_block_begin, 0);
        float* as = (float*)ba->get_scale(m, m_begin, k, 0);
        float* bs = (float*)bb->get_scale(n, n_block_begin, k, 0);
        
        __m512 sum = _mm512_setzero_ps();
        #define WORK_K_BLOCK(k_block) \
          { \
            __m256 abscale0 = _mm256_set1_ps(as[(k_block)*2] * bs[(k_block)*2]); \
            __m256 abscale1 = _mm256_set1_ps(as[(k_block)*2+1] * bs[(k_block)*2+1]); \
            __m512 abscale = _mm512_insertf32x8(_mm512_castps256_ps512(abscale0), abscale1, 1); \
            __m512i mul = _mm512_setzero_si512(); \
            mul = _mm512_dpbssd_epi32(mul, a512[k_block], compressed_int4_to_int8_avx512(b256[k_block])); \
            sum = _mm512_add_ps(sum, _mm512_mul_ps(abscale, _mm512_cvtepi32_ps(mul))); \
          }

        for (int k_block = 0; k_block < k / 64; k_block += 2) {
          WORK_K_BLOCK(k_block);
          WORK_K_BLOCK(k_block + 1);
        }

        c[n_block_begin - n_start] = _mm512_reduce_add_ps(sum) / 16;
      }
    }
  }
};

inline void vec_mul_kgroup(int m, int n, int k, int k_group_size, std::shared_ptr<GemmKernel224Int4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4SmallKGroup::BufferC> bc, int ith, int nth) {
  GemmKernel224Int4SmallKGroup::integer_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size, std::shared_ptr<GemmKernel224Int4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4SmallKGroup::BufferC> bc, int ith, int nth) {
  GemmKernel224Int4SmallKGroup::integer_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

// New k-group aware matrix multiplication function
template <typename K, bool amx_or_avx = true>
void integer_mat_mul_kgroup(int m, int n, int k, int k_group_size, typename K::BufferA* ba, typename K::BufferB* bb,
                            typename K::BufferC* bc, int ith, int nth) {
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);
  assert(k % k_group_size == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);
  // Process by k_groups
  for (int k_group_begin = 0; k_group_begin < k; k_group_begin += k_group_size) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);
        int32_t* int_c = bc->get_int_submat(m, n, m_begin, n_begin);

        // Initialize float c to zero at the very beginning
        if (k_group_begin == 0) {
          for (int i = 0; i < K::M_STEP && m_begin + i < m; i++) {
            for (int j = 0; j < K::N_STEP; j++) {
              c[i * K::N_STEP + j] = 0.0f;
            }
          }
        }
        for (int k_begin = k_group_begin; k_begin < std::min(k, k_group_begin + k_group_size); k_begin += K::K_STEP) {
          if constexpr (amx_or_avx && AMX_AVAILABLE) {
            K::amx_kernel(m, n, k, m_begin, n_begin, k_begin, int_c, ba, bb, k_group_size);
          } else {
            K::avx_kernel(m, n, k, m_begin, n_begin, k_begin, int_c, ba, bb, k_group_size);
          }
        }
        // }

        // Apply scale and accumulate to float buffer at end of k_group
        K::apply_scale_kgroup(m, n, m_begin, n_begin, k_group_begin, c, int_c, ba, bb, k, k_group_size);
      }
    }
  }
}

// Convenience functions for k-group kernels
inline void vec_mul_kgroup(int m, int n, int k, int k_group_size, std::shared_ptr<GemmKernel224Int4KGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4KGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4KGroup::BufferC> bc, int ith, int nth) {
  integer_mat_mul_kgroup<GemmKernel224Int4KGroup, false>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size, std::shared_ptr<GemmKernel224Int4KGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4KGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4KGroup::BufferC> bc, int ith, int nth) {
  integer_mat_mul_kgroup<GemmKernel224Int4KGroup, true>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

// Convenience functions for k-group kernels
inline void vec_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224Int4_1KGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4_1KGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4_1KGroup::BufferC> bc, int ith, int nth) {
  integer_mat_mul_kgroup<GemmKernel224Int4_1KGroup, false>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith,
                                                           nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224Int4_1KGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4_1KGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4_1KGroup::BufferC> bc, int ith, int nth) {
  integer_mat_mul_kgroup<GemmKernel224Int4_1KGroup, true>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith,
                                                          nth);
}

// Convenience functions for k-group kernels
inline void vec_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224Int4_1_LowKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4_1_LowKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4_1_LowKGroup::BufferC> bc, int ith, int nth) {
  integer_mat_mul_kgroup<GemmKernel224Int4_1_LowKGroup, false>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith,
                                                               nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224Int4_1_LowKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4_1_LowKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4_1_LowKGroup::BufferC> bc, int ith, int nth) {
  integer_mat_mul_kgroup<GemmKernel224Int4_1_LowKGroup, true>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith,
                                                              nth);
}

}  // namespace amx

#endif  // AMX_KERNELS_HPP
