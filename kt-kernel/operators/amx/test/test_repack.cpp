/**
 * @brief Unit tests for INT8 BufferB dynamic repack path.
 *
 * Tests the roundtrip: BF16 -> INT8 BufferB (from_mat) -> BF16 (to_mat)
 * and the full backward repack: forward INT8 -> to_mat -> BF16 workspace
 *   -> from_mat_transposed -> backward INT8.
 *
 * This is TDD — to_mat() on INT8 BufferB does not exist yet.
 * Once implemented in amx_kernels.hpp, this test should pass.
 *
 * Build (from kt-kernel/operators/amx/test):
 *   g++ -std=c++17 -O2 -march=native -mavx512f -mavx512bw -mavx512vl \
 *       -mamx-int8 -mamx-bf16 -mamx-tile \
 *       -I.. -I../la -I../../../third_party/ggml/include \
 *       test_repack.cpp -o test_repack -lm
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "../la/amx.hpp"
#include "../la/amx_kernels.hpp"

// ============================================================
// Helpers
// ============================================================

// --- INT8 helpers ---
using Int8Kernel = amx::GemmKernel224Int8;
using Int8BufferB = Int8Kernel::BufferB;

static void from_mat_all(Int8BufferB& bb, ggml_bf16_t* src) {
  int nth = Int8Kernel::recommended_nth(bb.n);
  for (int ith = 0; ith < nth; ith++) bb.from_mat(src, ith, nth);
}
static void to_mat_all(Int8BufferB& bb, ggml_bf16_t* dst) {
  int nth = Int8Kernel::recommended_nth(bb.n);
  for (int ith = 0; ith < nth; ith++) bb.to_mat(dst, ith, nth);
}
static void from_mat_transposed_all(Int8BufferB& bb, ggml_bf16_t* src, int src_n, int src_k) {
  int nth = Int8Kernel::recommended_nth(bb.n);
  for (int ith = 0; ith < nth; ith++) bb.from_mat_transposed(src, src_n, src_k, ith, nth);
}

// --- BF16 helpers ---
using BF16Kernel = amx::GemmKernel224BF;
using BF16BufferB = BF16Kernel::BufferB;

static void from_mat_all(BF16BufferB& bb, ggml_bf16_t* src) {
  int nth = BF16Kernel::recommended_nth(bb.n);
  for (int ith = 0; ith < nth; ith++) bb.from_mat(src, ith, nth);
}
static void to_mat_all(BF16BufferB& bb, ggml_bf16_t* dst) {
  int nth = BF16Kernel::recommended_nth(bb.n);
  for (int ith = 0; ith < nth; ith++) bb.to_mat(dst, ith, nth);
}
static void from_mat_transposed_all(BF16BufferB& bb, ggml_bf16_t* src, int src_n, int src_k) {
  int nth = BF16Kernel::recommended_nth(bb.n);
  for (int ith = 0; ith < nth; ith++) bb.from_mat_transposed(src, src_n, src_k, ith, nth);
}

// --- INT4 helpers ---
using Int4Kernel = amx::GemmKernel224Int4;
using Int4BufferB = Int4Kernel::BufferB;

static void from_mat_all(Int4BufferB& bb, ggml_bf16_t* src) {
  int nth = Int4Kernel::recommended_nth(bb.n);
  for (int ith = 0; ith < nth; ith++) bb.from_mat(src, ith, nth);
}
static void to_mat_all(Int4BufferB& bb, ggml_bf16_t* dst) {
  int nth = Int4Kernel::recommended_nth(bb.n);
  for (int ith = 0; ith < nth; ith++) bb.to_mat(dst, ith, nth);
}
static void from_mat_transposed_all(Int4BufferB& bb, ggml_bf16_t* src, int src_n, int src_k) {
  int nth = Int4Kernel::recommended_nth(bb.n);
  for (int ith = 0; ith < nth; ith++) bb.from_mat_transposed(src, src_n, src_k, ith, nth);
}

// --- from_bb_transposed helpers ---
// dst has shape (src.k, src.n), src has shape (src.n, src.k)
static void from_bb_transposed_all(BF16BufferB& dst, const BF16BufferB& src) {
  int nth = BF16Kernel::recommended_nth(dst.n);
  for (int ith = 0; ith < nth; ith++) dst.from_bb_transposed(src, ith, nth);
}
static void from_bb_transposed_all(Int8BufferB& dst, const Int8BufferB& src) {
  int nth = Int8Kernel::recommended_nth(dst.n);
  for (int ith = 0; ith < nth; ith++) dst.from_bb_transposed(src, ith, nth);
}

static int nth_for(int n) { return Int8Kernel::recommended_nth(n); }
static int bf16_nth_for(int n) { return BF16Kernel::recommended_nth(n); }

static float bf16_to_fp32(ggml_bf16_t v) { return GGML_BF16_TO_FP32(v); }
static ggml_bf16_t fp32_to_bf16(float v) { return GGML_FP32_TO_BF16(v); }

/// Fill BF16 buffer with random values in [-max_val, max_val].
static void fill_random_bf16(ggml_bf16_t* buf, size_t count, float max_val, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-max_val, max_val);
  for (size_t i = 0; i < count; i++) {
    buf[i] = fp32_to_bf16(dist(rng));
  }
}

/// Compute mean-absolute-error between two BF16 buffers.
static double compute_mae(const ggml_bf16_t* a, const ggml_bf16_t* b, size_t count) {
  double sum = 0.0;
  for (size_t i = 0; i < count; i++) {
    float va = bf16_to_fp32(a[i]);
    float vb = bf16_to_fp32(b[i]);
    sum += std::fabs(va - vb);
  }
  return sum / count;
}

/// Compute mean-absolute-value of a BF16 buffer.
static double compute_mean_abs(const ggml_bf16_t* buf, size_t count) {
  double sum = 0.0;
  for (size_t i = 0; i < count; i++) {
    sum += std::fabs(bf16_to_fp32(buf[i]));
  }
  return sum / count;
}

/// Compute max-absolute-error between two BF16 buffers.
static double compute_max_err(const ggml_bf16_t* a, const ggml_bf16_t* b, size_t count) {
  double max_err = 0.0;
  for (size_t i = 0; i < count; i++) {
    float va = bf16_to_fp32(a[i]);
    float vb = bf16_to_fp32(b[i]);
    double err = std::fabs(va - vb);
    if (err > max_err) max_err = err;
  }
  return max_err;
}

/// Compute relative error: MAE / mean_abs.
static double compute_relative_error(const ggml_bf16_t* ref, const ggml_bf16_t* test, size_t count) {
  double mae = compute_mae(ref, test, count);
  double mean_abs = compute_mean_abs(ref, count);
  if (mean_abs < 1e-10) return mae;
  return mae / mean_abs;
}

/// Transpose BF16 matrix [rows, cols] -> [cols, rows] (naive).
static void transpose_bf16(const ggml_bf16_t* src, ggml_bf16_t* dst, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      dst[c * rows + r] = src[r * cols + c];
    }
  }
}

// ============================================================
// Test 1: INT8 BufferB from_mat -> to_mat roundtrip
// ============================================================

static bool test_int8_bufferb_roundtrip(int n, int k, float max_val, double max_rel_err) {
  printf("  test_int8_bufferb_roundtrip(n=%d, k=%d, max_val=%.1f) ... ", n, k, max_val);

  size_t count = (size_t)n * k;

  // Allocate source BF16 matrix [n, k]
  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, max_val, /*seed=*/42);

  // Allocate INT8 BufferB
  size_t bb_size = Int8BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  Int8BufferB bb(n, k, bb_mem);

  // Pack: BF16 -> INT8 BufferB (all partitions)
  from_mat_all(bb, src.data());

  // Dequant: INT8 BufferB -> BF16 (all partitions)
  std::vector<ggml_bf16_t> recovered(count);
  to_mat_all(bb, recovered.data());

  // Compare
  double rel_err = compute_relative_error(src.data(), recovered.data(), count);
  double mae = compute_mae(src.data(), recovered.data(), count);
  double max_err = compute_max_err(src.data(), recovered.data(), count);

  std::free(bb_mem);

  bool pass = rel_err < max_rel_err;
  printf("rel_err=%.6f mae=%.6e max_err=%.6e  %s\n", rel_err, mae, max_err, pass ? "PASS" : "FAIL");

  if (!pass) {
    printf("    Sample values (src -> recovered):\n");
    for (int i = 0; i < std::min(8, (int)count); i++) {
      printf("      [%d] %.6f -> %.6f (err=%.6e)\n", i, bf16_to_fp32(src[i]), bf16_to_fp32(recovered[i]),
             bf16_to_fp32(src[i]) - bf16_to_fp32(recovered[i]));
    }
  }
  return pass;
}

// ============================================================
// Test 2: Full backward repack path
//   forward INT8 [n, k] -> to_mat -> BF16 workspace [n, k]
//   -> from_mat_transposed -> backward INT8 [k, n]
// vs.
//   direct from_mat_transposed on original BF16 -> backward INT8 [k, n]
// ============================================================

static bool test_full_repack_path(int n, int k, float max_val, double max_rel_err) {
  printf("  test_full_repack_path(n=%d, k=%d) ... ", n, k);

  size_t src_count = (size_t)n * k;
  size_t dst_count = (size_t)k * n;

  // Source BF16 matrix [n, k] (represents forward weight)
  std::vector<ggml_bf16_t> src(src_count);
  fill_random_bf16(src.data(), src_count, max_val, /*seed=*/123);

  // === Path A: Direct from_mat_transposed (ground truth for backward) ===
  size_t bb_bwd_size = Int8BufferB::required_size(k, n);
  void* bb_bwd_direct_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_direct_mem, 0, bb_bwd_size);
  Int8BufferB bb_bwd_direct(k, n, bb_bwd_direct_mem);
  from_mat_transposed_all(bb_bwd_direct, src.data(), n, k);

  // === Path B: Forward pack -> to_mat -> from_mat_transposed (the repack path) ===
  size_t bb_fwd_size = Int8BufferB::required_size(n, k);
  void* bb_fwd_mem = std::aligned_alloc(64, bb_fwd_size);
  memset(bb_fwd_mem, 0, bb_fwd_size);
  Int8BufferB bb_fwd(n, k, bb_fwd_mem);
  from_mat_all(bb_fwd, src.data());

  std::vector<ggml_bf16_t> workspace(src_count);
  to_mat_all(bb_fwd, workspace.data());

  void* bb_bwd_repack_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_repack_mem, 0, bb_bwd_size);
  Int8BufferB bb_bwd_repack(k, n, bb_bwd_repack_mem);
  from_mat_transposed_all(bb_bwd_repack, workspace.data(), n, k);

  // === Compare: Dequant both backward BufferBs and compare ===
  std::vector<ggml_bf16_t> bwd_direct_bf16(dst_count);
  to_mat_all(bb_bwd_direct, bwd_direct_bf16.data());

  std::vector<ggml_bf16_t> bwd_repack_bf16(dst_count);
  to_mat_all(bb_bwd_repack, bwd_repack_bf16.data());

  double rel_err = compute_relative_error(bwd_direct_bf16.data(), bwd_repack_bf16.data(), dst_count);
  double mae = compute_mae(bwd_direct_bf16.data(), bwd_repack_bf16.data(), dst_count);

  std::free(bb_fwd_mem);
  std::free(bb_bwd_direct_mem);
  std::free(bb_bwd_repack_mem);

  bool pass = rel_err < max_rel_err;
  printf("rel_err=%.6f mae=%.6e  %s\n", rel_err, mae, pass ? "PASS" : "FAIL");

  if (!pass) {
    printf("    Sample backward values (direct -> repack):\n");
    for (int i = 0; i < std::min(8, (int)dst_count); i++) {
      printf("      [%d] %.6f -> %.6f\n", i, bf16_to_fp32(bwd_direct_bf16[i]), bf16_to_fp32(bwd_repack_bf16[i]));
    }
  }
  return pass;
}

// ============================================================
// Test 3: to_mat with multi-threaded packing
//   Verify single-thread to_mat matches multi-thread from_mat.
// ============================================================

static bool test_int8_bufferb_roundtrip_multithread(int n, int k, double max_rel_err) {
  int nth = nth_for(n);
  printf("  test_int8_bufferb_roundtrip_multithread(n=%d, k=%d, nth=%d) ... ", n, k, nth);

  size_t count = (size_t)n * k;

  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, /*seed=*/77);

  size_t bb_size = Int8BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  Int8BufferB bb(n, k, bb_mem);

  // Pack with all partitions
  for (int ith = 0; ith < nth; ith++) {
    bb.from_mat(src.data(), ith, nth);
  }

  // Dequant with all partitions
  std::vector<ggml_bf16_t> recovered(count);
  to_mat_all(bb, recovered.data());

  double rel_err = compute_relative_error(src.data(), recovered.data(), count);

  std::free(bb_mem);

  bool pass = rel_err < max_rel_err;
  printf("rel_err=%.6f  %s\n", rel_err, pass ? "PASS" : "FAIL");
  return pass;
}

// ============================================================
// Test 4: Edge case — zero matrix
// ============================================================

static bool test_int8_bufferb_zero_matrix(int n, int k) {
  printf("  test_int8_bufferb_zero_matrix(n=%d, k=%d) ... ", n, k);

  size_t count = (size_t)n * k;

  std::vector<ggml_bf16_t> src(count);
  for (size_t i = 0; i < count; i++) src[i] = fp32_to_bf16(0.0f);

  size_t bb_size = Int8BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  Int8BufferB bb(n, k, bb_mem);

  from_mat_all(bb, src.data());

  std::vector<ggml_bf16_t> recovered(count);
  to_mat_all(bb, recovered.data());

  double max_err = compute_max_err(src.data(), recovered.data(), count);
  std::free(bb_mem);

  bool pass = max_err == 0.0;
  printf("max_err=%.6e  %s\n", max_err, pass ? "PASS" : "FAIL");
  return pass;
}

// ============================================================
// Test 5: to_mat multi-threaded dequant
//   to_mat itself should support ith/nth for parallelism.
// ============================================================

static bool test_int8_bufferb_to_mat_parallel(int n, int k, double max_rel_err) {
  int nth = nth_for(n);
  printf("  test_int8_bufferb_to_mat_parallel(n=%d, k=%d, nth=%d) ... ", n, k, nth);

  size_t count = (size_t)n * k;

  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, /*seed=*/99);

  size_t bb_size = Int8BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  Int8BufferB bb(n, k, bb_mem);
  from_mat_all(bb, src.data());

  // Dequant partition-by-partition into one buffer
  std::vector<ggml_bf16_t> recovered_partitioned(count);
  for (int ith = 0; ith < nth; ith++) {
    bb.to_mat(recovered_partitioned.data(), ith, nth);
  }

  // Also dequant via helper (should be identical)
  std::vector<ggml_bf16_t> recovered_all(count);
  to_mat_all(bb, recovered_all.data());

  double mae = compute_mae(recovered_partitioned.data(), recovered_all.data(), count);
  std::free(bb_mem);

  bool pass = mae == 0.0;
  printf("mae=%.6e  %s\n", mae, pass ? "PASS" : "FAIL");
  return pass;
}

// ============================================================
// BF16 BufferB Tests (lossless roundtrip)
// ============================================================

static bool test_bf16_bufferb_roundtrip(int n, int k, float max_val) {
  printf("  test_bf16_bufferb_roundtrip(n=%d, k=%d, max_val=%.1f) ... ", n, k, max_val);

  size_t count = (size_t)n * k;

  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, max_val, /*seed=*/42);

  size_t bb_size = BF16BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  BF16BufferB bb(n, k, bb_mem);

  from_mat_all(bb, src.data());

  std::vector<ggml_bf16_t> recovered(count);
  to_mat_all(bb, recovered.data());

  double mae = compute_mae(src.data(), recovered.data(), count);
  double max_err = compute_max_err(src.data(), recovered.data(), count);

  std::free(bb_mem);

  bool pass = mae == 0.0 && max_err == 0.0;
  printf("mae=%.6e max_err=%.6e  %s\n", mae, max_err, pass ? "PASS" : "FAIL");

  if (!pass) {
    printf("    Sample values (src -> recovered):\n");
    for (int i = 0; i < std::min(8, (int)count); i++) {
      printf("      [%d] %.6f -> %.6f\n", i, bf16_to_fp32(src[i]), bf16_to_fp32(recovered[i]));
    }
  }
  return pass;
}

static bool test_bf16_full_repack_path(int n, int k, float max_val) {
  printf("  test_bf16_full_repack_path(n=%d, k=%d) ... ", n, k);

  size_t src_count = (size_t)n * k;
  size_t dst_count = (size_t)k * n;

  std::vector<ggml_bf16_t> src(src_count);
  fill_random_bf16(src.data(), src_count, max_val, /*seed=*/123);

  // Path A: direct from_mat_transposed
  size_t bb_bwd_size = BF16BufferB::required_size(k, n);
  void* bb_bwd_direct_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_direct_mem, 0, bb_bwd_size);
  BF16BufferB bb_bwd_direct(k, n, bb_bwd_direct_mem);
  from_mat_transposed_all(bb_bwd_direct, src.data(), n, k);

  // Path B: from_mat -> to_mat -> from_mat_transposed
  size_t bb_fwd_size = BF16BufferB::required_size(n, k);
  void* bb_fwd_mem = std::aligned_alloc(64, bb_fwd_size);
  memset(bb_fwd_mem, 0, bb_fwd_size);
  BF16BufferB bb_fwd(n, k, bb_fwd_mem);
  from_mat_all(bb_fwd, src.data());

  std::vector<ggml_bf16_t> workspace(src_count);
  to_mat_all(bb_fwd, workspace.data());

  void* bb_bwd_repack_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_repack_mem, 0, bb_bwd_size);
  BF16BufferB bb_bwd_repack(k, n, bb_bwd_repack_mem);
  from_mat_transposed_all(bb_bwd_repack, workspace.data(), n, k);

  // Compare packed buffers directly (both should be bit-identical since BF16 is lossless)
  std::vector<ggml_bf16_t> bwd_direct_bf16(dst_count);
  to_mat_all(bb_bwd_direct, bwd_direct_bf16.data());

  std::vector<ggml_bf16_t> bwd_repack_bf16(dst_count);
  to_mat_all(bb_bwd_repack, bwd_repack_bf16.data());

  double mae = compute_mae(bwd_direct_bf16.data(), bwd_repack_bf16.data(), dst_count);

  std::free(bb_fwd_mem);
  std::free(bb_bwd_direct_mem);
  std::free(bb_bwd_repack_mem);

  bool pass = mae == 0.0;
  printf("mae=%.6e  %s\n", mae, pass ? "PASS" : "FAIL");
  return pass;
}

static bool test_bf16_bufferb_zero_matrix(int n, int k) {
  printf("  test_bf16_bufferb_zero_matrix(n=%d, k=%d) ... ", n, k);

  size_t count = (size_t)n * k;
  std::vector<ggml_bf16_t> src(count, fp32_to_bf16(0.0f));

  size_t bb_size = BF16BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  BF16BufferB bb(n, k, bb_mem);

  from_mat_all(bb, src.data());

  std::vector<ggml_bf16_t> recovered(count);
  to_mat_all(bb, recovered.data());

  double max_err = compute_max_err(src.data(), recovered.data(), count);
  std::free(bb_mem);

  bool pass = max_err == 0.0;
  printf("max_err=%.6e  %s\n", max_err, pass ? "PASS" : "FAIL");
  return pass;
}

// ============================================================
// INT4 BufferB Tests
// ============================================================

// INT4 constraints: n % N_STEP(32) == 0, k % B_K_STEP(128) == 0
// INT4 quantization: 4-bit signed [-8, 7], scale = amax / 112, ~14% relative error per roundtrip

static bool test_int4_bufferb_roundtrip(int n, int k, float max_val, double max_rel_err) {
  printf("  test_int4_bufferb_roundtrip(n=%d, k=%d, max_val=%.1f) ... ", n, k, max_val);

  size_t count = (size_t)n * k;

  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, max_val, /*seed=*/42);

  size_t bb_size = Int4BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  Int4BufferB bb(n, k, bb_mem);

  from_mat_all(bb, src.data());

  std::vector<ggml_bf16_t> recovered(count);
  to_mat_all(bb, recovered.data());

  double rel_err = compute_relative_error(src.data(), recovered.data(), count);
  double mae = compute_mae(src.data(), recovered.data(), count);
  double max_err = compute_max_err(src.data(), recovered.data(), count);

  std::free(bb_mem);

  bool pass = rel_err < max_rel_err;
  printf("rel_err=%.6f mae=%.6e max_err=%.6e  %s\n", rel_err, mae, max_err, pass ? "PASS" : "FAIL");

  if (!pass) {
    printf("    Sample values (src -> recovered):\n");
    for (int i = 0; i < std::min(8, (int)count); i++) {
      printf("      [%d] %.6f -> %.6f (err=%.6e)\n", i, bf16_to_fp32(src[i]), bf16_to_fp32(recovered[i]),
             bf16_to_fp32(src[i]) - bf16_to_fp32(recovered[i]));
    }
  }
  return pass;
}

static bool test_int4_full_repack_path(int n, int k, float max_val, double max_rel_err) {
  printf("  test_int4_full_repack_path(n=%d, k=%d) ... ", n, k);

  size_t src_count = (size_t)n * k;
  size_t dst_count = (size_t)k * n;

  std::vector<ggml_bf16_t> src(src_count);
  fill_random_bf16(src.data(), src_count, max_val, /*seed=*/123);

  // Path A: direct from_mat_transposed (ground truth)
  size_t bb_bwd_size = Int4BufferB::required_size(k, n);
  void* bb_bwd_direct_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_direct_mem, 0, bb_bwd_size);
  Int4BufferB bb_bwd_direct(k, n, bb_bwd_direct_mem);
  from_mat_transposed_all(bb_bwd_direct, src.data(), n, k);

  // Path B: from_mat -> to_mat -> from_mat_transposed (repack path)
  size_t bb_fwd_size = Int4BufferB::required_size(n, k);
  void* bb_fwd_mem = std::aligned_alloc(64, bb_fwd_size);
  memset(bb_fwd_mem, 0, bb_fwd_size);
  Int4BufferB bb_fwd(n, k, bb_fwd_mem);
  from_mat_all(bb_fwd, src.data());

  std::vector<ggml_bf16_t> workspace(src_count);
  to_mat_all(bb_fwd, workspace.data());

  void* bb_bwd_repack_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_repack_mem, 0, bb_bwd_size);
  Int4BufferB bb_bwd_repack(k, n, bb_bwd_repack_mem);
  from_mat_transposed_all(bb_bwd_repack, workspace.data(), n, k);

  // Compare: dequant both backward buffers
  std::vector<ggml_bf16_t> bwd_direct_bf16(dst_count);
  to_mat_all(bb_bwd_direct, bwd_direct_bf16.data());

  std::vector<ggml_bf16_t> bwd_repack_bf16(dst_count);
  to_mat_all(bb_bwd_repack, bwd_repack_bf16.data());

  double rel_err = compute_relative_error(bwd_direct_bf16.data(), bwd_repack_bf16.data(), dst_count);
  double mae = compute_mae(bwd_direct_bf16.data(), bwd_repack_bf16.data(), dst_count);

  std::free(bb_fwd_mem);
  std::free(bb_bwd_direct_mem);
  std::free(bb_bwd_repack_mem);

  bool pass = rel_err < max_rel_err;
  printf("rel_err=%.6f mae=%.6e  %s\n", rel_err, mae, pass ? "PASS" : "FAIL");

  if (!pass) {
    printf("    Sample backward values (direct -> repack):\n");
    for (int i = 0; i < std::min(8, (int)dst_count); i++) {
      printf("      [%d] %.6f -> %.6f\n", i, bf16_to_fp32(bwd_direct_bf16[i]), bf16_to_fp32(bwd_repack_bf16[i]));
    }
  }
  return pass;
}

static bool test_int4_bufferb_zero_matrix(int n, int k) {
  printf("  test_int4_bufferb_zero_matrix(n=%d, k=%d) ... ", n, k);

  size_t count = (size_t)n * k;
  std::vector<ggml_bf16_t> src(count, fp32_to_bf16(0.0f));

  size_t bb_size = Int4BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  Int4BufferB bb(n, k, bb_mem);

  from_mat_all(bb, src.data());

  std::vector<ggml_bf16_t> recovered(count);
  to_mat_all(bb, recovered.data());

  double max_err = compute_max_err(src.data(), recovered.data(), count);
  std::free(bb_mem);

  bool pass = max_err == 0.0;
  printf("max_err=%.6e  %s\n", max_err, pass ? "PASS" : "FAIL");
  return pass;
}

// ============================================================
// BF16 from_bb_transposed Tests (TDD — method not yet implemented)
// ============================================================

/**
 * Test BF16 from_bb_transposed against the ground truth path:
 *   Path A (ground truth): BF16 src → from_mat → fwd BB(n,k) → to_mat → workspace → from_mat_transposed → bwd BB(k,n)
 *   Path B (new):          BF16 src → from_mat → fwd BB(n,k) → from_bb_transposed → bwd BB(k,n)
 *
 * BF16 is lossless, so both paths should produce bit-identical results.
 */
static bool test_bf16_from_bb_transposed(int n, int k, float max_val) {
  printf("  test_bf16_from_bb_transposed(n=%d, k=%d) ... ", n, k);

  size_t src_count = (size_t)n * k;
  size_t dst_count = (size_t)k * n;

  // Source BF16 matrix [n, k]
  std::vector<ggml_bf16_t> src(src_count);
  fill_random_bf16(src.data(), src_count, max_val, /*seed=*/42);

  // Forward BB(n, k)
  size_t bb_fwd_size = BF16BufferB::required_size(n, k);
  void* bb_fwd_mem = std::aligned_alloc(64, bb_fwd_size);
  memset(bb_fwd_mem, 0, bb_fwd_size);
  BF16BufferB bb_fwd(n, k, bb_fwd_mem);
  from_mat_all(bb_fwd, src.data());

  // Path A: to_mat → from_mat_transposed
  size_t bb_bwd_size = BF16BufferB::required_size(k, n);
  std::vector<ggml_bf16_t> workspace(src_count);
  to_mat_all(bb_fwd, workspace.data());

  void* bb_bwd_a_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_a_mem, 0, bb_bwd_size);
  BF16BufferB bb_bwd_a(k, n, bb_bwd_a_mem);
  from_mat_transposed_all(bb_bwd_a, workspace.data(), n, k);

  // Path B: from_bb_transposed
  void* bb_bwd_b_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_b_mem, 0, bb_bwd_size);
  BF16BufferB bb_bwd_b(k, n, bb_bwd_b_mem);
  from_bb_transposed_all(bb_bwd_b, bb_fwd);

  // Compare: dequant both → compare BF16 values
  std::vector<ggml_bf16_t> bwd_a_bf16(dst_count);
  to_mat_all(bb_bwd_a, bwd_a_bf16.data());

  std::vector<ggml_bf16_t> bwd_b_bf16(dst_count);
  to_mat_all(bb_bwd_b, bwd_b_bf16.data());

  double mae = compute_mae(bwd_a_bf16.data(), bwd_b_bf16.data(), dst_count);
  double max_err = compute_max_err(bwd_a_bf16.data(), bwd_b_bf16.data(), dst_count);

  std::free(bb_fwd_mem);
  std::free(bb_bwd_a_mem);
  std::free(bb_bwd_b_mem);

  // BF16 → BF16 should be bit-exact
  bool pass = mae == 0.0 && max_err == 0.0;
  printf("mae=%.6e max_err=%.6e  %s\n", mae, max_err, pass ? "PASS" : "FAIL");

  if (!pass) {
    printf("    Sample (ground_truth -> from_bb_transposed):\n");
    for (int i = 0; i < std::min(8, (int)dst_count); i++) {
      printf("      [%d] %.6f -> %.6f\n", i, bf16_to_fp32(bwd_a_bf16[i]), bf16_to_fp32(bwd_b_bf16[i]));
    }
  }
  return pass;
}

/// BF16 from_bb_transposed with zero matrix.
static bool test_bf16_from_bb_transposed_zero(int n, int k) {
  printf("  test_bf16_from_bb_transposed_zero(n=%d, k=%d) ... ", n, k);

  size_t src_count = (size_t)n * k;
  size_t dst_count = (size_t)k * n;

  std::vector<ggml_bf16_t> src(src_count, fp32_to_bf16(0.0f));

  size_t bb_fwd_size = BF16BufferB::required_size(n, k);
  void* bb_fwd_mem = std::aligned_alloc(64, bb_fwd_size);
  memset(bb_fwd_mem, 0, bb_fwd_size);
  BF16BufferB bb_fwd(n, k, bb_fwd_mem);
  from_mat_all(bb_fwd, src.data());

  size_t bb_bwd_size = BF16BufferB::required_size(k, n);
  void* bb_bwd_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_mem, 0, bb_bwd_size);
  BF16BufferB bb_bwd(k, n, bb_bwd_mem);
  from_bb_transposed_all(bb_bwd, bb_fwd);

  std::vector<ggml_bf16_t> result(dst_count);
  to_mat_all(bb_bwd, result.data());

  // All values should be exactly zero
  double max_err = 0.0;
  for (size_t i = 0; i < dst_count; i++) {
    double v = std::fabs(bf16_to_fp32(result[i]));
    if (v > max_err) max_err = v;
  }

  std::free(bb_fwd_mem);
  std::free(bb_bwd_mem);

  bool pass = max_err == 0.0;
  printf("max_err=%.6e  %s\n", max_err, pass ? "PASS" : "FAIL");
  return pass;
}

// ============================================================
// INT8 from_bb_transposed Tests (TDD — method not yet implemented)
// ============================================================

/**
 * Test INT8 from_bb_transposed against the ground truth path:
 *   Path A: BF16 src → from_mat → fwd BB(n,k) → to_mat → workspace → from_mat_transposed → bwd BB(k,n)
 *   Path B: BF16 src → from_mat → fwd BB(n,k) → from_bb_transposed → bwd BB(k,n)
 *
 * INT8 involves quantization so paths may differ slightly (different intermediate precision).
 * We compare dequantized outputs with a tolerance.
 */
static bool test_int8_from_bb_transposed(int n, int k, float max_val, double max_rel_err) {
  printf("  test_int8_from_bb_transposed(n=%d, k=%d) ... ", n, k);

  size_t src_count = (size_t)n * k;
  size_t dst_count = (size_t)k * n;

  std::vector<ggml_bf16_t> src(src_count);
  fill_random_bf16(src.data(), src_count, max_val, /*seed=*/42);

  // Forward BB(n, k)
  size_t bb_fwd_size = Int8BufferB::required_size(n, k);
  void* bb_fwd_mem = std::aligned_alloc(64, bb_fwd_size);
  memset(bb_fwd_mem, 0, bb_fwd_size);
  Int8BufferB bb_fwd(n, k, bb_fwd_mem);
  from_mat_all(bb_fwd, src.data());

  // Path A: to_mat → from_mat_transposed
  size_t bb_bwd_size = Int8BufferB::required_size(k, n);
  std::vector<ggml_bf16_t> workspace(src_count);
  to_mat_all(bb_fwd, workspace.data());

  void* bb_bwd_a_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_a_mem, 0, bb_bwd_size);
  Int8BufferB bb_bwd_a(k, n, bb_bwd_a_mem);
  from_mat_transposed_all(bb_bwd_a, workspace.data(), n, k);

  // Path B: from_bb_transposed
  void* bb_bwd_b_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_b_mem, 0, bb_bwd_size);
  Int8BufferB bb_bwd_b(k, n, bb_bwd_b_mem);
  from_bb_transposed_all(bb_bwd_b, bb_fwd);

  // Compare dequantized outputs
  std::vector<ggml_bf16_t> bwd_a_bf16(dst_count);
  to_mat_all(bb_bwd_a, bwd_a_bf16.data());

  std::vector<ggml_bf16_t> bwd_b_bf16(dst_count);
  to_mat_all(bb_bwd_b, bwd_b_bf16.data());

  double rel_err = compute_relative_error(bwd_a_bf16.data(), bwd_b_bf16.data(), dst_count);
  double mae = compute_mae(bwd_a_bf16.data(), bwd_b_bf16.data(), dst_count);
  double max_err = compute_max_err(bwd_a_bf16.data(), bwd_b_bf16.data(), dst_count);

  std::free(bb_fwd_mem);
  std::free(bb_bwd_a_mem);
  std::free(bb_bwd_b_mem);

  bool pass = rel_err < max_rel_err;
  printf("rel_err=%.6f mae=%.6e max_err=%.6e  %s\n", rel_err, mae, max_err, pass ? "PASS" : "FAIL");

  if (!pass) {
    printf("    Sample (ground_truth -> from_bb_transposed):\n");
    for (int i = 0; i < std::min(8, (int)dst_count); i++) {
      printf("      [%d] %.6f -> %.6f\n", i, bf16_to_fp32(bwd_a_bf16[i]), bf16_to_fp32(bwd_b_bf16[i]));
    }
  }
  return pass;
}

/// INT8 from_bb_transposed with zero matrix.
static bool test_int8_from_bb_transposed_zero(int n, int k) {
  printf("  test_int8_from_bb_transposed_zero(n=%d, k=%d) ... ", n, k);

  size_t src_count = (size_t)n * k;
  size_t dst_count = (size_t)k * n;

  std::vector<ggml_bf16_t> src(src_count, fp32_to_bf16(0.0f));

  size_t bb_fwd_size = Int8BufferB::required_size(n, k);
  void* bb_fwd_mem = std::aligned_alloc(64, bb_fwd_size);
  memset(bb_fwd_mem, 0, bb_fwd_size);
  Int8BufferB bb_fwd(n, k, bb_fwd_mem);
  from_mat_all(bb_fwd, src.data());

  size_t bb_bwd_size = Int8BufferB::required_size(k, n);
  void* bb_bwd_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_mem, 0, bb_bwd_size);
  Int8BufferB bb_bwd(k, n, bb_bwd_mem);
  from_bb_transposed_all(bb_bwd, bb_fwd);

  std::vector<ggml_bf16_t> result(dst_count);
  to_mat_all(bb_bwd, result.data());

  double max_err = 0.0;
  for (size_t i = 0; i < dst_count; i++) {
    double v = std::fabs(bf16_to_fp32(result[i]));
    if (v > max_err) max_err = v;
  }

  std::free(bb_fwd_mem);
  std::free(bb_bwd_mem);

  bool pass = max_err == 0.0;
  printf("max_err=%.6e  %s\n", max_err, pass ? "PASS" : "FAIL");
  return pass;
}

/**
 * INT8 from_bb_transposed: verify against original BF16 source (end-to-end quality).
 * Compares the dequanted backward BB against the naively transposed original BF16.
 * Expected error: double quantization (~5%).
 */
static bool test_int8_from_bb_transposed_vs_original(int n, int k, float max_val, double max_rel_err) {
  printf("  test_int8_from_bb_transposed_vs_original(n=%d, k=%d) ... ", n, k);

  size_t src_count = (size_t)n * k;
  size_t dst_count = (size_t)k * n;

  std::vector<ggml_bf16_t> src(src_count);
  fill_random_bf16(src.data(), src_count, max_val, /*seed=*/77);

  // Forward BB(n, k)
  size_t bb_fwd_size = Int8BufferB::required_size(n, k);
  void* bb_fwd_mem = std::aligned_alloc(64, bb_fwd_size);
  memset(bb_fwd_mem, 0, bb_fwd_size);
  Int8BufferB bb_fwd(n, k, bb_fwd_mem);
  from_mat_all(bb_fwd, src.data());

  // from_bb_transposed → bwd BB(k, n)
  size_t bb_bwd_size = Int8BufferB::required_size(k, n);
  void* bb_bwd_mem = std::aligned_alloc(64, bb_bwd_size);
  memset(bb_bwd_mem, 0, bb_bwd_size);
  Int8BufferB bb_bwd(k, n, bb_bwd_mem);
  from_bb_transposed_all(bb_bwd, bb_fwd);

  // Dequant backward BB
  std::vector<ggml_bf16_t> bwd_bf16(dst_count);
  to_mat_all(bb_bwd, bwd_bf16.data());

  // Naive transpose of original
  std::vector<ggml_bf16_t> src_transposed(dst_count);
  transpose_bf16(src.data(), src_transposed.data(), n, k);

  double rel_err = compute_relative_error(src_transposed.data(), bwd_bf16.data(), dst_count);
  double mae = compute_mae(src_transposed.data(), bwd_bf16.data(), dst_count);

  std::free(bb_fwd_mem);
  std::free(bb_bwd_mem);

  bool pass = rel_err < max_rel_err;
  printf("rel_err=%.6f mae=%.6e  %s\n", rel_err, mae, pass ? "PASS" : "FAIL");
  return pass;
}

// ============================================================
// from_bb_transposed Performance Benchmarks
// ============================================================

#include <chrono>

/// Benchmark BF16 from_bb_transposed.
static void bench_bf16_from_bb_transposed(int n, int k, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t fwd_size = BF16BufferB::required_size(n, k);
  size_t bwd_size = BF16BufferB::required_size(k, n);

  void* fwd_mem = std::aligned_alloc(64, fwd_size);
  void* bwd_mem = std::aligned_alloc(64, bwd_size);
  memset(fwd_mem, 0, fwd_size);
  memset(bwd_mem, 0, bwd_size);

  BF16BufferB bb_fwd(n, k, fwd_mem);
  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  from_mat_all(bb_fwd, src.data());

  auto do_repack = [&]() {
    BF16BufferB bb_bwd(k, n, bwd_mem);
    from_bb_transposed_all(bb_bwd, bb_fwd);
  };

  for (int i = 0; i < warmup; i++) do_repack();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) do_repack();
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  printf("  bf16_from_bb_transposed(%d, %d) -> (%d, %d): %.1f us (%.3f ms)\n", n, k, k, n, us, us / 1000.0);

  std::free(fwd_mem);
  std::free(bwd_mem);
}

/// Benchmark INT8 from_bb_transposed.
static void bench_int8_from_bb_transposed(int n, int k, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t fwd_size = Int8BufferB::required_size(n, k);
  size_t bwd_size = Int8BufferB::required_size(k, n);

  void* fwd_mem = std::aligned_alloc(64, fwd_size);
  void* bwd_mem = std::aligned_alloc(64, bwd_size);
  memset(fwd_mem, 0, fwd_size);
  memset(bwd_mem, 0, bwd_size);

  Int8BufferB bb_fwd(n, k, fwd_mem);
  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  from_mat_all(bb_fwd, src.data());

  auto do_repack = [&]() {
    Int8BufferB bb_bwd(k, n, bwd_mem);
    from_bb_transposed_all(bb_bwd, bb_fwd);
  };

  for (int i = 0; i < warmup; i++) do_repack();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) do_repack();
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  printf("  int8_from_bb_transposed(%d, %d) -> (%d, %d): %.1f us (%.3f ms)\n", n, k, k, n, us, us / 1000.0);

  std::free(fwd_mem);
  std::free(bwd_mem);
}

// ============================================================
// Multithreaded from_bb_transposed benchmarks
// ============================================================

#include <thread>

template <typename Kernel, typename BB>
static void bench_from_bb_transposed_mt(const char* label, int n, int k, int num_threads, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t fwd_size = BB::required_size(n, k);
  size_t bwd_size = BB::required_size(k, n);

  void* fwd_mem = std::aligned_alloc(64, fwd_size);
  void* bwd_mem = std::aligned_alloc(64, bwd_size);
  memset(fwd_mem, 0, fwd_size);
  memset(bwd_mem, 0, bwd_size);

  BB bb_fwd(n, k, fwd_mem);
  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  {
    int nth = Kernel::recommended_nth(bb_fwd.n);
    for (int ith = 0; ith < nth; ith++) bb_fwd.from_mat(src.data(), ith, nth);
  }

  int nth = std::min(num_threads, Kernel::recommended_nth(k));  // dest.n = k

  auto do_repack = [&]() {
    BB bb_bwd(k, n, bwd_mem);
    std::vector<std::thread> threads;
    for (int t = 0; t < nth; t++) {
      threads.emplace_back([&, t]() { bb_bwd.from_bb_transposed(bb_fwd, t, nth); });
    }
    for (auto& t : threads) t.join();
  };

  for (int i = 0; i < warmup; i++) do_repack();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) do_repack();
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  printf("  %s_bb_trans_mt(%d,%d)->(%d,%d) nth=%d: %.1f us (%.3f ms)\n",
         label, n, k, k, n, nth, us, us / 1000.0);

  std::free(fwd_mem);
  std::free(bwd_mem);
}

/// Multithreaded old-path benchmark (to_mat + from_mat_transposed) for comparison.
template <typename Kernel, typename BB>
static void bench_old_repack_mt(const char* label, int n, int k, int num_threads, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t fwd_size = BB::required_size(n, k);
  size_t bwd_size = BB::required_size(k, n);

  void* fwd_mem = std::aligned_alloc(64, fwd_size);
  void* bwd_mem = std::aligned_alloc(64, bwd_size);
  memset(fwd_mem, 0, fwd_size);
  memset(bwd_mem, 0, bwd_size);

  BB bb_fwd(n, k, fwd_mem);
  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  {
    int nth = Kernel::recommended_nth(bb_fwd.n);
    for (int ith = 0; ith < nth; ith++) bb_fwd.from_mat(src.data(), ith, nth);
  }

  // to_mat parallelism uses fwd.n partitions, from_mat_transposed uses bwd.n=k partitions
  int fwd_nth = std::min(num_threads, Kernel::recommended_nth(n));
  int bwd_nth = std::min(num_threads, Kernel::recommended_nth(k));

  std::vector<ggml_bf16_t> workspace(count);

  auto do_repack = [&]() {
    // to_mat (parallel)
    {
      std::vector<std::thread> threads;
      for (int t = 0; t < fwd_nth; t++)
        threads.emplace_back([&, t]() { bb_fwd.to_mat(workspace.data(), t, fwd_nth); });
      for (auto& t : threads) t.join();
    }
    // from_mat_transposed (parallel)
    {
      BB bb_bwd(k, n, bwd_mem);
      std::vector<std::thread> threads;
      for (int t = 0; t < bwd_nth; t++)
        threads.emplace_back([&, t]() { bb_bwd.from_mat_transposed(workspace.data(), n, k, t, bwd_nth); });
      for (auto& t : threads) t.join();
    }
  };

  for (int i = 0; i < warmup; i++) do_repack();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) do_repack();
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  printf("  %s_old_mt(%d,%d)->(%d,%d) nth=%d/%d: %.1f us (%.3f ms)\n",
         label, n, k, k, n, fwd_nth, bwd_nth, us, us / 1000.0);

  std::free(fwd_mem);
  std::free(bwd_mem);
}

// ============================================================
// Performance Benchmarks
// ============================================================

// (chrono already included above)

/// Benchmark to_mat for a single BufferB[n, k].
static void bench_to_mat(int n, int k, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t bb_size = Int8BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  Int8BufferB bb(n, k, bb_mem);

  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  from_mat_all(bb, src.data());

  std::vector<ggml_bf16_t> dst(count);

  // Warmup
  for (int i = 0; i < warmup; i++) to_mat_all(bb, dst.data());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) to_mat_all(bb, dst.data());
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  double mb = (double)(count * sizeof(int8_t) + n * sizeof(float)) / (1024.0 * 1024.0);
  double gbps = mb / us * 1e6 / 1024.0;
  printf("  to_mat(%d, %d): %.1f us  (src %.2f MB, %.2f GB/s read)\n", n, k, us, mb, gbps);

  std::free(bb_mem);
}

/// Benchmark full repack for a single BufferB: to_mat + from_mat_transposed.
static void bench_full_repack(int n, int k, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t fwd_size = Int8BufferB::required_size(n, k);
  size_t bwd_size = Int8BufferB::required_size(k, n);

  void* fwd_mem = std::aligned_alloc(64, fwd_size);
  void* bwd_mem = std::aligned_alloc(64, bwd_size);
  memset(fwd_mem, 0, fwd_size);
  memset(bwd_mem, 0, bwd_size);

  Int8BufferB bb_fwd(n, k, fwd_mem);
  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  from_mat_all(bb_fwd, src.data());

  std::vector<ggml_bf16_t> workspace(count);

  auto do_repack = [&]() {
    to_mat_all(bb_fwd, workspace.data());
    Int8BufferB bb_bwd(k, n, bwd_mem);
    from_mat_transposed_all(bb_bwd, workspace.data(), n, k);
  };

  for (int i = 0; i < warmup; i++) do_repack();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) do_repack();
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  printf("  repack(%d, %d) -> (%d, %d): %.1f us (%.3f ms)\n", n, k, k, n, us, us / 1000.0);

  std::free(fwd_mem);
  std::free(bwd_mem);
}

/// Benchmark one layer's full repack: 128 experts × 3 projections (sequential, single-thread).
static void bench_layer_repack(int hidden, int inter, int num_experts, int warmup, int iters) {
  printf("\n  Layer repack: %d experts, gate/up[%d,%d] + down[%d,%d]\n",
         num_experts, inter, hidden, hidden, inter);

  // Pre-allocate forward BufferBs and backward memory
  size_t gate_up_fwd_size = Int8BufferB::required_size(inter, hidden);
  size_t down_fwd_size = Int8BufferB::required_size(hidden, inter);
  size_t gate_up_bwd_size = Int8BufferB::required_size(hidden, inter);
  size_t down_bwd_size = Int8BufferB::required_size(inter, hidden);

  struct ExpertBuffers {
    void* gate_fwd = nullptr;
    void* up_fwd = nullptr;
    void* down_fwd = nullptr;
    void* gate_bwd = nullptr;
    void* up_bwd = nullptr;
    void* down_bwd = nullptr;
  };
  std::vector<ExpertBuffers> experts(num_experts);

  for (int e = 0; e < num_experts; e++) {
    experts[e].gate_fwd = std::aligned_alloc(64, gate_up_fwd_size);
    experts[e].up_fwd = std::aligned_alloc(64, gate_up_fwd_size);
    experts[e].down_fwd = std::aligned_alloc(64, down_fwd_size);
    experts[e].gate_bwd = std::aligned_alloc(64, gate_up_bwd_size);
    experts[e].up_bwd = std::aligned_alloc(64, gate_up_bwd_size);
    experts[e].down_bwd = std::aligned_alloc(64, down_bwd_size);
    memset(experts[e].gate_fwd, 0, gate_up_fwd_size);
    memset(experts[e].up_fwd, 0, gate_up_fwd_size);
    memset(experts[e].down_fwd, 0, down_fwd_size);
    memset(experts[e].gate_bwd, 0, gate_up_bwd_size);
    memset(experts[e].up_bwd, 0, gate_up_bwd_size);
    memset(experts[e].down_bwd, 0, down_bwd_size);

    // Fill forward buffers with random data
    {
      size_t c = (size_t)inter * hidden;
      std::vector<ggml_bf16_t> tmp(c);
      fill_random_bf16(tmp.data(), c, 1.0f, 42 + e);
      Int8BufferB bb(inter, hidden, experts[e].gate_fwd);
      from_mat_all(bb, tmp.data());
    }
    {
      size_t c = (size_t)inter * hidden;
      std::vector<ggml_bf16_t> tmp(c);
      fill_random_bf16(tmp.data(), c, 1.0f, 1000 + e);
      Int8BufferB bb(inter, hidden, experts[e].up_fwd);
      from_mat_all(bb, tmp.data());
    }
    {
      size_t c = (size_t)hidden * inter;
      std::vector<ggml_bf16_t> tmp(c);
      fill_random_bf16(tmp.data(), c, 1.0f, 2000 + e);
      Int8BufferB bb(hidden, inter, experts[e].down_fwd);
      from_mat_all(bb, tmp.data());
    }
  }

  // Workspace for one expert at a time
  size_t ws_size = std::max((size_t)inter * hidden, (size_t)hidden * inter);
  std::vector<ggml_bf16_t> workspace(ws_size);

  auto do_layer_repack = [&]() {
    for (int e = 0; e < num_experts; e++) {
      // gate: fwd[inter, hidden] -> to_mat -> workspace[inter, hidden] -> from_mat_transposed -> bwd[hidden, inter]
      {
        Int8BufferB fwd(inter, hidden, experts[e].gate_fwd);
        to_mat_all(fwd, workspace.data());
        Int8BufferB bwd(hidden, inter, experts[e].gate_bwd);
        from_mat_transposed_all(bwd, workspace.data(), inter, hidden);
      }
      // up: same as gate
      {
        Int8BufferB fwd(inter, hidden, experts[e].up_fwd);
        to_mat_all(fwd, workspace.data());
        Int8BufferB bwd(hidden, inter, experts[e].up_bwd);
        from_mat_transposed_all(bwd, workspace.data(), inter, hidden);
      }
      // down: fwd[hidden, inter] -> to_mat -> workspace[hidden, inter] -> from_mat_transposed -> bwd[inter, hidden]
      {
        Int8BufferB fwd(hidden, inter, experts[e].down_fwd);
        to_mat_all(fwd, workspace.data());
        Int8BufferB bwd(inter, hidden, experts[e].down_bwd);
        from_mat_transposed_all(bwd, workspace.data(), hidden, inter);
      }
    }
  };

  for (int i = 0; i < warmup; i++) do_layer_repack();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) do_layer_repack();
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  double per_expert_ms = ms / num_experts;
  printf("  Layer total: %.1f ms  (%.3f ms/expert, %d experts)\n", ms, per_expert_ms, num_experts);
  printf("  Estimated per-step (94 layers): %.1f ms (%.2f s)\n", ms * 94, ms * 94 / 1000.0);

  // Cleanup
  for (int e = 0; e < num_experts; e++) {
    std::free(experts[e].gate_fwd);
    std::free(experts[e].up_fwd);
    std::free(experts[e].down_fwd);
    std::free(experts[e].gate_bwd);
    std::free(experts[e].up_bwd);
    std::free(experts[e].down_bwd);
  }
}

// ============================================================
// BF16 Performance Benchmarks
// ============================================================

/// Benchmark BF16 to_mat for a single BufferB[n, k].
static void bench_bf16_to_mat(int n, int k, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t bb_size = BF16BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  BF16BufferB bb(n, k, bb_mem);

  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  from_mat_all(bb, src.data());

  std::vector<ggml_bf16_t> dst(count);

  for (int i = 0; i < warmup; i++) to_mat_all(bb, dst.data());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) to_mat_all(bb, dst.data());
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  double mb = (double)(count * sizeof(ggml_bf16_t)) / (1024.0 * 1024.0);
  double gbps = mb / us * 1e6 / 1024.0;
  printf("  bf16_to_mat(%d, %d): %.1f us  (src %.2f MB, %.2f GB/s read)\n", n, k, us, mb, gbps);

  std::free(bb_mem);
}

/// Benchmark BF16 full repack: to_mat + from_mat_transposed.
static void bench_bf16_full_repack(int n, int k, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t fwd_size = BF16BufferB::required_size(n, k);
  size_t bwd_size = BF16BufferB::required_size(k, n);

  void* fwd_mem = std::aligned_alloc(64, fwd_size);
  void* bwd_mem = std::aligned_alloc(64, bwd_size);
  memset(fwd_mem, 0, fwd_size);
  memset(bwd_mem, 0, bwd_size);

  BF16BufferB bb_fwd(n, k, fwd_mem);
  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  from_mat_all(bb_fwd, src.data());

  std::vector<ggml_bf16_t> workspace(count);

  auto do_repack = [&]() {
    to_mat_all(bb_fwd, workspace.data());
    BF16BufferB bb_bwd(k, n, bwd_mem);
    from_mat_transposed_all(bb_bwd, workspace.data(), n, k);
  };

  for (int i = 0; i < warmup; i++) do_repack();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) do_repack();
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  printf("  bf16_repack(%d, %d) -> (%d, %d): %.1f us (%.3f ms)\n", n, k, k, n, us, us / 1000.0);

  std::free(fwd_mem);
  std::free(bwd_mem);
}

// ============================================================
// INT4 Performance Benchmarks
// ============================================================

/// Benchmark INT4 to_mat for a single BufferB[n, k].
static void bench_int4_to_mat(int n, int k, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t bb_size = Int4BufferB::required_size(n, k);
  void* bb_mem = std::aligned_alloc(64, bb_size);
  memset(bb_mem, 0, bb_size);
  Int4BufferB bb(n, k, bb_mem);

  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  from_mat_all(bb, src.data());

  std::vector<ggml_bf16_t> dst(count);

  for (int i = 0; i < warmup; i++) to_mat_all(bb, dst.data());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) to_mat_all(bb, dst.data());
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  double mb = (double)(count / 2 + n * sizeof(float)) / (1024.0 * 1024.0);
  double gbps = mb / us * 1e6 / 1024.0;
  printf("  int4_to_mat(%d, %d): %.1f us  (src %.2f MB, %.2f GB/s read)\n", n, k, us, mb, gbps);

  std::free(bb_mem);
}

/// Benchmark INT4 full repack: to_mat + from_mat_transposed.
static void bench_int4_full_repack(int n, int k, int warmup, int iters) {
  size_t count = (size_t)n * k;
  size_t fwd_size = Int4BufferB::required_size(n, k);
  size_t bwd_size = Int4BufferB::required_size(k, n);

  void* fwd_mem = std::aligned_alloc(64, fwd_size);
  void* bwd_mem = std::aligned_alloc(64, bwd_size);
  memset(fwd_mem, 0, fwd_size);
  memset(bwd_mem, 0, bwd_size);

  Int4BufferB bb_fwd(n, k, fwd_mem);
  std::vector<ggml_bf16_t> src(count);
  fill_random_bf16(src.data(), count, 1.0f, 42);
  from_mat_all(bb_fwd, src.data());

  std::vector<ggml_bf16_t> workspace(count);

  auto do_repack = [&]() {
    to_mat_all(bb_fwd, workspace.data());
    Int4BufferB bb_bwd(k, n, bwd_mem);
    memset(bwd_mem, 0, bwd_size);  // INT4 uses OR to pack, must zero first
    from_mat_transposed_all(bb_bwd, workspace.data(), n, k);
  };

  for (int i = 0; i < warmup; i++) do_repack();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) do_repack();
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  printf("  int4_repack(%d, %d) -> (%d, %d): %.1f us (%.3f ms)\n", n, k, k, n, us, us / 1000.0);

  std::free(fwd_mem);
  std::free(bwd_mem);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
  bool run_bench = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--bench") run_bench = true;
  }
  printf("=== INT8 BufferB Dynamic Repack Unit Tests ===\n\n");

  int pass_count = 0;
  int fail_count = 0;
  auto check = [&](bool result) {
    if (result)
      pass_count++;
    else
      fail_count++;
  };

  // INT8 quantization introduces ~1/127 ≈ 0.8% relative error per element.
  // Double quantization (INT8 -> BF16 -> INT8) adds another pass, so allow ~2%.
  constexpr double ROUNDTRIP_REL_ERR = 0.02;    // 2% for single roundtrip
  constexpr double REPACK_REL_ERR = 0.05;        // 5% for double-quant repack

  // INT8 BufferB constraints: n % N_STEP(32) == 0, k % K_STEP(64) == 0
  printf("[1] INT8 BufferB from_mat -> to_mat roundtrip\n");
  check(test_int8_bufferb_roundtrip(32, 64, 1.0f, ROUNDTRIP_REL_ERR));
  check(test_int8_bufferb_roundtrip(64, 128, 1.0f, ROUNDTRIP_REL_ERR));
  check(test_int8_bufferb_roundtrip(64, 128, 10.0f, ROUNDTRIP_REL_ERR));
  check(test_int8_bufferb_roundtrip(128, 3584, 1.0f, ROUNDTRIP_REL_ERR));  // partial K_BLOCK
  // Model dimensions (TP=2: intermediate_size/2=1024, hidden_size=7168)
  check(test_int8_bufferb_roundtrip(1024, 7168, 1.0f, ROUNDTRIP_REL_ERR));
  check(test_int8_bufferb_roundtrip(7168, 1024, 1.0f, ROUNDTRIP_REL_ERR));

  // Full repack: backward BufferB[k, n] requires k % 32 == 0 AND n % 64 == 0,
  // so both forward n and k must be multiples of 64.
  printf("\n[2] Full backward repack path (forward INT8 -> to_mat -> from_mat_transposed -> backward INT8)\n");
  check(test_full_repack_path(64, 64, 1.0f, REPACK_REL_ERR));
  check(test_full_repack_path(64, 128, 1.0f, REPACK_REL_ERR));
  check(test_full_repack_path(128, 3584, 1.0f, REPACK_REL_ERR));
  check(test_full_repack_path(1024, 7168, 1.0f, REPACK_REL_ERR));
  check(test_full_repack_path(7168, 1024, 1.0f, REPACK_REL_ERR));

  printf("\n[3] Multi-threaded from_mat -> to_mat roundtrip\n");
  check(test_int8_bufferb_roundtrip_multithread(64, 128, ROUNDTRIP_REL_ERR));
  check(test_int8_bufferb_roundtrip_multithread(1024, 7168, ROUNDTRIP_REL_ERR));

  printf("\n[4] Zero matrix edge case\n");
  check(test_int8_bufferb_zero_matrix(32, 64));
  check(test_int8_bufferb_zero_matrix(64, 128));
  check(test_int8_bufferb_zero_matrix(1024, 7168));

  printf("\n[5] to_mat parallel dequant consistency\n");
  check(test_int8_bufferb_to_mat_parallel(64, 128, ROUNDTRIP_REL_ERR));
  check(test_int8_bufferb_to_mat_parallel(1024, 7168, ROUNDTRIP_REL_ERR));

  // BF16 BufferB constraints: n % N_STEP(32) == 0, k % K_STEP(32) == 0
  printf("\n[6] BF16 BufferB from_mat -> to_mat roundtrip (lossless)\n");
  check(test_bf16_bufferb_roundtrip(32, 32, 1.0f));
  check(test_bf16_bufferb_roundtrip(64, 128, 1.0f));
  check(test_bf16_bufferb_roundtrip(256, 7168, 1.0f));
  check(test_bf16_bufferb_roundtrip(1024, 7168, 1.0f));
  check(test_bf16_bufferb_roundtrip(7168, 1024, 1.0f));

  printf("\n[7] BF16 full backward repack path (lossless)\n");
  check(test_bf16_full_repack_path(32, 32, 1.0f));
  check(test_bf16_full_repack_path(64, 128, 1.0f));
  check(test_bf16_full_repack_path(256, 7168, 1.0f));
  check(test_bf16_full_repack_path(1024, 7168, 1.0f));
  check(test_bf16_full_repack_path(7168, 1024, 1.0f));

  printf("\n[8] BF16 zero matrix edge case\n");
  check(test_bf16_bufferb_zero_matrix(32, 32));
  check(test_bf16_bufferb_zero_matrix(64, 128));
  check(test_bf16_bufferb_zero_matrix(1024, 7168));

  // INT4 quantization: 4-bit signed [-8,7], scale=amax/112
  // Single roundtrip: ~14% relative error. Double quant (repack): ~20%.
  constexpr double INT4_ROUNDTRIP_REL_ERR = 0.20;
  constexpr double INT4_REPACK_REL_ERR = 0.30;

  // INT4 BufferB constraints: n % N_STEP(32) == 0, k % B_K_STEP(128) == 0
  printf("\n[9] INT4 BufferB from_mat -> to_mat roundtrip\n");
  check(test_int4_bufferb_roundtrip(32, 128, 1.0f, INT4_ROUNDTRIP_REL_ERR));
  check(test_int4_bufferb_roundtrip(128, 128, 1.0f, INT4_ROUNDTRIP_REL_ERR));
  check(test_int4_bufferb_roundtrip(128, 3584, 1.0f, INT4_ROUNDTRIP_REL_ERR));
  check(test_int4_bufferb_roundtrip(1024, 7168, 1.0f, INT4_ROUNDTRIP_REL_ERR));
  check(test_int4_bufferb_roundtrip(7168, 1024, 1.0f, INT4_ROUNDTRIP_REL_ERR));

  // Full repack: backward [k, n] needs k % 32 == 0 AND n % 128 == 0
  // So both n and k must be multiples of 128
  printf("\n[10] INT4 full backward repack path\n");
  check(test_int4_full_repack_path(128, 128, 1.0f, INT4_REPACK_REL_ERR));
  check(test_int4_full_repack_path(128, 3584, 1.0f, INT4_REPACK_REL_ERR));
  check(test_int4_full_repack_path(1024, 7168, 1.0f, INT4_REPACK_REL_ERR));
  check(test_int4_full_repack_path(7168, 1024, 1.0f, INT4_REPACK_REL_ERR));

  printf("\n[11] INT4 zero matrix edge case\n");
  check(test_int4_bufferb_zero_matrix(32, 128));
  check(test_int4_bufferb_zero_matrix(128, 128));
  check(test_int4_bufferb_zero_matrix(1024, 7168));

  // from_bb_transposed tests (TDD — direct BB→BB transposed repack)
  // INT8 from_bb_transposed tolerance: path A goes through BF16 intermediate (to_mat),
  // path B goes through float intermediate. Allow ~5% relative error for the difference.
  constexpr double BB_TRANS_INT8_REL_ERR = 0.05;
  // End-to-end tolerance: double quantization (fwd quant + bwd quant) ~5%
  constexpr double BB_TRANS_INT8_E2E_REL_ERR = 0.05;

  // BF16 from_bb_transposed: n % 32 == 0, k % 32 == 0
  printf("\n[12] BF16 from_bb_transposed (bit-exact vs ground truth)\n");
  check(test_bf16_from_bb_transposed(32, 32, 1.0f));
  check(test_bf16_from_bb_transposed(64, 128, 1.0f));
  check(test_bf16_from_bb_transposed(256, 7168, 1.0f));
  check(test_bf16_from_bb_transposed(1024, 7168, 1.0f));
  check(test_bf16_from_bb_transposed(7168, 1024, 1.0f));

  printf("\n[13] BF16 from_bb_transposed zero matrix\n");
  check(test_bf16_from_bb_transposed_zero(32, 32));
  check(test_bf16_from_bb_transposed_zero(64, 128));
  check(test_bf16_from_bb_transposed_zero(1024, 7168));

  // INT8 from_bb_transposed: forward n % 32 == 0, k % 64 == 0
  // backward (k, n): k % 32 == 0 (auto), n % 64 == 0 → need forward n % 64 == 0
  printf("\n[14] INT8 from_bb_transposed (vs ground truth path)\n");
  check(test_int8_from_bb_transposed(64, 64, 1.0f, BB_TRANS_INT8_REL_ERR));
  check(test_int8_from_bb_transposed(64, 128, 1.0f, BB_TRANS_INT8_REL_ERR));
  check(test_int8_from_bb_transposed(128, 3584, 1.0f, BB_TRANS_INT8_REL_ERR));
  check(test_int8_from_bb_transposed(1024, 7168, 1.0f, BB_TRANS_INT8_REL_ERR));
  check(test_int8_from_bb_transposed(7168, 1024, 1.0f, BB_TRANS_INT8_REL_ERR));

  printf("\n[15] INT8 from_bb_transposed zero matrix\n");
  check(test_int8_from_bb_transposed_zero(64, 64));
  check(test_int8_from_bb_transposed_zero(64, 128));
  check(test_int8_from_bb_transposed_zero(1024, 7168));

  printf("\n[16] INT8 from_bb_transposed vs original BF16 (end-to-end quality)\n");
  check(test_int8_from_bb_transposed_vs_original(64, 64, 1.0f, BB_TRANS_INT8_E2E_REL_ERR));
  check(test_int8_from_bb_transposed_vs_original(64, 128, 1.0f, BB_TRANS_INT8_E2E_REL_ERR));
  check(test_int8_from_bb_transposed_vs_original(1024, 7168, 1.0f, BB_TRANS_INT8_E2E_REL_ERR));
  check(test_int8_from_bb_transposed_vs_original(7168, 1024, 1.0f, BB_TRANS_INT8_E2E_REL_ERR));

  printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);

  if (run_bench) {
    printf("\n=== Performance Benchmarks (single-thread, sequential) ===\n\n");

    constexpr int WARMUP = 3;
    constexpr int ITERS = 10;

    // DeepSeek R1 dims: hidden=7168, moe_intermediate=2048, 128 experts
    // TP=2: intermediate/2=1024
    printf("[A] to_mat latency (single BufferB dequant)\n");
    bench_to_mat(1024, 7168, WARMUP, ITERS);   // gate/up forward [inter/tp, hidden]
    bench_to_mat(7168, 1024, WARMUP, ITERS);   // down forward [hidden, inter/tp]
    bench_to_mat(2048, 7168, WARMUP, ITERS);   // gate/up forward TP=1
    bench_to_mat(7168, 2048, WARMUP, ITERS);   // down forward TP=1

    printf("\n[B] Full single-expert repack (to_mat + from_mat_transposed)\n");
    bench_full_repack(1024, 7168, WARMUP, ITERS);
    bench_full_repack(7168, 1024, WARMUP, ITERS);
    bench_full_repack(2048, 7168, WARMUP, ITERS);
    bench_full_repack(7168, 2048, WARMUP, ITERS);

    printf("\n[C] Full layer repack (128 experts × 3 projections, single-thread)\n");
    // TP=2: each TP partition handles all 128 experts with half the intermediate
    bench_layer_repack(7168, 1024, 128, 1, 3);  // TP=2
    bench_layer_repack(7168, 2048, 128, 1, 3);  // TP=1

    printf("\n=== BF16 Performance Benchmarks (single-thread) ===\n\n");

    printf("[D] BF16 to_mat latency (single BufferB)\n");
    bench_bf16_to_mat(1024, 7168, WARMUP, ITERS);
    bench_bf16_to_mat(7168, 1024, WARMUP, ITERS);
    bench_bf16_to_mat(2048, 7168, WARMUP, ITERS);
    bench_bf16_to_mat(7168, 2048, WARMUP, ITERS);

    printf("\n[E] BF16 full single-expert repack (to_mat + from_mat_transposed)\n");
    bench_bf16_full_repack(1024, 7168, WARMUP, ITERS);
    bench_bf16_full_repack(7168, 1024, WARMUP, ITERS);
    bench_bf16_full_repack(2048, 7168, WARMUP, ITERS);
    bench_bf16_full_repack(7168, 2048, WARMUP, ITERS);

    printf("\n=== INT4 Performance Benchmarks (single-thread) ===\n\n");

    printf("[F] INT4 to_mat latency (single BufferB)\n");
    bench_int4_to_mat(1024, 7168, WARMUP, ITERS);
    bench_int4_to_mat(7168, 1024, WARMUP, ITERS);
    bench_int4_to_mat(2048, 7168, WARMUP, ITERS);
    bench_int4_to_mat(7168, 2048, WARMUP, ITERS);

    printf("\n[G] INT4 full single-expert repack (to_mat + from_mat_transposed)\n");
    bench_int4_full_repack(1024, 7168, WARMUP, ITERS);
    bench_int4_full_repack(7168, 1024, WARMUP, ITERS);
    bench_int4_full_repack(2048, 7168, WARMUP, ITERS);
    bench_int4_full_repack(7168, 2048, WARMUP, ITERS);

    printf("\n=== from_bb_transposed Performance Benchmarks (single-thread) ===\n\n");

    printf("[H] BF16 from_bb_transposed (direct BB→BB repack)\n");
    bench_bf16_from_bb_transposed(1024, 7168, WARMUP, ITERS);
    bench_bf16_from_bb_transposed(7168, 1024, WARMUP, ITERS);
    bench_bf16_from_bb_transposed(2048, 7168, WARMUP, ITERS);
    bench_bf16_from_bb_transposed(7168, 2048, WARMUP, ITERS);

    printf("\n[I] INT8 from_bb_transposed (direct BB→BB repack)\n");
    bench_int8_from_bb_transposed(1024, 7168, WARMUP, ITERS);
    bench_int8_from_bb_transposed(7168, 1024, WARMUP, ITERS);
    bench_int8_from_bb_transposed(2048, 7168, WARMUP, ITERS);
    bench_int8_from_bb_transposed(7168, 2048, WARMUP, ITERS);

    printf("\n=== Multithreaded from_bb_transposed vs old path ===\n");
    for (int nth : {1, 2, 4, 8, 16}) {
      printf("\n--- %d threads ---\n", nth);
      bench_from_bb_transposed_mt<BF16Kernel, BF16BufferB>("bf16", 1024, 7168, nth, 2, 5);
      bench_old_repack_mt<BF16Kernel, BF16BufferB>("bf16", 1024, 7168, nth, 2, 5);
      bench_from_bb_transposed_mt<Int8Kernel, Int8BufferB>("int8", 1024, 7168, nth, 2, 5);
      bench_old_repack_mt<Int8Kernel, Int8BufferB>("int8", 1024, 7168, nth, 2, 5);
      bench_from_bb_transposed_mt<Int8Kernel, Int8BufferB>("int8", 7168, 1024, nth, 2, 5);
      bench_old_repack_mt<Int8Kernel, Int8BufferB>("int8", 7168, 1024, nth, 2, 5);
    }
  }

  return fail_count > 0 ? 1 : 0;
}
