/**
 * @Description  : Benchmark for NVFP4 kernel performance
 * @Author       : Claude & KVCache.AI Team
 * @Date         : 2025-01-04
 **/

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../la/nvfp4_kernel.hpp"
#include "../la/nvfp4_utils.hpp"

using namespace std::chrono;

// Benchmark configuration
struct BenchConfig {
  int m = 1;
  int n = 4096;
  int k = 4096;
  int warmup_iters = 10;
  int bench_iters = 100;
};

// Benchmark the LUT multiplication kernel (64 pairs)
void bench_lut_mul_64pairs(int iters) {
  std::cout << "\n=== Benchmark: nvfp4_mul_64pairs_avx512 ===" << std::endl;

  alignas(64) uint8_t a_vals[64];
  alignas(64) uint8_t b_vals[64];
  alignas(64) int16_t results[64];

  // Initialize with random values
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint8_t> dist(0, 15);
  for (int i = 0; i < 64; i++) {
    a_vals[i] = dist(gen);
    b_vals[i] = dist(gen);
  }

  __m512i a_vec = _mm512_load_si512((const __m512i*)a_vals);
  __m512i b_vec = _mm512_load_si512((const __m512i*)b_vals);

  // Warmup
  for (int i = 0; i < 100; i++) {
    nvfp4::nvfp4_mul_64pairs_avx512(a_vec, b_vec, results);
  }

  // Benchmark
  auto start = high_resolution_clock::now();
  for (int i = 0; i < iters; i++) {
    nvfp4::nvfp4_mul_64pairs_avx512(a_vec, b_vec, results);
  }
  auto end = high_resolution_clock::now();

  double elapsed_ns = duration_cast<nanoseconds>(end - start).count();
  double ns_per_call = elapsed_ns / iters;
  double calls_per_sec = 1e9 / ns_per_call;
  double pairs_per_sec = calls_per_sec * 64;

  printf("  Iterations: %d\n", iters);
  printf("  Time per call: %.2f ns\n", ns_per_call);
  printf("  Throughput: %.2f M pairs/sec\n", pairs_per_sec / 1e6);
  printf("  Throughput: %.2f M mul-ops/sec\n", pairs_per_sec / 1e6);
}

// Benchmark full matrix multiplication (with version selection)
void bench_matmul(const BenchConfig& cfg, int version = 0) {
  const char* version_names[] = {"nvfp4_matmul (baseline)", "nvfp4_matmul_opt", "nvfp4_matmul_opt2 (N-batch)",
                                 "nvfp4_matmul_opt3 (aggressive)", "nvfp4_matmul_opt4 (fused)"};
  std::cout << "\n=== Benchmark: " << version_names[version] << " ===" << std::endl;
  printf("  Matrix sizes: M=%d, N=%d, K=%d\n", cfg.m, cfg.n, cfg.k);

  // Allocate buffers
  size_t ba_size = nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(cfg.m, cfg.k);
  size_t bb_size = nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(cfg.n, cfg.k);
  size_t bc_size = nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(cfg.m, cfg.n);

  void* ba_buffer = std::aligned_alloc(64, ba_size);
  void* bb_buffer = std::aligned_alloc(64, bb_size);
  void* bc_buffer = std::aligned_alloc(64, bc_size);

  auto ba = std::make_shared<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>(cfg.m, cfg.k, ba_buffer);
  auto bb = std::make_shared<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>(cfg.n, cfg.k, bb_buffer);
  auto bc = std::make_shared<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>(cfg.m, cfg.n, bc_buffer);

  // Initialize with random data
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // Create activation (BF16)
  std::vector<ggml_bf16_t> a_bf16(cfg.m * cfg.k);
  for (int i = 0; i < cfg.m * cfg.k; i++) {
    a_bf16[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }

  // Create weights (quantized)
  std::vector<float> b_f32(cfg.n * cfg.k);
  for (int i = 0; i < cfg.n * cfg.k; i++) {
    b_f32[i] = dist(gen);
  }

  // Quantize weights
  float b_global_max = 0.0f;
  for (int i = 0; i < cfg.n * cfg.k; i++) {
    b_global_max = std::max(b_global_max, std::abs(b_f32[i]));
  }
  float b_tensor_scale = b_global_max / 448.0f;
  float b_ts_inv = (b_tensor_scale > 1e-10f) ? (1.0f / b_tensor_scale) : 1.0f;

  const int k_group_size = 16;
  int k_group_count = cfg.k / k_group_size;
  std::vector<uint8_t> b_fp4(cfg.n * cfg.k / 2);
  std::vector<uint8_t> b_scales_fp8(cfg.n * k_group_count);

  for (int n_i = 0; n_i < cfg.n; n_i++) {
    for (int kg = 0; kg < k_group_count; kg++) {
      int k_start = kg * k_group_size;
      float max_val = 0.0f;
      for (int k_i = 0; k_i < k_group_size; k_i++) {
        float val = b_f32[n_i * cfg.k + k_start + k_i] * b_ts_inv;
        max_val = std::max(max_val, std::abs(val));
      }
      float block_scale = max_val / 6.0f;
      b_scales_fp8[n_i * k_group_count + kg] = nvfp4::float_to_fp8_e4m3(block_scale);
      float scale_inv = (block_scale > 1e-10f) ? (1.0f / block_scale) : 0.0f;

      for (int k_i = 0; k_i < k_group_size; k_i += 2) {
        float val0 = b_f32[n_i * cfg.k + k_start + k_i] * b_ts_inv * scale_inv;
        float val1 = b_f32[n_i * cfg.k + k_start + k_i + 1] * b_ts_inv * scale_inv;
        uint8_t q0 = nvfp4::float_to_e2m1(val0);
        uint8_t q1 = nvfp4::float_to_e2m1(val1);
        b_fp4[n_i * cfg.k / 2 + (k_start + k_i) / 2] = q0 | (q1 << 4);
      }
    }
  }

  // Load buffers
  ba->from_bf16(cfg.m, a_bf16.data(), 0, 1);
  bb->from_raw_nvfp4(b_fp4.data(), b_scales_fp8.data(), b_tensor_scale, 0, 1);

  // Select kernel version
  auto run_kernel = [&]() {
    bc->clear(cfg.m);
    switch (version) {
      case 0:
        nvfp4::nvfp4_matmul(cfg.m, cfg.n, cfg.k, ba, bb, bc, 0, 1);
        break;
      case 1:
        nvfp4::nvfp4_matmul_opt(cfg.m, cfg.n, cfg.k, ba, bb, bc, 0, 1);
        break;
      case 2:
        nvfp4::nvfp4_matmul_opt2(cfg.m, cfg.n, cfg.k, ba, bb, bc, 0, 1);
        break;
      case 3:
        nvfp4::nvfp4_matmul_opt3(cfg.m, cfg.n, cfg.k, ba, bb, bc, 0, 1);
        break;
      case 4:
        nvfp4::nvfp4_matmul_opt4(cfg.m, cfg.n, cfg.k, ba, bb, bc, 0, 1);
        break;
    }
  };

  // Warmup
  printf("  Warmup: %d iterations\n", cfg.warmup_iters);
  for (int i = 0; i < cfg.warmup_iters; i++) {
    run_kernel();
  }

  // Benchmark
  printf("  Benchmark: %d iterations\n", cfg.bench_iters);
  auto start = high_resolution_clock::now();
  for (int i = 0; i < cfg.bench_iters; i++) {
    run_kernel();
  }
  auto end = high_resolution_clock::now();

  double elapsed_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
  double ms_per_iter = elapsed_ms / cfg.bench_iters;

  // Calculate FLOPS: 2 * M * N * K (multiply-add = 2 ops)
  double flops = 2.0 * cfg.m * cfg.n * cfg.k;
  double gflops = (flops / ms_per_iter) / 1e6;  // GFLOPS

  // Memory bandwidth (rough estimate)
  // Read: A (m*k/2 bytes) + B (n*k/2 bytes) + scales
  // Write: C (m*n*4 bytes)
  double read_bytes = (cfg.m * cfg.k / 2.0) + (cfg.n * cfg.k / 2.0) + (cfg.m * cfg.k / 16.0) + (cfg.n * cfg.k / 16.0);
  double write_bytes = cfg.m * cfg.n * 4.0;
  double total_bytes = read_bytes + write_bytes;
  double bandwidth_gbps = (total_bytes / ms_per_iter) / 1e6;

  printf("\n  Results:\n");
  printf("  ─────────────────────────────────\n");
  printf("  Total time:     %.3f ms\n", elapsed_ms);
  printf("  Time per iter:  %.3f ms\n", ms_per_iter);
  printf("  Throughput:     %.2f GFLOPS\n", gflops);
  printf("  Memory BW:      %.2f GB/s (estimated)\n", bandwidth_gbps);
  printf("  Arithmetic Int: %.2f FLOP/byte\n", flops / total_bytes);

  std::free(ba_buffer);
  std::free(bb_buffer);
  std::free(bc_buffer);
}

// Benchmark different matrix sizes
void bench_sweep() {
  std::cout << "\n=== Matrix Size Sweep ===" << std::endl;
  printf("%-8s %-8s %-8s %-12s %-12s\n", "M", "N", "K", "Time(ms)", "GFLOPS");
  printf("──────────────────────────────────────────────────\n");

  std::vector<std::tuple<int, int, int>> sizes = {
      {1, 1024, 1024}, {1, 2048, 2048},  {1, 4096, 4096},  {1, 4096, 14336}, {1, 14336, 4096}, {4, 4096, 4096},
      {8, 4096, 4096}, {16, 4096, 4096}, {32, 4096, 4096}, {1, 8192, 8192},  {1, 8192, 28672},
  };

  for (auto& [m, n, k] : sizes) {
    BenchConfig cfg;
    cfg.m = m;
    cfg.n = n;
    cfg.k = k;
    cfg.warmup_iters = 5;
    cfg.bench_iters = 20;

    // Quick inline benchmark
    size_t ba_size = nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(m, k);
    size_t bb_size = nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(n, k);
    size_t bc_size = nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(m, n);

    void* ba_buffer = std::aligned_alloc(64, ba_size);
    void* bb_buffer = std::aligned_alloc(64, bb_size);
    void* bc_buffer = std::aligned_alloc(64, bc_size);

    auto ba = std::make_shared<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>(m, k, ba_buffer);
    auto bb = std::make_shared<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>(n, k, bb_buffer);
    auto bc = std::make_shared<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>(m, n, bc_buffer);

    // Initialize with zeros (simplified)
    memset(ba_buffer, 0, ba_size);
    memset(bb_buffer, 0, bb_size);
    ba->tensor_scale = 1.0f;
    bb->tensor_scale = 1.0f;

    // Warmup
    for (int i = 0; i < cfg.warmup_iters; i++) {
      bc->clear(m);
      nvfp4::nvfp4_matmul(m, n, k, ba, bb, bc, 0, 1);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < cfg.bench_iters; i++) {
      bc->clear(m);
      nvfp4::nvfp4_matmul(m, n, k, ba, bb, bc, 0, 1);
    }
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    double ms_per_iter = elapsed_ms / cfg.bench_iters;
    double flops = 2.0 * m * n * k;
    double gflops = (flops / ms_per_iter) / 1e6;

    printf("%-8d %-8d %-8d %-12.3f %-12.2f\n", m, n, k, ms_per_iter, gflops);

    std::free(ba_buffer);
    std::free(bb_buffer);
    std::free(bc_buffer);
  }
}

int main(int argc, char** argv) {
  std::cout << "NVFP4 Kernel Benchmark" << std::endl;
  std::cout << "======================" << std::endl;

  // Parse command line
  BenchConfig cfg;
  bool run_sweep = false;
  bool run_lut = false;
  bool run_compare = false;
  int version = 0;  // 0=baseline, 1=opt, 2=opt2

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--sweep") == 0) {
      run_sweep = true;
    } else if (strcmp(argv[i], "--lut") == 0) {
      run_lut = true;
    } else if (strcmp(argv[i], "--compare") == 0) {
      run_compare = true;
    } else if (strcmp(argv[i], "--opt") == 0) {
      version = 1;
    } else if (strcmp(argv[i], "--opt2") == 0) {
      version = 2;
    } else if (strcmp(argv[i], "--opt3") == 0) {
      version = 3;
    } else if (strcmp(argv[i], "--opt4") == 0) {
      version = 4;
    } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      cfg.m = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      cfg.n = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
      cfg.k = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      cfg.bench_iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0) {
      printf("Usage: %s [options]\n", argv[0]);
      printf("Options:\n");
      printf("  -m <int>      M dimension (default: 1)\n");
      printf("  -n <int>      N dimension (default: 4096)\n");
      printf("  -k <int>      K dimension (default: 4096)\n");
      printf("  --iters <int> Benchmark iterations (default: 100)\n");
      printf("  --sweep       Run matrix size sweep\n");
      printf("  --lut         Benchmark LUT multiplication only\n");
      printf("  --opt         Use optimized version 1\n");
      printf("  --opt2        Use optimized version 2 (N-batch)\n");
      printf("  --opt3        Use optimized version 3 (aggressive)\n");
      printf("  --opt4        Use optimized version 4 (fused)\n");
      printf("  --compare     Compare all versions\n");
      printf("  --help        Show this help\n");
      return 0;
    }
  }

  if (run_lut) {
    bench_lut_mul_64pairs(1000000);
  }

  if (run_compare) {
    // Compare all versions
    std::cout << "\n========== Comparing All Versions ==========" << std::endl;
    bench_matmul(cfg, 0);  // baseline
    bench_matmul(cfg, 1);  // opt
    bench_matmul(cfg, 2);  // opt2
    bench_matmul(cfg, 3);  // opt3
    bench_matmul(cfg, 4);  // opt4 (fused)
  } else if (run_sweep) {
    bench_sweep();
  } else if (!run_lut) {
    bench_matmul(cfg, version);
  }

  return 0;
}
