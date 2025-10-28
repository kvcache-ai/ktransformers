#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "../la/amx.hpp"
#include "../la/amx_buffers.hpp"
#include "../la/amx_kernels.hpp"

void test_specific_dimensions() {
  std::cout << "=== Testing Specific Dimensions ===\n" << std::endl;

  const int m_original = 200;
  const int n = 512;
  const int k = 7168;
  const int k_group_size = 64;

  // Pad m to nearest multiple of 32 (M_STEP)
  const int M_STEP = 32;
  const int m = ((m_original + M_STEP - 1) / M_STEP) * M_STEP;  // Round up to 224

  std::cout << "Original dimensions: " << m_original << " x " << n << " x " << k << std::endl;
  std::cout << "Padded dimensions: " << m << " x " << n << " x " << k << std::endl;
  std::cout << "K-group size is: " << k_group_size << std::endl;
  std::cout << "Number of k-groups: " << k / k_group_size << std::endl;

  using Kernel = amx::GemmKernel224Int4KGroup;
  using Kernel_int4_1 = amx::GemmKernel224Int4_1;
  using Kernel_int4 = amx::GemmKernel224Int4;
  using Kernel_k_int4_1 = amx::GemmKernel224Int4_1KGroup;
  using Kernel_k_int4_1_low = amx::GemmKernel224Int4_1_LowKGroup;
  using BufferA = Kernel::BufferA;
  using BufferB = Kernel::BufferB;
  using BufferC = Kernel::BufferC;
  using BufferA_int4_1 = Kernel_int4_1::BufferA;
  using BufferB_int4_1 = Kernel_int4_1::BufferB;
  using BufferC_int4_1 = Kernel_int4_1::BufferC;
  using BufferA_int4 = Kernel_int4::BufferA;
  using BufferB_int4 = Kernel_int4::BufferB;
  using BufferC_int4 = Kernel_int4::BufferC;
  using BufferA_k_int4_1 = Kernel_k_int4_1::BufferA;
  using BufferB_k_int4_1 = Kernel_k_int4_1::BufferB;
  using BufferC_k_int4_1 = Kernel_k_int4_1::BufferC;
  using BufferA_k_int4_1_low = Kernel_k_int4_1_low::BufferA;
  using BufferB_k_int4_1_low = Kernel_k_int4_1_low::BufferB;
  using BufferC_k_int4_1_low = Kernel_k_int4_1_low::BufferC;

  void* buffer_a = std::aligned_alloc(64, BufferA::required_size(m, k, k_group_size));
  void* buffer_b = std::aligned_alloc(64, BufferB::required_size(n, k, k_group_size));
  void* buffer_c = std::aligned_alloc(64, BufferC::required_size(m, n));

  void* buffer_a_int4_1 = std::aligned_alloc(64, BufferA_int4_1::required_size(m, k));
  void* buffer_b_int4_1 = std::aligned_alloc(64, BufferB_int4_1::required_size(n, k));
  void* buffer_c_int4_1 = std::aligned_alloc(64, BufferC_int4_1::required_size(m, n));

  void* buffer_a_int4 = std::aligned_alloc(64, BufferA_int4::required_size(m, k));
  void* buffer_b_int4 = std::aligned_alloc(64, BufferB_int4::required_size(n, k));
  void* buffer_c_int4 = std::aligned_alloc(64, BufferC_int4::required_size(m, n));

  void* buffer_a_k_int4_1 = std::aligned_alloc(64, BufferA_k_int4_1::required_size(m, k, k_group_size));
  void* buffer_b_k_int4_1 = std::aligned_alloc(64, BufferB_k_int4_1::required_size(n, k, k_group_size));
  void* buffer_c_k_int4_1 = std::aligned_alloc(64, BufferC_k_int4_1::required_size(m, n));

  void* buffer_a_k_int4_1_low = std::aligned_alloc(64, BufferA_k_int4_1_low::required_size(m, k, k_group_size));
  void* buffer_b_k_int4_1_low = std::aligned_alloc(64, BufferB_k_int4_1_low::required_size(n, k, k_group_size));
  void* buffer_c_k_int4_1_low = std::aligned_alloc(64, BufferC_k_int4_1_low::required_size(m, n));

  auto ba = std::make_shared<BufferA>(m, k, k_group_size, buffer_a);
  printf("buffer_b ptr:%p\n", buffer_b);
  auto bb = std::make_shared<BufferB>(n, k, k_group_size, buffer_b);
  auto bc = std::make_shared<BufferC>(m, n, buffer_c);

  auto ba_int4_1 = std::make_shared<BufferA_int4_1>(m, k, buffer_a_int4_1);
  auto bb_int4_1 = std::make_shared<BufferB_int4_1>(n, k, buffer_b_int4_1);
  auto bc_int4_1 = std::make_shared<BufferC_int4_1>(m, n, buffer_c_int4_1);

  auto ba_int4 = std::make_shared<BufferA_int4>(m, k, buffer_a_int4);
  auto bb_int4 = std::make_shared<BufferB_int4>(n, k, buffer_b_int4);
  auto bc_int4 = std::make_shared<BufferC_int4>(m, n, buffer_c_int4);

  auto ba_k_int4_1 = std::make_shared<BufferA_k_int4_1>(m, k, k_group_size, buffer_a_k_int4_1);
  auto bb_k_int4_1 = std::make_shared<BufferB_k_int4_1>(n, k, k_group_size, buffer_b_k_int4_1);
  auto bc_k_int4_1 = std::make_shared<BufferC_k_int4_1>(m, n, buffer_c_k_int4_1);

  auto ba_k_int4_1_low = std::make_shared<BufferA_k_int4_1_low>(m, k, k_group_size, buffer_a_k_int4_1_low);
  auto bb_k_int4_1_low = std::make_shared<BufferB_k_int4_1_low>(n, k, k_group_size, buffer_b_k_int4_1_low);
  auto bc_k_int4_1_low = std::make_shared<BufferC_k_int4_1_low>(m, n, buffer_c_k_int4_1_low);

  // Create input matrices with realistic values
  std::vector<ggml_bf16_t> input_a(m * k);
  std::vector<ggml_bf16_t> input_b(k * n);

  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 0.1f);  // Normal distribution, mean=0, std=0.1

  std::cout << "\nGenerating input matrices..." << std::endl;
  // print input mat(first 10)
  // for (int i = 0; i < std::min(10, m * k); i++) {
  //   std::cout << "input_a[" << i << "] = " << ggml_compute_bf16_to_fp32(input_a[i]) << std::endl;
  // }
  // for (int i = 0; i < std::min(10, k * n); i++) {
  //   std::cout << "input_b[" << i << "] = " << ggml_compute_bf16_to_fp32(input_b[i]) << std::endl;
  // }
  for (int i = 0; i < m * k; i++) {
    input_a[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }
  for (int i = 0; i < k * n; i++) {
    input_b[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }

  // Compute reference result with float32 (sampling for speed, only use original m rows)
  std::cout << "Computing reference (sampling)..." << std::endl;
  const int sample_m = std::min(50, m_original);  // Use original m for reference
  const int sample_n = std::min(50, n);
  std::vector<float> ref_result(sample_m * sample_n, 0.0f);

  for (int i = 0; i < sample_m; i++) {
    for (int j = 0; j < sample_n; j++) {
      float sum = 0.0f;
      for (int l = 0; l < k; l++) {
        float a_val = ggml_compute_bf16_to_fp32(input_a[i * k + l]);
        float b_val = ggml_compute_bf16_to_fp32(input_b[j * k + l]);
        sum += a_val * b_val;
      }
      ref_result[i * sample_n + j] = sum;
    }
  }

  // Quantize and compute with k-group
  std::cout << "Quantizing matrices..." << std::endl;
  ba->from_mat(m, input_a.data(), 0, 1);
  int nth = Kernel::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bb->from_mat(input_b.data(), i, nth);
  }

  ba_int4_1->from_mat(m, input_a.data(), 0, 1);
  nth = Kernel_int4_1::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bb_int4_1->from_mat(input_b.data(), i, nth);
  }

  ba_int4->from_mat(m, input_a.data(), 0, 1);
  nth = Kernel_int4::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bb_int4->from_mat(input_b.data(), i, nth);
  }

  ba_k_int4_1->from_mat(m, input_a.data(), 0, 1);
  nth = Kernel_k_int4_1::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bb_k_int4_1->from_mat(input_b.data(), i, nth);
  }

  ba_k_int4_1_low->from_mat(m, input_a.data(), 0, 1);
  nth = Kernel_k_int4_1_low::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bb_k_int4_1_low->from_mat(input_b.data(), i, nth);
  }

  // Print some scale statistics
  std::cout << "\nScale statistics:" << std::endl;
  float min_a_scale = 1e10f, max_a_scale = 0.0f;
  float min_b_scale = 1e10f, max_b_scale = 0.0f;
  float min_a_scale_int4_1 = 1e10f, max_a_scale_int4_1 = 0.0f;
  float min_b_scale_int4_1 = 1e10f, max_b_scale_int4_1 = 0.0f;
  float min_b_min_int4_1 = 1e10f, max_b_min_int4_1 = -1e10f;
  float min_a_scale_int4 = 1e10f, max_a_scale_int4 = 0.0f;
  float min_b_scale_int4 = 1e10f, max_b_scale_int4 = 0.0f;
  float min_a_scale_k_int4_1 = 1e10f, max_a_scale_k_int4_1 = 0.0f;
  float min_b_scale_k_int4_1 = 1e10f, max_b_scale_k_int4_1 = 0.0f;
  float min_b_min_k_int4_1 = 1e10f, max_b_min_k_int4_1 = -1e10f;
  float min_a_scale_k_int4_1_low = 1e10f, max_a_scale_k_int4_1_low = 0.0f;
  float min_b_scale_k_int4_1_low = 1e10f, max_b_scale_k_int4_1_low = 0.0f;
  float min_b_min_k_int4_1_low = 1e10f, max_b_min_k_int4_1_low = -1e10f;

  for (int i = 0; i < std::min(10, m); i++) {
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *ba->get_scale(m, i, k, kg * k_group_size);
      min_a_scale = std::min(min_a_scale, scale);
      max_a_scale = std::max(max_a_scale, scale);
    }
  }

  for (int j = 0; j < std::min(10, n); j++) {
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *bb->get_scale(n, j, k, kg * k_group_size);
      min_b_scale = std::min(min_b_scale, scale);
      max_b_scale = std::max(max_b_scale, scale);
    }
  }
  for (int i = 0; i < std::min(10, m); i++) {
    float scale = *ba_int4_1->get_scale(m, i);
    min_a_scale_int4_1 = std::min(min_a_scale_int4_1, scale);
    max_a_scale_int4_1 = std::max(max_a_scale_int4_1, scale);
  }
  for (int j = 0; j < std::min(10, n); j++) {
    float scale = *bb_int4_1->get_scale(n, j);
    min_b_scale_int4_1 = std::min(min_b_scale_int4_1, scale);
    max_b_scale_int4_1 = std::max(max_b_scale_int4_1, scale);
    float b_min = *bb_int4_1->get_min(n, j);
    min_b_min_int4_1 = std::min(min_b_min_int4_1, b_min);
    max_b_min_int4_1 = std::max(max_b_min_int4_1, b_min);
  }

  for (int i = 0; i < std::min(10, m); i++) {
    float scale = *ba_int4->get_scale(m, i);
    min_a_scale_int4 = std::min(min_a_scale_int4, scale);
    max_a_scale_int4 = std::max(max_a_scale_int4, scale);
  }

  for (int j = 0; j < std::min(10, n); j++) {
    float scale = *bb_int4->get_scale(n, j);
    min_b_scale_int4 = std::min(min_b_scale_int4, scale);
    max_b_scale_int4 = std::max(max_b_scale_int4, scale);
  }

  for (int i = 0; i < std::min(10, m); i++) {
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *ba_k_int4_1->get_scale(m, i, k, kg * k_group_size);
      min_a_scale_k_int4_1 = std::min(min_a_scale_k_int4_1, scale);
      max_a_scale_k_int4_1 = std::max(max_a_scale_k_int4_1, scale);
    }
  }

  for (int j = 0; j < std::min(10, n); j++) {
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *bb_k_int4_1->get_scale(n, j, k, kg * k_group_size);
      min_b_scale_k_int4_1 = std::min(min_b_scale_k_int4_1, scale);
      max_b_scale_k_int4_1 = std::max(max_b_scale_k_int4_1, scale);
      float b_min = *bb_k_int4_1->get_min(n, j, k, kg * k_group_size);
      min_b_min_k_int4_1 = std::min(min_b_min_k_int4_1, b_min);
      max_b_min_k_int4_1 = std::max(max_b_min_k_int4_1, b_min);
    }
  }

  for (int i = 0; i < std::min(10, m); i++) {
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *ba_k_int4_1_low->get_scale(m, i, k, kg * k_group_size);
      min_a_scale_k_int4_1_low = std::min(min_a_scale_k_int4_1_low, scale);
      max_a_scale_k_int4_1_low = std::max(max_a_scale_k_int4_1_low, scale);
    }
  }

  for (int j = 0; j < std::min(10, n); j++) {
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *bb_k_int4_1_low->get_scale(n, j, k, kg * k_group_size);
      min_b_scale_k_int4_1_low = std::min(min_b_scale_k_int4_1_low, scale);
      max_b_scale_k_int4_1_low = std::max(max_b_scale_k_int4_1_low, scale);
      float b_min = *bb_k_int4_1_low->get_min(n, j, k, kg * k_group_size);
      min_b_min_k_int4_1_low = std::min(min_b_min_k_int4_1_low, b_min);
      max_b_min_k_int4_1_low = std::max(max_b_min_k_int4_1_low, b_min);
    }
  }
  std::cout << "  B_int4_1 scales: min=" << min_b_scale_int4_1 << ", max=" << max_b_scale_int4_1 << std::endl;
  std::cout << "  B_int4_1 min: min=" << min_b_min_int4_1 << ", max=" << max_b_min_int4_1 << std::endl;

  std::cout << "  A_int4 scales: min=" << min_a_scale_int4 << ", max=" << max_a_scale_int4 << std::endl;
  std::cout << "  B_int4 scales: min=" << min_b_scale_int4 << ", max=" << max_b_scale_int4 << std::endl;

  std::cout << "  A_k_int4_1 scales: min=" << min_a_scale_k_int4_1 << ", max=" << max_a_scale_k_int4_1 << std::endl;
  std::cout << "  B_k_int4_1 scales: min=" << min_b_scale_k_int4_1 << ", max=" << max_b_scale_k_int4_1 << std::endl;
  std::cout << "  B_k_int4_1 min: min=" << min_b_min_k_int4_1 << ", max=" << max_b_min_k_int4_1 << std::endl;

  std::cout << "  A_k_int4_1_low scales: min=" << min_a_scale_k_int4_1_low << ", max=" << max_a_scale_k_int4_1_low
            << std::endl;
  std::cout << "  B_k_int4_1_low scales: min=" << min_b_scale_k_int4_1_low << ", max=" << max_b_scale_k_int4_1_low
            << std::endl;
  std::cout << "  B_k_int4_1_low min: min=" << min_b_min_k_int4_1_low << ", max=" << max_b_min_k_int4_1_low
            << std::endl;

  Kernel::config();

  std::cout << "\nRunning k-group matrix multiplication..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  nth = Kernel::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, i, nth);
  }

  nth = Kernel_int4_1::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    amx::mat_mul(m, n, k, ba_int4_1, bb_int4_1, bc_int4_1, i, nth);
  }

  nth = Kernel_int4::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    amx::mat_mul(m, n, k, ba_int4, bb_int4, bc_int4, i, nth);
  }

  nth = Kernel_k_int4_1::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    amx::vec_mul_kgroup(m, n, k, k_group_size, ba_k_int4_1, bb_k_int4_1, bc_k_int4_1, i, nth);
  }

  nth = Kernel_k_int4_1_low::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    amx::vec_mul_kgroup(m, n, k, k_group_size, ba_k_int4_1_low, bb_k_int4_1_low, bc_k_int4_1_low, i, nth);
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Computation time: " << duration.count() / 1000.0 << " ms" << std::endl;

  // Calculate GFLOPS
  double ops = 2.0 * m * n * k;
  double gflops = ops / (duration.count() * 1000.0);
  std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

  std::vector<ggml_bf16_t> output(m * n);
  std::vector<ggml_bf16_t> output_int4_1(m * n);
  std::vector<ggml_bf16_t> output_int4(m * n);
  std::vector<ggml_bf16_t> output_k_int4_1(m * n);
  std::vector<ggml_bf16_t> output_k_int4_1_low(m * n);
  nth = Kernel::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bc->to_mat(m, output.data(), i, nth);
  }
  nth = Kernel_int4_1::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bc_int4_1->to_mat(m, output_int4_1.data(), i, nth);
  }
  nth = Kernel_int4::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bc_int4->to_mat(m, output_int4.data(), i, nth);
  }
  nth = Kernel_k_int4_1::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bc_k_int4_1->to_mat(m, output_k_int4_1.data(), i, nth);
  }
  nth = Kernel_k_int4_1_low::recommended_nth(n);
  for (int i = 0; i <= nth; i++) {
    bc_k_int4_1_low->to_mat(m, output_k_int4_1_low.data(), i, nth);
  }
  float thresh_hold = 2.0f;
  // Compute errors for sampled elements
  std::cout << "\nError analysis (sampled):" << std::endl;
  float max_abs_error = 0.0f;
  float total_abs_error = 0.0f;
  float max_rel_error = 0.0f;
  float total_rel_error = 0.0f;
  int count = 0;

  for (int i = 0; i < sample_m; i++) {
    for (int j = 0; j < sample_n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float abs_error = std::abs(actual - ref);
      float rel_error = std::abs(ref) > 1e-6 ? abs_error / std::abs(ref) : 0.0f;
      if (rel_error >= thresh_hold) {
        rel_error = thresh_hold;
      }
      max_abs_error = std::max(max_abs_error, abs_error);
      total_abs_error += abs_error;
      max_rel_error = std::max(max_rel_error, rel_error);
      total_rel_error += rel_error;
      count++;
    }
  }

  float avg_abs_error = total_abs_error / count;
  float avg_rel_error = total_rel_error / count;

  std::cout << "  Max absolute error: " << max_abs_error << std::endl;
  std::cout << "  Average absolute error: " << avg_abs_error << std::endl;
  std::cout << "  Max relative error: " << (max_rel_error * 100) << "%" << std::endl;
  std::cout << "  Average relative error: " << (avg_rel_error * 100) << "%" << std::endl;

  float max_abs_error_int4_1 = 0.0f;
  float total_abs_error_int4_1 = 0.0f;
  float max_rel_error_int4_1 = 0.0f;
  float total_rel_error_int4_1 = 0.0f;
  int count_int4_1 = 0;

  for (int i = 0; i < sample_m; i++) {
    for (int j = 0; j < sample_n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output_int4_1[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float abs_error = std::abs(actual - ref);
      float rel_error = std::abs(ref) > 1e-6 ? abs_error / std::abs(ref) : 0.0f;
      if (rel_error >= thresh_hold) {
        rel_error = thresh_hold;
      }

      max_abs_error_int4_1 = std::max(max_abs_error_int4_1, abs_error);
      total_abs_error_int4_1 += abs_error;
      max_rel_error_int4_1 = std::max(max_rel_error_int4_1, rel_error);
      total_rel_error_int4_1 += rel_error;
      count_int4_1++;
    }
  }

  float avg_abs_error_int4_1 = total_abs_error_int4_1 / count_int4_1;
  float avg_rel_error_int4_1 = total_rel_error_int4_1 / count_int4_1;
  std::cout << "\nINT4_1 Error analysis (sampled):" << std::endl;
  std::cout << "  Max absolute error: " << max_abs_error_int4_1 << std::endl;
  std::cout << "  Average absolute error: " << avg_abs_error_int4_1 << std::endl;
  std::cout << "  Max relative error: " << (max_rel_error_int4_1 * 100) << "%" << std::endl;
  std::cout << "  Average relative error: " << (avg_rel_error_int4_1 * 100) << "%" << std::endl;

  float max_abs_error_int4 = 0.0f;
  float total_abs_error_int4 = 0.0f;
  float max_rel_error_int4 = 0.0f;
  float total_rel_error_int4 = 0.0f;
  int count_int4 = 0;

  for (int i = 0; i < sample_m; i++) {
    for (int j = 0; j < sample_n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output_int4[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float abs_error = std::abs(actual - ref);
      float rel_error = std::abs(ref) > 1e-6 ? abs_error / std::abs(ref) : 0.0f;
      if (rel_error >= thresh_hold) {
        rel_error = thresh_hold;
      }

      max_abs_error_int4 = std::max(max_abs_error_int4, abs_error);
      total_abs_error_int4 += abs_error;
      max_rel_error_int4 = std::max(max_rel_error_int4, rel_error);
      total_rel_error_int4 += rel_error;
      count_int4++;
    }
  }

  float avg_abs_error_int4 = total_abs_error_int4 / count_int4;
  float avg_rel_error_int4 = total_rel_error_int4 / count_int4;
  std::cout << "\nINT4 Error analysis (sampled):" << std::endl;
  std::cout << "  Max absolute error: " << max_abs_error_int4 << std::endl;
  std::cout << "  Average absolute error: " << avg_abs_error_int4 << std::endl;
  std::cout << "  Max relative error: " << (max_rel_error_int4 * 100) << "%" << std::endl;
  std::cout << "  Average relative error: " << (avg_rel_error_int4 * 100) << "%" << std::endl;

  float max_abs_error_k_int4_1 = 0.0f;
  float total_abs_error_k_int4_1 = 0.0f;
  float max_rel_error_k_int4_1 = 0.0f;
  float total_rel_error_k_int4_1 = 0.0f;
  int count_k_int4_1 = 0;

  for (int i = 0; i < sample_m; i++) {
    for (int j = 0; j < sample_n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output_k_int4_1[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float abs_error = std::abs(actual - ref);
      float rel_error = std::abs(ref) > 1e-6 ? abs_error / std::abs(ref) : 0.0f;
      if (rel_error >= thresh_hold) {
        rel_error = thresh_hold;
      }

      max_abs_error_k_int4_1 = std::max(max_abs_error_k_int4_1, abs_error);
      total_abs_error_k_int4_1 += abs_error;
      max_rel_error_k_int4_1 = std::max(max_rel_error_k_int4_1, rel_error);
      total_rel_error_k_int4_1 += rel_error;
      count_k_int4_1++;
    }
  }
  float avg_abs_error_k_int4_1 = total_abs_error_k_int4_1 / count_k_int4_1;
  float avg_rel_error_k_int4_1 = total_rel_error_k_int4_1 / count_k_int4_1;
  std::cout << "\nINT4_1_k Error analysis (sampled):" << std::endl;
  std::cout << "  Max absolute error: " << max_abs_error_k_int4_1 << std::endl;
  std::cout << "  Average absolute error: " << avg_abs_error_k_int4_1 << std::endl;
  std::cout << "  Max relative error: " << (max_rel_error_k_int4_1 * 100) << "%" << std::endl;
  std::cout << "  Average relative error: " << (avg_rel_error_k_int4_1 * 100) << "%" << std::endl;

  float max_abs_error_k_int4_1_low = 0.0f;
  float total_abs_error_k_int4_1_low = 0.0f;
  float max_rel_error_k_int4_1_low = 0.0f;
  float total_rel_error_k_int4_1_low = 0.0f;
  int count_k_int4_1_low = 0;

  for (int i = 0; i < sample_m; i++) {
    for (int j = 0; j < sample_n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output_k_int4_1_low[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float abs_error = std::abs(actual - ref);
      float rel_error = std::abs(ref) > 1e-6 ? abs_error / std::abs(ref) : 0.0f;
      if (rel_error >= thresh_hold) {
        rel_error = thresh_hold;
      }

      max_abs_error_k_int4_1_low = std::max(max_abs_error_k_int4_1_low, abs_error);
      total_abs_error_k_int4_1_low += abs_error;
      max_rel_error_k_int4_1_low = std::max(max_rel_error_k_int4_1_low, rel_error);
      total_rel_error_k_int4_1_low += rel_error;
      count_k_int4_1_low++;
    }
  }

  float avg_abs_error_k_int4_1_low = total_abs_error_k_int4_1_low / count_k_int4_1_low;
  float avg_rel_error_k_int4_1_low = total_rel_error_k_int4_1_low / count_k_int4_1_low;
  std::cout << "\nINT4_1_k_low Error analysis (sampled):" << std::endl;
  std::cout << "  Max absolute error: " << max_abs_error_k_int4_1_low << std::endl;
  std::cout << "  Average absolute error: " << avg_abs_error_k_int4_1_low << std::endl;
  std::cout << "  Max relative error: " << (max_rel_error_k_int4_1_low * 100) << "%" << std::endl;
  std::cout << "  Average relative error: " << (avg_rel_error_k_int4_1_low * 100) << "%" << std::endl;

  // Print sample comparison
  std::cout << "\nSample comparison (first 10x10):" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 10; i < std::min(20, sample_m); i++) {
    for (int j = 10; j < std::min(20, sample_n); j++) {
      float actual = ggml_compute_bf16_to_fp32(output[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("%7.4f (%7.4f) [%+6.1f%%]  ", actual, ref, error_pct);
    }
    std::cout << std::endl;
  }
  std::cout << "\nint4_1 Sample comparison (first 10x10):" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 10; i < std::min(20, sample_m); i++) {
    for (int j = 10; j < std::min(20, sample_n); j++) {
      float actual = ggml_compute_bf16_to_fp32(output_int4_1[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("%7.4f (%7.4f) [%+6.1f%%]  ", actual, ref, error_pct);
    }
    std::cout << std::endl;
  }
  std::cout << "\nint4 Sample comparison (first 10x10):" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 10; i < std::min(20, sample_m); i++) {
    for (int j = 10; j < std::min(20, sample_n); j++) {
      float actual = ggml_compute_bf16_to_fp32(output_int4[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("%7.4f (%7.4f) [%+6.1f%%]  ", actual, ref, error_pct);
    }
    std::cout << std::endl;
  }

  std::cout << "\nint4_1_k Sample comparison (first 10x10):" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 10; i < std::min(20, sample_m); i++) {
    for (int j = 10; j < std::min(20, sample_n); j++) {
      float actual = ggml_compute_bf16_to_fp32(output_k_int4_1[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("%7.4f (%7.4f) [%+6.1f%%]  ", actual, ref, error_pct);
    }
    std::cout << std::endl;
  }

  std::cout << "\nint4_1_k_low Sample comparison (first 10x10):" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 10; i < std::min(20, sample_m); i++) {
    for (int j = 10; j < std::min(20, sample_n); j++) {
      float actual = ggml_compute_bf16_to_fp32(output_k_int4_1_low[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("%7.4f (%7.4f) [%+6.1f%%]  ", actual, ref, error_pct);
    }
    std::cout << std::endl;
  }

  std::cout << "\nint4 Sample comparison:" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output_int4[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("j:%d, %7.4f (%7.4f) [%+6.1f%%]  ", j, actual, ref, error_pct);
    }
    std::cout << std::endl;
  }

  std::cout << "\nSample comparison:" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("j:%d, %7.4f (%7.4f) [%+6.1f%%]  ", j, actual, ref, error_pct);
    }
    std::cout << std::endl;
  }

  std::cout << "\nint4_1_k Sample comparison:" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output_k_int4_1[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("j:%d, %7.4f (%7.4f) [%+6.1f%%]  ", j, actual, ref, error_pct);
    }
    std::cout << std::endl;
  }

  std::cout << "\nint4_1 Sample comparison:" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output_int4_1[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("j:%d, %7.4f (%7.4f) [%+6.1f%%]  ", j, actual, ref, error_pct);
    }
    std::cout << std::endl;
  }

  std::cout << "\nint4_1_k_low Sample comparison:" << std::endl;
  std::cout << "Format: actual (reference) [error%]" << std::endl;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output_k_int4_1_low[i * n + j]);
      float ref = ref_result[i * sample_n + j];
      float error_pct = std::abs(ref) > 1e-6 ? (actual - ref) / ref * 100 : 0.0f;
      printf("j:%d, %7.4f (%7.4f) [%+6.1f%%]  ", j, actual, ref, error_pct);
    }
    std::cout << std::endl;
  }

  // Check if accuracy is acceptable for INT4
  if (avg_rel_error < 0.2f) {
    std::cout << "\n✓ Excellent accuracy (<20% average error)" << std::endl;
  } else if (avg_rel_error < 0.3f) {
    std::cout << "\n✓ Acceptable accuracy (20-30% average error)" << std::endl;
  } else if (avg_rel_error < 0.4f) {
    std::cout << "\n⚠ Marginal accuracy (30-40% average error)" << std::endl;
  } else {
    std::cout << "\n✗ Poor accuracy (>40% average error)" << std::endl;
  }

  if (avg_rel_error_int4_1 < 0.2f) {
    std::cout << "\n✓ Excellent accuracy for INT4 quantization (<20% average error)" << std::endl;
  } else if (avg_rel_error_int4_1 < 0.3f) {
    std::cout << "\n✓ Acceptable accuracy for INT4 quantization (20-30% average error)" << std::endl;
  } else if (avg_rel_error_int4_1 < 0.4f) {
    std::cout << "\n⚠ Marginal accuracy for INT4 quantization (30-40% average error)" << std::endl;
  } else {
    std::cout << "\n✗ Poor accuracy for INT4 quantization (>40% average error)" << std::endl;
  }

  if (avg_rel_error_int4 < 0.2f) {
    std::cout << "\n✓ Excellent accuracy for INT4 quantization (<20% average error)" << std::endl;
  } else if (avg_rel_error_int4 < 0.3f) {
    std::cout << "\n✓ Acceptable accuracy for INT4 quantization (20-30% average error)" << std::endl;
  } else if (avg_rel_error_int4 < 0.4f) {
    std::cout << "\n⚠ Marginal accuracy for INT4 quantization (30-40% average error)" << std::endl;
  } else {
    std::cout << "\n✗ Poor accuracy for INT4 quantization (>40% average error)" << std::endl;
  }

  if (avg_rel_error_k_int4_1 < 0.2f) {
    std::cout << "\n✓ Excellent accuracy for INT4 k-group quantization (<20% average error)" << std::endl;
  } else if (avg_rel_error_k_int4_1 < 0.3f) {
    std::cout << "\n✓ Acceptable accuracy for INT4 k-group quantization (20-30% average error)" << std::endl;
  } else if (avg_rel_error_k_int4_1 < 0.4f) {
    std::cout << "\n⚠ Marginal accuracy for INT4 k-group quantization (30-40% average error)" << std::endl;
  } else {
    std::cout << "\n✗ Poor accuracy for INT4 k-group quantization (>40% average error)" << std::endl;
  }

  if (avg_rel_error_k_int4_1_low < 0.2f) {
    std::cout << "\n✓ Excellent accuracy for INT4 k-group low quantization (<20% average error)" << std::endl;
  } else if (avg_rel_error_k_int4_1_low < 0.3f) {
    std::cout << "\n✓ Acceptable accuracy for INT4 k-group low quantization (20-30% average error)" << std::endl;
  } else if (avg_rel_error_k_int4_1_low < 0.4f) {
    std::cout << "\n⚠ Marginal accuracy for INT4 k-group low quantization (30-40% average error)" << std::endl;
  } else {
    std::cout << "\n✗ Poor accuracy for INT4 k-group low quantization (>40% average error)" << std::endl;
  }

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);
}

int main() {
  test_specific_dimensions();
  return 0;
}