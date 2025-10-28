#include <random>
#include <stdexcept>

#include "amx.hpp"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"

int main() {
  // init GGML
  struct ggml_init_params params = {
      0,
      NULL,
      true,
  };

  auto ctx_eval = ggml_init(params);

  if (!ctx_eval) {
    throw std::runtime_error("Failed to create ggml context");
  }

  // Allocate Memory
  int m = 1000, n = 8, k = 512;
  float* a = new float[m * k];  // m x k, Row Major
  float* b = new float[k * n];  // k x n, Column Major
  size_t c_row_size = n * sizeof(float);
  c_row_size = (c_row_size + 63) / 64 * 64;  // pad C row
  float* c = new (std::align_val_t(64)) float[m * c_row_size];
  memset(c, 0, m * c_row_size * sizeof(float));
  size_t ldc = c_row_size * sizeof(float);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dis(0, 16);
  for (int i = 0; i < m * k; i++) {
    a[i] = dis(gen);
  }
  for (int i = 0; i < k * n; i++) {
    b[i] = dis(gen);
  }

  // Convert to BF16
  // QA and QB must be aligned to 64 for BF16
  // k is a multiple of 32, so no need for padding
  ggml_bf16_t* qa = new (std::align_val_t(64)) ggml_bf16_t[m * k];
  size_t lda = k * sizeof(ggml_bf16_t);
  ggml_bf16_t* qb = new (std::align_val_t(64)) ggml_bf16_t[k * n];
  size_t ldb = k * sizeof(ggml_bf16_t);
  ggml_fp32_to_bf16_row(a, qa, m * k);
  ggml_fp32_to_bf16_row(b, qb, k * n);

  // AMX Computation
  amx::init_tile(GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32);
  int nth = amx::recommended_nth(m, n, k, GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32);

#pragma omp parallel for
  for (int ith = 0; ith < nth; ith++) {
    amx::gemm(m, n, k, qa, lda, GGML_TYPE_BF16, qb, ldb, GGML_TYPE_BF16, c, ldc, GGML_TYPE_F32, ith, nth);
  }

  // Check
  float* d = new float[m * n];
  memset(d, 0, m * n * sizeof(float));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int kk = 0; kk < k; kk++) {
        d[i * n + j] += a[i * k + kk] * b[j * k + kk];
      }
    }
  }

  float max_error = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      max_error = std::max(max_error, std::abs(d[i * n + j] - c[i * c_row_size + j]) / std::abs(d[i * n + j]));
      // printf("%.2f ",c[i*c_row_size+j]);
    }
    // printf("\n");
  }
  printf("Max Error %f%%\n", max_error * 100);

  return 0;
}
