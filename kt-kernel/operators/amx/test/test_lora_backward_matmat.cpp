#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../la/avx_kernels.hpp"

namespace {

float random_value(std::mt19937& rng) {
  static std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  return dist(rng);
}

bool close(const char* name, const std::vector<float>& actual, const std::vector<float>& expected) {
  float max_abs = 0.0f;
  double diff2 = 0.0;
  double ref2 = 0.0;
  for (size_t i = 0; i < actual.size(); i++) {
    const float diff = actual[i] - expected[i];
    max_abs = std::max(max_abs, std::abs(diff));
    diff2 += static_cast<double>(diff) * diff;
    ref2 += static_cast<double>(expected[i]) * expected[i];
  }
  const double rel_l2 = std::sqrt(diff2 / std::max(ref2, 1e-30));
  const bool ok = max_abs <= 2e-3f && rel_l2 <= 1e-4;
  std::printf("%s max_abs=%g rel_l2=%g %s\n", name, max_abs, rel_l2, ok ? "PASS" : "FAIL");
  return ok;
}

bool run_case(int m, int dim) {
  constexpr int rank = 8;
  constexpr float scale = 0.25f;
  std::mt19937 rng(1000 + m * 17 + dim);

  std::vector<float> grad(static_cast<size_t>(m) * dim);
  std::vector<ggml_bf16_t> b_t(static_cast<size_t>(rank) * dim);
  for (float& value : grad) value = random_value(rng);
  for (ggml_bf16_t& value : b_t) value = GGML_FP32_TO_BF16(random_value(rng));

  std::vector<float> du(static_cast<size_t>(m) * rank, 0.0f);
  std::vector<float> du_ref(du.size(), 0.0f);
  avx::lora_backward_du_rank8_matmat(grad.data(), b_t.data(), du.data(), m, dim);
  for (int t = 0; t < m; t++) {
    for (int r = 0; r < rank; r++) {
      for (int n = 0; n < dim; n++) {
        du_ref[static_cast<size_t>(t) * rank + r] +=
            grad[static_cast<size_t>(t) * dim + n] * GGML_BF16_TO_FP32(b_t[static_cast<size_t>(r) * dim + n]);
      }
    }
  }

  std::vector<ggml_bf16_t> a(static_cast<size_t>(rank) * dim);
  for (ggml_bf16_t& value : a) value = GGML_FP32_TO_BF16(random_value(rng));
  std::vector<float> dx(static_cast<size_t>(m) * dim);
  for (float& value : dx) value = random_value(rng);
  std::vector<float> dx_ref = dx;
  avx::lora_backward_dx_rank8_matmat(du.data(), a.data(), dx.data(), m, dim, scale);
  for (int t = 0; t < m; t++) {
    for (int k = 0; k < dim; k++) {
      for (int r = 0; r < rank; r++) {
        dx_ref[static_cast<size_t>(t) * dim + k] +=
            du[static_cast<size_t>(t) * rank + r] * GGML_BF16_TO_FP32(a[static_cast<size_t>(r) * dim + k]) * scale;
      }
    }
  }

  std::vector<ggml_bf16_t> input(static_cast<size_t>(m) * dim);
  std::vector<int> rows(m);
  for (ggml_bf16_t& value : input) value = GGML_FP32_TO_BF16(random_value(rng));
  for (int t = 0; t < m; t++) rows[t] = m - 1 - t;
  std::vector<float> da(static_cast<size_t>(rank) * dim);
  for (float& value : da) value = random_value(rng);
  std::vector<float> da_ref = da;
  avx::lora_backward_da_rank8_matmat(input.data(), rows.data(), du.data(), da.data(), m, dim, scale);
  for (int r = 0; r < rank; r++) {
    for (int k = 0; k < dim; k++) {
      for (int t = 0; t < m; t++) {
        da_ref[static_cast<size_t>(r) * dim + k] +=
            du[static_cast<size_t>(t) * rank + r] *
            GGML_BF16_TO_FP32(input[static_cast<size_t>(rows[t]) * dim + k]) * scale;
      }
    }
  }

  std::vector<float> u(static_cast<size_t>(m) * rank);
  for (float& value : u) value = random_value(rng);
  std::vector<float> db(static_cast<size_t>(dim) * rank);
  for (float& value : db) value = random_value(rng);
  std::vector<float> db_ref = db;
  avx::lora_backward_db_rank8_matmat(u.data(), grad.data(), db.data(), m, dim, scale);
  for (int n = 0; n < dim; n++) {
    for (int r = 0; r < rank; r++) {
      for (int t = 0; t < m; t++) {
        db_ref[static_cast<size_t>(n) * rank + r] +=
            u[static_cast<size_t>(t) * rank + r] * grad[static_cast<size_t>(t) * dim + n] * scale;
      }
    }
  }

  char label[64];
  bool ok = true;
  std::snprintf(label, sizeof(label), "du_m%d_n%d", m, dim);
  ok &= close(label, du, du_ref);
  std::snprintf(label, sizeof(label), "dx_m%d_n%d", m, dim);
  ok &= close(label, dx, dx_ref);
  std::snprintf(label, sizeof(label), "da_m%d_n%d", m, dim);
  ok &= close(label, da, da_ref);
  std::snprintf(label, sizeof(label), "db_m%d_n%d", m, dim);
  ok &= close(label, db, db_ref);
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  for (int m : {1, 2, 3, 4, 7, 8, 9, 42, 64}) ok &= run_case(m, 64);
  ok &= run_case(7, 2048);
  ok &= run_case(42, 7168);
  return ok ? 0 : 1;
}
