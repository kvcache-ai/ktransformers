// GGUF strip dequantization tests: kt::gguf::dequant_rows_bf16 vs ggml.
//
// For each supported GGML type: quantize random f32 with ggml's reference
// quantizer, dequantize with the kt::gguf kernels (AVX-512 and scalar paths),
// and require bit-identical BF16 output to ggml's dequantize_row_* followed
// by GGML_FP32_TO_BF16. Also covers tail rows, non-multiple-of-N_BLOCK row
// counts and column slices (down-projection NUMA slicing).
//
// Build: KTRANSFORMERS_CPU_DEBUG=ON builds operators/amx/test/*.cpp against
// the llama static lib (see CMakeLists.txt).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "llama.cpp/ggml-quants.h"
#include "../../llamafile/conversion.h"
#include "../../gguf/dequant.hpp"

namespace {

using kt::gguf::dequant_rows_bf16;
using kt::gguf::fp32_to_bf16_ggml;
using kt::gguf::gguf_row_bytes;

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    printf("FAIL: %s\n", what);
    failures++;
  }
}

std::vector<float> make_random(int64_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(gen);
  return v;
}

// Quantize a flat f32 vector into a GGML block buffer.
std::vector<uint8_t> quantize(const std::vector<float>& f32, ggml_type type) {
  const int64_t n = (int64_t)f32.size();
  const int64_t row_size = ggml_row_size(type, n);
  std::vector<uint8_t> out(row_size);
  switch (type) {
    case GGML_TYPE_Q4_K: quantize_row_q4_K_reference(f32.data(), reinterpret_cast<block_q4_K*>(out.data()), n); break;
    case GGML_TYPE_Q5_K: quantize_row_q5_K_reference(f32.data(), reinterpret_cast<block_q5_K*>(out.data()), n); break;
    case GGML_TYPE_Q6_K: quantize_row_q6_K_reference(f32.data(), reinterpret_cast<block_q6_K*>(out.data()), n); break;
    case GGML_TYPE_Q8_0: quantize_row_q8_0_reference(f32.data(), reinterpret_cast<block_q8_0*>(out.data()), n); break;
    case GGML_TYPE_Q4_0: quantize_row_q4_0_reference(f32.data(), reinterpret_cast<block_q4_0*>(out.data()), n); break;
    default: std::abort();
  }
  return out;
}

// ggml reference: to_float -> fp32_to_bf16 (per element).
std::vector<ggml_bf16_t> reference_bf16(const std::vector<uint8_t>& raw, ggml_type type, int64_t n) {
  std::vector<float> f32(n);
  std::vector<ggml_bf16_t> out(n);
  switch (type) {
    case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K*)raw.data(), f32.data(), n); break;
    case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K*)raw.data(), f32.data(), n); break;
    case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K*)raw.data(), f32.data(), n); break;
    case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0*)raw.data(), f32.data(), n); break;
    case GGML_TYPE_Q4_0: dequantize_row_q4_0((const block_q4_0*)raw.data(), f32.data(), n); break;
    default: std::abort();
  }
  for (int64_t i = 0; i < n; i++) out[i] = fp32_to_bf16_ggml(f32[i]);
  return out;
}

void run_type(ggml_type type, int64_t k, int64_t nrows, unsigned seed) {
  char what[256];
  const int64_t n = k * nrows;
  std::vector<float> f32 = make_random(n, seed);
  std::vector<uint8_t> raw = quantize(f32, type);

  // --- full columns ---
  std::vector<ggml_bf16_t> got(n);
  dequant_rows_bf16(raw.data(), type, k, 0, nrows, 0, k, got.data());
  std::vector<ggml_bf16_t> ref = reference_bf16(raw, type, n);

  // compare via the raw src (which is (nrows, k) row-major)
  bool ok = true;
  for (int64_t i = 0; i < n; i++) {
    if (got[i].bits != ref[i].bits) {
      snprintf(what, sizeof(what), "%s full: element %lld (row %lld col %lld): got %04x ref %04x",
               ggml_type_name(type), (long long)i, (long long)(i / k), (long long)(i % k), got[i].bits, ref[i].bits);
      ok = false;
      break;
    }
  }
  check(ok, ok ? "" : what);

  // --- row sub-ranges ---
  for (auto [r0, r1] : {std::pair<int64_t, int64_t>{1, std::min((int64_t)3, nrows)}, {nrows - 2, nrows}, {0, std::min(nrows, (int64_t)63)}}) {
    if (r0 >= r1) continue;
    std::vector<ggml_bf16_t> strip((r1 - r0) * k);
    dequant_rows_bf16(raw.data(), type, k, r0, r1, 0, k, strip.data());
    ok = true;
    for (int64_t r = r0; r < r1; r++) {
      for (int64_t c = 0; c < k; c++) {
        if (strip[(r - r0) * k + c].bits != ref[r * k + c].bits) {
          snprintf(what, sizeof(what), "%s rows [%lld,%lld): mismatch at (%lld,%lld)", ggml_type_name(type),
                   (long long)r0, (long long)r1, (long long)r, (long long)c);
          ok = false;
          break;
        }
      }
      if (!ok) break;
    }
    check(ok, ok ? "" : what);
  }

  // --- column slices (down-projection NUMA slicing) ---
  const int64_t dcol = k / 2;
  for (auto [c0, c1] : {std::pair<int64_t, int64_t>{0, dcol}, {dcol, k}, {256, 512}, {100, 200}}) {
    c0 = std::max((int64_t)0, std::min(c0, k));
    c1 = std::max(c0, std::min(c1, k));
    if (c0 >= c1) continue;
    std::vector<ggml_bf16_t> strip(nrows * (c1 - c0));
    dequant_rows_bf16(raw.data(), type, k, 0, nrows, c0, c1, strip.data());
    ok = true;
    for (int64_t r = 0; r < nrows && ok; r++) {
      for (int64_t c = c0; c < c1; c++) {
        if (strip[r * (c1 - c0) + (c - c0)].bits != ref[r * k + c].bits) {
          snprintf(what, sizeof(what), "%s cols [%lld,%lld): mismatch at (%lld,%lld)", ggml_type_name(type),
                   (long long)c0, (long long)c1, (long long)r, (long long)c);
          ok = false;
          break;
        }
      }
    }
    check(ok, ok ? "" : what);
  }
}

void test_passthrough_types() {
  // BF16: byte copy; F16/F32: value conversion.
  const int64_t k = 1024, nrows = 3;
  std::vector<float> f32 = make_random(k * nrows, 42);
  std::vector<ggml_bf16_t> ref(k * nrows), got(k * nrows);
  for (int64_t i = 0; i < k * nrows; i++) ref[i] = fp32_to_bf16_ggml(f32[i]);

  // F32
  dequant_rows_bf16(f32.data(), GGML_TYPE_F32, k, 0, nrows, 0, k, got.data());
  check(std::memcmp(got.data(), ref.data(), k * nrows * 2) == 0, "F32 passthrough bit-exact");

  // F16: fp16 -> fp32 is exact; the reference must be bf16 of the fp16-rounded
  // value (double rounding), not bf16 of the original f32. Use the
  // self-contained bit-trick (GGML_FP16_TO_FP32 reads ggml's table).
  std::vector<ggml_fp16_t> f16(k * nrows);
  std::vector<ggml_bf16_t> ref16(k * nrows);
  for (int64_t i = 0; i < k * nrows; i++) {
    const uint16_t bits = GGML_FP32_TO_FP16(f32[i]);
    f16[i] = bits;
    ref16[i] = fp32_to_bf16_ggml(kt::gguf::fp16_to_fp32_bits(bits));
  }
  dequant_rows_bf16(f16.data(), GGML_TYPE_F16, k, 0, nrows, 0, k, got.data());
  check(std::memcmp(got.data(), ref16.data(), k * nrows * 2) == 0, "F16 passthrough bit-exact");

  // BF16
  std::vector<ggml_bf16_t> bf16(k * nrows);
  for (int64_t i = 0; i < k * nrows; i++) bf16[i] = GGML_FP32_TO_BF16(f32[i]);
  dequant_rows_bf16(bf16.data(), GGML_TYPE_BF16, k, 0, nrows, 0, k, got.data());
  check(std::memcmp(got.data(), bf16.data(), k * nrows * 2) == 0, "BF16 passthrough bit-exact");
}

}  // namespace

int main() {
  // ggml_table_f32_f16 is all zeros until ggml_init() runs, and the
  // dequantize_row_* reference functions read it via GGML_FP16_TO_FP32.
  struct ggml_init_params params = {0, NULL, true};
  ggml_init(params);
  printf("test_gguf_dequant: Q4_K/Q5_K/Q6_K/Q8_0 (AVX-512 + scalar) vs ggml\n");
  run_type(GGML_TYPE_Q4_K, 7168, 5, 1);
  run_type(GGML_TYPE_Q4_K, 7168, 80, 2);  // non-multiple of N_BLOCK=64
  run_type(GGML_TYPE_Q5_K, 1024, 3, 3);
  run_type(GGML_TYPE_Q6_K, 7168, 2, 4);
  run_type(GGML_TYPE_Q8_0, 512, 80, 5);
  run_type(GGML_TYPE_Q4_0, 1024, 5, 6);  // generic fallback
  test_passthrough_types();

  if (failures == 0) {
    printf("PASS: all gguf dequant tests passed\n");
    return 0;
  }
  printf("FAIL: %d checks failed\n", failures);
  return 1;
}