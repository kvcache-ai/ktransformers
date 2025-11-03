#include <arm_sve.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "../../reduce.hpp"
#include "../../rms-norm.hpp"
#include "../../rope.hpp"
#include "../../softmax.hpp"
#include "../la/arm_kml.hpp"
#include "llama.cpp/ggml-common.h"
#include "llama.cpp/ggml.h"

void bf16_to_fp16(const ggml_bf16_t* src, ggml_fp16_t* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float x = ggml_bf16_to_fp32(src[i]);
    dst[i] = ggml_fp32_to_fp16(x);
  }
}

void debug_rope() {
  float16_t* fp16 = new float16_t[1024 * 64];

  for (size_t i = 0; i < 1024 * 64; i++) {
    fp16[i] = static_cast<double>(std::rand()) / RAND_MAX;
  }
  std::ofstream("before_rope", std::ios::binary).write((char*)fp16, 1024 * 64 * sizeof(float16_t));

  DeepseekV3YarnRotaryEmbedding rope(64, 163840, 10000, 40, 4096, 32, 1, 1, 1);

  rope.init(1024);

  Rope<DeepseekV3YarnRotaryEmbedding, float16_t> rope_applier;
  rope_applier.apply_multiple(rope, fp16, 64, 64, 0, 1024);

  std::ofstream("cos", std::ios::binary).write((char*)rope.cos(0), 1024 * 32 * sizeof(float));
  std::ofstream("sin", std::ios::binary).write((char*)rope.sin(0), 1024 * 32 * sizeof(float));

  std::ofstream("after_rope", std::ios::binary).write((char*)fp16, 1024 * 64 * sizeof(float16_t));
}

void debug_softmax() {
  float16_t* fp16 = new float16_t[64 * 1024];

  for (size_t i = 0; i < 1024 * 64; i++) {
    fp16[i] = static_cast<double>(std::rand()) / RAND_MAX * 10;
    if (i % 12 == 0) {
      fp16[i] -= std::numeric_limits<float16_t>::infinity();
    }
  }
  std::ofstream("before_softmax", std::ios::binary).write((char*)fp16, 1024 * 64 * sizeof(float16_t));

  Softmax<float16_t>::apply_multiple(64, fp16, 1024, 1024);
  std::ofstream("after_softmax", std::ios::binary).write((char*)fp16, 1024 * 64 * sizeof(float16_t));
}

void debug_inf() {
  float16_t x, y;
  // x = std::numeric_limits<float16_t>::infinity(); // 0.00
  // y = -std::numeric_limits<float16_t>::infinity(); // -0.00
  // x = 1e10;
  x = std::numeric_limits<float>::infinity();   // inf
  y = -std::numeric_limits<float>::infinity();  // -inf
  printf("x = %f, y = %f\n", x, y);
}

void debug_reduce() {
  std::vector<float16_t*> fp16s(128);
  for (size_t i = 0; i < 128; i++) {
    fp16s[i] = new float16_t[1024];
    for (size_t j = 0; j < 1024; j++) {
      fp16s[i][j] = i;
    }
  }

  reduce_sum(fp16s.data(), 128, 0, 10);
  for (int i = 0; i < 10; i++) {
    printf("%f ", fp16s[0][i]);
  }
}

int main() {
  debug_reduce();

  return 0;
}
