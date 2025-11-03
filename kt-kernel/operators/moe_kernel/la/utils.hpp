#pragma once
// #include <arm_sve.h>
#include <cstdint>
#include <cstring>

// 简单截断模式：直接丢弃低 16 位
static inline uint16_t float_to_bf16_trunc(float f) {
  uint32_t u;
  // 按位拷贝，避免 strict‑aliasing UB
  memcpy(&u, &f, sizeof(u));   // :contentReference[oaicite:3]{index=3}
  return (uint16_t)(u >> 16);  // 截断得到高 16 位 :contentReference[oaicite:4]{index=4}
}

static inline void convert_32fp32_to_32bf16_pure_c(const float* src, uint16_t* dst) {
  // src 已偏移至 token_nth * hidden_size
  for (int e = 0; e < 32; e++) {  // 共 32 个元素
    // 选择截断或四舍五入
    dst[e] = float_to_bf16_trunc(src[e]);
  }
}

// 把 32 个 bf16 元素转换成 32 个 fp32 元素

static inline void convert_32bf16_to_32fp32_pure_c(const uint16_t* src, float* dst) {
  for (int e = 0; e < 32; e++) {
    uint32_t temp = ((uint32_t)src[e]) << 16;  // 将 BF16 左移 16 位
    memcpy(&dst[e], &temp, sizeof(float));     // 将结果复制到 FP32 变量中
  }
}