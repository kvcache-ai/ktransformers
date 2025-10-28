// #pragma once
#ifdef TEST_UTIL
#include <arm_neon.h>
#include <arm_sve.h>
#include <stdio.h>

static inline void sve_32xbf16_to_32xfp32(const bfloat16_t* src, float* dst0, float* dst1) {
#ifdef __ARM_FEATURE_SVE
  // 全真谓词，对应每个 16‑bit 元素
#else
// fallback: scalar or NEON
#endif
}

static inline void neon_32xbf16_to_32xfp32(const uint16_t* src, float* dst0, float* dst1) {
  // src 指向 32 个连续的 BF16（uint16_t）
  // dst0、dst1 各指向 16 个 float 的缓冲

  for (int block = 0; block < 4; ++block) {
    // 每次处理 8 个 BF16 → 8 个 FP32（拆为两次 4→4 存储）
    uint16x8_t v_bf16 = vld1q_u16(src + block * 8);  // load 8×BF16 :contentReference[oaicite:6]{index=6}

    // 拆低半、高半各 4 个到 u32
    uint32x4_t lo_u32 = vmovl_u16(vget_low_u16(v_bf16));   // lower 4 → u32 :contentReference[oaicite:7]{index=7}
    uint32x4_t hi_u32 = vmovl_u16(vget_high_u16(v_bf16));  // upper 4 → u32 :contentReference[oaicite:8]{index=8}

    // 左移 16 位，相当于将 BF16 的 16 位 mantissa+exp 放到 FP32 高位
    lo_u32 = vshlq_n_u32(lo_u32, 16);  // shift left 16 :contentReference[oaicite:9]{index=9}
    hi_u32 = vshlq_n_u32(hi_u32, 16);  // shift left 16 :contentReference[oaicite:10]{index=10}

    // 重新解释为 float32x4_t
    float32x4_t lo_f32 = vreinterpretq_f32_u32(lo_u32);  // bits → FP32 :contentReference[oaicite:11]{index=11}
    float32x4_t hi_f32 = vreinterpretq_f32_u32(hi_u32);  // bits → FP32 :contentReference[oaicite:12]{index=12}

    // 存储到 dst0 或 dst1，每次存 8 个
    if (block < 2) {
      vst1q_f32(dst0 + block * 4, lo_f32);      // store 4 floats :contentReference[oaicite:13]{index=13}
      vst1q_f32(dst0 + block * 4 + 4, hi_f32);  // store next 4 floats :contentReference[oaicite:14]{index=14}
    } else {
      int b = block - 2;
      vst1q_f32(dst1 + b * 4, lo_f32);      // store 4 floats :contentReference[oaicite:15]{index=15}
      vst1q_f32(dst1 + b * 4 + 4, hi_f32);  // store next 4 floats :contentReference[oaicite:16]{index=16}
    }
  }
}

int main() {
  // 测试代码
  uint16_t bf16_data[32] = {0};  // 假设这里填充了一些 BF16 数据
  float f32_data0[16] = {0};
  float f32_data1[16] = {0};

  neon_32xbf16_to_32xfp32(bf16_data, f32_data0, f32_data1);

  // 打印结果
  for (int i = 0; i < 16; ++i) {
    printf("f32_data0[%d]: %f\n", i, f32_data0[i]);
    printf("f32_data1[%d]: %f\n", i, f32_data1[i]);
  }

  return 0;
}
#endif
