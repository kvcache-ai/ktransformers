#ifndef AMX_RAW_UTILS_HPP
#define AMX_RAW_UTILS_HPP

/**
 * @file amx_raw_utils.hpp
 * @brief Raw data format conversion utilities (FP8 e4m3, BF16, etc.)
 *
 * 本文件实现 FP8 e4m3 格式与 BF16/FP32 之间的转换。
 * 由于 Intel AMX 没有原生 FP8 指令，需要先将 FP8 转换为 BF16，
 * 然后使用 _tile_dpbf16ps 进行计算。
 *
 * FP8 e4m3 格式：
 *   - 1 位符号
 *   - 4 位指数 (bias = 7)
 *   - 3 位尾数
 *   - 范围: [-240, 240]
 *   - 特殊值: 无穷大和 NaN 被禁用
 *
 * UE8M0 Scale 格式：
 *   - 8 位无符号指数，无尾数
 *   - 表示 2^k，其中 k 是整数
 */

#include <cstdint>
#include <cmath>
#include <immintrin.h>
#include "amx_config.hpp"
#include "llama.cpp/ggml-impl.h"

namespace amx {
namespace fp8 {

// FP8 e4m3 格式常量
constexpr int FP8_E4M3_EXPONENT_BITS = 4;
constexpr int FP8_E4M3_MANTISSA_BITS = 3;
constexpr int FP8_E4M3_EXPONENT_BIAS = 7;
constexpr float FP8_E4M3_MAX = 240.0f;
constexpr float FP8_E4M3_MIN = -240.0f;

// 块量化常量
constexpr int FP8_BLOCK_SIZE = 128;  // 128x128 块量化

// ============================================================================
// 标量 FP8 转换函数
// ============================================================================

/**
 * @brief 将单个 FP8 e4m3 值转换为 FP32
 * @param fp8_val FP8 值（8位）
 * @return 对应的 FP32 值
 */
ALWAYS_INLINE float fp8_e4m3_to_fp32(uint8_t fp8_val) {
    // 提取符号、指数、尾数
    uint8_t sign = (fp8_val >> 7) & 0x1;
    uint8_t exp = (fp8_val >> 3) & 0xF;  // 4 位指数
    uint8_t mant = fp8_val & 0x7;        // 3 位尾数

    // 处理特殊情况：零
    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }

    // 处理次正规数
    if (exp == 0) {
        // 次正规数：值 = (-1)^sign * 2^(1-bias) * (0.mant)
        float mantissa = mant / 8.0f;  // 0.xxx
        float value = std::ldexp(mantissa, 1 - FP8_E4M3_EXPONENT_BIAS);
        return sign ? -value : value;
    }

    // 正规数：值 = (-1)^sign * 2^(exp-bias) * (1.mant)
    float mantissa = 1.0f + mant / 8.0f;  // 1.xxx
    float value = std::ldexp(mantissa, exp - FP8_E4M3_EXPONENT_BIAS);
    return sign ? -value : value;
}

/**
 * @brief 将 FP32 值转换为 FP8 e4m3
 * @param fp32_val FP32 值
 * @return 对应的 FP8 值（8位）
 */
ALWAYS_INLINE uint8_t fp32_to_fp8_e4m3(float fp32_val) {
    // 处理特殊情况
    if (fp32_val == 0.0f) {
        return std::signbit(fp32_val) ? 0x80 : 0x00;
    }

    // 裁剪到 FP8 范围
    if (fp32_val > FP8_E4M3_MAX) fp32_val = FP8_E4M3_MAX;
    if (fp32_val < FP8_E4M3_MIN) fp32_val = FP8_E4M3_MIN;

    uint8_t sign = std::signbit(fp32_val) ? 1 : 0;
    float abs_val = std::fabs(fp32_val);

    // 获取指数
    int exp;
    float mant = std::frexp(abs_val, &exp);
    // frexp 返回 [0.5, 1.0)，我们需要 [1.0, 2.0)
    mant *= 2.0f;
    exp -= 1;

    // 加上偏置
    int biased_exp = exp + FP8_E4M3_EXPONENT_BIAS;

    // 处理下溢（次正规数）
    if (biased_exp <= 0) {
        // 次正规数处理
        int shift = 1 - biased_exp;
        mant = mant / (1 << shift);
        biased_exp = 0;
    }

    // 处理上溢
    if (biased_exp >= 15) {
        // 最大值
        return (sign << 7) | 0x7E;  // 最大有限值
    }

    // 量化尾数到 3 位
    uint8_t mant_bits;
    if (biased_exp == 0) {
        // 次正规数：mant 是 0.xxx 形式
        mant_bits = static_cast<uint8_t>(std::round(mant * 8.0f));
    } else {
        // 正规数：mant 是 1.xxx 形式，去掉隐含的 1
        mant_bits = static_cast<uint8_t>(std::round((mant - 1.0f) * 8.0f));
    }

    if (mant_bits > 7) mant_bits = 7;

    return (sign << 7) | (biased_exp << 3) | mant_bits;
}

/**
 * @brief 将 FP8 e4m3 转换为 BF16
 * @param fp8_val FP8 值
 * @return BF16 值
 */
ALWAYS_INLINE ggml_bf16_t fp8_e4m3_to_bf16(uint8_t fp8_val) {
    float fp32 = fp8_e4m3_to_fp32(fp8_val);
    return GGML_FP32_TO_BF16(fp32);
}

/**
 * @brief 将 BF16 转换为 FP8 e4m3
 * @param bf16_val BF16 值
 * @return FP8 值
 */
ALWAYS_INLINE uint8_t bf16_to_fp8_e4m3(ggml_bf16_t bf16_val) {
    float fp32 = GGML_BF16_TO_FP32(bf16_val);
    return fp32_to_fp8_e4m3(fp32);
}

// ============================================================================
// 纯位运算 FP8 -> BF16 转换 (高性能，无浮点运算)
// ============================================================================

/**
 * @brief FP8 e4m3 -> BF16 纯位运算转换 (标量版本)
 *
 * 位运算原理:
 *   FP8 e4m3:  S EEEE MMM      (1+4+3 bits, bias=7)
 *   BF16:      S EEEEEEEE MMMMMMM (1+8+7 bits, bias=127)
 *
 *   转换步骤:
 *   1. 符号: S_bf16 = S_fp8 << 8  (从bit7移到bit15)
 *   2. 指数: E_bf16 = E_fp8 + 120 (127-7=120), 左移7位
 *   3. 尾数: M_bf16 = M_fp8 << 4  (3位扩展到7位)
 *
 * @param fp8_val FP8 e4m3 值
 * @return BF16 的位表示 (uint16_t)
 */
ALWAYS_INLINE uint16_t fp8_e4m3_to_bf16_bits(uint8_t fp8_val) {
    // 处理零
    if ((fp8_val & 0x7F) == 0) {
        return (uint16_t)(fp8_val & 0x80) << 8;
    }

    // 提取各部分并转换
    uint16_t sign = (uint16_t)(fp8_val & 0x80) << 8;    // bit7 -> bit15
    uint16_t exp = ((fp8_val >> 3) & 0xF) + 120;        // 指数 + 偏置差
    uint16_t mant = (uint16_t)(fp8_val & 0x07) << 4;    // 尾数左移4位

    return sign | (exp << 7) | mant;
}

/**
 * @brief AVX-512 批量 FP8 e4m3 -> BF16 转换 (32个元素，无次正规数处理)
 *
 * 来自 sglang/sgl-kernel/csrc/cpu/vec.h 的高效实现
 * 注意: 此版本不处理次正规数 (0.0019 ~ 0.0137 范围)，适用于权重推理
 *
 * @param fp8_vec 256-bit 向量，包含 32 个 FP8 值
 * @return 512-bit BF16 向量
 */
ALWAYS_INLINE __m512bh fp8x32_to_bf16x32_bitwise(__m256i fp8_vec) {
    // 1. 将 8-bit FP8 扩展为 16-bit
    const __m512i x = _mm512_cvtepu8_epi16(fp8_vec);

    // 2. 提取尾数: (x & 0x07) << 4
    const __m512i mant = _mm512_slli_epi16(
        _mm512_and_si512(x, _mm512_set1_epi16(0x07)), 4);

    // 3. 提取指数: ((x >> 3) & 0x0F)
    const __m512i raw_exp = _mm512_srli_epi16(
        _mm512_and_si512(x, _mm512_set1_epi16(0x78)), 3);

    // 4. 调整指数偏置并左移: (exp + 120) << 7
    const __m512i exp = _mm512_slli_epi16(
        _mm512_add_epi16(raw_exp, _mm512_set1_epi16(120)), 7);

    // 5. 合并指数和尾数
    const __m512i nonsign = _mm512_or_si512(exp, mant);

    // 6. 提取符号: (x & 0x80) << 8
    const __m512i sign = _mm512_slli_epi16(
        _mm512_and_si512(x, _mm512_set1_epi16(0x80)), 8);

    // 7. 合并符号和非符号部分
    const __m512i combined = _mm512_or_si512(nonsign, sign);

    // 8. 处理零值: 若输入为0则输出也为0
    const __mmask32 is_nonzero = _mm512_cmpneq_epi16_mask(x, _mm512_setzero_si512());
    return (__m512bh)_mm512_maskz_mov_epi16(is_nonzero, combined);
}

/**
 * @brief AVX-512 批量 FP8 e4m3 -> BF16 转换 (带次正规数处理)
 *
 * 完整处理次正规数和 NaN，适用于需要精确转换的场景
 *
 * @param fp8_vec 256-bit 向量，包含 32 个 FP8 值
 * @return 512-bit BF16 向量
 */
ALWAYS_INLINE __m512bh fp8x32_to_bf16x32_bitwise_full(__m256i fp8_vec) {
    __m512i x = _mm512_cvtepu8_epi16(fp8_vec);

    // 计算 lg2(mantissa) 用于次正规数处理
    __m512i lg2mant = _mm512_mask_mov_epi16(
        _mm512_mask_mov_epi16(
            _mm512_setzero_si512(),
            _mm512_test_epi16_mask(x, _mm512_set1_epi16(2)),
            _mm512_set1_epi16(1)),
        _mm512_test_epi16_mask(x, _mm512_set1_epi16(4)),
        _mm512_set1_epi16(2));

    return (__m512bh)(_mm512_or_si512(
        _mm512_maskz_mov_epi16(
            _mm512_cmpneq_epi16_mask(
                _mm512_and_si512(x, _mm512_set1_epi16(127)),
                _mm512_setzero_si512()),
            _mm512_mask_blend_epi16(
                _mm512_test_epi16_mask(x, _mm512_set1_epi16(120)),
                // 次正规数处理
                _mm512_or_si512(
                    _mm512_and_si512(
                        _mm512_sllv_epi16(
                            _mm512_and_si512(x, _mm512_set1_epi16(3)),
                            _mm512_sub_epi16(_mm512_set1_epi16(7), lg2mant)),
                        _mm512_set1_epi16(0x007f)),
                    _mm512_slli_epi16(
                        _mm512_add_epi16(lg2mant, _mm512_set1_epi16(118)), 7)),
                // 正规数处理
                _mm512_or_si512(
                    _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(7)), 4),
                    _mm512_slli_epi16(
                        _mm512_add_epi16(
                            _mm512_srli_epi16(
                                _mm512_and_si512(x, _mm512_set1_epi16(120)), 3),
                            _mm512_set1_epi16(120)),
                        7)))),
        // 符号位
        _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(128)), 8)));
}

/**
 * @brief 批量 FP8 -> BF16 转换并存储 (32个元素，带 scale)
 *
 * @param fp8_ptr 输入 FP8 数据指针 (32 bytes)
 * @param bf16_ptr 输出 BF16 数据指针 (64 bytes)
 * @param scale 缩放因子
 */
ALWAYS_INLINE void fp8x32_to_bf16x32_scaled(__m256i fp8_vec, ggml_bf16_t* bf16_ptr, float scale) {
    // 使用位运算转换
    __m512bh bf16_raw = fp8x32_to_bf16x32_bitwise(fp8_vec);

    // 应用 scale: 需要先转 FP32，乘以 scale，再转回 BF16
    __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32((__m512i)bf16_raw, 0)), 16));
    __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32((__m512i)bf16_raw, 1)), 16));

    __m512 vscale = _mm512_set1_ps(scale);
    f0 = _mm512_mul_ps(f0, vscale);
    f1 = _mm512_mul_ps(f1, vscale);

    // FP32 -> BF16
    __m512i result = (__m512i)_mm512_cvtne2ps_pbh(f1, f0);
    _mm512_storeu_si512((__m512i*)bf16_ptr, result);
}

/**
 * @brief 批量 FP8 -> BF16 转换并存储 (无 scale，直接位运算)
 *
 * @param fp8_ptr 输入 FP8 数据指针 (32 bytes)
 * @param bf16_ptr 输出 BF16 数据指针 (64 bytes)
 */
ALWAYS_INLINE void fp8x32_to_bf16x32_direct(const uint8_t* fp8_ptr, ggml_bf16_t* bf16_ptr) {
    __m256i fp8_vec = _mm256_loadu_si256((__m256i*)fp8_ptr);
    __m512bh bf16_raw = fp8x32_to_bf16x32_bitwise(fp8_vec);
    _mm512_storeu_si512((__m512i*)bf16_ptr, (__m512i)bf16_raw);
}

// ============================================================================
// SIMD FP8 转换函数 (AVX-512) - 查找表版本
// ============================================================================

/**
 * @brief 预计算的 FP8 到 FP32 查找表
 *
 * 由于 FP8 只有 256 个可能的值，使用查找表是最快的转换方式。
 */
struct FP8ToFP32LUT {
    float table[256];

    FP8ToFP32LUT() {
        for (int i = 0; i < 256; i++) {
            table[i] = fp8_e4m3_to_fp32(static_cast<uint8_t>(i));
        }
    }

    ALWAYS_INLINE float operator[](uint8_t idx) const {
        return table[idx];
    }
};

// 全局查找表实例
inline const FP8ToFP32LUT& get_fp8_to_fp32_lut() {
    static FP8ToFP32LUT lut;
    return lut;
}

/**
 * @brief 使用查找表将 FP8 转换为 FP32（快速版本）
 * @param fp8_val FP8 值
 * @return FP32 值
 */
ALWAYS_INLINE float fp8_e4m3_to_fp32_fast(uint8_t fp8_val) {
    return get_fp8_to_fp32_lut()[fp8_val];
}

/**
 * @brief 将 16 个 FP8 值转换为 16 个 FP32 值
 * @param fp8_ptr 指向 16 个 FP8 值的指针
 * @param scale 缩放因子
 * @return 包含 16 个 FP32 值的 __m512
 */
ALWAYS_INLINE __m512 fp8x16_to_fp32x16(const uint8_t* fp8_ptr, float scale) {
    const auto& lut = get_fp8_to_fp32_lut();
    alignas(64) float result[16];

    // 使用查找表转换
    for (int i = 0; i < 16; i++) {
        result[i] = lut[fp8_ptr[i]] * scale;
    }

    return _mm512_load_ps(result);
}

/**
 * @brief 将 32 个 FP8 值转换为 32 个 BF16 值（用于 AMX）
 * @param fp8_ptr 指向 32 个 FP8 值的指针
 * @param bf16_ptr 输出 32 个 BF16 值的指针
 * @param scale 缩放因子
 */
ALWAYS_INLINE void fp8x32_to_bf16x32(const uint8_t* fp8_ptr, ggml_bf16_t* bf16_ptr, float scale) {
    const auto& lut = get_fp8_to_fp32_lut();

    // 转换前 16 个
    alignas(64) float fp32_0[16], fp32_1[16];
    for (int i = 0; i < 16; i++) {
        fp32_0[i] = lut[fp8_ptr[i]] * scale;
        fp32_1[i] = lut[fp8_ptr[i + 16]] * scale;
    }

    __m512 v0 = _mm512_load_ps(fp32_0);
    __m512 v1 = _mm512_load_ps(fp32_1);

    // FP32 -> BF16
    __m512i bf16_combined = _mm512_cvtne2ps_pbh(v1, v0);
    _mm512_store_si512((__m512i*)bf16_ptr, bf16_combined);
}

/**
 * @brief 将 64 个 FP8 值转换为 64 个 BF16 值（一次处理一行 AMX tile）
 * @param fp8_ptr 指向 64 个 FP8 值的指针
 * @param bf16_ptr 输出 64 个 BF16 值的指针
 * @param scale 缩放因子
 */
ALWAYS_INLINE void fp8x64_to_bf16x64(const uint8_t* fp8_ptr, ggml_bf16_t* bf16_ptr, float scale) {
    fp8x32_to_bf16x32(fp8_ptr, bf16_ptr, scale);
    fp8x32_to_bf16x32(fp8_ptr + 32, bf16_ptr + 32, scale);
}

// ============================================================================
// UE8M0 Scale 格式处理
// ============================================================================

/**
 * @brief 将 FP32 scale 转换为 UE8M0 格式
 *
 * UE8M0 格式只存储指数，表示 2^k
 * @param fp32_scale FP32 缩放因子
 * @return UE8M0 编码的指数值
 */
ALWAYS_INLINE uint8_t fp32_to_ue8m0(float fp32_scale) {
    if (fp32_scale == 0.0f) return 0;

    int exp;
    std::frexp(fp32_scale, &exp);
    // 向上取整到最近的 2 的幂次
    float ceil_val = std::pow(2.0f, std::ceil(std::log2(std::fabs(fp32_scale))));

    std::frexp(ceil_val, &exp);
    // UE8M0 偏置为 127（与 FP32 相同）
    int biased_exp = exp + 126;  // frexp 返回的 exp 需要调整

    if (biased_exp < 0) biased_exp = 0;
    if (biased_exp > 255) biased_exp = 255;

    return static_cast<uint8_t>(biased_exp);
}

/**
 * @brief 将 UE8M0 格式转换为 FP32 scale
 * @param ue8m0_val UE8M0 编码值
 * @return FP32 缩放因子
 */
ALWAYS_INLINE float ue8m0_to_fp32(uint8_t ue8m0_val) {
    if (ue8m0_val == 0) return 0.0f;

    // UE8M0 表示 2^(val - 127)
    int exp = static_cast<int>(ue8m0_val) - 127;
    return std::ldexp(1.0f, exp);
}

/**
 * @brief 计算 128x128 块的缩放因子
 *
 * 对于给定的 128x128 块，计算其 amax 并返回 UE8M0 格式的 scale
 * @param data 指向 128x128 FP32 数据的指针
 * @param stride 行步长
 * @return UE8M0 格式的 scale
 */
ALWAYS_INLINE float compute_block_scale_fp8(const float* data, int rows, int cols, int stride) {
    float amax = 0.0f;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float abs_val = std::fabs(data[i * stride + j]);
            if (abs_val > amax) amax = abs_val;
        }
    }

    // scale = amax / 240 (FP8 最大值)，向上取整到 2 的幂次
    if (amax == 0.0f) return 1.0f;

    float scale = amax / FP8_E4M3_MAX;
    // 向上取整到 2 的幂次
    return std::pow(2.0f, std::ceil(std::log2(scale)));
}

/**
 * @brief 使用 AVX-512 计算块的最大绝对值
 * @param data 数据指针
 * @param count 元素数量（必须是 16 的倍数）
 * @return 最大绝对值
 */
ALWAYS_INLINE float compute_amax_avx512(const float* data, int count) {
    __m512 amax_v = _mm512_setzero_ps();
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);

    for (int i = 0; i < count; i += 16) {
        __m512 v = _mm512_loadu_ps(data + i);
        __m512 abs_v = _mm512_andnot_ps(sign_mask, v);
        amax_v = _mm512_max_ps(amax_v, abs_v);
    }

    return _mm512_reduce_max_ps(amax_v);
}

/**
 * @brief 使用 AVX-512 计算 BF16 块的最大绝对值
 */
ALWAYS_INLINE float compute_amax_bf16_avx512(const ggml_bf16_t* data, int count) {
    __m512 amax_v = _mm512_setzero_ps();
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);

    for (int i = 0; i < count; i += 32) {
        __m512 f0, f1;
        __m512i bf16_data = _mm512_loadu_si512((__m512i*)(data + i));

        // BF16 -> FP32 转换
        __m256i lo = _mm512_extracti32x8_epi32(bf16_data, 0);
        __m256i hi = _mm512_extracti32x8_epi32(bf16_data, 1);

        f0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16));
        f1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16));

        __m512 abs_f0 = _mm512_andnot_ps(sign_mask, f0);
        __m512 abs_f1 = _mm512_andnot_ps(sign_mask, f1);

        amax_v = _mm512_max_ps(amax_v, abs_f0);
        amax_v = _mm512_max_ps(amax_v, abs_f1);
    }

    return _mm512_reduce_max_ps(amax_v);
}

}  // namespace fp8
}  // namespace amx

#endif  // AMX_RAW_UTILS_HPP
