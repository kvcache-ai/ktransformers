#ifndef AMX_RAW_BUFFERS_HPP
#define AMX_RAW_BUFFERS_HPP

/**
 * @file amx_raw_buffers.hpp
 * @brief Raw data format buffer management (FP8, BF16, etc.)
 *
 * 本文件实现原精度格式的缓冲区管理，用于 DeepSeek V3.2 等原精度推理。
 *
 * 缓冲区类型：
 * - BufferAFP8Impl: 输入激活缓冲区，支持动态 FP8 量化
 * - BufferBFP8Impl: 权重缓冲区，FP8 格式 + 128x128 块缩放
 * - BufferBFP8BlockImpl: 优化的块量化权重缓冲区
 *
 * 内存布局：
 * - FP8 数据：1 字节/元素
 * - Scale：4 字节/块（BufferB 每 128x128 块一个，BufferA 每 128 行一个）
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "amx_config.hpp"
#include "amx_raw_utils.hpp"
#include "amx_utils.hpp"
#include "llama.cpp/ggml-impl.h"
#include "pack.hpp"
#include "utils.hpp"

namespace amx {

// ============================================================================
// BufferAFP8Impl: FP8 激活缓冲区（支持动态量化）
// ============================================================================

/**
 * @brief FP8 激活缓冲区模板
 *
 * 支持两种模式：
 * 1. 存储 BF16 格式（与现有 BF16 kernel 兼容）
 * 2. 存储 FP8 格式 + 行缩放因子（用于 FP8 原精度）
 *
 * @tparam K Kernel 类型，提供 M_STEP, K_STEP, K_BLOCK 常量
 */
template <typename K>
struct BufferAFP8Impl {
    uint8_t* a;    // FP8 数据
    float* d;      // 每 128 行的缩放因子
    int max_m, k;

    static constexpr int M_STEP = K::M_STEP;
    static constexpr int K_STEP = K::K_STEP;
    static constexpr int K_BLOCK = K::K_BLOCK;
    static constexpr int FP8_BLOCK_SIZE = 128;

    /**
     * @brief 计算所需内存大小
     *
     * @param max_m 最大 M 维度
     * @param k K 维度
     * @return 所需字节数
     */
    static size_t required_size(int max_m, int k) {
        int n_scale_blocks = (max_m + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        return sizeof(uint8_t) * max_m * k + sizeof(float) * n_scale_blocks;
    }

    /**
     * @brief 构造函数
     */
    BufferAFP8Impl(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
        assert(max_m % M_STEP == 0);
        assert(k % K_STEP == 0);
        if (max_m % M_STEP || k % K_STEP) {
            printf("max_m = %d, k = %d, M_STEP = %d, K_STEP = %d\n", max_m, k, M_STEP, K_STEP);
            throw std::runtime_error("BufferAFP8Impl: max_m and k must be multiple of M_STEP and K_STEP");
        }
        set_data(ptr);
    }

    void set_data(void* ptr) {
        assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
        a = reinterpret_cast<uint8_t*>(ptr);
        d = reinterpret_cast<float*>(a + max_m * k);
    }

    /**
     * @brief 从 BF16 源进行动态 FP8 量化
     *
     * 计算每 128 行的 amax，量化到 FP8 格式
     *
     * @param m 实际行数
     * @param src BF16 源数据
     * @param ith 线程索引
     * @param nth 总线程数
     */
    void from_bf16(int m, ggml_bf16_t* src, int ith, int nth) {
        assert(m <= max_m);
        assert(ith == 0 && nth == 1);

        // 计算每 128 行的 scale
        for (int block_m = 0; block_m < m; block_m += FP8_BLOCK_SIZE) {
            int block_rows = std::min(FP8_BLOCK_SIZE, m - block_m);
            float amax = 0.0f;

            // 计算 amax
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < k; j += 32) {
                    __m512 f0, f1;
                    avx512_32xbf16_to_32xfp32(
                        (__m512i*)(src + (block_m + i) * k + j), &f0, &f1
                    );
                    amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
                    amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
                }
            }

            // 计算 scale（向上取整到 2 的幂次）
            float scale = (amax > 0.0f) ? amax / fp8::FP8_E4M3_MAX : 1.0f;
            scale = std::pow(2.0f, std::ceil(std::log2(scale)));
            d[block_m / FP8_BLOCK_SIZE] = scale;

            // 量化到 FP8
            float inv_scale = 1.0f / scale;
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < k; j++) {
                    float val = GGML_BF16_TO_FP32(src[(block_m + i) * k + j]);
                    a[(block_m + i) * k + j] = fp8::fp32_to_fp8_e4m3(val * inv_scale);
                }
            }
        }
    }

    /**
     * @brief 将 FP8 数据转换为 BF16 用于 AMX 计算
     *
     * @param dst BF16 目标缓冲区
     * @param m_begin 起始行
     * @param k_begin 起始列
     * @param rows 行数
     * @param cols 列数
     */
    void to_bf16_tiled(ggml_bf16_t* dst, int m_begin, int k_begin, int rows, int cols) {
        int block_idx = m_begin / FP8_BLOCK_SIZE;
        float scale = d[block_idx];

        for (int i = 0; i < rows; i++) {
            fp8::fp8x32_to_bf16x32(
                a + (m_begin + i) * k + k_begin,
                dst + i * cols,
                scale
            );
        }
    }

    /**
     * @brief 获取子矩阵指针
     */
    uint8_t* get_submat(int m, int k_dim, int m_begin, int k_begin) {
        int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
        int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
        k_begin -= k_block_begin;
        int k_block_size = std::min(K_BLOCK, k_dim - k_block_begin);
        return a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
    }

    /**
     * @brief 获取指定行的 scale
     */
    float get_scale(int m_begin) const {
        return d[m_begin / FP8_BLOCK_SIZE];
    }
};

template <typename K>
struct BufferABF16Impl {
    ggml_bf16_t* a;
    int max_m, k;

    static constexpr int M_STEP = K::M_STEP;
    static constexpr int K_STEP = K::K_STEP;
    static constexpr int K_BLOCK = K::K_BLOCK;
    static constexpr int FP8_BLOCK_SIZE = 128;

    /**
     * @brief 计算所需内存大小
     *
     * @param max_m 最大 M 维度
     * @param k K 维度
     * @return 所需字节数
     */
    static size_t required_size(int max_m, int k) {
        return sizeof(ggml_bf16_t) * max_m * k;
    }

    /**
     * @brief 构造函数
     */
    BufferABF16Impl(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
        assert(max_m % M_STEP == 0);
        assert(k % K_STEP == 0);
        if (max_m % M_STEP || k % K_STEP) {
            printf("max_m = %d, k = %d, M_STEP = %d, K_STEP = %d\n", max_m, k, M_STEP, K_STEP);
            throw std::runtime_error("BufferAFP8Impl: max_m and k must be multiple of M_STEP and K_STEP");
        }
        set_data(ptr);
    }

    void set_data(void* ptr) {
        assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
        a = reinterpret_cast<ggml_bf16_t*>(ptr);
    }

    /**
     * @brief 从 BF16 源数据 到 BF16 BufferA BF16 格式
     *
     * @param m 实际行数
     * @param src BF16 源数据
     * @param ith 线程索引
     * @param nth 总线程数
     */
    void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
        assert(m <= max_m);
        assert(ith == 0 && nth == 1);
        // TODO: 
    }

    /**
     * @brief TODO: 目前不知道这是在干什么
     *
     * @param dst BF16 目标缓冲区
     * @param m_begin 起始行
     * @param k_begin 起始列
     * @param rows 行数
     * @param cols 列数
     */
    void to_bf16_tiled(ggml_bf16_t* dst, int m_begin, int k_begin, int rows, int cols) {
        int block_idx = m_begin / FP8_BLOCK_SIZE;
        float scale = d[block_idx];

        for (int i = 0; i < rows; i++) {
            fp8::fp8x32_to_bf16x32(
                a + (m_begin + i) * k + k_begin,
                dst + i * cols,
                scale
            );
        }
    }

    /**
     * @brief 获取子矩阵指针
     */
    ggml_bf16_t* get_submat(int m, int k_dim, int m_begin, int k_begin) {
        // TODO: BufferA 格式还未确定。
    }
};

// ============================================================================
// BufferBFP8Impl: FP8 权重缓冲区（128x128 块量化）
// ============================================================================

/**
 * @brief FP8 权重缓冲区
 *
 * 存储 FP8 格式的权重矩阵，每个 128x128 块有一个缩放因子。
 * 这与 DeepSeek V3.2 的原精度格式匹配。
 *
 * @tparam K Kernel 类型
 */
template <typename K>
struct BufferBFP8Impl {
    uint8_t* b;    // FP8 权重数据
    float* d;      // 块缩放因子 [n_blocks_n, n_blocks_k]
    int n, k;

    static constexpr int N_STEP = K::N_STEP;
    static constexpr int K_STEP = K::K_STEP;
    static constexpr int N_BLOCK = K::N_BLOCK;
    static constexpr int K_BLOCK = K::K_BLOCK;
    static constexpr int FP8_BLOCK_SIZE = 128;

    static constexpr bool SCALE = true;

    int n_blocks_n;  // N 方向块数
    int n_blocks_k;  // K 方向块数

    /**
     * @brief 计算所需内存大小
     */
    static size_t required_size(int n, int k) {
        int n_blocks_n = (n + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        int n_blocks_k = (k + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        return sizeof(uint8_t) * n * k +
               sizeof(float) * n_blocks_n * n_blocks_k;
    }

    /**
     * @brief 构造函数
     */
    BufferBFP8Impl(int n, int k, void* ptr) : n(n), k(k) {
        n_blocks_n = (n + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        n_blocks_k = (k + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        set_data(ptr);
    }

    void set_data(void* ptr) {
        assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
        b = reinterpret_cast<uint8_t*>(ptr);
        d = reinterpret_cast<float*>(b + n * k);
    }

    /**
     * @brief 从原始 FP8 权重加载（已经是量化格式）
     *
     * @param src FP8 源数据
     * @param scales 块缩放因子
     */
    void load_fp8_weights(const uint8_t* src, const float* scales) {
        // 复制权重
        std::memcpy(b, src, sizeof(uint8_t) * n * k);

        // 复制缩放因子
        std::memcpy(d, scales, sizeof(float) * n_blocks_n * n_blocks_k);
    }

    /**
     * @brief 从 BF16 权重量化到 FP8（离线量化）
     *
     * @param src BF16 源权重
     */
    void from_bf16(const ggml_bf16_t* src) {
        // 对每个 128x128 块进行量化
        for (int bn = 0; bn < n; bn += FP8_BLOCK_SIZE) {
            for (int bk = 0; bk < k; bk += FP8_BLOCK_SIZE) {
                int block_n = std::min(FP8_BLOCK_SIZE, n - bn);
                int block_k = std::min(FP8_BLOCK_SIZE, k - bk);

                // 计算块的 amax
                float amax = 0.0f;
                for (int i = 0; i < block_n; i++) {
                    for (int j = 0; j < block_k; j++) {
                        float val = std::fabs(GGML_BF16_TO_FP32(src[(bn + i) * k + bk + j]));
                        amax = std::max(amax, val);
                    }
                }

                // 计算 scale（向上取整到 2 的幂次）
                float scale = (amax > 0.0f) ? amax / fp8::FP8_E4M3_MAX : 1.0f;
                scale = std::pow(2.0f, std::ceil(std::log2(scale)));

                int block_idx_n = bn / FP8_BLOCK_SIZE;
                int block_idx_k = bk / FP8_BLOCK_SIZE;
                d[block_idx_n * n_blocks_k + block_idx_k] = scale;

                // 量化到 FP8
                float inv_scale = 1.0f / scale;
                for (int i = 0; i < block_n; i++) {
                    for (int j = 0; j < block_k; j++) {
                        float val = GGML_BF16_TO_FP32(src[(bn + i) * k + bk + j]);
                        b[(bn + i) * k + bk + j] = fp8::fp32_to_fp8_e4m3(val * inv_scale);
                    }
                }
            }
        }
    }

    /**
     * @brief 获取指定块的缩放因子
     */
    float get_block_scale(int n_idx, int k_idx) const {
        int block_n = n_idx / FP8_BLOCK_SIZE;
        int block_k = k_idx / FP8_BLOCK_SIZE;
        return d[block_n * n_blocks_k + block_k];
    }

    /**
     * @brief 设置指定块的缩放因子
     */
    void set_block_scale(int block_n, int block_k, float scale) {
        d[block_n * n_blocks_k + block_k] = scale;
    }

    /**
     * @brief 获取子矩阵指针
     */
    uint8_t* get_submat(int n_begin, int k_begin) const {
        return b + n_begin * k + k_begin;
    }

    /**
     * @brief 将指定区域的 FP8 权重转换为 BF16
     *
     * @param dst BF16 输出缓冲区
     * @param n_begin N 起始索引
     * @param k_begin K 起始索引
     * @param rows 行数
     * @param cols 列数
     */
    void to_bf16_block(ggml_bf16_t* dst, int n_begin, int k_begin, int rows, int cols) const {
        float scale = get_block_scale(n_begin, k_begin);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j += 32) {
                int count = std::min(32, cols - j);
                if (count == 32) {
                    fp8::fp8x32_to_bf16x32(
                        b + (n_begin + i) * k + k_begin + j,
                        dst + i * cols + j,
                        scale
                    );
                } else {
                    // 处理边界
                    for (int jj = 0; jj < count; jj++) {
                        float fp32 = fp8::fp8_e4m3_to_fp32(b[(n_begin + i) * k + k_begin + j + jj]) * scale;
                        dst[i * cols + j + jj] = GGML_FP32_TO_BF16(fp32);
                    }
                }
            }
        }
    }

    /**
     * @brief 准备 AMX tiling 格式
     *
     * 将权重重新排列为 AMX 友好的内存布局
     */
    void prepare_for_amx() {
        // TODO: 实现 VNNI 格式转换
        // 对于 BF16 VNNI 格式，需要将相邻的 2 个元素打包
    }
};

// ============================================================================
// BufferCFP8Impl: FP8 输出缓冲区
// ============================================================================

/**
 * @brief FP8 输出缓冲区
 *
 * 存储 FP32 格式的累加器，支持转换为 BF16 输出
 *
 * @tparam K Kernel 类型
 */
template <typename K>
struct BufferCFP8Impl {
    float* c;
    int max_m, n;

    static constexpr int M_STEP = K::M_STEP;
    static constexpr int N_STEP = K::N_STEP;

    /**
     * @brief 计算所需内存大小
     */
    static size_t required_size(int max_m, int n) {
        return sizeof(float) * max_m * n;
    }

    /**
     * @brief 构造函数
     */
    BufferCFP8Impl(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
        assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
        c = reinterpret_cast<float*>(ptr);
    }

    void set_data(void* ptr) {
        assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
        c = reinterpret_cast<float*>(ptr);
    }

    /**
     * @brief 清零
     */
    void clear() {
        std::memset(c, 0, sizeof(float) * max_m * n);
    }

    /**
     * @brief 将 FP32 结果转换为 BF16 输出
     */
    void to_bf16(int m, ggml_bf16_t* dst, int ith, int nth) {
        assert(m <= max_m);
        assert(ith == 0 && nth == 1);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j += 32) {
                int count = std::min(32, n - j);
                if (count == 32) {
                    __m512 f0 = _mm512_load_ps(c + i * n + j);
                    __m512 f1 = _mm512_load_ps(c + i * n + j + 16);
                    __m512i bf16 = _mm512_cvtne2ps_pbh(f1, f0);
                    _mm512_store_si512((__m512i*)(dst + i * n + j), bf16);
                } else {
                    for (int jj = 0; jj < count; jj++) {
                        dst[i * n + j + jj] = GGML_FP32_TO_BF16(c[i * n + j + jj]);
                    }
                }
            }
        }
    }

    /**
     * @brief 获取子矩阵指针
     */
    float* get_submat(int m_begin, int n_begin) {
        return c + m_begin * n + n_begin;
    }

    /**
     * @brief 累加操作（用于多专家合并）
     */
    void accumulate(const float* src, int m, float weight) {
        __m512 w = _mm512_set1_ps(weight);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j += 16) {
                __m512 dst_v = _mm512_load_ps(c + i * n + j);
                __m512 src_v = _mm512_loadu_ps(src + i * n + j);
                dst_v = _mm512_fmadd_ps(src_v, w, dst_v);
                _mm512_store_ps(c + i * n + j, dst_v);
            }
        }
    }
};

// ============================================================================
// 类型别名（用于兼容性）
// ============================================================================

// 前向声明 GemmKernel224FP8
struct GemmKernel224FP8;

using BufferAFP8 = BufferAFP8Impl<GemmKernel224FP8>;
using BufferBFP8 = BufferBFP8Impl<GemmKernel224FP8>;
using BufferCFP8 = BufferCFP8Impl<GemmKernel224FP8>;

}  // namespace amx

#endif  // AMX_RAW_BUFFERS_HPP
