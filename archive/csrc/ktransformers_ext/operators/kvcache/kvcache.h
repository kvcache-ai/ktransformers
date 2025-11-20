/**
 * @Description  :
 * @Author       : Jianwei Dong
 * @Date         : 2024-08-26 22:47:06
 * @Version      : 1.0.0
 * @LastEditors  : Jianwei Dong
 * @LastEditTime : 2024-08-26 22:47:06
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#ifndef CPUINFER_OPERATOR_KVCACHE_H
#define CPUINFER_OPERATOR_KVCACHE_H

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../cpu_backend/backend.h"
#include "llama.cpp/ggml-common.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

#define CHUNK_SIZE 32

/**
 * @brief Converts a ggml_type enum value to its corresponding string
 * representation.
 *
 * This function provides a human-readable string representation for a given
 * ggml_type enum value. The string can be used for logging, debugging, or
 * displaying information in a user interface.
 *
 * @param type The ggml_type enum value to convert.
 * @return A string representation of the enum value.
 */
std::string ggml_type_to_string(ggml_type type);

/**
 * @enum AnchorType
 * @brief Defines the types of anchors used in attention mechanisms.
 *
 * This enum specifies different types of anchors that can be used in attention
 * mechanisms, such as fixed anchors, dynamic anchors, or special anchors like
 * QUEST, BLOCK_MEAN, or BLOCK_MAX.
 */
enum AnchorType {
    FIXED_ANCHOR, /**< A fixed anchor that does not change. */
    DYNAMIC,      /**< A dynamic anchor that can change over time. */
    QUEST, /**< A special anchor type used for QUEST (Query and Embedding Space
              Transformation). */
    BLOCK_MEAN, /**< An anchor based on the mean of a block of data. */
    BLOCK_MAX /**< An anchor based on the maximum value within a block of data.
               */
};

/**
 * @brief Converts an AnchorType enum value to its corresponding string
 * representation.
 *
 * This function provides a human-readable string representation for a given
 * AnchorType enum value. The string can be used for logging, debugging, or
 * displaying information in a user interface.
 *
 * @param anchor_type The AnchorType enum value to convert.
 * @return A string representation of the enum value.
 */
std::string AnchorTypeToString(AnchorType anchor_type);

/**
 * @enum RetrievalType
 * @brief Defines the types of retrieval strategies in attention mechanisms.
 *
 * This enum specifies different retrieval strategies that can be used in
 * attention mechanisms, such as layer-level retrieval, key-value head-level
 * retrieval, or query head-level retrieval.
 */
enum RetrievalType {
    LAYER,  /**< Retrieval at the layer level. */
    KVHEAD, /**< Retrieval at the key-value head level. */
    QHEAD   /**< Retrieval at the query head level. */
};

/**
 * @brief Converts a RetrievalType enum value to its corresponding string
 * representation.
 *
 * This function provides a human-readable string representation for a given
 * RetrievalType enum value. The string can be used for logging, debugging, or
 * displaying information in a user interface.
 *
 * @param retrieval_type The RetrievalType enum value to convert.
 * @return A string representation of the enum value.
 */
std::string RetrievalTypeToString(RetrievalType retrieval_type);

/**
 * @struct KVCacheConfig
 * @brief Configuration structure for Key-Value (KV) Cache.
 *
 * This structure holds configuration parameters for setting up and managing
 * a Key-Value (KV) Cache used in various attention mechanisms. It includes
 * parameters such as the number of layers, the number of heads, the dimension
 * of each head, block length, anchor information, and memory-related settings.
 */
struct KVCacheConfig {
    int layer_num;   /**< Number of layers in the model. */
    int kv_head_num; /**< Number of heads in the KV Cache. */
    int q_head_num;  /**< Number of heads in the query. */
    int head_dim;    /**< Dimension of each head. */
    int block_len;   /**< Length of each block in the cache. */
    int anchor_num;  /**< Number of anchors used in attention. */

    ggml_type kv_type; /**< Data type of the KV Cache (e.g., fp16, q8_0). */

    // Controls the pre-allocated memory size
    int max_block_num;  /**< Maximum number of blocks that can be allocated. */
    int max_batch_size; /**< Maximum batch size that can be processed. */
    int max_thread_num; /**< Maximum number of threads that can be used. */

    AnchorType
        anchor_type; /**< Type of anchors used in the attention mechanism. */
    RetrievalType
        retrieval_type; /**< Type of retrieval strategy used in the cache. */

    int layer_step;   /**< Step size between layers. */
    int token_step;   /**< Step size between tokens. */
    int layer_offset; /**< Offset value for layers. */

    /**
     * @brief Default constructor for KVCacheConfig.
     *
     * Initializes the configuration with default values. This constructor
     * does not initialize any member variables explicitly.
     */
    KVCacheConfig() = default;

    /**
     * @brief Parameterized constructor for KVCacheConfig.
     *
     * This constructor initializes the configuration with specific values
     * for all member variables.
     *
     * @param layer_num The number of layers in the model.
     * @param kv_head_num The number of heads in the KV Cache.
     * @param q_head_num The number of heads in the query.
     * @param head_dim The dimension of each head.
     * @param block_len The length of each block in the cache.
     * @param anchor_num The number of anchors used in attention.
     * @param anchor_type The type of anchors used in the attention mechanism.
     * @param kv_type The data type of the KV Cache (e.g., fp16, q8_0).
     * @param retrieval_type The type of retrieval strategy used in the cache.
     * @param layer_step The step size between layers.
     * @param token_step The step size between tokens.
     * @param layer_offset The offset value for layers.
     * @param max_block_num The maximum number of blocks that can be allocated.
     * @param max_batch_size The maximum batch size that can be processed.
     * @param max_thread_num The maximum number of threads that can be used.
     */
    KVCacheConfig(int layer_num, int kv_head_num, int q_head_num, int head_dim,
                  int block_len, int anchor_num, AnchorType anchor_type,
                  ggml_type kv_type, RetrievalType retrieval_type,
                  int layer_step, int token_step, int layer_offset,
                  int max_block_num, int max_batch_size, int max_thread_num);
};

/**
 * @class KVCache
 * @brief Manages the Key-Value (KV) Cache used in attention mechanisms.
 *
 * The KVCache class provides functionality for managing the Key-Value Cache,
 * including resizing the cache, retrieving configuration parameters, and
 * updating internal states. This class is typically used in transformer models
 * to store and manage past key and value states for efficient attention
 * computations.
 */
class KVCache {
  public:
    /**
     * @brief Constructs a KVCache object with the given configuration.
     *
     * Initializes the KVCache with the specified configuration parameters,
     * such as the number of layers, heads, head dimensions, and other
     * relevant settings.
     *
     * @param config The configuration object containing initialization
     * parameters.
     */
    KVCache(KVCacheConfig config);

    /**
     * @brief Resizes the number of threads used by the cache.
     *
     * This function adjusts the number of threads that the cache can utilize.
     * It allows dynamic reconfiguration of the parallel processing capabilities
     * based on the current workload or system resources.
     *
     * @param thread_num The new number of threads to use.
     */
    void ThreadResize(int thread_num);

    /**
     * @brief Resizes the batch size managed by the cache.
     *
     * This function adjusts the batch size that the cache can handle. It
     * is useful when the input batch size changes dynamically, allowing
     * the cache to be reconfigured accordingly.
     *
     * @param batch_size The new batch size.
     */
    void BatchResize(int batch_size);

    /**
     * @brief Resizes the number of blocks managed by the cache.
     *
     * This function adjusts the number of blocks that the cache can manage.
     * It allows dynamic reconfiguration of the block structure based on the
     * current sequence length or other factors.
     *
     * @param block_num The new number of blocks.
     */
    void BlockResize(int block_num);

    /**
     * @brief Gets the number of layers in the cache.
     *
     * @return The number of layers configured in the cache.
     */
    int get_layer_num() { return config_.layer_num; }

    /**
     * @brief Gets the number of KV heads in the cache.
     *
     * @return The number of KV heads configured in the cache.
     */
    int get_kv_head_num() { return config_.kv_head_num; }

    /**
     * @brief Gets the number of query heads in the cache.
     *
     * @return The number of query heads configured in the cache.
     */
    int get_q_head_num() { return config_.q_head_num; }

    /**
     * @brief Gets the dimension of each head in the cache.
     *
     * @return The dimension of each head.
     */
    int get_head_dim() { return config_.head_dim; }

    /**
     * @brief Gets the length of each block in the cache.
     *
     * @return The length of each block.
     */
    int get_block_len() { return config_.block_len; }

    /**
     * @brief Gets the number of blocks for a specific layer.
     *
     * @param layer_id The ID of the layer for which to retrieve the block
     * number.
     * @return The number of blocks in the specified layer.
     */
    int get_block_num(int layer_id) { return past_block_num_[layer_id]; }

    /**
     * @brief Gets the number of anchors in the cache.
     *
     * @return The number of anchors configured in the cache.
     */
    int get_anchor_num() { return config_.anchor_num; }

    /**
     * @brief Gets the total length of the cache.
     *
     * @return The total length of the cache.
     */
    int get_cache_total_len() { return cache_total_len_; }

    /**
     * @brief Gets the total number of blocks in the cache.
     *
     * This function computes and returns the total number of blocks in the
     * cache based on the total cache length and the block length configuration.
     *
     * @return The total number of blocks in the cache.
     */
    int get_cache_total_block_num() {
        return (cache_total_len_ + config_.block_len - 1) / config_.block_len;
    }

    /**
     * @brief Updates the total length of the cache.
     *
     * This function sets a new total length for the cache, allowing dynamic
     * adjustment of the cache size during runtime.
     *
     * @param cache_total_len The new total length of the cache.
     */
    void update_cache_total_len(int cache_total_len) {
        cache_total_len_ = cache_total_len;
    }
    void attn(const ggml_fp16_t *q_in, ggml_fp16_t *output, float *attn_lse,
              int layer_idx, int generate_token_idx, int q_len, int batch_size,
              int max_block_num, int *block_table, int *cache_seqlens,
              int pick_block_num, int init_block_num, int local_block_num,
              Backend *backend);

    void update_kvcache_one_block_fp16(const ggml_fp16_t *k_in,
                                       const ggml_fp16_t *v_in, int layer_id,
                                       int block_idx, Backend *backend);

    void get_kvcache_one_block_fp16(ggml_fp16_t *k_in, ggml_fp16_t *v_in,
                                    int layer_id, int block_idx,
                                    Backend *backend);

    void update_importance_one_block(const ggml_fp16_t *importance,
                                     int layer_id, int block_idx,
                                     Backend *backend);
    void get_importance_one_block(ggml_fp16_t *importance, int layer_id,
                                  int block_idx, Backend *backend);

    void get_anchor_one_block(ggml_fp16_t *anchor, int layer_id, int block_idx,
                              Backend *backend);

    void update_anchor_one_block(const ggml_fp16_t *anchor, int layer_id,
                                 int block_idx, Backend *backend);

    void calc_anchor_all_layers(int *block_table, int *cache_seqlens,
                                int batch_size, int max_block_num,
                                Backend *backend);

    void load_kvcache(std::string tensor_file_path, Backend *backend);
    void dump_kvcache(int *block_table, int cache_total_len,
                      std::string tensor_file_path, Backend *backend);

    void get_and_update_kvcache_fp16(ggml_fp16_t *k_in, ggml_fp16_t *v_in,
                                     int layer_id, int *block_table,
                                     int batch_size, int max_block_num,
                                     int *cache_seqlens, int q_len,
                                     Backend *backend);

    void get_kvcache_fp16(ggml_fp16_t *k_in, ggml_fp16_t *v_in, int layer_id,
                          int *block_table, int batch_size, int max_block_num,
                          int *cache_seqlens, Backend *backend);

    void update_kvcache_fp16(const ggml_fp16_t *k_in, const ggml_fp16_t *v_in,
                             int layer_id, int *block_table, int batch_size,
                             int max_block_num, int *cache_seqlens, int q_len,
                             Backend *backend);

    void update_importance(const ggml_fp16_t *importance, int layer_id,
                           int *block_table, int batch_size, int max_block_num,
                           int *offset, int width, Backend *backend);

    void attn_with_kvcache(const ggml_fp16_t *q_in, const ggml_fp16_t *k_in,
                           const ggml_fp16_t *v_in, ggml_fp16_t *output,
                           float *attn_lse, int layer_idx,
                           int generate_token_idx, int q_len, int batch_size,
                           int max_block_num, int *block_table,
                           int *cache_seqlens, int topk, int local,
                           Backend *backend);

    void clear_importance_all_layers(int *block_table, int *cache_seqlens,
                                     int batch_size, int max_block_num,
                                     Backend *backend);

    void clear_kvcache_all_layers(int *block_table, int *cache_seqlens,
                                  int batch_size, int max_block_num,
                                  Backend *backend);

    void get_sincos(ggml_fp16_t *sin, ggml_fp16_t *cos, int seqlen);

    void get_attn_sparsity(const ggml_fp16_t *q_in, float *attn_sparsity,
                           int layer_idx, int generate_token_idx, int q_len,
                           int batch_size, int max_block_num, int *block_table,
                           int *cache_seqlens, int *block_table_origin,
                           int *cache_seqlens_origin, int max_block_num_origin,
                           int topk, int local, Backend *backend);

    void get_all_kvcache_one_layer(int layer_id, ggml_fp16_t *k_in,
                                   ggml_fp16_t *v_in, Backend *backend);

  private:
    // Persistent data
    KVCacheConfig config_;
    int n_gqa_;                            // q_head_num / kv_head_num
    int cache_total_len_;                  // Number of tokens in cache
    std::vector<uint64_t> past_block_num_; // [layer_num]
    std::vector<std::vector<std::vector<std::vector<block_q4_0>>>>
        k_cache_q4; // [layer_num, kv_head_num, past_block_num, block_len *
                    // (head_dim / QK_4)]
    std::vector<std::vector<std::vector<std::vector<block_q4_0>>>>
        v_cache_q4; // [layer_num, kv_head_num, past_block_num, head_dim *
                    // (block_len / QK_4)]
    std::vector<std::vector<std::vector<std::vector<block_q8_0>>>>
        k_cache_q8; // [layer_num, kv_head_num, past_block_num, block_len *
                    // (head_dim / QK_8)]
    std::vector<std::vector<std::vector<std::vector<block_q8_0>>>>
        v_cache_q8; // [layer_num, kv_head_num, past_block_num, head_dim *
                    // (block_len / QK_8)]

    std::vector<std::vector<std::vector<std::vector<ggml_fp16_t>>>>
        k_cache_fp16_; // [layer_num, kv_head_num, past_block_num, block_len *
                       // head_dim]
    std::vector<std::vector<std::vector<std::vector<ggml_fp16_t>>>>
        v_cache_fp16_; // [layer_num, kv_head_num, past_block_num, head_dim *
                       // block_len]

    std::vector<std::vector<std::vector<std::vector<ggml_fp16_t>>>>
        importance_; // [layer_num, past_block_num, block_len,
                     // attention_head_num]

    std::vector<ggml_fp16_t>
        anchor_; // [layer_num * past_block_num * anchor_num *
                 // attention_head_num * head_dim]

    // Runtime data
    int64_t layer_id_;
    int64_t block_idx_;
    int *block_table_;
    uint64_t block_num_;
    int max_block_num_after_retrieval_;

    // Rotary positional embeddings
    std::vector<std::vector<ggml_fp16_t>> sin_; // [seq_len, head_dim]
    std::vector<std::vector<ggml_fp16_t>> cos_; // [seq_len, head_dim]

    // update/get
    int seq_len_;
    uint16_t *k_scales_;        // q4_0
    uint8_t *k_in_;             // q4_0
    uint16_t *v_scales_;        // q4_0
    uint8_t *v_in_;             // q4_0
    uint16_t *k_data_;          // fp16
    uint16_t *v_data_;          // fp16
    uint16_t *importance_data_; // fp16
    uint16_t *anchor_data_;     // fp16

    // sparsity = (sigma(block lse / lse))
    std::vector<std::vector<std::vector<float>>>
        block_lse_; // [batch_size, max_block_num, q_head_num]
    std::vector<std::vector<float>> attn_sparsity_; // [batch_size, q_head_num]

    // attn
    std::vector<std::vector<float>>
        avg_q; // [batch_size, q_head_num * head_dim]

    std::vector<std::vector<ggml_fp16_t>>
        avg_q_fp16; // [batch_size, q_head_num * head_dim]
    std::vector<
        std::priority_queue<std::pair<float, int>,
                            std::vector<std::pair<float, int>>, std::greater<>>>
        top_similar_block_;

    std::vector<std::vector<float>> block_similar_;
    std::vector<std::vector<std::vector<float>>> block_similar_kv_head_;
    std::vector<std::vector<std::vector<float>>> block_similar_q_head_;

    std::vector<int> cache_seqlens_;               // [batch_size]
    std::vector<int> selected_blocks_num_history_; // [layer_num // layer_step]

    std::vector<std::vector<std::vector<int>>> selected_blocks_history_;
    // [layer_num // layer_step, batch_size, max_block_num]

    std::vector<std::vector<std::vector<std::vector<int>>>>
        selected_blocks_history_kvhead_; // [layer_num // layer_step,
                                         // batch_size, max_block_num,
                                         // kv_head_num]

    std::vector<std::vector<int>>
        block_table_before_retrieval_; // [batch_size, max_block_num]
    std::vector<std::vector<int>>
        block_table_after_retrieval_; // [batch_size, pick_block_num]

    std::vector<std::vector<std::vector<int>>>
        block_table_before_retrieval_qhead_; // [batch_size, max_block_num,
                                             // q_head_num]
    std::vector<std::vector<std::vector<int>>>
        block_table_after_retrieval_qhead_; // [batch_size, pick_block_num,
                                            // q_head_num]

    std::vector<std::vector<std::vector<int>>>
        block_table_before_retrieval_kvhead_; // [batch_size, max_block_num,
                                              // kv_head_num]
    std::vector<std::vector<std::vector<int>>>
        block_table_after_retrieval_kvhead_; // [batch_size, pick_block_num,
                                             // kv_head_num]

    std::vector<std::vector<std::unique_ptr<std::mutex>>>
        mutex_; // [batch_size, kv_head_num]
    std::vector<std::vector<std::vector<block_q8_0>>>
        q_q8_0_; // [batch_size, kv_head_num, n_gqa * head_dim / QK8_0]
    std::vector<std::vector<std::vector<float>>>
        q_fp32_; // [batch_size, kv_head_num, n_gqa * head_dim]

    std::vector<std::vector<std::vector<float>>>
        output_fp32_; // [batch_size, kv_head_num, n_gqa * head_dim]
    std::vector<std::vector<std::vector<float>>>
        attn_lse_; // [batch_size, kv_head_num, n_gqa]

    std::vector<std::pair<int, int>> thread_cur_head_idx_; // [thread_num]

    std::vector<std::vector<block_q8_0>>
        thread_local_output_q8_0_; // [thread_num, n_gqa * head_dim / QK8_0]
    std::vector<std::vector<float>>
        thread_local_attn_score_; // [thread_num, n_gqa * block_len]
    std::vector<std::vector<float>>
        thread_local_output_fp32_; // [thread_num, n_gqa * head_dim]
    std::vector<std::vector<float>>
        thread_local_attn_lse_; // [thread_num, n_gqa]
    std::vector<std::vector<float>>
        thread_local_cur_output_fp32_; // [thread_num, n_gqa * head_dim]
    std::vector<std::vector<float>>
        thread_local_cur_attn_lse_; // [thread_num, n_gqa]
    std::vector<std::vector<uint8_t>>
        thread_local_attn_mask_; // [thread_num, block_len // 8]
    std::vector<std::vector<char>>
        thread_local_draft_; // [thread_num, 2 * n_gqa * block_len + 6 * n_gqa *
                             // head_dim + 2 * block_len * head_dim]

    // tmp space
    std::vector<float> q_fp32; // [n_gqa * head_dim]

    void quantize_q_(const uint16_t *q_in_data, int batch_size);
    void attn_initialize_layer_(int batch_size, int layer_idx, int *block_table,
                                int &max_block_num, int *cache_seqlens);
    void attn_initialize_kvhead_(int batch_size, int layer_idx,
                                 int *block_table, int &max_block_num,
                                 int *cache_seqlens);
    void retrieval_kvcache_layer_(const uint16_t *q_in_data, int init_block_num,
                                  int local_block_num, int pick_block_num,
                                  int q_len, int generate_token_idx,
                                  int batch_size, int layer_idx,
                                  int *cache_seqlens, int &max_block_num,
                                  Backend *backend);
    void retrieval_kvcache_kvhead_(const uint16_t *q_in_data,
                                   int init_block_num, int local_block_num,
                                   int pick_block_num, int q_len,
                                   int generate_token_idx, int batch_size,
                                   int layer_idx, int *cache_seqlens,
                                   int &max_block_num, Backend *backend);

    void calculate_block_similarity_layer_(
        const uint16_t *q_in_data, int batch_size, int layer_idx, int q_len,
        int max_block_num, int *cache_seqlens, int init_block_num,
        int local_block_num, int pick_block_num, Backend *backend);
    void calculate_block_similarity_kvhead_(
        const uint16_t *q_in_data, int batch_size, int layer_idx, int q_len,
        int max_block_num, int *cache_seqlens, int init_block_num,
        int local_block_num, int pick_block_num, Backend *backend);

    void select_block_layer_(int batch_size, int layer_idx, int max_block_num,
                             int init_block_num, int local_block_num,
                             int pick_block_num);
    void select_block_kvhead_(int batch_size, int layer_idx, int max_block_num,
                              int init_block_num, int local_block_num,
                              int pick_block_num);

    void calculate_sparsity_layer_(const uint16_t *q_in_data,
                                   float *attn_sparsity, int batch_size,
                                   int max_block_num, int *block_table,
                                   int *cache_seqlens, Backend *backend);
    void calculate_sparsity_kvhead_(const uint16_t *q_in_data,
                                    float *attn_sparsity, int batch_size,
                                    int max_block_num, int *block_table,
                                    int *cache_seqlens, Backend *backend);

    void attention_kvhead_(const uint16_t *q_in_data, ggml_fp16_t *output,
                           float *attn_lse, int batch_size, Backend *backend);
    void attention_layer_(const uint16_t *q_in_data, ggml_fp16_t *output,
                          float *attn_lse, int batch_size, Backend *backend);

    /**
     * @brief Computes attention with KV cache for one block.
     *
     * This function performs attention computation for one block using KV
     * cache. The function supports different data types for Q, K, and V caches,
     * and provides options for quantization. The function does not perform any
     * dynamic memory allocation internally, so all necessary buffers must be
     * pre-allocated externally.
     *
     * @param head_dim The dimension of the head.
     * @param bsz The batch size.
     * @param q_type The data type of Q (GGML data type). Only supports fp16 and
     * q8_0.
     * @param q Pointer to the Q tensor [bsz, head_dim]. The quantization is
     *          always applied along the head_dim dimension. The size must be
     *          bsz * head_dim/32 * qtype_size. If head_dim % 32 != 0, an error
     *          will be raised.
     * @param past_kv_len The length of the past KV cache.
     * @param past_kv_offset The offset in the past KV cache.
     * @param is_full_attn Boolean flag indicating whether to use full attention
     *                     (true for full 1 mask).
     * @param attn_mask Pointer to the attention mask [bsz, past_kv_len]. If
     *                  is_full_attn = false, a bit matrix is passed to
     * represent the mask.
     * @param k_type The data type of K cache (GGML data type). Only supports
     *               fp16, q4_0, and q8_0.
     * @param k_quant_type Quantization type for K cache. 0 for per_token, 1 for
     *                     per_channel. Other values will raise an error.
     * @param k_cache Pointer to the K cache tensor [seq_len, head_dim]. If
     *                quant_type == 0, head_dim % 32 must be 0. If quant_type ==
     * 1, seq_len % 32 must be 0.
     * @param num_k_anchor The number of K anchors. If num_k_anchor == 0, it
     * means no anchor is present.
     * @param k_cache_anchors Pointer to the K cache anchors [num_k_anchor,
     * head_dim]. The k_anchor_type must be fp16.
     * @param k_cache_anchor_pos Pointer to the K cache anchor positions. Each
     * token is associated with the nearest previous anchor position.
     * @param v_type The data type of V cache (GGML data type).
     * @param v_quant_type Quantization type for V cache.
     * @param v_cache Pointer to the V cache tensor [head_dim, seq_len].
     * @param num_v_anchor The number of V anchors.
     * @param v_cache_anchors Pointer to the V cache anchors.
     * @param v_cache_anchor_pos Pointer to the V cache anchor positions.
     * @param attn_score Pre-allocated buffer for attention scores [bsz,
     * past_kv_len].
     * @param output Output tensor [bsz, head_dim] with the same type as q_type.
     * @param lse Pre-allocated buffer [bsz] for the log-sum-exp of the
     * attention scores.
     * @param draft Pre-allocated temporary buffer. The buffer size should be
     * enough to hold (2 * bsz * past_kv_len + 6 * bsz * head_dim + 2 *
     *              past_kv_len * head_dim + past_kv_len * head_dim / 32) bytes.
     * @param rotary_angle Pointer to the rotary angle tensor.
     * @param rotary_cos Pointer to the cosine values for rotary embedding.
     * @param rotary_sin Pointer to the sine values for rotary embedding.
     */
    void attn_with_kvcache_one_block_(
        int head_dim, int bsz,
        ggml_type q_type, // GGML data type of `Q`, only supports fp16 and q8_0
        // [bsz, head_dim]
        // Quantization is always on the head_dim dimension (per_token). If
        // head_dim % 32 != 0, an error will be raised. The size must be bsz *
        // head_dim/32 * qtype_size.
        const void *q,

        int past_kv_len, int past_kv_offset,
        bool is_full_attn, // true indicates a full 1 mask
        // If is_full_attn = false, a bit matrix representing the mask is
        // passed. [bsz, past_kv_len]
        const uint8_t *attn_mask,

        ggml_type k_type, // GGML data type of `K Cache`, only supports fp16,
                          // q4_0, q8_0
        int k_quant_type, // 0 for per_token, 1 for per_channel, others raise an
                          // error
        // [seq_len, head_dim]
        // If quant_type == 0, head_dim % 32 must be 0.
        // If quant_type == 1, seq_len % 32 must be 0.
        const void *k_cache,

        // k_anchor_type must be fp16
        int num_k_anchor, // num_k_anchor == 0 indicates no anchor
        // [num_k_anchor, head_dim]
        const void *k_cache_anchors,
        // Each token is associated with the nearest previous position's anchor,
        // with the same distance.
        const int *k_cache_anchor_pos,

        // v_cache similar to k_cache
        ggml_type v_type, int v_quant_type,
        // [head_dim, seq_len]
        const void *v_cache, int num_v_anchor, const void *v_cache_anchors,
        const int *v_cache_anchor_pos,

        // Pre-allocated buffer for intermediate calculations [bsz,
        // past_kv_len]. No malloc is performed inside this function.
        float *attn_score,

        // Output: [bsz, head_dim], with the same type as q_type
        void *output,
        // [bsz]
        float *lse,

        // Pre-allocated temporary buffer with sufficient size:
        // (2 * bsz * past_kv_len + 6 * bsz * head_dim + 2 * past_kv_len *
        // head_dim + past_kv_len * head_dim / 32) bytes.
        void *draft,

        // Apply rotary embedding online
        const int *rotary_angle, const void *rotary_cos, const void *rotary_sin
        // rotary_cos=None,
        // rotary_sin=None,
        // cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
        // cache_batch_idx: Optional[torch.Tensor] = None,
        // rotary_interleaved=True,

        // // Not supported for now
        // window_size=(-1, -1),  # -1 means infinite context window
        // alibi_slopes=None,
    );
};

/**
 * @brief Scales a float32 vector by a given scalar value.
 *
 * This function multiplies each element of the input vector `y` by a scalar
 * `v`. It uses platform-specific optimizations if available, such as Apple's
 * Accelerate framework or SIMD instructions. If no specific optimization is
 * available, the function falls back to a simple scalar multiplication loop.
 *
 * @param n The number of elements in the vector `y`.
 * @param y The input vector to be scaled. The result will be stored in the same
 * vector.
 * @param v The scalar value by which to scale the vector.
 */
void ggml_vec_scale_f32(const int n, float *y, const float v);
#endif