/**
 * @Description  :
 * @Author       : Jianwei Dong
 * @Date         : 2024-08-26 22:47:06
 * @Version      : 1.0.0
 * @LastEditors  : Jianwei Dong
 * @LastEditTime : 2024-08-26 22:47:06
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "kvcache.h"

#include <chrono>

std::string ggml_type_to_string(ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32:
        return "GGML_TYPE_F32";
    case GGML_TYPE_F16:
        return "GGML_TYPE_F16";
    case GGML_TYPE_Q4_0:
        return "GGML_TYPE_Q4_0";
    case GGML_TYPE_Q8_0:
        return "GGML_TYPE_Q8_0";
    }
    return "UNDIFINED";
}
std::string AnchorTypeToString(AnchorType type) {
    switch (type) {
    case AnchorType::DYNAMIC:
        return "DYNAMIC";
    case AnchorType::BLOCK_MEAN:
        return "BLOCK_MEAN";
    case AnchorType::BLOCK_MAX:
        return "BLOCK_MAX";
    case AnchorType::FIXED_ANCHOR:
        return "FIXED_ANCHOR";
    case AnchorType::QUEST:
        return "QUEST";
    }
    return "UNDIFINED";
}
std::string RetrievalTypeToString(RetrievalType type) {
    switch (type) {
    case RetrievalType::LAYER:
        return "SHARED";
    case RetrievalType::KVHEAD:
        return "SEPARATE";
    case RetrievalType::QHEAD:
        return "INDIVIDUAL";
    }
    return "UNDIFINED";
}
KVCacheConfig::KVCacheConfig(int layer_num, int kv_head_num, int q_head_num,
                             int head_dim, int block_len, int anchor_num,
                             AnchorType anchor_type, ggml_type kv_type,
                             RetrievalType retrieval_type, int layer_step,
                             int token_step, int layer_offset,
                             int max_block_num, int max_batch_size,
                             int max_thread_num)
    : layer_num(layer_num), kv_head_num(kv_head_num), q_head_num(q_head_num),
      head_dim(head_dim), block_len(block_len), anchor_num(anchor_num),
      anchor_type(anchor_type), kv_type(kv_type),
      retrieval_type(retrieval_type), layer_step(layer_step),
      token_step(token_step), layer_offset(layer_offset),
      max_block_num(max_block_num), max_batch_size(max_batch_size),
      max_thread_num(max_thread_num) {
    printf(
        "layer_num: %d, kv_head_num: %d, q_head_num: %d, head_dim: %d, "
        "block_len: %d, anchor_num: %d, anchor_type: %s, kv_type: %s, "
        "retrieval_type: %s, layer_step: %d, token_step: %d, layer_offset: %d,"
        "max_block_num: %d, max_batch_size: %d, max_thread_num: %d\n",
        layer_num, kv_head_num, q_head_num, head_dim, block_len, anchor_num,
        AnchorTypeToString(anchor_type).c_str(),
        ggml_type_to_string(kv_type).c_str(),
        RetrievalTypeToString(retrieval_type).c_str(), layer_step, token_step,
        layer_offset, max_block_num, max_batch_size, max_thread_num);
    assert(q_head_num % kv_head_num == 0);
}
KVCache::KVCache(KVCacheConfig config) {
    this->config_ = config;

    n_gqa_ = config_.q_head_num / config_.kv_head_num;
    if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
        // TODO: Elegant implement
        k_cache_fp16_.resize(config_.layer_num);
        v_cache_fp16_.resize(config_.layer_num);
        selected_blocks_num_history_.resize(config_.layer_num /
                                            config_.layer_step);
        if (config_.retrieval_type == RetrievalType::LAYER) {
            selected_blocks_history_.resize(config_.layer_num /
                                            config_.layer_step);
        } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
            selected_blocks_history_kvhead_.resize(config_.layer_num /
                                                   config_.layer_step);
        } else if (config_.retrieval_type == RetrievalType::QHEAD) {
        }
    } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
        k_cache_q4.resize(config.layer_num);
        v_cache_q4.resize(config.layer_num);
    } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
        k_cache_q8.resize(config.layer_num);
        v_cache_q8.resize(config.layer_num);
    } else {
        assert(false);
    }
    anchor_.resize(config.layer_num * config.max_block_num * config.anchor_num *
                   config.q_head_num * config.head_dim);
    importance_.resize(config.layer_num);
    past_block_num_.resize(config.layer_num);
    for (int i = 0; i < config.layer_num; i++) {
        past_block_num_[i] = 0;
    }

    ThreadResize(config.max_thread_num);
    BatchResize(config.max_batch_size);
    BlockResize(config.max_block_num);
    q_fp32.resize(n_gqa_ * config.head_dim);
}

void KVCache::ThreadResize(int thread_num) {
    thread_local_output_q8_0_.resize(thread_num);
    thread_local_attn_score_.resize(thread_num);
    thread_local_output_fp32_.resize(thread_num);
    thread_local_attn_lse_.resize(thread_num);
    thread_local_cur_output_fp32_.resize(thread_num);
    thread_local_cur_attn_lse_.resize(thread_num);
    thread_local_draft_.resize(thread_num);
    thread_cur_head_idx_.resize(thread_num);
    thread_local_attn_mask_.resize(thread_num);
    for (int i = 0; i < thread_num; i++) {
        thread_local_output_q8_0_[i].resize(n_gqa_ * config_.head_dim / QK8_0);
        thread_local_attn_score_[i].resize(n_gqa_ * config_.block_len);
        thread_local_output_fp32_[i].resize(n_gqa_ * config_.head_dim);
        thread_local_attn_lse_[i].resize(n_gqa_);
        thread_local_cur_output_fp32_[i].resize(n_gqa_ * config_.head_dim);
        thread_local_cur_attn_lse_[i].resize(n_gqa_);
        thread_local_draft_[i].resize(
            2 * n_gqa_ * config_.block_len + 6 * n_gqa_ * config_.head_dim +
            2 * config_.block_len * config_.head_dim +
            config_.block_len * config_.head_dim / QK4_0);
        thread_local_attn_mask_[i].resize(config_.block_len / 8);
    }
}
void KVCache::BatchResize(int batch_size) {
    mutex_.resize(batch_size);
    q_q8_0_.resize(batch_size);
    q_fp32_.resize(batch_size);
    output_fp32_.resize(batch_size);
    attn_lse_.resize(batch_size);
    block_lse_.resize(batch_size);
    attn_sparsity_.resize(batch_size);

    if (config_.retrieval_type == RetrievalType::LAYER) {
        block_table_before_retrieval_.resize(batch_size);
        block_table_after_retrieval_.resize(batch_size);

        for (int i = 0; i < config_.layer_num / config_.layer_step; i++) {
            selected_blocks_history_[i].resize(batch_size);
        }

    } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
        block_table_before_retrieval_kvhead_.resize(batch_size);
        block_table_after_retrieval_kvhead_.resize(batch_size);
        for (int i = 0; i < config_.layer_num / config_.layer_step; i++) {
            selected_blocks_history_kvhead_[i].resize(batch_size);
        }
    } else if (config_.retrieval_type == RetrievalType::QHEAD) {
        block_table_before_retrieval_qhead_.resize(batch_size);
        block_table_after_retrieval_qhead_.resize(batch_size);
    }
    cache_seqlens_.resize(batch_size);
    if (config_.retrieval_type == RetrievalType::LAYER) {
        block_similar_.resize(batch_size);
    } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
        block_similar_kv_head_.resize(batch_size);
    } else if (config_.retrieval_type == RetrievalType::QHEAD) {
        block_similar_q_head_.resize(batch_size);
    }
    for (int i = 0; i < batch_size; i++) {
        top_similar_block_.resize(batch_size);

        mutex_[i].resize(config_.kv_head_num);
        q_q8_0_[i].resize(config_.kv_head_num);
        q_fp32_[i].resize(config_.kv_head_num);
        output_fp32_[i].resize(config_.kv_head_num);
        attn_lse_[i].resize(config_.kv_head_num);

        for (int j = 0; j < config_.kv_head_num; j++) {
            if (!mutex_[i][j]) {
                mutex_[i][j] = std::make_unique<std::mutex>();
            }
            q_q8_0_[i][j].resize(n_gqa_ * config_.head_dim / QK8_0);
            q_fp32_[i][j].resize(n_gqa_ * config_.head_dim);
            output_fp32_[i][j].resize(n_gqa_ * config_.head_dim);
            attn_lse_[i][j].resize(n_gqa_);
        }
    }
    avg_q.resize(batch_size);
    avg_q_fp16.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        attn_sparsity_[i].resize(config_.q_head_num);
        avg_q[i].resize(config_.q_head_num * config_.head_dim);
        avg_q_fp16[i].resize(config_.q_head_num * config_.head_dim);
    }
}

void KVCache::BlockResize(int max_block_num) {
    sin_.resize(max_block_num * config_.block_len);
    cos_.resize(max_block_num * config_.block_len);
    for (int i = 0; i < max_block_num * config_.block_len; i++) {
        sin_[i].resize(config_.head_dim);
        cos_[i].resize(config_.head_dim);
    }

    for (int i = 0; i < config_.layer_num / config_.layer_step; i++) {
        for (int j = 0; j < config_.max_batch_size; j++) {
            if (config_.retrieval_type == RetrievalType::LAYER) {
                selected_blocks_history_[i][j].resize(max_block_num);
            } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
                selected_blocks_history_kvhead_[i][j].resize(max_block_num);
                for (int k = 0; k < config_.max_block_num; k++) {
                    selected_blocks_history_kvhead_[i][j][k].resize(
                        config_.kv_head_num);
                }
            } else if (config_.retrieval_type == RetrievalType::QHEAD) {
            }
        }
    }

    for (int layer_id = 0; layer_id < config_.layer_num; layer_id++) {
        importance_[layer_id].resize(max_block_num);

        if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            // TODO: Elegant implement
            k_cache_fp16_[layer_id].resize(config_.kv_head_num);
            v_cache_fp16_[layer_id].resize(config_.kv_head_num);

            for (int i = 0; i < config_.kv_head_num; i++) {
                k_cache_fp16_[layer_id][i].resize(max_block_num);
                v_cache_fp16_[layer_id][i].resize(max_block_num);

                for (int j = 0; j < max_block_num; j++) {
                    k_cache_fp16_[layer_id][i][j].resize(config_.block_len *
                                                         config_.head_dim);
                    v_cache_fp16_[layer_id][i][j].resize(config_.block_len *
                                                         config_.head_dim);
                }
            }

        } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            k_cache_q4[layer_id].resize(config_.kv_head_num);
            v_cache_q4[layer_id].resize(config_.kv_head_num);
            for (int i = 0; i < config_.kv_head_num; i++) {
                k_cache_q4[layer_id][i].resize(max_block_num);
                v_cache_q4[layer_id][i].resize(max_block_num);

                for (int j = 0; j < max_block_num; j++) {
                    k_cache_q4[layer_id][i][j].resize(config_.block_len *
                                                      config_.head_dim / 32);
                    v_cache_q4[layer_id][i][j].resize(config_.block_len *
                                                      config_.head_dim / 32);
                }
            }
        } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            k_cache_q8[layer_id].resize(config_.kv_head_num);
            v_cache_q8[layer_id].resize(config_.kv_head_num);
            for (int i = 0; i < config_.kv_head_num; i++) {
                k_cache_q8[layer_id][i].resize(max_block_num);
                v_cache_q8[layer_id][i].resize(max_block_num);

                for (int j = 0; j < max_block_num; j++) {
                    k_cache_q8[layer_id][i][j].resize(config_.block_len *
                                                      config_.head_dim / 32);
                    v_cache_q8[layer_id][i][j].resize(config_.block_len *
                                                      config_.head_dim / 32);
                }
            }
        } else {
            assert(false);
        }
        for (int i = 0; i < config_.max_batch_size; i++) {
            if (config_.retrieval_type == RetrievalType::LAYER) {
                block_similar_[i].resize(max_block_num);
                block_table_before_retrieval_[i].resize(max_block_num);
                block_table_after_retrieval_[i].resize(max_block_num);
            } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
                block_similar_kv_head_[i].resize(max_block_num);
                block_table_before_retrieval_kvhead_[i].resize(max_block_num);
                block_table_after_retrieval_kvhead_[i].resize(max_block_num);
                for (int j = 0; j < max_block_num; j++) {
                    block_similar_kv_head_[i][j].resize(config_.kv_head_num);
                    block_table_before_retrieval_kvhead_[i][j].resize(
                        config_.kv_head_num);
                    block_table_after_retrieval_kvhead_[i][j].resize(
                        config_.kv_head_num);
                }
            } else if (config_.retrieval_type == RetrievalType::QHEAD) {
                block_similar_q_head_[i].resize(max_block_num);
                block_table_before_retrieval_qhead_[i].resize(max_block_num);
                block_table_after_retrieval_qhead_[i].resize(max_block_num);
                for (int j = 0; j < max_block_num; j++) {
                    block_similar_q_head_[i][j].resize(config_.q_head_num);
                    block_table_before_retrieval_qhead_[i][j].resize(
                        config_.q_head_num);
                    block_table_after_retrieval_qhead_[i][j].resize(
                        config_.q_head_num);
                }
            }
            block_lse_[i].resize(max_block_num);
            for (int j = 0; j < max_block_num; j++) {
                block_lse_[i][j].resize(config_.q_head_num);
            }
        }

        for (int i = 0; i < max_block_num; i++) {
            importance_[layer_id][i].resize(config_.block_len);
            for (int j = 0; j < config_.block_len; j++) {
                importance_[layer_id][i][j].resize(config_.q_head_num);
            }
        }
    }
}

void KVCache::calc_anchor_all_layers(int *block_table, int *cache_seqlens,
                                     int batch_size, int max_block_num,
                                     Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    // Each task updates the importance of a certain block
    seq_len_ = config_.block_len;
    backend->do_work_stealing_job(
        config_.layer_num * batch_size * max_block_num, nullptr,
        [&](int task_id) {
            int layer_id = task_id / (batch_size * max_block_num);
            int batch_id = (task_id / max_block_num) % batch_size;
            int block_id = task_id % max_block_num;
            // If the block is out of the sequence length, skip it. In
            // particular, the last block of the sequence that is shorter than
            // the block length should be skipped.

            if (cache_seqlens[batch_id] / config_.block_len < block_id) {
                return;
            }
            int block_idx = block_table[batch_id * max_block_num + block_id];

            std::vector<float> block_fp32(32);
            if (config_.anchor_type == AnchorType::DYNAMIC) {

                // clear anchor_
                for (int anchor_id = 0; anchor_id < 1; anchor_id++) {
                    for (int head_id = 0; head_id < config_.q_head_num;
                         head_id++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            anchor_[layer_id * config_.max_block_num *
                                        config_.anchor_num *
                                        config_.q_head_num * config_.head_dim +
                                    block_idx * config_.anchor_num *
                                        config_.q_head_num * config_.head_dim +
                                    anchor_id * config_.q_head_num *
                                        config_.head_dim +
                                    head_id * config_.head_dim + l] = 0;
                        }
                    }
                }

                // find top anchor_num importances and their corresponding
                // positions in the importance_ tensor
                // TODO: Move top_importances to the class member to avoid
                // repeated memory allocation
                std::priority_queue<
                    std::pair<float, std::pair<int, int>>,
                    std::vector<std::pair<float, std::pair<int, int>>>,
                    std::greater<>>
                    top_importances;
                for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
                    for (int k = 0; k < seq_len_; k++) {
                        top_importances.push(std::make_pair(
                            GGML_FP16_TO_FP32(
                                importance_[layer_id][block_idx][k][head_id]),
                            std::make_pair(block_idx, k)));
                        // TODO: change to config_ item
                        if (top_importances.size() > config_.anchor_num) {
                            top_importances.pop();
                        }
                    }

                    // fill anchor_

                    for (int l = 0; l < config_.head_dim; l++) {
                        anchor_[layer_id * config_.max_block_num *
                                    config_.anchor_num * config_.q_head_num *
                                    config_.head_dim +
                                block_idx * config_.anchor_num *
                                    config_.q_head_num * config_.head_dim +
                                0 * config_.q_head_num * config_.head_dim +
                                head_id * config_.head_dim + l] = 0;
                    }
                    for (int k = 0; k < config_.anchor_num; k++) {
                        int top_indice = top_importances.top().second.second;
                        int top_block_idx = top_importances.top().second.first;

                        if (config_.kv_type == ggml_type::GGML_TYPE_F16) {

                            for (int l = 0; l < config_.head_dim; l++) {
                                anchor_[layer_id * config_.max_block_num *
                                            config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        top_block_idx * config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        0 * config_.q_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(
                                        GGML_FP16_TO_FP32(
                                            anchor_[layer_id *
                                                        config_.max_block_num *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    top_block_idx *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    0 * config_.q_head_num *
                                                        config_.head_dim +
                                                    head_id * config_.head_dim +
                                                    l]) +
                                        GGML_FP16_TO_FP32(
                                            k_cache_fp16_[layer_id]
                                                         [head_id / n_gqa_]
                                                         [top_block_idx]
                                                         [top_indice *
                                                              config_.head_dim +
                                                          l]));
                            }

                        } else if (config_.kv_type ==
                                   ggml_type::GGML_TYPE_Q4_0) {
                            for (int l = 0; l < config_.head_dim / 32; l++) {
                                block_q4_0 block = k_cache_q4
                                    [layer_id][head_id / n_gqa_][top_block_idx]
                                    [top_indice * config_.head_dim / 32 + l];
                                dequantize_row_q4_0(&block, block_fp32.data(),
                                                    32);
                                for (int m = 0; m < 32; m++) {
                                    anchor_[layer_id * config_.max_block_num *
                                                config_.anchor_num *
                                                config_.q_head_num *
                                                config_.head_dim +
                                            top_block_idx * config_.anchor_num *
                                                config_.q_head_num *
                                                config_.head_dim +
                                            0 * config_.q_head_num *
                                                config_.head_dim +
                                            head_id * config_.head_dim +
                                            l * 32 + m] =
                                        GGML_FP32_TO_FP16(
                                            block_fp32[m] / 4 +
                                            GGML_FP16_TO_FP32(
                                                anchor_[layer_id *
                                                            config_
                                                                .max_block_num *
                                                            config_.anchor_num *
                                                            config_.q_head_num *
                                                            config_.head_dim +
                                                        top_block_idx *
                                                            config_.anchor_num *
                                                            config_.q_head_num *
                                                            config_.head_dim +
                                                        0 * config_.q_head_num *
                                                            config_.head_dim +
                                                        head_id *
                                                            config_.head_dim +
                                                        l * 32 + m]));
                                }
                            }
                        } else if (config_.kv_type ==
                                   ggml_type::GGML_TYPE_Q8_0) {
                            for (int l = 0; l < config_.head_dim / 32; l++) {
                                block_q8_0 block = k_cache_q8
                                    [layer_id][head_id / n_gqa_][top_block_idx]
                                    [top_indice * config_.head_dim / 32 + l];
                                dequantize_row_q8_0(&block, block_fp32.data(),
                                                    32);
                                for (int m = 0; m < 32; m++) {
                                    anchor_[layer_id * config_.max_block_num *
                                                config_.anchor_num *
                                                config_.q_head_num *
                                                config_.head_dim +
                                            top_block_idx * config_.anchor_num *
                                                config_.q_head_num *
                                                config_.head_dim +
                                            0 * config_.q_head_num *
                                                config_.head_dim +
                                            head_id * config_.head_dim +
                                            l * 32 + m] =
                                        GGML_FP32_TO_FP16(
                                            block_fp32[m] / 4 +
                                            GGML_FP16_TO_FP32(
                                                anchor_[layer_id *
                                                            config_
                                                                .max_block_num *
                                                            config_.anchor_num *
                                                            config_.q_head_num *
                                                            config_.head_dim +
                                                        top_block_idx *
                                                            config_.anchor_num *
                                                            config_.q_head_num *
                                                            config_.head_dim +
                                                        0 * config_.q_head_num *
                                                            config_.head_dim +
                                                        head_id *
                                                            config_.head_dim +
                                                        l * 32 + m]));
                                }
                            }
                        }
                        top_importances.pop();
                    }
                }
            } else if (config_.anchor_type == AnchorType::BLOCK_MEAN) {
                // clear anchor_
                for (int anchor_id = 0; anchor_id < config_.anchor_num;
                     anchor_id++) {
                    for (int head_id = 0; head_id < config_.q_head_num;
                         head_id++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            anchor_[layer_id * config_.max_block_num *
                                        config_.anchor_num *
                                        config_.q_head_num * config_.head_dim +
                                    block_idx * config_.anchor_num *
                                        config_.q_head_num * config_.head_dim +
                                    anchor_id * config_.q_head_num *
                                        config_.head_dim +
                                    head_id * config_.head_dim + l] = 0;
                        }
                    }
                }

                // fill anchor_
                if (config_.kv_type == ggml_type::GGML_TYPE_F16) {

                    for (int head_id = 0; head_id < config_.q_head_num;
                         head_id++) {
                        for (int k = 0; k < config_.block_len; k++) {
                            for (int l = 0; l < config_.head_dim; l++) {
                                anchor_[layer_id * config_.max_block_num *
                                            config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        block_idx * config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        0 * config_.q_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(
                                        GGML_FP16_TO_FP32(
                                            anchor_[layer_id *
                                                        config_.max_block_num *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    block_idx *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    0 * config_.q_head_num *
                                                        config_.head_dim +
                                                    head_id * config_.head_dim +
                                                    l]) +
                                        GGML_FP16_TO_FP32(
                                            k_cache_fp16_[layer_id]
                                                         [head_id / n_gqa_]
                                                         [block_idx]
                                                         [k * config_.head_dim +
                                                          l]) /
                                            config_.block_len);
                            }
                        }
                    }
                }
            } else if (config_.anchor_type == AnchorType::BLOCK_MAX) {
                // clear anchor_
                for (int anchor_id = 0; anchor_id < config_.anchor_num;
                     anchor_id++) {
                    for (int head_id = 0; head_id < config_.q_head_num;
                         head_id++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            anchor_[layer_id * config_.max_block_num *
                                        config_.anchor_num *
                                        config_.q_head_num * config_.head_dim +
                                    block_idx * config_.anchor_num *
                                        config_.q_head_num * config_.head_dim +
                                    anchor_id * config_.q_head_num *
                                        config_.head_dim +
                                    head_id * config_.head_dim + l] = 0;
                        }
                    }
                }

                // fill anchor_
                if (config_.kv_type == ggml_type::GGML_TYPE_F16) {

                    for (int head_id = 0; head_id < config_.q_head_num;
                         head_id++) {
                        for (int k = 0; k < config_.block_len; k++) {
                            for (int l = 0; l < config_.head_dim; l++) {
                                anchor_[layer_id * config_.max_block_num *
                                            config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        block_idx * config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        0 * config_.q_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(std::max(
                                        GGML_FP16_TO_FP32(
                                            anchor_[layer_id *
                                                        config_.max_block_num *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    block_idx *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    0 * config_.q_head_num *
                                                        config_.head_dim +
                                                    head_id * config_.head_dim +
                                                    l]),
                                        GGML_FP16_TO_FP32(
                                            k_cache_fp16_
                                                [layer_id][head_id / n_gqa_]
                                                [block_idx]
                                                [k * config_.head_dim + l])));
                            }
                        }
                    }
                }
            } else if (config_.anchor_type == AnchorType::FIXED_ANCHOR) {
                // clear anchor_
                for (int anchor_id = 0; anchor_id < 1; anchor_id++) {
                    for (int head_id = 0; head_id < config_.q_head_num;
                         head_id++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            anchor_[layer_id * config_.max_block_num *
                                        config_.anchor_num *
                                        config_.q_head_num * config_.head_dim +
                                    block_idx * config_.anchor_num *
                                        config_.q_head_num * config_.head_dim +
                                    anchor_id * config_.q_head_num *
                                        config_.head_dim +
                                    head_id * config_.head_dim + l] = 0;
                        }
                    }
                }

                // fill anchor_
                if (config_.kv_type == ggml_type::GGML_TYPE_F16) {

                    int stride = config_.block_len / config_.anchor_num;
                    for (int head_id = 0; head_id < config_.q_head_num;
                         head_id++) {
                        for (int k = 0, tot = 0;
                             k < config_.block_len, tot < config_.anchor_num;
                             k += stride, tot++) {
                            for (int l = 0; l < config_.head_dim; l++) {
                                anchor_[layer_id * config_.max_block_num *
                                            config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        block_idx * config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        0 * config_.q_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(
                                        GGML_FP16_TO_FP32(
                                            anchor_[layer_id *
                                                        config_.max_block_num *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    block_idx *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    0 * config_.q_head_num *
                                                        config_.head_dim +
                                                    head_id * config_.head_dim +
                                                    l]) +
                                        GGML_FP16_TO_FP32(
                                            k_cache_fp16_[layer_id]
                                                         [head_id / n_gqa_]
                                                         [block_idx]
                                                         [k * config_.head_dim +
                                                          l]) /
                                            config_.anchor_num);
                            }
                        }
                    }
                }

            } else if (config_.anchor_type == AnchorType::QUEST) {
                // clear anchor_
                for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
                    for (int l = 0; l < config_.head_dim; l++) {
                        anchor_[layer_id * config_.max_block_num *
                                    config_.anchor_num * config_.q_head_num *
                                    config_.head_dim +
                                block_idx * config_.anchor_num *
                                    config_.q_head_num * config_.head_dim +
                                1 * config_.q_head_num * config_.head_dim +
                                head_id * config_.head_dim + l] =
                            GGML_FP32_TO_FP16(
                                std::numeric_limits<float>::max());

                        anchor_[layer_id * config_.max_block_num *
                                    config_.anchor_num * config_.q_head_num *
                                    config_.head_dim +
                                block_idx * config_.anchor_num *
                                    config_.q_head_num * config_.head_dim +
                                0 * config_.q_head_num * config_.head_dim +
                                head_id * config_.head_dim + l] =
                            GGML_FP32_TO_FP16(
                                std::numeric_limits<float>::min());
                    }
                }

                // fill anchor_

                if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
                    for (int indice = 0; indice < seq_len_; indice++) {
                        for (int head_id = 0; head_id < config_.kv_head_num;
                             head_id++) {
                            for (int l = 0; l < config_.head_dim; l++) {
                                anchor_[layer_id * config_.max_block_num *
                                            config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        block_idx * config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        0 * config_.q_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(std::max(
                                        GGML_FP16_TO_FP32(
                                            k_cache_fp16_
                                                [layer_id][head_id][block_idx]
                                                [indice * config_.head_dim +
                                                 l]),
                                        GGML_FP16_TO_FP32(
                                            anchor_[layer_id *
                                                        config_.max_block_num *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    block_idx *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    0 * config_.q_head_num *
                                                        config_.head_dim +
                                                    head_id * config_.head_dim +
                                                    l])));

                                anchor_[layer_id * config_.max_block_num *
                                            config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        block_idx * config_.anchor_num *
                                            config_.q_head_num *
                                            config_.head_dim +
                                        1 * config_.q_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(std::min(
                                        GGML_FP16_TO_FP32(
                                            k_cache_fp16_
                                                [layer_id][head_id][block_idx]
                                                [indice * config_.head_dim +
                                                 l]),
                                        GGML_FP16_TO_FP32(
                                            anchor_[layer_id *
                                                        config_.max_block_num *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    block_idx *
                                                        config_.anchor_num *
                                                        config_.q_head_num *
                                                        config_.head_dim +
                                                    1 * config_.q_head_num *
                                                        config_.head_dim +
                                                    head_id * config_.head_dim +
                                                    l])));
                            }
                        }
                    }

                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                    for (int indice = 0; indice < seq_len_; indice++) {
                        for (int head_id = 0; head_id < config_.kv_head_num;
                             head_id++) {
                            for (int l = 0; l < config_.head_dim / 32; l++) {
                                block_q4_0 block =
                                    k_cache_q4[layer_id][head_id][block_idx]
                                              [indice * config_.head_dim / 32 +
                                               l];
                                dequantize_row_q4_0(&block, block_fp32.data(),
                                                    32);

                                for (int m = 0; m < 32; m++) {
                                    for (int gqa_idx = 0; gqa_idx < n_gqa_;
                                         gqa_idx++) {

                                        anchor_[layer_id *
                                                    config_.max_block_num *
                                                    config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                block_idx * config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                0 * config_.q_head_num *
                                                    config_.head_dim +
                                                head_id * config_.head_dim +
                                                l * 32 + m] =
                                            GGML_FP32_TO_FP16(std::max(
                                                block_fp32[m],
                                                GGML_FP16_TO_FP32(
                                                    anchor_
                                                        [layer_id *
                                                             config_
                                                                 .max_block_num *
                                                             config_
                                                                 .anchor_num *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         block_idx *
                                                             config_
                                                                 .anchor_num *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         0 *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         head_id *
                                                             config_.head_dim +
                                                         l * 32 + m])));

                                        anchor_[layer_id *
                                                    config_.max_block_num *
                                                    config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                block_idx * config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                1 * config_.q_head_num *
                                                    config_.head_dim +
                                                head_id * config_.head_dim +
                                                l * 32 + m] =
                                            GGML_FP32_TO_FP16(std::min(
                                                block_fp32[m],
                                                GGML_FP16_TO_FP32(
                                                    anchor_
                                                        [layer_id *
                                                             config_
                                                                 .max_block_num *
                                                             config_
                                                                 .anchor_num *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         block_idx *
                                                             config_
                                                                 .anchor_num *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         1 *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         head_id *
                                                             config_.head_dim +
                                                         l * 32 + m])));
                                    }
                                }
                            }
                        }
                    }
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                    for (int indice = 0; indice < seq_len_; indice++) {
                        for (int head_id = 0; head_id < config_.kv_head_num;
                             head_id++) {
                            for (int l = 0; l < config_.head_dim / 32; l++) {
                                block_q8_0 block =
                                    k_cache_q8[layer_id][head_id][block_idx]
                                              [indice * config_.head_dim / 32 +
                                               l];
                                dequantize_row_q8_0(&block, block_fp32.data(),
                                                    32);

                                for (int m = 0; m < 32; m++) {
                                    for (int gqa_idx = 0; gqa_idx < n_gqa_;
                                         gqa_idx++) {

                                        anchor_[layer_id *
                                                    config_.max_block_num *
                                                    config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                block_idx * config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                0 * config_.q_head_num *
                                                    config_.head_dim +
                                                head_id * config_.head_dim +
                                                l * 32 + m] =
                                            GGML_FP32_TO_FP16(std::max(
                                                block_fp32[m],
                                                GGML_FP16_TO_FP32(
                                                    anchor_
                                                        [layer_id *
                                                             config_
                                                                 .max_block_num *
                                                             config_
                                                                 .anchor_num *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         block_idx *
                                                             config_
                                                                 .anchor_num *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         0 *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         head_id *
                                                             config_.head_dim +
                                                         l * 32 + m])));

                                        anchor_[layer_id *
                                                    config_.max_block_num *
                                                    config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                block_idx * config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                1 * config_.q_head_num *
                                                    config_.head_dim +
                                                head_id * config_.head_dim +
                                                l * 32 + m] =
                                            GGML_FP32_TO_FP16(std::min(
                                                block_fp32[m],
                                                GGML_FP16_TO_FP32(
                                                    anchor_
                                                        [layer_id *
                                                             config_
                                                                 .max_block_num *
                                                             config_
                                                                 .anchor_num *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         block_idx *
                                                             config_
                                                                 .anchor_num *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         1 *
                                                             config_
                                                                 .q_head_num *
                                                             config_.head_dim +
                                                         head_id *
                                                             config_.head_dim +
                                                         l * 32 + m])));
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                assert(false);
            }
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    //    printf("time of calc_anchor_all_layers: %f s\n", duration.count());
}

void KVCache::clear_importance_all_layers(int *block_table, int *cache_seqlens,
                                          int batch_size, int max_block_num,
                                          Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    // Each task updates the importance of a certain block
    seq_len_ = config_.block_len;
    backend->do_work_stealing_job(
        config_.layer_num * batch_size * max_block_num, nullptr,
        [&](int task_id) {
            int layer_id = task_id / (batch_size * max_block_num);
            int batch_id = (task_id / max_block_num) % batch_size;
            int block_id = task_id % max_block_num;
            // If the block is out of the sequence length, skip it. In
            // particular, the last block of the sequence that is shorter than
            // the block length should be skipped.

            if (cache_seqlens[batch_id] / config_.block_len < block_id) {
                return;
            }
            int block_idx = block_table[batch_id * max_block_num + block_id];

            if (config_.anchor_type == AnchorType::DYNAMIC) {

                // clear anchor_
                for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
                    for (int l = 0; l < config_.block_len; l++) {
                        importance_[layer_id][block_idx][l][head_id] = 0;
                    }
                }
            }
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    //    printf("time of clear_importance_all_layerssssss: %f s\n",
    //    duration.count());
}

void KVCache::clear_kvcache_all_layers(int *block_table, int *cache_seqlens,
                                       int batch_size, int max_block_num,
                                       Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    // Each task updates the importance of a certain block
    seq_len_ = config_.block_len;
    backend->do_work_stealing_job(
        config_.layer_num * batch_size * max_block_num * config_.kv_head_num,
        nullptr,
        [&](int task_id) {
            int layer_id =
                task_id / (batch_size * max_block_num * config_.kv_head_num);
            int batch_id =
                (task_id / (max_block_num * config_.kv_head_num)) % batch_size;
            int block_id = task_id / config_.kv_head_num % max_block_num;
            int head_id = task_id % config_.kv_head_num;
            // If the block is out of the sequence length, skip it. In
            // particular, the last block of the sequence that is shorter than
            // the block length should be skipped.
            if (cache_seqlens[batch_id] / config_.block_len < block_id) {
                return;
            }
            int block_idx = block_table[batch_id * max_block_num + block_id];

            if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
                for (int l = 0; l < config_.block_len * config_.head_dim; l++) {
                    k_cache_fp16_[layer_id][head_id][block_idx][l] = 0;
                    v_cache_fp16_[layer_id][head_id][block_idx][l] = 0;
                }
            } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                for (int l = 0; l < config_.block_len * config_.head_dim / 32;
                     l++) {
                    k_cache_q4[layer_id][head_id][block_idx][l].d = 0;
                    v_cache_q4[layer_id][head_id][block_idx][l].d = 0;
                }
            } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                for (int l = 0; l < config_.block_len * config_.head_dim / 32;
                     l++) {
                    k_cache_q8[layer_id][head_id][block_idx][l].d = 0;
                    v_cache_q8[layer_id][head_id][block_idx][l].d = 0;
                }
            }
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    //    printf("time of clear_kvcache_all_layers: %f s\n", duration.count());
}

void KVCache::get_sincos(ggml_fp16_t *sin, ggml_fp16_t *cos, int seqlen) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    const uint16_t *sin_data = const_cast<const uint16_t *>(sin);
    const uint16_t *cos_data = const_cast<const uint16_t *>(cos);

    for (int i = 0; i < seqlen; i++) {
        for (int j = 0; j < config_.head_dim; j++) {
            sin_[i][j] = sin_data[i * config_.head_dim + j];
            cos_[i][j] = cos_data[i * config_.head_dim + j];
        }
    }

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("time of get_sincos: %f s\n", duration.count());
}

void ggml_vec_scale_f32(const int n, float *y, const float v) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

            GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}