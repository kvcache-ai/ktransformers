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

void KVCache::get_anchor_one_block(ggml_fp16_t *anchor, int layer_id,
                                   int block_idx, Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    anchor_data_ = const_cast<uint16_t *>(anchor);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of reading anchor: %f s\n", layer_id,
           block_idx, duration.count());
}

void KVCache::update_anchor_one_block(const ggml_fp16_t *anchor, int layer_id,
                                      int block_idx, Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    anchor_data_ = const_cast<uint16_t *>(anchor);

    // Each task updates the anchor of a certain position
    // backend->do_work_stealing_job(config_.anchor_num, [&](int task_id) {
    //     int k = task_id % config_.anchor_num;
    //     int head_id = task_id / config_.anchor_num;
    //     memcpy(anchor_[layer_id_][head_id][block_idx].data() +
    //                k * config_.head_dim,
    //            anchor_data_ + k * config_.head_dim,
    //            sizeof(uint16_t) * config_.head_dim);
    // });

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of writting anchor: %f s\n", layer_id,
           block_idx, duration.count());
}

void KVCache::update_importance_one_block(const ggml_fp16_t *importance,
                                          int layer_id, int block_idx,
                                          Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    importance_data_ = const_cast<uint16_t *>(importance);

    // Each task updates the importance of a certain position
    backend->do_work_stealing_job(
        config_.block_len, nullptr,
        [&](int task_id) {
            int k = task_id;
            memcpy(importance_[layer_id_][block_idx].data() + k,
                   importance_data_ + k, sizeof(uint16_t));
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of writting importance: %f s\n", layer_id,
           block_idx, duration.count());
}

void KVCache::get_importance_one_block(ggml_fp16_t *importance, int layer_id,
                                       int block_idx, Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    importance_data_ = const_cast<uint16_t *>(importance);

    // Each task updates the importance of a certain position
    backend->do_work_stealing_job(
        config_.block_len, nullptr,
        [&](int task_id) {
            int k = task_id;
            memcpy(importance_data_ + k,
                   importance_[layer_id_][block_idx].data() + k,
                   sizeof(uint16_t));
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of reading importance: %f s\n", layer_id,
           block_idx, duration.count());
}

void KVCache::update_kvcache_one_block_fp16(const ggml_fp16_t *k_in,
                                            const ggml_fp16_t *v_in,
                                            int layer_id, int block_idx,
                                            Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    k_data_ = const_cast<uint16_t *>(k_in);
    v_data_ = const_cast<uint16_t *>(v_in);

    int new_block_num = std::max((int)past_block_num_[layer_id], block_idx + 1);

    importance_[layer_id_].resize(new_block_num);

    for (int i = 0; i < config_.kv_head_num; i++) {
        k_cache_q4[layer_id][i].resize(new_block_num);
        v_cache_q4[layer_id][i].resize(new_block_num);
        // anchor_[layer_id][i].resize(new_block_num);
    }

    for (int i = 0; i < new_block_num; i++) {
        importance_[layer_id][i].resize(config_.block_len);
    }

    // Each task updates the k cache or v cache of a certain header
    backend->do_work_stealing_job(
        config_.kv_head_num * 2, nullptr,
        [&](int task_id) {
            std::vector<float> block_fp32(32);
            int head_id = task_id / 2;
            if (task_id & 1) {
                // fill k_cache_
                k_cache_q4[layer_id_][head_id][block_idx].resize(
                    config_.block_len * config_.head_dim / 32);
                for (int k = 0; k < config_.block_len; k++) {
                    for (int l = 0; l < config_.head_dim / 32; l++) {
                        block_q4_0 block;
                        for (int m = 0; m < 32; m++) {

                            block_fp32[m] = GGML_FP16_TO_FP32(
                                k_data_[((0 * config_.kv_head_num + head_id) *
                                             seq_len_ +
                                         0 * config_.block_len + k) *
                                            config_.head_dim +
                                        l * 32 + m]);
                        }
                        quantize_row_q4_0(block_fp32.data(), &block, 32);
                        k_cache_q4[layer_id_][head_id][block_idx]
                                  [k * config_.head_dim / 32 + l] = block;
                    }
                }
            } else {
                // fill v_cache_
                v_cache_q4[layer_id_][head_id][block_idx].resize(
                    config_.head_dim * config_.block_len / 32);
                for (int k = 0; k < config_.block_len / 32; k++) {
                    for (int l = 0; l < config_.head_dim; l++) {
                        block_q4_0 block;
                        for (int m = 0; m < 32; m++) {

                            block_fp32[m] = GGML_FP16_TO_FP32(
                                v_data_[((0 * config_.kv_head_num + head_id) *
                                             seq_len_ +
                                         0 * config_.block_len + k * 32 + m) *
                                            config_.head_dim +
                                        l]);
                        }
                        quantize_row_q4_0(block_fp32.data(), &block, 32);
                        v_cache_q4[layer_id_][head_id][block_idx]
                                  [l * config_.block_len / 32 + k] = block;
                    }
                }
            }
        },
        nullptr);
    past_block_num_[layer_id] = new_block_num;

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of writting KV Cache: %f s\n", layer_id,
           block_idx, duration.count());
    // printf("get_one_block_fp16 duration: %ld\n", duration);
}

void KVCache::get_kvcache_one_block_fp16(ggml_fp16_t *k_in, ggml_fp16_t *v_in,
                                         int layer_id, int block_idx,
                                         Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    seq_len_ = config_.block_len;
    k_data_ = reinterpret_cast<uint16_t *>(k_in);
    v_data_ = reinterpret_cast<uint16_t *>(v_in);

    // printf("layer_id: %d, block_idx: %d\n", layer_id, block_idx);
    // Each task gets the k cache or v cache of a certain header
    backend->do_work_stealing_job(
        config_.kv_head_num * 2, nullptr,
        [&](int task_id) {
            std::vector<float> block_fp32(32);
            int head_id = task_id / 2;
            if (task_id & 1) {
                // get k_cache_
                for (int k = 0; k < config_.block_len; k++) {
                    for (int l = 0; l < config_.head_dim / 32; l++) {
                        block_q4_0 block =
                            k_cache_q4[layer_id_][head_id][block_idx]
                                      [k * config_.head_dim / 32 + l];
                        dequantize_row_q4_0(&block, block_fp32.data(), 32);
                        for (int m = 0; m < 32; m++) {

                            k_data_[((0 * config_.kv_head_num + head_id) *
                                         seq_len_ +
                                     0 * config_.block_len + k) *
                                        config_.head_dim +
                                    l * 32 + m] =
                                GGML_FP32_TO_FP16(block_fp32[m]);
                        }
                    }
                }
            } else {
                // get v_cache_
                for (int k = 0; k < config_.block_len / 32; k++) {
                    for (int l = 0; l < config_.head_dim; l++) {
                        block_q4_0 block =
                            v_cache_q4[layer_id_][head_id][block_idx]
                                      [l * config_.block_len / 32 + k];
                        dequantize_row_q4_0(&block, block_fp32.data(), 32);
                        for (int m = 0; m < 32; m++) {

                            v_data_[((0 * config_.kv_head_num + head_id) *
                                         seq_len_ +
                                     0 * config_.block_len + k * 32 + m) *
                                        config_.head_dim +
                                    l] = GGML_FP32_TO_FP16(block_fp32[m]);
                        }
                    }
                }
            }
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of reading KV Cache: %f s\n", layer_id,
           block_idx, duration.count());
    // printf("get_one_block_fp16 duration: %ld\n", duration);
}

// k_in: (batch_size, seq_len, head_num, head_dim)
// v_in: (batch_size, seq_len, head_num, head_dim)
void KVCache::get_and_update_kvcache_fp16(ggml_fp16_t *k_in, ggml_fp16_t *v_in,
                                          int layer_id, int *block_table,
                                          int batch_size, int max_block_num,
                                          int *cache_seqlens, int q_len,
                                          Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    k_data_ = const_cast<uint16_t *>(k_in);
    v_data_ = const_cast<uint16_t *>(v_in);

    // Each task updates the k cache and v cache of a certain header
    backend->do_work_stealing_job(
        config_.kv_head_num * max_block_num * batch_size, nullptr,
        [&](int task_id) {
            // printf("block_idx: %d, task_id: %d\n", block_idx, task_id);
            std::vector<float> block_fp32(32);
            int batch_id = task_id / (config_.kv_head_num * max_block_num);
            int block_id = (task_id / config_.kv_head_num) % max_block_num;
            int head_id = task_id % config_.kv_head_num;
            int block_idx = block_table[batch_id * max_block_num + block_id];
            int seq_len = cache_seqlens[batch_id];
            int block_l = block_id * config_.block_len;
            int block_r = block_id * config_.block_len + config_.block_len;

            if (block_l < seq_len) {
                if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
                    for (int k = 0; k < config_.block_len; k++) {
                        if (block_id * config_.block_len + k >= seq_len)
                            break;
                        for (int l = 0; l < config_.head_dim; l++) {
                            k_data_
                                [batch_id *
                                     (max_block_num * config_.block_len *
                                      config_.kv_head_num * config_.head_dim) +
                                 block_id *
                                     (config_.block_len * config_.kv_head_num *
                                      config_.head_dim) +
                                 k * (config_.kv_head_num * config_.head_dim) +
                                 head_id * config_.head_dim + l] =
                                    k_cache_fp16_[layer_id_][head_id][block_idx]
                                                 [k * config_.head_dim + l];
                            v_data_
                                [batch_id *
                                     (max_block_num * config_.block_len *
                                      config_.kv_head_num * config_.head_dim) +
                                 block_id *
                                     (config_.block_len * config_.kv_head_num *
                                      config_.head_dim) +
                                 k * (config_.kv_head_num * config_.head_dim) +
                                 head_id * config_.head_dim + l] =
                                    v_cache_fp16_[layer_id_][head_id][block_idx]
                                                 [l * config_.block_len + k];
                        }
                    }
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                    // get k_cache_
                    for (int k = 0; k < config_.block_len; k++) {
                        if (block_id * config_.block_len + k >= seq_len)
                            break;
                        for (int l = 0; l < config_.head_dim / 32; l++) {
                            block_q4_0 block =
                                k_cache_q4[layer_id_][head_id][block_idx]
                                          [k * config_.head_dim / 32 + l];
                            dequantize_row_q4_0(&block, block_fp32.data(), 32);
                            for (int m = 0; m < 32; m++) {

                                k_data_[batch_id *
                                            (max_block_num * config_.block_len *
                                             config_.kv_head_num *
                                             config_.head_dim) +
                                        block_id * (config_.block_len *
                                                    config_.kv_head_num *
                                                    config_.head_dim) +
                                        k * (config_.kv_head_num *
                                             config_.head_dim) +
                                        head_id * config_.head_dim + l * 32 +
                                        m] = GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                    }
                    // get v_cache_
                    for (int k = 0; k < config_.block_len / 32; k++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            block_q4_0 block =
                                v_cache_q4[layer_id_][head_id][block_idx]
                                          [l * config_.block_len / 32 + k];
                            dequantize_row_q4_0(&block, block_fp32.data(), 32);
                            for (int m = 0; m < 32; m++) {

                                if (block_id * config_.block_len + k * 32 + m >=
                                    seq_len)
                                    break;
                                v_data_[batch_id *
                                            (max_block_num * config_.block_len *
                                             config_.kv_head_num *
                                             config_.head_dim) +
                                        block_id * (config_.block_len *
                                                    config_.kv_head_num *
                                                    config_.head_dim) +
                                        (k * 32 + m) * config_.kv_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                    }
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                    // get k_cache_
                    for (int k = 0; k < config_.block_len; k++) {
                        if (block_id * config_.block_len + k >= seq_len)
                            break;
                        for (int l = 0; l < config_.head_dim / 32; l++) {
                            block_q8_0 block =
                                k_cache_q8[layer_id_][head_id][block_idx]
                                          [k * config_.head_dim / 32 + l];
                            dequantize_row_q8_0(&block, block_fp32.data(), 32);
                            for (int m = 0; m < 32; m++) {

                                k_data_[batch_id *
                                            (max_block_num * config_.block_len *
                                             config_.kv_head_num *
                                             config_.head_dim) +
                                        block_id * (config_.block_len *
                                                    config_.kv_head_num *
                                                    config_.head_dim) +
                                        k * (config_.kv_head_num *
                                             config_.head_dim) +
                                        head_id * config_.head_dim + l * 32 +
                                        m] = GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                    }
                    // get v_cache_
                    for (int k = 0; k < config_.block_len / 32; k++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            block_q8_0 block =
                                v_cache_q8[layer_id_][head_id][block_idx]
                                          [l * config_.block_len / 32 + k];
                            dequantize_row_q8_0(&block, block_fp32.data(), 32);
                            for (int m = 0; m < 32; m++) {

                                if (block_id * config_.block_len + k * 32 + m >=
                                    seq_len)
                                    break;
                                v_data_[batch_id *
                                            (max_block_num * config_.block_len *
                                             config_.kv_head_num *
                                             config_.head_dim) +
                                        block_id * (config_.block_len *
                                                    config_.kv_head_num *
                                                    config_.head_dim) +
                                        (k * 32 + m) * config_.kv_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                    }
                }
            }
            if (block_r > seq_len && block_l < seq_len + q_len) {
                if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
                    for (int k = 0; k < config_.block_len; k++) {
                        if (block_id * config_.block_len + k >=
                                seq_len + q_len ||
                            block_id * config_.block_len + k < seq_len)
                            continue;
                        for (int l = 0; l < config_.head_dim; l++) {
                            k_cache_fp16_[layer_id_][head_id][block_idx]
                                         [k * config_.head_dim + l] = k_data_
                                             [batch_id * (max_block_num *
                                                          config_.block_len *
                                                          config_.kv_head_num *
                                                          config_.head_dim) +
                                              block_id * (config_.block_len *
                                                          config_.kv_head_num *
                                                          config_.head_dim) +
                                              k * (config_.kv_head_num *
                                                   config_.head_dim) +
                                              head_id * config_.head_dim + l];
                            v_cache_fp16_[layer_id_][head_id][block_idx]
                                         [l * config_.block_len + k] = v_data_
                                             [batch_id * (max_block_num *
                                                          config_.block_len *
                                                          config_.kv_head_num *
                                                          config_.head_dim) +
                                              block_id * (config_.block_len *
                                                          config_.kv_head_num *
                                                          config_.head_dim) +
                                              k * (config_.kv_head_num *
                                                   config_.head_dim) +
                                              head_id * config_.head_dim + l];
                        }
                    }
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                    // fill k_cache_
                    for (int k = 0; k < config_.block_len; k++) {
                        if (block_id * config_.block_len + k >=
                                seq_len + q_len ||
                            block_id * config_.block_len + k < seq_len)
                            continue;
                        for (int l = 0; l < config_.head_dim / 32; l++) {
                            block_q4_0 block;
                            for (int m = 0; m < 32; m++) {

                                block_fp32[m] = GGML_FP16_TO_FP32(
                                    k_data_[batch_id * (max_block_num *
                                                        config_.block_len *
                                                        config_.kv_head_num *
                                                        config_.head_dim) +
                                            block_id * (config_.block_len *
                                                        config_.kv_head_num *
                                                        config_.head_dim) +
                                            k * (config_.kv_head_num *
                                                 config_.head_dim) +
                                            head_id * config_.head_dim +
                                            l * 32 + m]);
                            }
                            quantize_row_q4_0(block_fp32.data(), &block, 32);
                            k_cache_q4[layer_id_][head_id][block_idx]
                                      [k * config_.head_dim / 32 + l] = block;
                        }
                    }

                    // fill v_cache_
                    for (int k = 0; k < config_.block_len / 32; k++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            block_q4_0 block;
                            for (int m = 0; m < 32; m++) {

                                if (block_id * config_.block_len + k * 32 + m >=
                                    seq_len + q_len) {
                                    block_fp32[m] = 0;
                                    continue;
                                }
                                block_fp32[m] = GGML_FP16_TO_FP32(
                                    v_data_[batch_id * (max_block_num *
                                                        config_.block_len *
                                                        config_.kv_head_num *
                                                        config_.head_dim) +
                                            block_id * (config_.block_len *
                                                        config_.kv_head_num *
                                                        config_.head_dim) +
                                            (k * 32 + m) * config_.kv_head_num *
                                                config_.head_dim +
                                            head_id * config_.head_dim + l]);
                            }
                            quantize_row_q4_0(block_fp32.data(), &block, 32);
                            v_cache_q4[layer_id_][head_id][block_idx]
                                      [l * config_.block_len / 32 + k] = block;
                        }
                    }
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                    // fill k_cache_
                    for (int k = 0; k < config_.block_len; k++) {
                        if (block_id * config_.block_len + k >=
                                seq_len + q_len ||
                            block_id * config_.block_len + k < seq_len)
                            continue;
                        for (int l = 0; l < config_.head_dim / 32; l++) {
                            block_q8_0 block;
                            for (int m = 0; m < 32; m++) {

                                block_fp32[m] = GGML_FP16_TO_FP32(
                                    k_data_[batch_id * (max_block_num *
                                                        config_.block_len *
                                                        config_.kv_head_num *
                                                        config_.head_dim) +
                                            block_id * (config_.block_len *
                                                        config_.kv_head_num *
                                                        config_.head_dim) +
                                            k * (config_.kv_head_num *
                                                 config_.head_dim) +
                                            head_id * config_.head_dim +
                                            l * 32 + m]);
                            }
                            quantize_row_q8_0(block_fp32.data(), &block, 32);
                            k_cache_q8[layer_id_][head_id][block_idx]
                                      [k * config_.head_dim / 32 + l] = block;
                        }
                    }

                    // fill v_cache_
                    for (int k = 0; k < config_.block_len / 32; k++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            block_q8_0 block;
                            for (int m = 0; m < 32; m++) {

                                if (block_id * config_.block_len + k * 32 + m >=
                                    seq_len + q_len) {
                                    block_fp32[m] = 0;
                                    continue;
                                }
                                block_fp32[m] = GGML_FP16_TO_FP32(
                                    v_data_[batch_id * (max_block_num *
                                                        config_.block_len *
                                                        config_.kv_head_num *
                                                        config_.head_dim) +
                                            block_id * (config_.block_len *
                                                        config_.kv_head_num *
                                                        config_.head_dim) +
                                            (k * 32 + m) * config_.kv_head_num *
                                                config_.head_dim +
                                            head_id * config_.head_dim + l]);
                            }
                            quantize_row_q8_0(block_fp32.data(), &block, 32);
                            v_cache_q8[layer_id_][head_id][block_idx]
                                      [l * config_.block_len / 32 + k] = block;
                        }
                    }
                }
            }
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // printf("layer %d time of reading and updating KV Cache: %f s\n",
    // layer_id,
    //        duration.count());
}

void KVCache::update_importance(const ggml_fp16_t *importance, int layer_id,
                                int *block_table, int batch_size,
                                int max_block_num, int *offset, int width,
                                Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    importance_data_ = const_cast<uint16_t *>(importance);

    // Each task updates the importance of a certain position
    backend->do_work_stealing_job(
        max_block_num * batch_size, nullptr,
        [&](int task_id) {
            int block_id = task_id % max_block_num;
            int batch_id = task_id / max_block_num;
            int block_idx = block_table[batch_id * max_block_num + block_id];
            if (block_id > (offset[batch_id] + width) / config_.block_len) {
                return;
            }
            for (int k = 0; k < config_.block_len; k++) {
                for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
                    importance_[layer_id_][block_idx][k][head_id] =
                        GGML_FP32_TO_FP16(
                            GGML_FP16_TO_FP32(
                                importance_data_[batch_id * max_block_num *
                                                     config_.block_len *
                                                     config_.q_head_num +
                                                 (block_id * config_.block_len +
                                                  k) *
                                                     config_.q_head_num +
                                                 head_id]) +
                            GGML_FP16_TO_FP32(
                                importance_[layer_id_][block_idx][k][head_id]));
                }
            }
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // printf("layer %d time of updating importance: %f s\n", layer_id,
    //        duration.count());
}

void KVCache::get_kvcache_fp16(ggml_fp16_t *k_in, ggml_fp16_t *v_in,
                               int layer_id, int *block_table, int batch_size,
                               int max_block_num, int *cache_seqlens,
                               Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    k_data_ = const_cast<uint16_t *>(k_in);
    v_data_ = const_cast<uint16_t *>(v_in);

    // Each task updates the k cache and v cache of a certain header
    backend->do_work_stealing_job(
        config_.kv_head_num * max_block_num * batch_size, nullptr,
        [&](int task_id) {
            // printf("block_idx: %d, task_id: %d\n", block_idx, task_id);
            std::vector<float> block_fp32(32);
            int batch_id = task_id / (config_.kv_head_num * max_block_num);
            int block_id = (task_id / config_.kv_head_num) % max_block_num;
            int head_id = task_id % config_.kv_head_num;
            int block_idx = block_table[batch_id * max_block_num + block_id];
            int seq_len = cache_seqlens[batch_id];
            int block_l = block_id * config_.block_len;
            int block_r = block_id * config_.block_len + config_.block_len;

            if (block_l < seq_len) {
                if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
                    for (int k = 0; k < config_.block_len; k++) {
                        if (block_id * config_.block_len + k >= seq_len)
                            break;
                        for (int l = 0; l < config_.head_dim; l++) {
                            k_data_
                                [batch_id *
                                     (max_block_num * config_.block_len *
                                      config_.kv_head_num * config_.head_dim) +
                                 block_id *
                                     (config_.block_len * config_.kv_head_num *
                                      config_.head_dim) +
                                 k * (config_.kv_head_num * config_.head_dim) +
                                 head_id * config_.head_dim + l] =
                                    k_cache_fp16_[layer_id_][head_id][block_idx]
                                                 [k * config_.head_dim + l];
                            v_data_
                                [batch_id *
                                     (max_block_num * config_.block_len *
                                      config_.kv_head_num * config_.head_dim) +
                                 block_id *
                                     (config_.block_len * config_.kv_head_num *
                                      config_.head_dim) +
                                 k * (config_.kv_head_num * config_.head_dim) +
                                 head_id * config_.head_dim + l] =
                                    v_cache_fp16_[layer_id_][head_id][block_idx]
                                                 [l * config_.block_len + k];
                        }
                    }
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                    // get k_cache_
                    for (int k = 0; k < config_.block_len; k++) {
                        if (block_id * config_.block_len + k >= seq_len)
                            break;
                        for (int l = 0; l < config_.head_dim / 32; l++) {
                            block_q4_0 block =
                                k_cache_q4[layer_id_][head_id][block_idx]
                                          [k * config_.head_dim / 32 + l];
                            dequantize_row_q4_0(&block, block_fp32.data(), 32);
                            for (int m = 0; m < 32; m++) {

                                k_data_[batch_id *
                                            (max_block_num * config_.block_len *
                                             config_.kv_head_num *
                                             config_.head_dim) +
                                        block_id * (config_.block_len *
                                                    config_.kv_head_num *
                                                    config_.head_dim) +
                                        k * (config_.kv_head_num *
                                             config_.head_dim) +
                                        head_id * config_.head_dim + l * 32 +
                                        m] = GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                    }
                    // get v_cache_
                    for (int k = 0; k < config_.block_len / 32; k++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            block_q4_0 block =
                                v_cache_q4[layer_id_][head_id][block_idx]
                                          [l * config_.block_len / 32 + k];
                            dequantize_row_q4_0(&block, block_fp32.data(), 32);
                            for (int m = 0; m < 32; m++) {

                                if (block_id * config_.block_len + k * 32 + m >=
                                    seq_len)
                                    break;
                                v_data_[batch_id *
                                            (max_block_num * config_.block_len *
                                             config_.kv_head_num *
                                             config_.head_dim) +
                                        block_id * (config_.block_len *
                                                    config_.kv_head_num *
                                                    config_.head_dim) +
                                        (k * 32 + m) * config_.kv_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                    }
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                    // get k_cache_
                    for (int k = 0; k < config_.block_len; k++) {
                        if (block_id * config_.block_len + k >= seq_len)
                            break;
                        for (int l = 0; l < config_.head_dim / 32; l++) {
                            block_q8_0 block =
                                k_cache_q8[layer_id_][head_id][block_idx]
                                          [k * config_.head_dim / 32 + l];
                            dequantize_row_q8_0(&block, block_fp32.data(), 32);
                            for (int m = 0; m < 32; m++) {

                                k_data_[batch_id *
                                            (max_block_num * config_.block_len *
                                             config_.kv_head_num *
                                             config_.head_dim) +
                                        block_id * (config_.block_len *
                                                    config_.kv_head_num *
                                                    config_.head_dim) +
                                        k * (config_.kv_head_num *
                                             config_.head_dim) +
                                        head_id * config_.head_dim + l * 32 +
                                        m] = GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                    }
                    // get v_cache_
                    for (int k = 0; k < config_.block_len / 32; k++) {
                        for (int l = 0; l < config_.head_dim; l++) {
                            block_q8_0 block =
                                v_cache_q8[layer_id_][head_id][block_idx]
                                          [l * config_.block_len / 32 + k];
                            dequantize_row_q8_0(&block, block_fp32.data(), 32);
                            for (int m = 0; m < 32; m++) {

                                if (block_id * config_.block_len + k * 32 + m >=
                                    seq_len)
                                    break;
                                v_data_[batch_id *
                                            (max_block_num * config_.block_len *
                                             config_.kv_head_num *
                                             config_.head_dim) +
                                        block_id * (config_.block_len *
                                                    config_.kv_head_num *
                                                    config_.head_dim) +
                                        (k * 32 + m) * config_.kv_head_num *
                                            config_.head_dim +
                                        head_id * config_.head_dim + l] =
                                    GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                    }
                }
            }
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
}

void KVCache::update_kvcache_fp16(const ggml_fp16_t *k_in,
                                  const ggml_fp16_t *v_in, int layer_id,
                                  int *block_table, int batch_size,
                                  int max_block_num, int *cache_seqlens,
                                  int q_len, Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    k_data_ = const_cast<uint16_t *>(k_in);
    v_data_ = const_cast<uint16_t *>(v_in);
    // Each task updates the k cache and v cache of a certain header
    backend->do_work_stealing_job(
        batch_size * config_.kv_head_num * q_len, nullptr,
        [&](int task_id) {
            int batch_id = task_id / (config_.kv_head_num * q_len);
            int head_id = task_id / q_len % config_.kv_head_num;
            int seq_len = cache_seqlens[batch_id] + task_id % q_len;
            int q_offset = task_id % q_len;

            int block_id = seq_len / config_.block_len;
            int block_idx = block_table[batch_id * max_block_num + block_id];
            int pos_in_block = seq_len % config_.block_len;

            if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
                for (int l = 0; l < config_.head_dim; l++) {
                    k_cache_fp16_[layer_id_][head_id][block_idx]
                                 [pos_in_block * config_.head_dim + l] =
                                     k_data_[batch_id *
                                                 (q_len * config_.kv_head_num *
                                                  config_.head_dim) +
                                             q_offset * config_.kv_head_num *
                                                 config_.head_dim +
                                             head_id * config_.head_dim + l];
                    v_cache_fp16_[layer_id_][head_id][block_idx]
                                 [l * config_.block_len + pos_in_block] =
                                     v_data_[batch_id *
                                                 (q_len * config_.kv_head_num *
                                                  config_.head_dim) +
                                             q_offset * config_.kv_head_num *
                                                 config_.head_dim +
                                             head_id * config_.head_dim + l];
                }
            } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                std::vector<float> block_fp32(32);
                // fill k_cache_
                for (int l = 0; l < config_.head_dim / 32; l++) {
                    block_q4_0 block;
                    for (int m = 0; m < 32; m++) {

                        block_fp32[m] = GGML_FP16_TO_FP32(
                            k_data_[batch_id * (q_len * config_.kv_head_num *
                                                config_.head_dim) +
                                    head_id * config_.head_dim + l * 32 + m]);
                    }
                    quantize_row_q4_0(block_fp32.data(), &block, 32);

                    k_cache_q4[layer_id_][head_id][block_idx]
                              [pos_in_block * config_.head_dim / 32 + l] =
                                  block;
                }

                // fill v_cache_
                for (int l = 0; l < config_.head_dim; l++) {
                    block_q4_0 block = v_cache_q4[layer_id_][head_id][block_idx]
                                                 [l * config_.block_len / 32 +
                                                  pos_in_block / 32];
                    dequantize_row_q4_0(&block, block_fp32.data(), 32);
                    block_fp32[pos_in_block % 32] = GGML_FP16_TO_FP32(
                        v_data_[batch_id * (q_len * config_.kv_head_num *
                                            config_.head_dim) +
                                head_id * config_.head_dim + l]);
                    quantize_row_q4_0(block_fp32.data(), &block, 32);
                    v_cache_q4[layer_id_][head_id][block_idx]
                              [l * config_.block_len / 32 + pos_in_block / 32] =
                                  block;
                }
            } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                std::vector<float> block_fp32(32);
                // fill k_cache_
                for (int l = 0; l < config_.head_dim / 32; l++) {
                    block_q8_0 block;
                    for (int m = 0; m < 32; m++) {

                        block_fp32[m] = GGML_FP16_TO_FP32(
                            k_data_[batch_id * (q_len * config_.kv_head_num *
                                                config_.head_dim) +
                                    head_id * config_.head_dim + l * 32 + m]);
                    }
                    quantize_row_q8_0(block_fp32.data(), &block, 32);

                    k_cache_q8[layer_id_][head_id][block_idx]
                              [pos_in_block * config_.head_dim / 32 + l] =
                                  block;
                }

                // fill v_cache_
                for (int l = 0; l < config_.head_dim; l++) {
                    block_q8_0 block = v_cache_q8[layer_id_][head_id][block_idx]
                                                 [l * config_.block_len / 32 +
                                                  pos_in_block / 32];
                    dequantize_row_q8_0(&block, block_fp32.data(), 32);
                    block_fp32[pos_in_block % 32] = GGML_FP16_TO_FP32(
                        v_data_[batch_id * (q_len * config_.kv_head_num *
                                            config_.head_dim) +
                                head_id * config_.head_dim + l]);
                    quantize_row_q8_0(block_fp32.data(), &block, 32);
                    v_cache_q8[layer_id_][head_id][block_idx]
                              [l * config_.block_len / 32 + pos_in_block / 32] =
                                  block;
                }
            }
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // printf("layer %d time of reading KV Cache: %f s\n", layer_id,
    //        duration.count());
}

void KVCache::get_all_kvcache_one_layer(int layer_id, ggml_fp16_t *k_in,
                                        ggml_fp16_t *v_in, Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    seq_len_ = config_.block_len;
    block_num_ = get_cache_total_block_num();
    k_data_ = reinterpret_cast<uint16_t *>(k_in);
    v_data_ = reinterpret_cast<uint16_t *>(v_in);

    // Each task gets the k cache or v cache of a certain header
    backend->do_work_stealing_job(
        config_.kv_head_num * past_block_num_[layer_id] * 2, nullptr,
        [&](int task_id) {
            std::vector<float> block_fp32(32);
            int head_id = task_id / 2 / past_block_num_[layer_id];
            int block_idx = task_id / 2 % past_block_num_[layer_id];
            if (block_idx >= block_num_)
                return;

            int max_offset = 0;
            if (task_id & 1) {
                // get k_cache_
                for (int k = 0; k < config_.block_len; k++) {
                    if (block_idx * seq_len_ + k >= cache_total_len_)
                        break;
                    for (int l = 0; l < config_.head_dim / 32; l++) {
                        block_q4_0 block =
                            k_cache_q4[layer_id_][head_id][block_idx]
                                      [k * config_.head_dim / 32 + l];
                        dequantize_row_q4_0(&block, block_fp32.data(), 32);
                        for (int m = 0; m < 32; m++) {

                            k_data_[(head_id * cache_total_len_ +
                                     block_idx * config_.block_len + k) *
                                        config_.head_dim +
                                    l * 32 + m] =
                                GGML_FP32_TO_FP16(block_fp32[m]);
                            max_offset = std::max(
                                max_offset,
                                (int)(head_id * cache_total_len_ +
                                      block_idx * config_.block_len + k) *
                                        config_.head_dim +
                                    l * 32 + m);
                        }
                    }
                }
            } else {
                // get v_cache_
                for (int k = 0; k < config_.block_len / 32; k++) {
                    for (int l = 0; l < config_.head_dim; l++) {
                        block_q4_0 block =
                            v_cache_q4[layer_id_][head_id][block_idx]
                                      [l * config_.block_len / 32 + k];
                        dequantize_row_q4_0(&block, block_fp32.data(), 32);
                        for (int m = 0; m < 32; m++) {

                            if (block_idx * seq_len_ + k * 32 + m >=
                                cache_total_len_)
                                break;
                            v_data_[(head_id * cache_total_len_ +
                                     block_idx * config_.block_len + k * 32 +
                                     m) *
                                        config_.head_dim +
                                    l] = GGML_FP32_TO_FP16(block_fp32[m]);
                            max_offset =
                                std::max(max_offset,
                                         (int)((head_id * cache_total_len_ +
                                                block_idx * config_.block_len +
                                                k * 32 + m) *
                                                   config_.head_dim +
                                               l));
                        }
                    }
                }
            }
        },
        nullptr);

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // printf("layer %d block num %d time of reading all KV Cache: %f s\n",
    //        layer_id, block_num_, duration.count());
}
