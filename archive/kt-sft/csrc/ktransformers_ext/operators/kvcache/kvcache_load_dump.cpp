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

void KVCache::load_kvcache(std::string tensor_file_path, Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream ifs_tensor(tensor_file_path, std::ios::binary);
    if (!ifs_tensor) {
        throw std::runtime_error("Failed to open tensor file");
    }
    ifs_tensor.read(reinterpret_cast<char *>(&cache_total_len_),
                    sizeof(cache_total_len_));
    int past_block_num =
        (cache_total_len_ + config_.block_len - 1) / config_.block_len;
    printf("cache_total_len: %d, past_block_num: %d\n", cache_total_len_,
           past_block_num);
    for (int i = 0; i < config_.layer_num; ++i) {
        past_block_num_[i] = past_block_num;
    }
    ifs_tensor.read(reinterpret_cast<char *>(anchor_.data()),
                    anchor_.size() * sizeof(ggml_fp16_t));
    for (int i = 0; i < config_.layer_num; ++i) {
        for (int j = 0; j < config_.kv_head_num; ++j) {
            for (int k = 0; k < past_block_num_[i]; ++k) {
                if (config_.kv_type == GGML_TYPE_F16) {
                    ifs_tensor.read(
                        reinterpret_cast<char *>(k_cache_fp16_[i][j][k].data()),
                        k_cache_fp16_[i][j][k].size() * sizeof(ggml_fp16_t));
                    ifs_tensor.read(
                        reinterpret_cast<char *>(v_cache_fp16_[i][j][k].data()),
                        v_cache_fp16_[i][j][k].size() * sizeof(ggml_fp16_t));
                } else if (config_.kv_type == GGML_TYPE_Q4_0) {
                    ifs_tensor.read(
                        reinterpret_cast<char *>(k_cache_q4[i][j][k].data()),
                        k_cache_q4[i][j][k].size() * sizeof(block_q4_0));
                    ifs_tensor.read(
                        reinterpret_cast<char *>(v_cache_q4[i][j][k].data()),
                        v_cache_q4[i][j][k].size() * sizeof(block_q4_0));
                }
            }
        }
        for (int k = 0; k < past_block_num_[i]; ++k) {
            for (int l = 0; l < config_.block_len; l++) {
                ifs_tensor.read(
                    reinterpret_cast<char *>(importance_[i][k][l].data()),
                    importance_[i][k][l].size() * sizeof(ggml_fp16_t));
            }
        }
    }
    ifs_tensor.close();
    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("time of load: %f s\n", diff.count());
}
void KVCache::dump_kvcache(int *block_table, int cache_total_len,
                           std::string tensor_file_path, Backend *backend) {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();
    std::ofstream ofs(tensor_file_path, std::ios::binary);
    printf("dump_kvcache: %s\n", tensor_file_path.c_str());
    if (!ofs.is_open()) {
        std::cerr << "Cannot open file " << tensor_file_path << std::endl;
        return;
    }
    ofs.write(reinterpret_cast<const char *>(&cache_total_len),
              sizeof(cache_total_len));
    int past_block_num =
        (cache_total_len + config_.block_len - 1) / config_.block_len;
    printf("cache_total_len: %d, past_block_num: %d\n", cache_total_len,
           past_block_num);
    ofs.write(reinterpret_cast<const char *>(anchor_.data()),
              anchor_.size() * sizeof(ggml_fp16_t));
    for (int i = 0; i < config_.layer_num; ++i) {
        for (int j = 0; j < config_.kv_head_num; ++j) {
            for (int k = 0; k < past_block_num; ++k) {
                int block_idx = block_table[k];
                if (config_.kv_type == GGML_TYPE_F16) {
                    ofs.write(reinterpret_cast<const char *>(
                                  k_cache_fp16_[i][j][block_idx].data()),
                              k_cache_fp16_[i][j][block_idx].size() *
                                  sizeof(ggml_fp16_t));
                    ofs.write(reinterpret_cast<const char *>(
                                  v_cache_fp16_[i][j][block_idx].data()),
                              v_cache_fp16_[i][j][block_idx].size() *
                                  sizeof(ggml_fp16_t));

                } else if (config_.kv_type == GGML_TYPE_Q4_0) {
                    ofs.write(reinterpret_cast<const char *>(
                                  k_cache_q4[i][j][block_idx].data()),
                              k_cache_q4[i][j][block_idx].size() *
                                  sizeof(block_q4_0));
                    ofs.write(reinterpret_cast<const char *>(
                                  v_cache_q4[i][j][block_idx].data()),
                              v_cache_q4[i][j][block_idx].size() *
                                  sizeof(block_q4_0));
                }
            }
        }
        for (int k = 0; k < past_block_num; ++k) {
            int block_idx = block_table[k];
            for (int l = 0; l < config_.block_len; l++) {
                ofs.write(reinterpret_cast<const char *>(
                              importance_[i][block_idx][l].data()),
                          importance_[i][block_idx][l].size() *
                              sizeof(ggml_fp16_t));
            }
        }
    }
    ofs.close();
    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("time of dump: %f s\n", diff.count());
}