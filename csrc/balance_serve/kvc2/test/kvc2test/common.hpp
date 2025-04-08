/**
 * @Description  :
 * @Author       : Xie Weiyu
 * @Date         : 2024-11-22 06:02:41
 * @Version      : 1.0.0
 * @LastEditors  : Xie Weiyu
 * @LastEditTime : 2024-12-11 07:34:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#pragma once
#include <random>
#include <thread>
#include "kvc2.h"
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#define FMT_HEADER_ONLY
#include "spdlog/spdlog.h"

using namespace kvc2;

template <typename T>
T* offset_by_bytes(T* t, size_t n) {
  return reinterpret_cast<T*>(reinterpret_cast<size_t>(t) + n);
}

std::string FLAGS_disk_cache_path;

kvc2::KVC2Config config;
kvc2::GPUPageCacheConfig qw25_7B_gpu_config{
    .gpu_only = false,
    .gpu_devices_id = {0, 1},
    .layer_count = 28,
    .total_kvcache_pages = 40,
    .num_token_per_page = 256,
    .num_k_heads = 4,
    .k_head_dim = 896,
    .full_kv_cache_on_each_gpu = false,
    .k_cache_on = true,
    .v_cache_on = true,
    .tensor_type = torch::kBFloat16,
    .num_streams_per_device = 4,
};

ModelName test_model_name = "Qwen2.5-7B-Instruct";
QuantType test_quant_type = "FP16";
CacheInfo test_cache_info{
    .model_name = test_model_name,
    .is_key_cache = true,
    .quant_type = test_quant_type,
};

void init(int argc, char* argv[]) {
  if (argc != 2) {
    fmt::print("Usage: {} <disk_cache_path>\n", argv[0]);
    exit(1);
  }
  load_quant_configs("./config/quant_configs.json");
  load_model_configs("./config/model_configs.json");

  FLAGS_disk_cache_path = argv[1];
  if (FLAGS_disk_cache_path.empty()) {
    fmt::print("disk_cache_path is empty\n");
    exit(1);
  }
  config.path = FLAGS_disk_cache_path;
  config.config_path = "./config";
  config.gpu_cache_config = qw25_7B_gpu_config;
}

data_block_ptr empty_block() {
  auto re = new (std::align_val_t(4096)) std::byte[test_cache_info.element_size(config.num_token_per_page)];
  memset(re, 0, test_cache_info.element_size(config.num_token_per_page));
  return reinterpret_cast<data_block_ptr>(re);
}

data_block_ptr random_block(std::mt19937& gen) {
  auto re = empty_block();
  uint64_t* d = (uint64_t*)re;
  for (size_t i = 0; i < test_cache_info.element_size(config.num_token_per_page) / 8; i++) {
    d[i] = gen();
  }
  return re;
}
layer_data random_blocks(size_t block_count, size_t seed) {
  std::mt19937 gen(seed);
  layer_data re;
  for (size_t i = 0; i < block_count; i++) {
    re.push_back(random_block(gen));
  }
  return re;
}

layer_data empty_blocks(size_t block_count) {
  layer_data re;
  for (size_t i = 0; i < block_count; i++) {
    re.push_back(empty_block());
  }
  return re;
}

void copy_kvcache(std::vector<layer_data>& from, std::vector<layer_data>& to, size_t block_start, size_t length) {
  for (size_t i = 0; i < from.size(); i++) {
    for (size_t j = 0; j < length; j++) {
      to[i][block_start + j] = from[i][block_start + j];
    }
  }
}

std::vector<layer_data> random_kvcache(size_t block_count, std::mt19937& gen) {
  std::vector<layer_data> re;
  re.resize(test_cache_info.hidden_layer_count());
  fmt::print("Generating random kvcache, layer {}\n", test_cache_info.hidden_layer_count());
  std::vector<std::mt19937> gens;
  for (size_t i = 0; i < test_cache_info.hidden_layer_count(); i++) {
    gens.push_back(std::mt19937(gen()));
  }
#pragma omp parallel for
  for (size_t i = 0; i < test_cache_info.hidden_layer_count(); i++) {
    re[i] = random_blocks(block_count, gens[i]());
  }
  return re;
}

std::vector<layer_data> empty_kvcache(size_t block_count) {
  std::vector<layer_data> re;
  re.resize(test_cache_info.hidden_layer_count());
  fmt::print("Generating empty kvcache, layer {}\n", test_cache_info.hidden_layer_count());
#pragma omp parallel for
  for (size_t i = 0; i < test_cache_info.hidden_layer_count(); i++) {
    re[i] = empty_blocks(block_count);
  }
  return re;
}

std::vector<Token> random_ids(size_t length, std::mt19937& gen) {
  std::vector<Token> re;
  for (size_t i = 0; i < length; i++) {
    re.push_back(gen());
  }
  return re;
}

std::vector<layer_data> slice(std::vector<layer_data>& h1, size_t start, size_t end) {
  std::vector<layer_data> re;
  for (auto& l : h1) {
    layer_data new_layer;
    new_layer.insert(new_layer.end(), l.begin() + start, l.begin() + end);
    re.push_back(new_layer);
  }
  return re;
}

void cmp_handle_data(std::vector<layer_data> h1, std::vector<layer_data> h2,
                     std::optional<size_t> blocks = std::nullopt) {
  assert(h1.size() == h2.size());

  for (size_t i = 0; i < h1.size(); i++) {
    auto& b1 = h1[i];
    auto& b2 = h2[i];
    if (blocks.has_value() == false) {
      assert(b1.size() == b2.size());
    }
    int cmp_to = blocks.has_value() ? blocks.value() : b1.size();
    for (int j = 0; j < cmp_to; j++) {
      auto e1 = reinterpret_cast<void*>(b1[j]);
      auto e2 = reinterpret_cast<void*>(b2[j]);
      assert(memcmp(e1, e2, test_cache_info.element_size(config.num_token_per_page)) == 0);
    }
  }
  fmt::print("KVCacheHandle cmp ok\n");
}

void copy_gpu_cpu(std::vector<size_t>& block_idx, std::vector<torch::Tensor>& kcache,
                  std::vector<torch::Tensor>& vcache, std::vector<layer_data>& k_cpu, std::vector<layer_data>& v_cpu,
                  size_t at) {
  size_t gpu_count = config.gpu_cache_config->gpu_devices_id.size();
  size_t element_size_per_gpu = test_cache_info.element_size(config.num_token_per_page) / gpu_count;

  for (size_t layer = 0; layer < test_cache_info.hidden_layer_count(); layer++) {
    for (size_t gpu_idx = 0; gpu_idx < gpu_count; gpu_idx++) {
      {
        auto kt = kcache[gpu_idx][layer][block_idx[at]].to(torch::kCPU);
        void* src = kt.data_ptr();
        void* dst = offset_by_bytes(k_cpu[layer][at], gpu_idx * element_size_per_gpu);
        memcpy(dst, src, element_size_per_gpu);
      }
      {
        auto vt = vcache[gpu_idx][layer][block_idx[at]].to(torch::kCPU);
        void* src = vt.data_ptr();
        void* dst = offset_by_bytes(v_cpu[layer][at], gpu_idx * element_size_per_gpu);
        memcpy(dst, src, element_size_per_gpu);
      }
    }
  }
}

void copy_cpu_gpu(std::vector<size_t>& block_idx, std::vector<torch::Tensor>& kcache,
                  std::vector<torch::Tensor>& vcache, std::vector<layer_data>& k_cpu, std::vector<layer_data>& v_cpu,
                  size_t at) {
  size_t gpu_count = config.gpu_cache_config->gpu_devices_id.size();
  size_t element_size_per_gpu = test_cache_info.element_size(config.num_token_per_page) / gpu_count;

  for (size_t layer = 0; layer < test_cache_info.hidden_layer_count(); layer++) {
    for (size_t gpu_idx = 0; gpu_idx < gpu_count; gpu_idx++) {
      {
        auto kt = kcache[gpu_idx][layer][block_idx[at]].to(torch::kCPU);
        void* dst = kt.data_ptr();
        void* src = offset_by_bytes(k_cpu[layer][at], gpu_idx * element_size_per_gpu);
        memcpy(dst, src, element_size_per_gpu);
        kcache[gpu_idx][layer][block_idx[at]].copy_(kt);
      }
      {
        auto vt = vcache[gpu_idx][layer][block_idx[at]].to(torch::kCPU);
        void* dst = vt.data_ptr();
        void* src = offset_by_bytes(v_cpu[layer][at], gpu_idx * element_size_per_gpu);
        memcpy(dst, src, element_size_per_gpu);
        vcache[gpu_idx][layer][block_idx[at]].copy_(vt);
      }
    }
  }
}

void cmp_handle_gpu(std::vector<size_t>& block_idx, std::vector<torch::Tensor>& kcache,
                    std::vector<torch::Tensor>& vcache, std::vector<layer_data>& k1, std::vector<layer_data>& v1,
                    size_t num_blocks) {
  auto k_from_gpu = empty_kvcache(num_blocks);
  auto v_from_gpu = empty_kvcache(num_blocks);

  for (size_t j = 0; j < std::min(block_idx.size(), num_blocks); j++) {
    copy_gpu_cpu(block_idx, kcache, vcache, k_from_gpu, v_from_gpu, j);
  }
  cmp_handle_data(k1, k_from_gpu, num_blocks);
  cmp_handle_data(v1, v_from_gpu, num_blocks);
}
