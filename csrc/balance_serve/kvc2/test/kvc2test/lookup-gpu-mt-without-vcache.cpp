/**
 * @Description  :
 * @Author       : Xie Weiyu
 * @Date         : 2024-11-22 09:52:48
 * @Version      : 1.0.0
 * @LastEditors  : Xie Weiyu
 * @LastEditTime : 2024-11-25 07:51:09
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "common.hpp"

int main(int argc, char* argv[]) {
  qw25_7B_gpu_config.v_cache_on = false;
  config.gpu_cache_config = qw25_7B_gpu_config;
  config.v_cache_on = false;

  init(argc, argv);
  spdlog::set_level(spdlog::level::debug);
  auto kvc2 = kvc2::create_kvc2(config);

  std::mt19937 gen(123);
  auto ids1 = random_ids(10 * config.num_token_per_page, gen);
  auto k1 = random_kvcache(10, gen);

  kvc2->raw_insert(test_model_name, test_quant_type, ids1.data(), ids1.size(), k1, {});

// complete same
#pragma omp parallel for
  for (size_t ti = 0; ti < 3; ti++) {
    auto h = kvc2->lookup_to_gpu(test_model_name, test_quant_type, ids1.data(), ids1.size(),
                                 ids1.size() + 2 * config.num_token_per_page);
    auto k = h->handle_data(true);
    cmp_handle_data(k1, k, 10);

    auto block_idx = h->get_gpu_block_idx();
    auto [kcache, vcache] = kvc2->get_kvcache();

    auto k_from_gpu = empty_kvcache(15);

    size_t gpu_count = config.gpu_cache_config->gpu_devices_id.size();
    size_t element_size_per_gpu = test_cache_info.element_size(config.num_token_per_page) / gpu_count;
    for (size_t i = 0; i < k_from_gpu.size(); i++) {
      for (size_t j = 0; j < block_idx.size(); j++) {
        size_t b_idx = block_idx[j];
        for (size_t gpu_idx = 0; gpu_idx < gpu_count; gpu_idx++) {
          {
            auto kt = kcache[gpu_idx][i][b_idx].to(torch::kCPU);
            void* src = kt.data_ptr();
            void* dst = offset_by_bytes(k_from_gpu[i][j], gpu_idx * element_size_per_gpu);
            memcpy(dst, src, element_size_per_gpu);
          }
        }
      }
    }
    cmp_handle_data(k1, k_from_gpu, 10);
  }

  SPDLOG_CRITICAL("All Test Passed: {}", argv[0]);
  return 0;
}
