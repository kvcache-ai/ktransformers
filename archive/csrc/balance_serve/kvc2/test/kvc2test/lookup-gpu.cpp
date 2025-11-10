/**
 * @Description  :
 * @Author       : Xie Weiyu
 * @Date         : 2024-11-22 09:52:48
 * @Version      : 1.0.0
 * @LastEditors  : Xie Weiyu
 * @LastEditTime : 2024-11-25 08:38:33
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "common.hpp"

int main(int argc, char* argv[]) {
  init(argc, argv);
  spdlog::set_level(spdlog::level::debug);
  auto kvc2 = kvc2::create_kvc2(config);

  std::mt19937 gen(123);
  auto ids1 = random_ids(10 * config.num_token_per_page, gen);
  auto k1 = random_kvcache(10, gen);
  auto v1 = random_kvcache(10, gen);

  kvc2->raw_insert(test_model_name, test_quant_type, ids1.data(), ids1.size(), k1, v1);

  // complete same
  {
    auto h = kvc2->lookup_to_gpu(test_model_name, test_quant_type, ids1.data(), ids1.size(),
                                 ids1.size() + 5 * config.num_token_per_page);
    auto k = h->handle_data(true);
    auto v = h->handle_data(false);
    cmp_handle_data(k1, k, 10);
    cmp_handle_data(v1, v, 10);

    auto block_idx = h->get_gpu_block_idx();
    auto [kcache, vcache] = kvc2->get_kvcache();

    auto k_from_gpu = empty_kvcache(15);
    auto v_from_gpu = empty_kvcache(15);

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
          {
            auto vt = vcache[gpu_idx][i][b_idx].to(torch::kCPU);
            void* src = vt.data_ptr();
            void* dst = offset_by_bytes(v_from_gpu[i][j], gpu_idx * element_size_per_gpu);
            memcpy(dst, src, element_size_per_gpu);
          }
        }
      }
    }
    cmp_handle_data(k1, k_from_gpu, 10);
    cmp_handle_data(v1, v_from_gpu, 10);
  }

  // prefix and evict
  {
    auto h = kvc2->lookup_to_gpu(test_model_name, test_quant_type, ids1.data(), config.num_token_per_page * 3,
                                 config.gpu_cache_config->total_kvcache_pages * config.num_token_per_page);
    auto k = h->handle_data(true);
    auto v = h->handle_data(false);
    cmp_handle_data(k1, k, 3);
    cmp_handle_data(v1, v, 3);

    auto block_idx = h->get_gpu_block_idx();
    auto [kcache, vcache] = kvc2->get_kvcache();

    auto k_from_gpu = empty_kvcache(3);
    auto v_from_gpu = empty_kvcache(3);

    size_t gpu_count = config.gpu_cache_config->gpu_devices_id.size();
    size_t element_size_per_gpu = test_cache_info.element_size(config.num_token_per_page) / gpu_count;
    for (size_t i = 0; i < k_from_gpu.size(); i++) {
      for (size_t j = 0; j < 3; j++) {
        size_t b_idx = block_idx[j];
        for (size_t gpu_idx = 0; gpu_idx < gpu_count; gpu_idx++) {
          {
            auto kt = kcache[gpu_idx][i][b_idx].to(torch::kCPU);
            void* src = kt.data_ptr();
            void* dst = offset_by_bytes(k_from_gpu[i][j], gpu_idx * element_size_per_gpu);
            memcpy(dst, src, element_size_per_gpu);
          }
          {
            auto vt = vcache[gpu_idx][i][b_idx].to(torch::kCPU);
            void* src = vt.data_ptr();
            void* dst = offset_by_bytes(v_from_gpu[i][j], gpu_idx * element_size_per_gpu);
            memcpy(dst, src, element_size_per_gpu);
          }
        }
      }
    }
    cmp_handle_data(k1, k_from_gpu, 3);
    cmp_handle_data(v1, v_from_gpu, 3);
  }

  // // complete prefix
  // {
  //   std::vector<Token> ids2(ids1.begin(), ids1.begin() + 3 * config.num_token_per_page);
  //   auto h = kvc2->lookup(test_model_name, test_quant_type, ids2.data(), ids2.size(),
  //                         ids2.size() + 3 * config.num_token_per_page);
  //   auto k = h->handle_data(true);
  //   auto v = h->handle_data(false);
  //   cmp_handle_data(k1, k, 3);
  //   cmp_handle_data(v1, v, 3);
  // }

  // // common prefix
  // {
  //   std::vector<Token> ids2(ids1.begin(), ids1.begin() + 3 * config.num_token_per_page);
  //   auto rids = random_ids(config.num_token_per_page * 2 + config.num_token_per_page / 2, gen);
  //   ids2.insert(ids2.end(), rids.begin(), rids.end());

  //   auto h = kvc2->lookup(test_model_name, test_quant_type, ids2.data(), ids2.size(), ids2.size());
  //   auto k = h->handle_data(true);
  //   auto v = h->handle_data(false);
  //   cmp_handle_data(k1, k, 3);
  //   cmp_handle_data(v1, v, 3);
  // }

  // // no prefix
  // {
  //   std::vector<Token> ids2 = random_ids(config.num_token_per_page, gen);
  //   auto h = kvc2->lookup(test_model_name, test_quant_type, ids2.data(), ids2.size(), ids2.size());
  //   assert(h->matched_length() == 0);
  // }

  // // insert partly new
  // auto k2 = random_kvcache(10, gen);
  // auto v2 = random_kvcache(10, gen);
  // copy_kvcache(k1, k2, 0, 5);
  // copy_kvcache(v1, v2, 0, 5);
  // auto ids2 = random_ids(10 * config.num_token_per_page, gen);
  // for (size_t i = 0; i < 5 * config.num_token_per_page; i++) {
  //   ids2[i] = ids1[i];
  // }
  // kvc2->raw_insert(test_model_name, test_quant_type, ids2.data(), ids2.size(), k2, v2);

  // // read new part
  // {
  //   std::vector<Token> ids(ids2.begin(), ids2.begin() + 7 * config.num_token_per_page);
  //   auto h = kvc2->lookup(test_model_name, test_quant_type, ids.data(), ids.size(),
  //                         ids.size() + 7 * config.num_token_per_page);
  //   auto k = h->handle_data(true);
  //   auto v = h->handle_data(false);
  //   cmp_handle_data(k, k2, 7);
  //   cmp_handle_data(v, v2, 7);
  // }

  SPDLOG_CRITICAL("All Test Passed: {}", argv[0]);
  return 0;
}
