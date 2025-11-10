/**
 * @Description  :
 * @Author       : Xie Weiyu
 * @Date         : 2024-11-22 08:29:45
 * @Version      : 1.0.0
 * @LastEditors  : Xie Weiyu
 * @LastEditTime : 2024-11-22 09:56:12
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
  // auto v1 = random_kvcache(10, gen);

  kvc2->raw_insert(test_model_name, test_quant_type, ids1.data(), ids1.size(), k1, {});

  // complete same
  {
    auto h = kvc2->lookup(test_model_name, test_quant_type, ids1.data(), ids1.size(),
                          ids1.size() + 10 * config.num_token_per_page);
    auto k = h->handle_data(true);
    cmp_handle_data(k1, k, 10);
  }

  // complete prefix
  {
    std::vector<Token> ids2(ids1.begin(), ids1.begin() + 3 * config.num_token_per_page);
    auto h = kvc2->lookup(test_model_name, test_quant_type, ids2.data(), ids2.size(),
                          ids2.size() + 3 * config.num_token_per_page);
    auto k = h->handle_data(true);
    cmp_handle_data(k1, k, 3);
  }

  // common prefix
  {
    std::vector<Token> ids2(ids1.begin(), ids1.begin() + 3 * config.num_token_per_page);
    auto rids = random_ids(config.num_token_per_page * 2 + config.num_token_per_page / 2, gen);
    ids2.insert(ids2.end(), rids.begin(), rids.end());

    auto h = kvc2->lookup(test_model_name, test_quant_type, ids2.data(), ids2.size(), ids2.size());
    auto k = h->handle_data(true);
    cmp_handle_data(k1, k, 3);
  }

  // no prefix
  {
    std::vector<Token> ids2 = random_ids(config.num_token_per_page, gen);
    auto h = kvc2->lookup(test_model_name, test_quant_type, ids2.data(), ids2.size(), ids2.size());
    assert(h->matched_length() == 0);
  }

  // insert partly new
  auto k2 = random_kvcache(10, gen);
  copy_kvcache(k1, k2, 0, 5);
  auto ids2 = random_ids(10 * config.num_token_per_page, gen);
  for (size_t i = 0; i < 5 * config.num_token_per_page; i++) {
    ids2[i] = ids1[i];
  }
  kvc2->raw_insert(test_model_name, test_quant_type, ids2.data(), ids2.size(), k2, {});

  // read new part
  {
    std::vector<Token> ids(ids2.begin(), ids2.begin() + 7 * config.num_token_per_page);
    auto h = kvc2->lookup(test_model_name, test_quant_type, ids.data(), ids.size(),
                          ids.size() + 7 * config.num_token_per_page);
    auto k = h->handle_data(true);
    cmp_handle_data(k, k2, 7);
  }

  SPDLOG_CRITICAL("All Test Passed: {}", argv[0]);
  return 0;
}
