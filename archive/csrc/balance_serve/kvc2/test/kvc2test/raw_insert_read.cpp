/**
 * @Description  :
 * @Author       : Xie Weiyu
 * @Date         : 2024-11-22 06:00:16
 * @Version      : 1.0.0
 * @LastEditors  : Xie Weiyu
 * @LastEditTime : 2024-11-22 07:30:46
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
    auto k2 = empty_kvcache(10);
    auto v2 = empty_kvcache(10);
    auto l2 = kvc2->raw_read(test_model_name, test_quant_type, ids1.data(), ids1.size(), k2, v2);
    assert(l2 == ids1.size());

    cmp_handle_data(k1, k2);
    cmp_handle_data(v1, v2);
  }

  // complete prefix
  {
    auto k2 = empty_kvcache(10);
    auto v2 = empty_kvcache(10);
    std::vector<Token> ids2 = std::vector<Token>(ids1.begin(), ids1.begin() + 3 * config.num_token_per_page);
    auto l2 = kvc2->raw_read(test_model_name, test_quant_type, ids2.data(), ids2.size(), k2, v2);
    assert(l2 == 3 * config.num_token_per_page);

    cmp_handle_data(k1, k2, 3);
    cmp_handle_data(v1, v2, 3);
  }

  // common prefix
  {
    auto k2 = empty_kvcache(10);
    auto v2 = empty_kvcache(10);
    std::vector<Token> ids2 = std::vector<Token>(ids1.begin(), ids1.begin() + 3 * config.num_token_per_page);
    auto rids = random_ids(config.num_token_per_page * 2 + config.num_token_per_page / 2, gen);
    ids2.insert(ids2.end(), rids.begin(), rids.end());

    auto l2 = kvc2->raw_read(test_model_name, test_quant_type, ids2.data(), ids2.size(), k2, v2);
    assert(l2 == 3 * config.num_token_per_page);

    cmp_handle_data(k1, k2, 3);
    cmp_handle_data(v1, v2, 3);
  }

  // no prefix
  {
    auto k2 = empty_kvcache(1);
    auto v2 = empty_kvcache(1);
    std::vector<Token> ids2 = random_ids(config.num_token_per_page, gen);
    auto l2 = kvc2->raw_read(test_model_name, test_quant_type, ids2.data(), ids2.size(), k2, v2);
    assert(l2 == 0);
  }

  // insert partly new
  auto k2 = random_kvcache(10, gen);
  auto v2 = random_kvcache(10, gen);
  copy_kvcache(k1, k2, 0, 5);
  copy_kvcache(v1, v2, 0, 5);
  auto ids2 = random_ids(10 * config.num_token_per_page, gen);
  for (size_t i = 0; i < 5 * config.num_token_per_page; i++) {
    ids2[i] = ids1[i];
  }
  kvc2->raw_insert(test_model_name, test_quant_type, ids2.data(), ids2.size(), k2, v2);

  // read new part
  {
    auto k = empty_kvcache(10);
    auto v = empty_kvcache(10);
    std::vector<Token> ids = std::vector<Token>(ids2.begin(), ids2.begin() + 7 * config.num_token_per_page);

    auto l = kvc2->raw_read(test_model_name, test_quant_type, ids.data(), ids.size(), k, v);
    assert(l == 7 * config.num_token_per_page);

    cmp_handle_data(k, k2, 7);
    cmp_handle_data(v, v2, 7);
  }

  SPDLOG_CRITICAL("All Test Passed: {}", argv[0]);

  return 0;
}
