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
  init(argc, argv);
  spdlog::set_level(spdlog::level::trace);
  auto kvc2 = kvc2::create_kvc2(config);

  std::mt19937 gen(123);

  std::vector<std::vector<Token>> ids;

  std::vector<std::vector<layer_data>> k, v;
  for (size_t i = 0; i < 10; i++) {
    ids.push_back(random_ids(1 * config.num_token_per_page, gen));
    k.push_back(random_kvcache(1, gen));
    v.push_back(random_kvcache(1, gen));
    kvc2->raw_insert(test_model_name, test_quant_type, ids[i].data(), ids[i].size(), k[i], v[i]);
  }

  kvc2->debug();
  {
    // all match
    std::vector<Token*> chunks;
    std::vector<TokenLength> lengths;
    for (size_t i = 0; i < 10; i++) {
      chunks.push_back(ids[i].data());
      lengths.push_back(ids[i].size());
    }

    auto h = kvc2->lookup_alt(test_model_name, test_quant_type, chunks, lengths, 15 * config.num_token_per_page);
    auto hk = h->handle_data(true);
    auto hv = h->handle_data(false);

    for (size_t i = 0; i < 10; i++) {
      cmp_handle_data(slice(hk, i, i + 1), k[i], 1);
      cmp_handle_data(slice(hv, i, i + 1), v[i], 1);
    }
  }

  {
    // no match in the middle
    std::vector<Token*> chunks;
    std::vector<TokenLength> lengths;

    std::vector<std::vector<Token>> new_ids;
    for (size_t i = 0; i < 10; i++) {
      new_ids.push_back(random_ids(1 * config.num_token_per_page, gen));
    }

    for (size_t i = 0; i < 10; i++) {
      if (i == 1 || i == 5 || i == 6) {
        chunks.push_back(new_ids[i].data());
      } else {
        chunks.push_back(ids[i].data());
      }
      lengths.push_back(ids[i].size());
    }

    auto h = kvc2->lookup_alt(test_model_name, test_quant_type, chunks, lengths, 15 * config.num_token_per_page);
    auto statuses = h->matched_status();
    for (size_t i = 0; i < 10; i++) {
      if (i == 1) {
        assert(statuses[i] == MatchStatus::NotMatchExact);
      } else if (i == 5 || i == 6) {
        assert(statuses[i] == MatchStatus::NotMatchPartial);
      } else if (i == 0) {
        assert(statuses[i] == MatchStatus::Exact);
      } else {
        assert(statuses[i] == MatchStatus::Partial);
      }
    }

    auto hk = h->handle_data(true);
    auto hv = h->handle_data(false);

    for (size_t i = 0; i < 10; i++) {
      if (i == 1 || i == 5 || i == 6) {
      } else {
        cmp_handle_data(slice(hk, i, i + 1), k[i], 1);
        cmp_handle_data(slice(hv, i, i + 1), v[i], 1);
      }
    }
  }

  SPDLOG_CRITICAL("All Test Passed: {}", argv[0]);
  return 0;
}
