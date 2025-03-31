/**
 * @Description  :
 * @Author       : Xie Weiyu
 * @Date         : 2024-11-22 09:52:48
 * @Version      : 1.0.0
 * @LastEditors  : Xie Weiyu
 * @LastEditTime : 2024-11-25 07:51:09
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include <future>
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
#pragma omp parallel for
  for (size_t ti = 0; ti < 3; ti++) {
    std::promise<std::shared_ptr<DoubleCacheHandleInterface>> p;
    kvc2->lookup_to_gpu_async(test_model_name, test_quant_type, ids1.data(), ids1.size(),
                              ids1.size() + 2 * config.num_token_per_page,
                              [&p](std::shared_ptr<DoubleCacheHandleInterface> h) { p.set_value(h); });
    auto fut = p.get_future();
    fut.wait();
    auto h = fut.get();
    auto k = h->handle_data(true);
    auto v = h->handle_data(false);
    cmp_handle_data(k1, k, 10);
    cmp_handle_data(v1, v, 10);

    auto block_idx = h->get_gpu_block_idx();
    auto [kcache, vcache] = kvc2->get_kvcache();

    cmp_handle_gpu(block_idx, kcache, vcache, k1, v1, 10);
  }

  SPDLOG_CRITICAL("All Test Passed: {}", argv[0]);
  return 0;
}
