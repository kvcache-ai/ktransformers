#include <future>
#include "common.hpp"

int main(int argc, char* argv[]) {
  init(argc, argv);
  spdlog::set_level(spdlog::level::debug);
  auto kvc2 = kvc2::create_kvc2(config);

#pragma omp parallel for
  for (size_t ti = 0; ti < 3; ti++) {
    auto [kcache, vcache] = kvc2->get_kvcache();
    std::mt19937 gen(ti + 123);
    size_t total_page = 10;
    TokenLength total_length = total_page * config.num_token_per_page;
    auto tokens = random_ids(total_length, gen);
    TokenLength prompt_length = 3 * config.num_token_per_page;
    auto k1 = random_kvcache(total_page, gen);
    auto v1 = random_kvcache(total_page, gen);
    {
      std::promise<std::shared_ptr<DoubleCacheHandleInterface>> p;
      kvc2->lookup_to_gpu_async(test_model_name, test_quant_type, tokens.data(), prompt_length, total_length,
                                [&p](std::shared_ptr<DoubleCacheHandleInterface> h) { p.set_value(h); });
      auto fut = p.get_future();
      fut.wait();
      auto h = fut.get();
      assert(h->matched_length() % config.num_token_per_page == 0);
      size_t matched_block = h->matched_length() / config.num_token_per_page;
      auto block_idx = h->get_gpu_block_idx();
      cmp_handle_gpu(block_idx, kcache, vcache, k1, v1, matched_block);
      for (size_t at = matched_block; at < block_idx.size(); at++) {
        copy_cpu_gpu(block_idx, kcache, vcache, k1, v1, at);
      }
      h->append_tokens(tokens.data(), total_length);
      cmp_handle_gpu(block_idx, kcache, vcache, k1, v1, total_page);
    }

    {
      std::promise<std::shared_ptr<DoubleCacheHandleInterface>> p;
      kvc2->lookup_to_gpu_async(test_model_name, test_quant_type, tokens.data(), total_length, total_length,
                                [&p](std::shared_ptr<DoubleCacheHandleInterface> h) { p.set_value(h); });
      auto fut = p.get_future();
      fut.wait();
      auto h = fut.get();
      assert(h->matched_length() == total_length);
      size_t matched_block = h->matched_length() / config.num_token_per_page;
      auto block_idx = h->get_gpu_block_idx();
      cmp_handle_gpu(block_idx, kcache, vcache, k1, v1, matched_block);
    }
  }
  SPDLOG_CRITICAL("All Test Passed: {}", argv[0]);
  return 0;
}
