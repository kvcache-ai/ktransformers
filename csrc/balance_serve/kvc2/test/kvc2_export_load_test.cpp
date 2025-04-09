#include "kvc2.h"
#include "kvc2_test_utils.cpp"

int main(int argc, char* argv[]) {
  init(argc, argv);
  spdlog::set_level(spdlog::level::debug);
  std::mt19937 gen(123);

  KVC2Config config = {
      .path = FLAGS_disk_cache_path,
      .block_length = BlockLength,
      .memory_pool_size = size_t(10e9),
      .evict_count = 20,
  };
  auto kvcc = create_kvc2(config);
  kvcc->load();

  auto io = kvcc->start_io_thread();

  SPDLOG_INFO("Disk Test");
  auto ids = random_ids(10 * BlockLength, gen);
  auto h1 = empty_kvcache(qwen_cache_info, 10);
  // kvcc->raw_insert(qwen_cache_info, reinterpret_cast<IDptr>(ids.data()), ids.size(), h1);

  // complete same
  {
    // auto h2 = empty_kvcache(qwen_cache_info, 10);
    kvcc->raw_read(qwen_cache_info, reinterpret_cast<TokenPtr>(ids.data()), ids.size(), h1);
    // cmp_handle_data(qwen_cache_info, h1, h2);
  }

  // complete prefix
  {
    auto h2 = empty_kvcache(qwen_cache_info, 10);
    auto ids2 = std::vector<Token>(ids.begin(), ids.begin() + 3 * BlockLength);
    kvcc->raw_read(qwen_cache_info, reinterpret_cast<TokenPtr>(ids2.data()), ids2.size(), h2);
    cmp_handle_data(qwen_cache_info, h1, h2, 3);
  }

  // common prefix
  {
    auto h2 = empty_kvcache(qwen_cache_info, 10);
    auto ids2 = std::vector<Token>(ids.begin(), ids.begin() + 5 * BlockLength);
    auto rids = random_ids(BlockLength * 2 + BlockLength / 2, gen);
    ids2.insert(ids2.end(), rids.begin(), rids.end());

    kvcc->raw_read(qwen_cache_info, reinterpret_cast<TokenPtr>(ids2.data()), ids2.size(), h2);

    cmp_handle_data(qwen_cache_info, h1, h2, 5);
  }

  // no prefix
  {
    auto h2 = empty_kvcache(qwen_cache_info, 10);

    auto ids2 = random_ids(10 * BlockLength, gen);

    kvcc->raw_read(qwen_cache_info, reinterpret_cast<TokenPtr>(ids2.data()), ids2.size(), h2);
  }

  // insert partly new
  auto h2 = random_kvcache(qwen_cache_info, 10, gen);
  copy_kvcache(h1, h2, 0, 5);
  auto ids2 = random_ids(10 * BlockLength, gen);
  for (size_t i = 0; i < 5 * BlockLength; i++) {
    ids2[i] = ids[i];
  }

  kvcc->raw_insert(qwen_cache_info, reinterpret_cast<TokenPtr>(ids2.data()), ids2.size(), h2);

  // read new part
  {
    auto h3 = empty_kvcache(qwen_cache_info, 10);
    auto ids3 = std::vector<Token>(ids2.begin(), ids2.begin() + 7 * BlockLength);
    ids3.push_back(123);

    kvcc->raw_read(qwen_cache_info, reinterpret_cast<TokenPtr>(ids3.data()), ids3.size(), h3);
    cmp_handle_data(qwen_cache_info, h3, h2, 7);
  }

  kvcc->stop_io_thread();
  io.join();

  SPDLOG_WARN("{} Test Passed", __FILE__);

  return 0;
}