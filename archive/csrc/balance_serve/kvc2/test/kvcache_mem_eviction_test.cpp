#include "kvcache_test_utils.cpp"

int main(int argc, char* argv[]) {
  parse_and_check(argc, argv);
  spdlog::set_level(spdlog::level::debug);
  std::mt19937 gen(123);

  KVC2 kvc2(FLAGS_disk_cache_path);
  auto io = kvc2.io_dealer->start_io_thread();

  SPDLOG_WARN("Insert 10 x 10 KVCache");
  std::vector<KVCacheHandle> handles(10);
  for (int i = 0; i < 10; i++) {
    handles[i] = random_kvcache(qwen_cache_info, 10, gen);
    auto& h1 = handles[i];
    h1.ids = random_ids(10 * BlockLength, gen);
    kvc2.raw_insert(h1);
  }

  SPDLOG_WARN("Cache Eviction Test");
  {
    for (int i = 0; i < 10; i++) {
      auto& h = handles[i];
      SPDLOG_WARN("Lookup {}", i);
      auto x = kvc2.lookup(qwen_cache_info, h.ids.data(), h.ids.size());
      cmp_handle_data(h, *x);
    }
    SPDLOG_WARN("Simple Eviction OK");
  }

  {
    std::vector<std::shared_ptr<KVCacheHandle>> lookup_handles;
    for (int i = 0; i < 10; i++) {
      auto& h = handles[i];
      SPDLOG_WARN("Lookup {}", i);
      auto x = kvc2.lookup(qwen_cache_info, h.ids.data(), h.ids.size());
      if (i >= 5) {
        assert(x == nullptr);
        continue;
      }
      lookup_handles.push_back(x);
      cmp_handle_data(h, *x);
    }
    SPDLOG_WARN("Cannot Eviction OK");
  }

  kvc2.io_dealer->stop();
  io.join();

  SPDLOG_WARN("{} Test Passed", __FILE__);
  return 0;
}