#include <optional>
#include <random>
#include "kvc2.h"
#define FMT_HEADER_ONLY
#include <spdlog/spdlog.h>

const int BlockLength = 256;

std::string FLAGS_disk_cache_path;

void init(int argc, char* argv[]) {
  if (argc != 2) {
    fmt::print("Usage: {} --disk_cache_path=xxx\n", argv[0]);
    exit(1);
  }
  FLAGS_disk_cache_path = argv[1];
  if (FLAGS_disk_cache_path.empty()) {
    fmt::print("disk_cache_path is empty");
    exit(1);
  }
}

using namespace kvc2;

data_block_ptr empty_block(CacheInfo info) {
  auto re = new (std::align_val_t(4096)) std::byte[info.element_size(BlockLength)];
  return reinterpret_cast<data_block_ptr>(re);
}

data_block_ptr random_block(CacheInfo info, std::mt19937& gen) {
  auto re = empty_block(info);
  uint64_t* d = (uint64_t*)re;
  for (size_t i = 0; i < info.element_size(BlockLength) / 8; i++) {
    d[i] = gen();
  }
  return re;
}
layer_data random_blocks(CacheInfo info, size_t block_count, size_t seed) {
  std::mt19937 gen(seed);
  layer_data re;
  for (size_t i = 0; i < block_count; i++) {
    re.push_back(random_block(info, gen));
  }
  return re;
}

layer_data empty_blocks(CacheInfo info, size_t block_count) {
  layer_data re;
  for (size_t i = 0; i < block_count; i++) {
    re.push_back(empty_block(info));
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

std::vector<layer_data> random_kvcache(CacheInfo info, size_t block_count, std::mt19937& gen) {
  std::vector<layer_data> re;
  re.resize(info.hidden_layer_count());
  fmt::print("Generating random kvcache, layer {}\n", info.hidden_layer_count());
#pragma omp parallel for
  for (size_t i = 0; i < info.hidden_layer_count(); i++) {
    re[i] = random_blocks(info, block_count, gen());
  }
  return re;
}

std::vector<layer_data> empty_kvcache(CacheInfo info, size_t block_count) {
  std::vector<layer_data> re;
  re.resize(info.hidden_layer_count());
  fmt::print("Generating empty kvcache, layer {}\n", info.hidden_layer_count());
#pragma omp parallel for
  for (size_t i = 0; i < info.hidden_layer_count(); i++) {
    re[i] = empty_blocks(info, block_count);
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

CacheInfo qwen_cache_info = {
    .model_name = "qwen2-72b-instruct",
    .is_key_cache = true,
    .quant_type = "BF16",
};

void cmp_handle_data(CacheInfo info, std::vector<layer_data>& h1, std::vector<layer_data>& h2,
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
      assert(memcmp(e1, e2, info.element_size(BlockLength)) == 0);
    }
  }
  fmt::print("KVCacheHandle cmp ok\n");
}
