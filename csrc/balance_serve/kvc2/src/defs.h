#ifndef __DEFS_H_
#define __DEFS_H_

#include <cstdint>
#include <optional>
#include <vector>
#include "model_config.h"

namespace kvc2 {
using kvc2_ptr = void*;
// using data_block_ptr = std::intptr_t;
using data_block_ptr = void*;
using layer_data = std::vector<data_block_ptr>;
using kvc2_handle = void*;

using Token = uint32_t;
using Tokens = std::vector<Token>;
using TokenPtr = std::intptr_t;
using TokenLength = size_t;
using BlockLength = size_t;

struct CacheInfo {
  ModelName model_name;
  bool is_key_cache;
  QuantType quant_type;

  size_t hidden_layer_count();
  std::filesystem::path path(std::optional<size_t> which_layer = std::nullopt);
  bool operator==(const CacheInfo& other) const;
  size_t element_size(size_t block_length);
  size_t hash_value() const;
};

};  // namespace kvc2
#endif
