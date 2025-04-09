#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <optional>
#include <vector>
#include "defs.h"
#include "model_config.h"

namespace kvc2 {
struct GPUPageCacheConfig {
  bool gpu_only;
  std::vector<size_t> gpu_devices_id;

  size_t layer_count;
  size_t total_kvcache_pages;
  size_t num_token_per_page;
  size_t num_k_heads;
  size_t k_head_dim;

  bool full_kv_cache_on_each_gpu = false;
  bool k_cache_on = true;
  bool v_cache_on = true;
  torch::ScalarType tensor_type;

  // for cuda stream manager
  size_t num_streams_per_device = 4;
};

struct KVC2Config {
  bool k_cache_on = true;
  bool v_cache_on = true;
  bool gpu_only = false;
  bool load_from_disk = true;
  bool save_to_disk = true;
  std::string path;
  std::string config_path;
  TokenLength num_token_per_page = 256;
  size_t memory_pool_size = 10e9;
  size_t evict_count = 20;
  std::optional<GPUPageCacheConfig> gpu_cache_config = std::nullopt;
  size_t metrics_port;
  double recompute_ratio = 0.2;
};

class DoubleCacheHandleInterface;
class KVC2Interface {
 public:
  virtual ~KVC2Interface() = default;

  virtual void load() = 0;
  virtual void save() = 0;
  /*
Raw Insert
Insert kvcache from kvcache_data to disk.

info: cache info
id: start pointer of token array
length: length of token array
kvcache_data: data of kvcache

This will firstly match the ID array with the existing kvcache, and then insert the unmatched kvcache to disk.
*/
  virtual void raw_insert(ModelName model_name, QuantType quant_type, Token* id, TokenLength length,
                          const std::vector<layer_data>& k_cache, const std::vector<layer_data>& v_cache) = 0;

  /*
Raw Read
Read kvcache from disk to user specified pointers.

info: cache info
id: start pointer of token array
length: length of token array
kvcache_data: data of kvcache
Return:  matched length of prefix, in tokens

This will not read from memory pool, it directly read from disk.
*/
  virtual TokenLength raw_read(ModelName model_name, QuantType quant_type, Token* id, TokenLength length,
                               const std::vector<layer_data>& k_cache, const std::vector<layer_data>& v_cache) = 0;

  /*
  Lookup
  Lookup kvcache and load it from disk to memory pool if needed.

  info: cache info
  id: start pointer of token array
  length: length of token array

  Return:  kvc2_handle, holds kvcache until being released.
           if not found, matched_length will return 0.
           if memory pool is full, return nullptr
  */
  virtual std::shared_ptr<DoubleCacheHandleInterface> lookup(ModelName model_name, QuantType quant_type, Token* id,
                                                             TokenLength length, TokenLength estimated_length) = 0;

  /*
  Lookup and allocate to gpu
  info.is_k_cache does not matter here
  */
  virtual std::shared_ptr<DoubleCacheHandleInterface> lookup_to_gpu(ModelName model_name, QuantType quant_type,
                                                                    Token* id, TokenLength length,
                                                                    TokenLength estimated_length) = 0;

  virtual void lookup_to_gpu_async(ModelName model_name, QuantType quant_type, Token* id, TokenLength length,
                                   TokenLength estimated_length,
                                   std::function<void(std::shared_ptr<DoubleCacheHandleInterface>)> call_back) = 0;

  virtual std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_kvcache() = 0;

  virtual void debug() = 0;
};

std::shared_ptr<KVC2Interface> create_kvc2(KVC2Config config);

enum MatchStatus {
  Exact,
  Partial,
  NotMatchExact,
  NotMatchPartial,
};

class DoubleCacheHandleInterface {
 public:
  virtual ~DoubleCacheHandleInterface() = default;
  virtual TokenLength matched_length() = 0;
  virtual std::vector<MatchStatus> matched_status() = 0;
  virtual std::vector<layer_data> handle_data(bool is_key_cache) = 0;
  virtual bool to_gpu() = 0;
  virtual void to_gpu_async(std::function<void(bool)> call_back) = 0;
  virtual std::vector<size_t> get_gpu_block_idx() = 0;
  virtual std::vector<size_t> get_gpu_attached_block_idx() = 0;

  virtual void append_tokens(Token* tokens, TokenLength length) = 0;  // update generated tokens

  virtual void debug() = 0;
};

};  // namespace kvc2
