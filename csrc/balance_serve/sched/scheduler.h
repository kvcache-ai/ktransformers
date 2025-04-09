#pragma once
#include "model_config.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <torch/torch.h>
#include <vector>

namespace scheduler {

using Token = uint32_t;
using QueryID = uint64_t;
constexpr QueryID NoQueryID = 0;

using TokenLength = size_t;
using BatchID = uint64_t;

using PageCount = size_t;

struct ModelSettings {
  std::string model_path;
  size_t params_count;
  size_t layer_count;
  size_t num_k_heads;
  size_t k_head_dim;

  double bytes_per_params;
  double bytes_per_kv_cache_element;

  inline size_t params_nbytes() { return params_count * bytes_per_params; }
  inline size_t bytes_per_token_kv_cache() {
    return bytes_per_kv_cache_element * num_k_heads * k_head_dim;
  }
};

struct SampleOptions {
  double temperature = 1.0;
  double top_p = 1.0;
};

struct Settings {
  // something is aukward here, kvc2 only use model_name and quant_type to get
  // model infos.
  ModelName model_name;
  QuantType quant_type;
  // model_setting is ignore by kvc2
  ModelSettings model_settings;

  size_t page_size = 256;            // how many token in a page
  std::vector<size_t> gpu_device_id; //
  size_t gpu_memory_size;            // memory size in bytes of each GPU, each
  double memory_utilization_percentage;

  size_t max_batch_size = 256;

  size_t recommended_chunk_prefill_token_count;
  SampleOptions sample_options;
  size_t sched_metrics_port;

  // for kvc2
  bool gpu_only;
  bool use_self_defined_head_dim = false;
  size_t self_defined_head_dim;
  bool full_kv_cache_on_each_gpu = false;
  bool k_cache_on = true;
  bool v_cache_on = true;
  std::string kvc2_config_path;
  std::string kvc2_root_path;
  double memory_pool_size_GB = 100;
  size_t evict_count = 20;
  size_t kvc2_metrics_port;
  bool load_from_disk = false;
  bool save_to_disk = false;

  // for strategy
  std::string strategy_name;

  // derived
  size_t gpu_device_count;
  std::optional<size_t> total_kvcache_pages;
  std::vector<torch::Device> devices;
  void auto_derive();
};

using PrefillTask =
    std::tuple<QueryID, TokenLength, TokenLength>; // id, start, length

struct BatchQueryTodo {
  // query
  std::vector<QueryID> query_ids;
  std::vector<torch::Tensor> query_tokens;
  std::vector<TokenLength> query_lengths;
  std::vector<torch::Tensor>
      block_indexes; // (max_num_blocks_per_seq), dtype torch.int32.
  std::optional<torch::Tensor> attn_masks;
  std::optional<torch::Tensor> rope_ranges;
  std::vector<SampleOptions> sample_options;
  std::vector<std::vector<std::vector<int>>> stop_criteria;

  // mini batches, adjacent two mini batches are executed together
  // tasks count must be <=2, because of flash infer attention
  std::vector<PrefillTask>
      prefill_mini_batches; // prefill minibatch only has 1 prefill
  std::vector<std::vector<QueryID>>
      decode_mini_batches; // decode minibatch has multiple decode

  std::string debug();
  bool empty();
};

struct QueryUpdate {
  QueryID id;
  bool ok;
  bool is_prefill;
  bool decode_done;            // no use for now
  TokenLength active_position; // the position where no kvcache now,
                               // kvcache[active_position] == None

  Token generated_token;

  std::string debug() const;
};

using BatchQueryUpdate = std::vector<QueryUpdate>;

struct InferenceContext {
  std::vector<torch::Tensor> k_cache; // [gpu num] (layer_count, num blocks,
                                      // page size, kheadnum, head_dim)
  std::vector<torch::Tensor> v_cache;
};

using UserID = int64_t;
constexpr UserID NoUser = -1;
const int MAX_SLO_TIME = 1e9;

struct QueryAdd {
  std::vector<Token> query_token; // int here
  // torch::Tensor attn_mask;
  TokenLength query_length;
  TokenLength estimated_length;

  std::vector<std::vector<int>> stop_criteria;

  SampleOptions sample_options;

  UserID user_id;
  int SLO_TTFT_ms = MAX_SLO_TIME;
  int SLO_TBT_ms = MAX_SLO_TIME;

  std::string serialize();
  static QueryAdd deserialize(const std::string &input);
};

class Scheduler {
public:
  virtual void init(Settings settings) = 0;

  virtual void run() = 0;
  virtual void stop() = 0;

  // webserver call this
  virtual QueryID add_query(QueryAdd query) = 0;
  virtual void cancel_query(QueryID id) = 0;

  // inference loop call this
  virtual std::shared_ptr<BatchQueryTodo>
  update_last_batch(BatchQueryUpdate updates) = 0;
  virtual InferenceContext get_inference_context() = 0;

  virtual ~Scheduler() = default;
};

std::shared_ptr<Scheduler> create_scheduler(Settings settings);

}; // namespace scheduler