#ifndef CPUINFER_OPERATOR_MLA_HPP
#define CPUINFER_OPERATOR_MLA_HPP

#include "common.hpp"

template <typename T>
// qlens: token count for each query
// cache_pages: kv_cache for all queries in the current layer
// page_tables: kv_cache page table for each query ([query_idx][page_idx])
// kv_lens: kv_cache length for each query
// input: input tensor, shape [qlen, hidden_size]
// output: output tensor, shape [qlen, hidden_size]
// config: GeneralMLAConfig
// tp_idx: thread pool index
// T must have the following methods:
concept MLA_TP_PART =
    requires(T t, std::vector<int> qlens, std::vector<void*> kv_lora_pages, std::vector<void*> pe_pages,
             std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens, const void* input, void* output,
             GeneralMLAConfig config, int tp_idx, int page_count, std::vector<void*> attention_masks) {
      typename T::output_t;
      { new T(config, tp_idx) } -> std::same_as<T*>;
      { t.set_pages(kv_lora_pages, pe_pages) } -> std::same_as<void>;
      { t.set_local_pages(page_count) } -> std::same_as<void>;
      { t.forward(qlens, page_tables, kv_lens, input, output) } -> std::same_as<void>;
      { t.forward(qlens, page_tables, kv_lens, attention_masks, input, output) } -> std::same_as<void>;
    };

template <MLA_TP_PART T>
class TP_MLA_Common : public MLA_Interface {
 protected:
  GeneralMLAConfig config;
  std::vector<GeneralMLAConfig> tp_configs;
  int tp_count;
  int me_numa_id;
  std::vector<std::unique_ptr<T>> tps;

  std::vector<typename T::output_t*> local_output_numa;

  bool weights_loaded = false;
#ifdef FORWARD_TIME_REPORT
  size_t forward_time_sum_ns = 0;
  size_t forward_count = 0;
#endif

 public:
  TP_MLA_Common(GeneralMLAConfig config) : config(config) {
    printf("TP MLA layer %d, pool: 0x%lx\n", config.layer_idx, (intptr_t)config.pool);
    if (config.pool == nullptr) {
      printf("TP MLA layer %d, no worker pool\n", config.layer_idx);
      throw std::runtime_error("no worker pool");
    }

    this->config = config;
    tp_count = config.pool->config.subpool_count;
    if (config.hidden_size % tp_count != 0) {
      printf("hidden_size %d, tp count %d\n", config.hidden_size, tp_count);
      throw std::runtime_error(
          "For TP, hidden_size must be a "
          "multiple of NUMA node count");
    }

    for (auto i = 0; i < tp_count; i++) {
      tps.push_back(nullptr);
    }

    tp_configs.resize(tp_count);
    config.pool->dispense_backend()->do_numa_job([this, config](int i) {
      tp_configs[i] = config;
      tp_configs[i].num_heads /= tp_count;
      tps[i] = std::move(std::unique_ptr<T>(new T(tp_configs[i], i)));
    });

    local_output_numa.resize(tp_count, nullptr);
    MemoryRequest mem_requests;
    for (auto i = 0; i < tp_count; i++) {
      mem_requests.append_pointer(&local_output_numa[i],
                                  sizeof(typename T::output_t) * tp_configs[i].max_qlen * tp_configs[i].hidden_size);
    }
    shared_mem_buffer.alloc(this, mem_requests);
  }

  void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
               const void* input, void* output) override {
    if (weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Not Loaded");
    }
#ifdef FORWARD_TIME_REPORT
    auto start = std::chrono::high_resolution_clock::now();
#endif

    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, pool, qlens, page_tables, kv_lens, input](int numa_id) {
      tps[numa_id]->forward(qlens, page_tables, kv_lens, input, this->local_output_numa[numa_id]);
    });
    int qlen_sum = 0;
    for (auto i = 0; i < qlens.size(); i++) {
      qlen_sum += qlens[i];
    }

    merge_results(qlen_sum, output);

#ifdef FORWARD_TIME_REPORT
    auto end = std::chrono::high_resolution_clock::now();
    auto forward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto band_width = (1.0 * config.routed_expert_num * config.hidden_size * config.intermediate_size * 3 / 1e9) /
                      (1.0 * forward_time / 1e6);
    auto GFLOPS =
        (1.0 * config.hidden_size * config.intermediate_size * qlen * 3 * config.routed_expert_num * 2 / 1e9) /
        (1.0 * forward_time / 1e6);
    if (qlen <= 10) {
      forward_time_sum_ns += forward_time;
      forward_count++;
    }
    auto average_bandwidth =
        (1.0 * forward_count * config.routed_expert_num * config.hidden_size * config.intermediate_size * 3 / 1e9) /
        (1.0 * forward_time_sum_ns / 1e6);
    printf(
        "forward time %ld, time stamp:%ld, band width %f GElement/s, ave bandwidth %f GElement/s (only "
        "decode), %f GFLOPS, me numa: %d\n",
        forward_time, end.time_since_epoch().count() / 1000 % 100000000, band_width, average_bandwidth, GFLOPS,
        numa_node_of_cpu(sched_getcpu()));
#endif
  }

  void set_pages(std::vector<std::vector<void*>> kv_lora_pages, std::vector<std::vector<void*>> pe_pages) {
    for (auto i = 0; i < tp_count; i++) {
      tps[i]->set_pages(kv_lora_pages[i], pe_pages[i]);
    }
  }

  void set_local_pages(int page_count) {
    config.pool->dispense_backend()->do_numa_job(
        [this, page_count](int tp_idx) { tps[tp_idx]->set_local_pages(page_count); });
  }

  virtual void load_weights() = 0;
  virtual void merge_results(int qlen, void* output) = 0;
};

template <MLA_TP_PART T>
class TP_MLA : public TP_MLA_Common<T> {
 public:
  using TP_MLA_Common<T>::TP_MLA_Common;
  void load_weights() { throw std::runtime_error("Not Implemented"); }
  void merge_results(int qlen, void* output) { throw std::runtime_error("Not Implemented"); }
};

#endif