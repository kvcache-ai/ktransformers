#ifndef CPUINFER_OPERATOR_MOE_HPP
#define CPUINFER_OPERATOR_MOE_HPP

// #define CHECK

#include <cstdint>
#include <cstring>
// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../cpu_backend/shared_mem_buffer.h"
#include "../cpu_backend/worker_pool.h"
#include "llama.cpp/ggml.h"

struct GeneralMOEConfig {
  // Basic Config
  int expert_num;
  int routed_expert_num;
  int hidden_size;
  int intermediate_size;
  int layer_idx = 0;
  WorkerPool *pool = nullptr;

  void *gate_proj;
  void *up_proj;
  void *down_proj;

  // for amx
  int max_len = 0;
  std::vector<std::vector<void *>> gate_projs;
  std::vector<std::vector<void *>> up_projs;
  std::vector<std::vector<void *>> down_projs;
  std::vector<std::vector<void *>> gate_scales;
  std::vector<std::vector<void *>> up_scales;
  std::vector<std::vector<void *>> down_scales;

  std::string path;
  bool save;
  bool load;

  // for llamafile
  int m_block = 4;
  int group_min_len = 0;
  int group_max_len = 0;
  int gate_type;
  int up_type;
  int down_type;
  int hidden_type;

  GeneralMOEConfig() {}

  GeneralMOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size)
      : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size),
        intermediate_size(intermediate_size) {}

  int max_possible_qlen() { return std::max(max_len, group_max_len); }
};

template <typename T>
concept MOE_TP_PART = requires(T t, int qlen, int k, const uint64_t *expert_ids, const float *weights,
                               const void *input, void *output, GeneralMOEConfig config, int tp_idx) {
  typename T::output_t;
  { new T(config, tp_idx) } -> std::same_as<T *>;
  { t.forward(qlen, k, expert_ids, weights, input, output) } -> std::same_as<void>;
  // { t.load_weights() } -> std::same_as<void>;
};

inline void debug_bf16(ggml_bf16_t *x) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", ggml_bf16_to_fp32(x[i]));
  }
  printf("\n");
}
inline void debug_f32(float *x) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", x[i]);
  }
  printf("\n");
}

template <MOE_TP_PART T> class TP_MOE_Common {
protected:
  GeneralMOEConfig config;
  std::vector<GeneralMOEConfig> tp_configs;
  int tp_count;
  int me_numa_id;
  std::vector<std::unique_ptr<T>> tps;

  std::vector<typename T::output_t *> local_output_numa;
  T::output_t *local_output = nullptr;

  bool weights_loaded = false;

#ifdef FORWARD_TIME_REPORT
  size_t forward_time_sum_ns = 0;
  size_t forward_count = 0;
#endif
public:
  TP_MOE_Common(GeneralMOEConfig config) : config(config) {
    // printf("TP MOE layer %d, pool: 0x%lx\n", config.layer_idx, (intptr_t)config.pool);
    if (config.pool == nullptr) {
      printf("TP MOE layer %d, no worker pool\n", config.layer_idx);
      throw std::runtime_error("no worker pool");
    }

    this->config = config;
    tp_count = config.pool->config.subpool_count;
    if (config.intermediate_size % tp_count != 0) {
      printf("intermediate_size %d, tp count %d\n", config.intermediate_size, tp_count);
      throw std::runtime_error("For TP, intermediate_size must be a "
                               "multiple of NUMA node count");
    }

    for (auto i = 0; i < tp_count; i++) {
      tps.push_back(nullptr);
      GeneralMOEConfig tp_config = config;
      tp_config.intermediate_size /= tp_count;
      tp_configs.push_back(tp_config);
    }

    config.pool->dispense_backend()->do_numa_job(
        [this, config](int i) { tps[i] = std::move(std::unique_ptr<T>(new T(tp_configs[i], i))); });

    local_output_numa.resize(tp_count, nullptr);
    std::vector<std::pair<void **, uint64_t>> m_mem_requests;
    for (auto i = 0; i < tp_count; i++) {
      m_mem_requests.push_back(
          {(void **)&local_output_numa[i],
           (size_t)sizeof(typename T::output_t) * tp_configs[i].max_possible_qlen() * tp_configs[i].hidden_size});
    }
    m_mem_requests.push_back({(void **)&local_output, sizeof(typename T::output_t) * tp_configs[0].max_possible_qlen() *
                                                          tp_configs[0].hidden_size});
    // printf("local output tp, %d,\n", tp_configs[0].max_possible_qlen());
    shared_mem_buffer.alloc(this, m_mem_requests);
  }

  void warm_up() {
    int qlen = config.max_possible_qlen();
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<uint64_t> expert_ids(qlen * config.routed_expert_num);
    std::vector<float> weights(qlen * config.routed_expert_num);
    for (int i = 0; i < qlen * config.routed_expert_num; i++) {
      expert_ids[i] = i % config.expert_num;
      weights[i] = 0.01;
    }
    forward(&qlen, config.routed_expert_num, expert_ids.data(), weights.data(), input.data(), output.data(), false);
  }

  void forward(int *qlen_ptr, int k, const uint64_t *expert_ids, const float *weights, const void *input, void *output,
               bool incremental) {
    if (weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Not Loaded");
    }
#ifdef FORWARD_TIME_REPORT
    auto start = std::chrono::high_resolution_clock::now();
#endif
    int qlen = *qlen_ptr;

    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) {
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
    });

    merge_results(qlen, output, incremental);

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
    printf("forward time %ld, time stamp:%ld, band width %f GElement/s, ave bandwidth %f GElement/s (only "
           "decode), %f GFLOPS, me numa: %d\n",
           forward_time, end.time_since_epoch().count() / 1000 % 100000000, band_width, average_bandwidth, GFLOPS,
           numa_node_of_cpu(sched_getcpu()));
#endif
  }

  virtual void load_weights() = 0;
  virtual void merge_results(int qlen, void *output, bool incremental) = 0;
};

template <MOE_TP_PART T> class TP_MOE : public TP_MOE_Common<T> {
public:
  using TP_MOE_Common<T>::TP_MOE_Common;
  void load_weights() { throw std::runtime_error("Not Implemented"); }
  void merge_results(int qlen, void *output, bool incremental) { throw std::runtime_error("Not Implemented"); }
};

#endif
