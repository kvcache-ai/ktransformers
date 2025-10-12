#ifndef CPUINFER_OPERATOR_COMMON_HPP
#define CPUINFER_OPERATOR_COMMON_HPP

#include "../cpu_backend/shared_mem_buffer.h"
#include "../cpu_backend/worker_pool.h"
#include "llama.cpp/ggml.h"

#if defined(__aarch64__) && defined(CPU_USE_KML)
#include <arm_sve.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#define ASSERT_RELEASE(x, text)                                                            \
  do {                                                                                     \
    if (!(x)) {                                                                            \
      fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", #x, __FILE__, __LINE__); \
      fprintf(stderr, "Error message: %s\n", (text));                                      \
      throw std::runtime_error((text));                                                    \
    }                                                                                      \
  } while (0)

#define PUSH_MEM_REQ(ptr, size) mem_requests.append_pointer(&(ptr), (size))

#define PROFILE_RECORD_TIME_STAMP(name)                                                             \
  do {                                                                                              \
    auto end_time = std::chrono::high_resolution_clock::now();                                      \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - last).count(); \
    time_map[(name)] = duration;                                                                    \
    last = end_time;                                                                                \
  } while (0)

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T>
T* offset_pointer(T* ptr, size_t byte_offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + byte_offset);
}

template <typename T>
size_t pointer_offset(T* ptr, T* b) {
  return reinterpret_cast<size_t>(b) - reinterpret_cast<size_t>(ptr);
}

template <typename T>
const T* offset_pointer(const T* ptr, size_t byte_offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(ptr) + byte_offset);
}

template <typename T>
T* offset_pointer_row_major(T* t, int row, int col, size_t ld) {
  return offset_pointer(t, row * ld) + col;
}

template <typename T>
T* offset_pointer_col_major(T* t, int row, int col, size_t ld) {
  return offset_pointer(t, col * ld) + row;
}

class TimePerf {
 protected:
  std::string time_perf_name;
  std::map<std::string, long> time_map;
  std::chrono::time_point<std::chrono::high_resolution_clock> last;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

  void forward_perf_start() {
    start_time = std::chrono::high_resolution_clock::now();
    last = start_time;
  }

  void perf_report() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::string output = time_perf_name + ", forward time: " + std::to_string(duration.count()) + " us";
    // for (auto [name, t] : time_map) {
    //   double p = 100.0 * t / duration.count();
    //   // if (p < 1.0) {
    //   //   continue; // Skip if the percentage is less than 1%
    //   // }
    //   output += ", " + name + ": " + std::to_string(t) + " us(" + std::to_string(size_t(round(p))) + "%)";
    // }
    // 反向遍历
    for (auto it = time_map.rbegin(); it != time_map.rend(); ++it) {
      const std::string& name = it->first;
      long t = it->second;
      double p = 100.0 * t / duration.count();
      // if (p < 1.0) {
      //   continue; // Skip if the percentage is less than 1%
      // }
      output += ", " + name + ": " + std::to_string(t) + " us(" + std::to_string(size_t(round(p))) + "%)";
    }
    printf("%s\n", output.c_str());
  }
};

struct TaskCounter {
  std::vector<size_t> fold = {}, card = {};

  TaskCounter(std::initializer_list<size_t> i) {
    card.push_back(1);
    for (auto j : i) {
      push_back(j);
    }
  }

  void push_back(size_t i) {
    fold.push_back(i);
    for (auto& c : card) {
      c *= i;
    }
    card.push_back(1);
  }
  void push_back(std::vector<size_t> i) {
    for (auto j : i) {
      push_back(j);
    }
  }
  size_t count() { return card[0]; }
  size_t at(size_t id, size_t which) { return id % card.at(which) / card.at(which + 1); }
};

struct GeneralConfig {
  size_t vocab_size;
  size_t hidden_size;

  size_t num_experts_per_tok;
  size_t n_routed_experts;
  size_t n_shared_experts;
  size_t max_qlen = 4096;

  void* lm_heads_ptr;
  ggml_type lm_heads_type;
  void* norm_weights_ptr;
  ggml_type norm_weights_type;
  void* token_embd_ptr;
  ggml_type token_embd_type;
  WorkerPool* pool = nullptr;
  GeneralConfig() {}
};

struct GeneralMLAConfig {
  size_t hidden_size;
  size_t q_lora_rank;
  size_t num_heads;
  size_t nope_size;
  size_t rope_size;
  size_t kv_lora_rank;

  int layer_idx = 0;
  WorkerPool* pool = nullptr;
  size_t token_count_in_page = 256;  // token count in a page
  size_t max_qlen = 1024;
  size_t max_kvlen = 4096;

  // rope
  size_t max_position_embeddings;
  double rope_scaling_factor = 1.0;
  double rope_theta = 10000.0;
  double rope_scaling_beta_fast;
  double rope_scaling_beta_slow;
  double rope_scaling_mscale;
  double rope_scaling_mscale_all_dim;
  double rope_scaling_original_max_position_embeddings;

  void* q_a_proj;
  void* q_a_norm = nullptr;
  void* q_b_proj;
  void* kv_a_proj_with_mqa;
  void* kv_a_norm = nullptr;
  void* kv_b_proj;
  void* o_proj;

  // for llamafile
  ggml_type q_a_proj_type;
  ggml_type q_a_norm_type;
  ggml_type q_b_proj_type;
  ggml_type kv_a_proj_with_mqa_type;
  ggml_type kv_a_norm_type;
  ggml_type kv_b_proj_type;
  ggml_type w_o_type;

  ggml_type input_type = GGML_TYPE_F32;
  ggml_type output_type = GGML_TYPE_F32;

  size_t m_block = 4;
  size_t n_block = 4;
  // for kvcache
  size_t page_count = 200;  // page count for kv cache

  GeneralMLAConfig() {}
  GeneralMLAConfig(size_t hidden_size, size_t q_lora_rank, size_t kv_lora_rank, size_t num_heads, size_t nope_size,
                   size_t rope_size)
      : hidden_size(hidden_size),
        q_lora_rank(q_lora_rank),
        kv_lora_rank(kv_lora_rank),
        num_heads(num_heads),
        nope_size(nope_size),
        rope_size(rope_size) {}
};

struct QuantConfig {
  std::string quant_method = "";
  int bits = 0;
  int group_size = 0;
  bool zero_point;
};

struct GeneralMOEConfig {
  // Basic Config
  int expert_num;
  int num_experts_per_tok;
  int hidden_size;
  int intermediate_size;

  int layer_idx = 0;
  WorkerPool* pool = nullptr;

  // SGLang offload
  int num_gpu_experts = 0;
  void* physical_to_logical_map = nullptr;

  void* gate_proj;
  void* up_proj;
  void* down_proj;

  void* gate_scale;
  void* up_scale;
  void* down_scale;

  void* gate_zero;
  void* up_zero;
  void* down_zero;

  QuantConfig quant_config;

  // for amx
  int max_len = 0;
  std::vector<std::vector<void*>> gate_projs;
  std::vector<std::vector<void*>> up_projs;
  std::vector<std::vector<void*>> down_projs;
  std::vector<std::vector<void*>> gate_scales;
  std::vector<std::vector<void*>> up_scales;
  std::vector<std::vector<void*>> down_scales;
  std::vector<std::vector<void*>> gate_zeros;
  std::vector<std::vector<void*>> up_zeros;
  std::vector<std::vector<void*>> down_zeros;

  std::string path;
  bool save = false;
  bool load = false;

  // for llamafile
  int m_block = 4;
  int group_min_len = 0;
  int group_max_len = 0;
  int gate_type;
  int up_type;
  int down_type;
  int hidden_type;

  GeneralMOEConfig() {}

  GeneralMOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int num_gpu_experts)
      : expert_num(expert_num),
        num_experts_per_tok(routed_expert_num),
        hidden_size(hidden_size),
        intermediate_size(intermediate_size),
        num_gpu_experts(num_gpu_experts) {}

  int max_possible_qlen() { return std::max(max_len, group_max_len); }
};

struct GeneralGateConfig {
  size_t hidden_size;
  size_t num_experts_per_tok;
  size_t n_routed_experts;
  size_t n_group;
  size_t topk_group;

  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;

  std::string scoring_func = "sigmoid";
  std::string topk_method = "noaux_tc";

  int layer_idx = 0;
  WorkerPool* pool = nullptr;

  void* weight = nullptr;
  ggml_type weight_type;
  void* e_score_correction_bias = nullptr;
  ggml_type e_score_correction_bias_type;

  size_t max_seqlen = 25600;

  GeneralGateConfig() = default;

  GeneralGateConfig(int hidden_size, int num_experts_per_tok, int n_routed_experts, int n_group, int topk_group)
      : hidden_size(hidden_size),
        num_experts_per_tok(num_experts_per_tok),
        n_routed_experts(n_routed_experts),
        n_group(n_group),
        topk_group(topk_group) {}
};

class MLA_Interface {
 public:
  virtual void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
                       const void* input, void* output) = 0;
};

class MoE_Interface {
 public:
  virtual void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output, bool incremental = false) = 0;
};
inline void init_ggml() {
  static bool inited = false;
  if (inited) {
    return;
  }
  struct ggml_init_params params = {
      0,
      NULL,
      true,
  };

  auto ctx_eval = ggml_init(params);

  if (!ctx_eval) {
    throw std::runtime_error("Failed to create ggml context");
  }
  inited = true;
}

template <typename A, typename B>
void convert_or_copy(A* dst, const B* src, size_t count) {
  if constexpr (std::is_same_v<A, B>) {
    // printf("Direct copy\n");
    memcpy(dst, src, sizeof(A) * count);
  } else {
    if constexpr (std::is_same_v<A, float>) {
      if constexpr (std::is_same_v<B, ggml_bf16_t>) {
        // printf("Converting ggml_bf16_t to float\n");
        ggml_bf16_to_fp32_row(src, dst, count);
      } else if constexpr (std::is_same_v<B, ggml_fp16_t>) {
        ggml_fp16_to_fp32_row(src, dst, count);
      } else {
        throw std::runtime_error("Unsupported conversion");
      }
    } else if constexpr (std::is_same_v<A, ggml_bf16_t>) {
      if constexpr (std::is_same_v<B, float>) {
        // printf("Converting float to ggml_bf16_t\n");
        ggml_fp32_to_bf16_row(src, dst, count);
      } else {
        throw std::runtime_error("Unsupported conversion");
      }
    }

    else {
      throw std::runtime_error("Unsupported conversion");
    }
  }
}

template <typename A>
void convert_or_copy(A* dst, void* src, ggml_type type, size_t count) {
  switch (type) {
    case GGML_TYPE_BF16: {
      auto src_bf16 = (ggml_bf16_t*)src;
      convert_or_copy(dst, src_bf16, count);
      break;
    }
    case GGML_TYPE_F16: {
#if defined(__aarch64__) && defined(CPU_USE_KML)
      auto src_fp16 = (float16_t*)src;
      convert_or_copy(dst, src_fp16, count);
#else
      throw std::runtime_error("GGML_TYPE_F16 is not supported on this platform");
#endif
      break;
    }
    case GGML_TYPE_F32: {
      auto src_f32 = (float*)src;
      convert_or_copy(dst, src_f32, count);
      break;
    }
    default:
      throw std::runtime_error("Unsupported type for conversion");
  }
}

template <typename A>
void check_numerics(A* data, size_t count) {
  for (size_t i = 0; i < count; i++) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      printf("Numerics check failed at index %zu: value = %f\n", i, data[i]);
      throw std::runtime_error("Numerics check failed");
    }
  }
  printf("Numerics check passed for %zu elements.\n", count);
}

inline void debug_bf16(ggml_bf16_t* x) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", ggml_bf16_to_fp32(x[i]));
  }
  printf("\n");
}
inline void debug_f32(float* x) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", x[i]);
  }
  printf("\n");
}

inline void debug_f32(float* x, size_t count) {
  if (count < 10) {
    for (size_t i = 0; i < count; i++) {
      printf("%f ", x[i]);
    }
  } else {
    for (size_t i = 0; i < 3; i++) {
      printf("%f ", x[i]);
    }
    printf("...");
    for (size_t i = count - 3; i < count; i++) {
      printf("%f ", x[i]);
    }
    printf("\n");
  }
}

#endif
