/**
 * @Description : SYCL GPTQ INT4 MoE backend for Intel GPUs
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Supports symmetric GPTQ INT4 weights stored as qweight [K/8, N] and
 * scales [K/group_size, N]. Weights are reordered once at load time to an
 * output-major layout. Decode uses packed SG16 gate/up kernels, per-expert
 * asynchronous submission, and a two-subgroup down-projection work-group.
 */
#ifndef CPUINFER_OPERATOR_SYCL_GPTQ_INT4_MOE_H
#define CPUINFER_OPERATOR_SYCL_GPTQ_INT4_MOE_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "../avx2/moe_base.hpp"

namespace sycl_int4 {

constexpr int kSubgroupSize = 16;
constexpr int kDownRowsPerWorkGroup = 2;

inline sycl::queue& queue() {
  static sycl::queue q([] {
    auto async_handler = [](sycl::exception_list exceptions) {
      for (const auto& exception : exceptions) {
        try {
          std::rethrow_exception(exception);
        } catch (const sycl::exception& error) {
          std::fprintf(stderr, "SYCL GPTQ INT4 asynchronous error: %s\n", error.what());
        }
      }
    };

    try {
      const char* filter = std::getenv("KT_SYCL_DEVICE_FILTER");
      if (filter != nullptr && filter[0] != '\0') {
        return sycl::queue(sycl::ext::oneapi::filter_selector(filter), async_handler);
      }

      const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
      if (devices.empty()) {
        throw std::runtime_error("No SYCL GPU device is available");
      }
      for (const auto& device : devices) {
        if (device.get_platform().get_backend() == sycl::backend::ext_oneapi_level_zero) {
          return sycl::queue(device, async_handler);
        }
      }
      return sycl::queue(devices.front(), async_handler);
    } catch (const sycl::exception& error) {
      throw std::runtime_error(
          std::string("Failed to create the SYCL GPTQ INT4 queue. Check sycl-ls, "
                      "the render-group permission, or set KT_SYCL_DEVICE_FILTER. Original error: ") +
          error.what());
    }
  }());
  return q;
}

template <typename T>
inline T* usm_alloc(size_t elements, const char* name) {
  elements = std::max<size_t>(elements, 1);
  T* pointer = sycl::malloc_shared<T>(elements, queue());
  if (pointer == nullptr) {
    throw std::runtime_error(std::string("SYCL shared-USM allocation failed for ") + name);
  }
  return pointer;
}

inline void usm_free(void* pointer) {
  if (pointer != nullptr) sycl::free(pointer, queue());
}

inline float bf16_to_fp32(uint16_t value) { return sycl::bit_cast<float>(static_cast<uint32_t>(value) << 16); }

inline uint16_t fp32_to_bf16(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

struct DecodeScratch {
  std::vector<sycl::event> gate_up_events;
  std::vector<int> gate_up_experts;
  int active_experts = 0;
  bool gate_up_pending = false;

  uint32_t* gate_qweight = nullptr;
  uint32_t* up_qweight = nullptr;
  uint32_t* down_qweight = nullptr;
  float* gate_scales = nullptr;
  float* up_scales = nullptr;
  float* down_scales = nullptr;
  size_t gate_up_qweight_stride = 0;
  size_t gate_up_scale_stride = 0;
  size_t down_qweight_stride = 0;
  size_t down_scale_stride = 0;
  bool weights_ready = false;
};

struct GemmKernelSYCLGPTQInt4 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 1;
  static constexpr int K_STEP = 1;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {
    static std::once_flag once;
    std::call_once(once, [] {
      const auto device = queue().get_device();
      if (!device.get_info<sycl::info::device::usm_shared_allocations>()) {
        throw std::runtime_error("SYCL GPTQ INT4 requires shared-USM support");
      }
    });
  }

  static int recommended_nth(int) { return 1; }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return avx2::split_range(n, ith, nth); }

  struct BufferA {
    uint16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;
    size_t capacity_m = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void*) : max_m(m), k(k_) {}
    BufferA(const BufferA&) = delete;
    BufferA& operator=(const BufferA&) = delete;
    ~BufferA() { usm_free(data); }

    static size_t required_size(size_t, size_t) { return 1; }

    void ensure(size_t m) {
      if (m <= capacity_m) return;
      usm_free(data);
      data = usm_alloc<uint16_t>(m * k, "GPTQ INT4 activation buffer");
      capacity_m = m;
    }

    void set_data(void*) { ensure(max_m); }

    void from_mat(int m, const ggml_bf16_t* source, int ith, int nth) {
      ensure(static_cast<size_t>(m));
      if (ith == 0 && nth == 1) {
        std::memcpy(data, source, static_cast<size_t>(m) * k * sizeof(uint16_t));
        return;
      }
      auto [begin, end] = avx2::split_range(m, ith, nth);
      std::memcpy(data + static_cast<size_t>(begin) * k, source + static_cast<size_t>(begin) * k,
                  static_cast<size_t>(end - begin) * k * sizeof(uint16_t));
    }
  };

  struct BufferB {
    uint32_t* qweight = nullptr;
    float* scales = nullptr;
    int n = 0;
    int k = 0;
    int group_size = 128;
    int num_groups = 0;
    int k_packed = 0;
    bool owns_storage = true;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int group_size_, void*)
        : n(static_cast<int>(n_)), k(static_cast<int>(k_)), group_size(group_size_) {
      if (group_size <= 0 || (k % 8) != 0 || (k % group_size) != 0) {
        throw std::runtime_error("SYCL GPTQ INT4 requires K divisible by 8 and by group_size");
      }
      k_packed = k / 8;
      num_groups = k / group_size;
    }

    BufferB(const BufferB&) = delete;
    BufferB& operator=(const BufferB&) = delete;

    ~BufferB() {
      if (owns_storage) {
        usm_free(qweight);
        usm_free(scales);
      }
    }

    static size_t required_size(size_t, size_t, int) { return 1; }
    size_t qweight_elements() const { return static_cast<size_t>(n) * k_packed; }
    size_t scale_elements() const { return static_cast<size_t>(n) * num_groups; }

    void bind_external(uint32_t* qweight_pointer, float* scale_pointer, bool take_ownership) {
      if (owns_storage) {
        usm_free(qweight);
        usm_free(scales);
      }
      qweight = qweight_pointer;
      scales = scale_pointer;
      owns_storage = take_ownership;
    }

    void ensure() {
      if (qweight == nullptr) {
        qweight = usm_alloc<uint32_t>(qweight_elements(), "GPTQ INT4 qweight");
      }
      if (scales == nullptr) {
        scales = usm_alloc<float>(scale_elements(), "GPTQ INT4 scales");
      }
    }

    // Source layout is [K/8, N] and [K/group_size, N]. Device layout is
    // output-major [N, K/8] and [N, K/group_size].
    void from_mat(const uint32_t* source_qweight, const float* source_scales, int ith, int nth) {
      ensure();
      auto [begin, end] = avx2::split_range(n, ith, nth);
      for (int output = begin; output < end; ++output) {
        for (int packed_k = 0; packed_k < k_packed; ++packed_k) {
          qweight[static_cast<size_t>(output) * k_packed + packed_k] =
              source_qweight[static_cast<size_t>(packed_k) * n + output];
        }
        for (int group = 0; group < num_groups; ++group) {
          scales[static_cast<size_t>(output) * num_groups + group] =
              source_scales[static_cast<size_t>(group) * n + output];
        }
      }
    }
  };

  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0;
    size_t n = 0;
    size_t capacity_m = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void*) : max_m(m), n(n_) {}
    BufferC(const BufferC&) = delete;
    BufferC& operator=(const BufferC&) = delete;
    ~BufferC() { usm_free(data); }

    static size_t required_size(size_t, size_t) { return 1; }

    void ensure(size_t m) {
      if (m <= capacity_m) return;
      usm_free(data);
      data = usm_alloc<float>(m * n, "GPTQ INT4 output buffer");
      capacity_m = m;
    }

    void set_data(void*) { ensure(max_m); }

    void to_mat(int m, ggml_bf16_t* destination, int ith, int nth) {
      ensure(static_cast<size_t>(m));
      auto [begin, end] = avx2::split_range(static_cast<int>(n), ith, nth);
      for (int row = 0; row < m; ++row) {
        const float* source = data + static_cast<size_t>(row) * n;
        ggml_bf16_t* output = destination + static_cast<size_t>(row) * n;
        for (int column = begin; column < end; ++column) {
          output[column] = GGML_FP32_TO_BF16(source[column]);
        }
      }
    }
  };
};

// Correctness-oriented fallback used by the existing prefill flow.
inline void gemm_generic(int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& input,
                         GemmKernelSYCLGPTQInt4::BufferB& weight, GemmKernelSYCLGPTQInt4::BufferC& output, int ith,
                         int nth) {
  if (m <= 0 || n <= 0 || k <= 0) return;
  auto [begin, end] = avx2::split_range(n, ith, nth);
  if (begin >= end) return;

  const int num_groups = weight.num_groups;
  const int group_size = weight.group_size;
  const int k_packed = weight.k_packed;
  const size_t input_stride = input.k;
  const size_t output_stride = output.n;
  const int output_count = end - begin;
  const uint16_t* input_data = input.data;
  const uint32_t* qweight = weight.qweight;
  const float* scales = weight.scales;
  float* output_data = output.data;

  queue()
      .submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<2>(static_cast<size_t>(m), static_cast<size_t>(output_count)),
                             [=](sycl::id<2> index) {
                               const int row = static_cast<int>(index[0]);
                               const int column = begin + static_cast<int>(index[1]);
                               const uint16_t* row_input = input_data + static_cast<size_t>(row) * input_stride;
                               float accumulator = 0.0f;
                               for (int group = 0; group < num_groups; ++group) {
                                 float group_sum = 0.0f;
                                 const int group_begin = group * group_size;
                                 for (int offset = 0; offset < group_size; ++offset) {
                                   const int kk = group_begin + offset;
                                   const uint32_t packed = qweight[static_cast<size_t>(column) * k_packed + (kk >> 3)];
                                   const int quantized = static_cast<int>((packed >> ((kk & 7) * 4)) & 0x0fu) - 8;
                                   group_sum += bf16_to_fp32(row_input[kk]) * static_cast<float>(quantized);
                                 }
                                 accumulator += group_sum * scales[static_cast<size_t>(column) * num_groups + group];
                               }
                               output_data[static_cast<size_t>(row) * output_stride + column] = accumulator;
                             });
      })
      .wait_and_throw();
}

inline sycl::event submit_gate_up_decode(int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& input,
                                         GemmKernelSYCLGPTQInt4::BufferB& gate_weight,
                                         GemmKernelSYCLGPTQInt4::BufferB& up_weight,
                                         GemmKernelSYCLGPTQInt4::BufferA& output, float swiglu_limit,
                                         float swiglu_alpha) {
  if (m <= 0 || n <= 0 || k <= 0) return sycl::event{};
  if (gate_weight.n != n || up_weight.n != n || gate_weight.k != k || up_weight.k != k ||
      gate_weight.group_size != up_weight.group_size) {
    throw std::runtime_error("Incompatible gate/up shapes for SYCL GPTQ INT4");
  }

  constexpr int subgroup_size = kSubgroupSize;
  const int num_groups = gate_weight.num_groups;
  const int packed_per_group = gate_weight.group_size / 8;
  const int k_packed = gate_weight.k_packed;
  const size_t input_stride = input.k;
  const size_t output_stride = output.k;
  const size_t work_groups = static_cast<size_t>(m) * n;
  const uint16_t* input_data = input.data;
  const uint32_t* gate_qweight = gate_weight.qweight;
  const uint32_t* up_qweight = up_weight.qweight;
  const float* gate_scales = gate_weight.scales;
  const float* up_scales = up_weight.scales;
  uint16_t* output_data = output.data;

  return queue().submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>(work_groups * subgroup_size, subgroup_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(subgroup_size)]] {
          const auto subgroup = item.get_sub_group();
          const int group_id = static_cast<int>(item.get_group(0));
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const int row = group_id / n;
          const int column = group_id - row * n;
          const uint16_t* row_input = input_data + static_cast<size_t>(row) * input_stride;

          float gate_accumulator = 0.0f;
          float up_accumulator = 0.0f;
          for (int group = 0; group < num_groups; ++group) {
            float gate_partial = 0.0f;
            float up_partial = 0.0f;
            const int packed_begin = group * packed_per_group;
            for (int packed_offset = lane; packed_offset < packed_per_group; packed_offset += subgroup_size) {
              const int packed_k = packed_begin + packed_offset;
              const uint32_t gate_packed = gate_qweight[static_cast<size_t>(column) * k_packed + packed_k];
              const uint32_t up_packed = up_qweight[static_cast<size_t>(column) * k_packed + packed_k];
              const uint16_t* activation = row_input + static_cast<size_t>(packed_k) * 8;
#pragma unroll
              for (int index = 0; index < 8; ++index) {
                const float value = bf16_to_fp32(activation[index]);
                const int shift = index * 4;
                gate_partial += value * static_cast<float>(static_cast<int>((gate_packed >> shift) & 0x0fu) - 8);
                up_partial += value * static_cast<float>(static_cast<int>((up_packed >> shift) & 0x0fu) - 8);
              }
            }

            const float gate_sum = sycl::reduce_over_group(subgroup, gate_partial, sycl::plus<float>());
            const float up_sum = sycl::reduce_over_group(subgroup, up_partial, sycl::plus<float>());
            if (lane == 0) {
              const size_t scale_offset = static_cast<size_t>(column) * num_groups + group;
              gate_accumulator += gate_sum * gate_scales[scale_offset];
              up_accumulator += up_sum * up_scales[scale_offset];
            }
          }

          if (lane == 0) {
            float gate_value = gate_accumulator;
            float up_value = up_accumulator;
            float activated;
            if (swiglu_alpha > 0.0f) {
              if (swiglu_limit > 0.0f) {
                gate_value = sycl::fmin(sycl::fmax(gate_value, -swiglu_limit), swiglu_limit);
                up_value = sycl::fmin(sycl::fmax(up_value, -swiglu_limit), swiglu_limit);
              }
              const float sigmoid = 1.0f / (1.0f + sycl::native::exp(-gate_value * swiglu_alpha));
              activated = gate_value * sigmoid * (up_value + 1.0f);
            } else {
              if (swiglu_limit > 0.0f) {
                gate_value = sycl::fmin(gate_value, swiglu_limit);
                up_value = sycl::fmin(sycl::fmax(up_value, -swiglu_limit), swiglu_limit);
              }
              const float sigmoid = 1.0f / (1.0f + sycl::native::exp(-gate_value));
              activated = gate_value * sigmoid * up_value;
            }
            output_data[static_cast<size_t>(row) * output_stride + column] = fp32_to_bf16(activated);
          }
        });
  });
}

inline sycl::event submit_down_decode(int m, int n, int k, GemmKernelSYCLGPTQInt4::BufferA& input,
                                      GemmKernelSYCLGPTQInt4::BufferB& weight, GemmKernelSYCLGPTQInt4::BufferC& output,
                                      const sycl::event& dependency) {
  if (m <= 0 || n <= 0 || k <= 0) return sycl::event{};

  constexpr int subgroup_size = kSubgroupSize;
  constexpr int rows_per_work_group = kDownRowsPerWorkGroup;
  constexpr size_t local_size = subgroup_size * rows_per_work_group;
  const int num_groups = weight.num_groups;
  const int packed_per_group = weight.group_size / 8;
  const int k_packed = weight.k_packed;
  const size_t input_stride = input.k;
  const size_t output_stride = output.n;
  const size_t output_rows = static_cast<size_t>(m) * n;
  const size_t work_groups = (output_rows + rows_per_work_group - 1) / rows_per_work_group;
  const uint16_t* input_data = input.data;
  const uint32_t* qweight = weight.qweight;
  const float* scales = weight.scales;
  float* output_data = output.data;

  return queue().submit([&](sycl::handler& handler) {
    handler.depends_on(dependency);
    handler.parallel_for(sycl::nd_range<1>(work_groups * local_size, local_size),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(subgroup_size)]] {
                           const auto subgroup = item.get_sub_group();
                           const int row_in_work_group = static_cast<int>(subgroup.get_group_linear_id());
                           const int lane = static_cast<int>(subgroup.get_local_linear_id());
                           const size_t linear_row = item.get_group(0) * rows_per_work_group + row_in_work_group;
                           if (linear_row >= output_rows) return;

                           const int input_row = static_cast<int>(linear_row / static_cast<size_t>(n));
                           const int column = static_cast<int>(linear_row - static_cast<size_t>(input_row) * n);
                           const uint16_t* row_input = input_data + static_cast<size_t>(input_row) * input_stride;
                           float accumulator = 0.0f;

                           for (int group = 0; group < num_groups; ++group) {
                             float partial = 0.0f;
                             const int packed_begin = group * packed_per_group;
                             for (int packed_offset = lane; packed_offset < packed_per_group;
                                  packed_offset += subgroup_size) {
                               const int packed_k = packed_begin + packed_offset;
                               const uint32_t packed = qweight[static_cast<size_t>(column) * k_packed + packed_k];
                               const uint16_t* activation = row_input + static_cast<size_t>(packed_k) * 8;
#pragma unroll
                               for (int index = 0; index < 8; ++index) {
                                 const int quantized = static_cast<int>((packed >> (index * 4)) & 0x0fu) - 8;
                                 partial += bf16_to_fp32(activation[index]) * static_cast<float>(quantized);
                               }
                             }

                             const float group_sum = sycl::reduce_over_group(subgroup, partial, sycl::plus<float>());
                             if (lane == 0) {
                               accumulator += group_sum * scales[static_cast<size_t>(column) * num_groups + group];
                             }
                           }

                           if (lane == 0) {
                             output_data[static_cast<size_t>(input_row) * output_stride + column] = accumulator;
                           }
                         });
  });
}

}  // namespace sycl_int4

template <class T = sycl_int4::GemmKernelSYCLGPTQInt4>
class SYCL_GPTQ_INT4_MOE_TP : public AVX2_MOE_BASE<T, SYCL_GPTQ_INT4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, SYCL_GPTQ_INT4_MOE_TP<T>>;
  using Base::backend_scratch_;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_expert_id_map_;
  using Base::m_local_down_output_ptr_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  SYCL_GPTQ_INT4_MOE_TP() = default;
  SYCL_GPTQ_INT4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    T::config();
    const int group_size = config_.quant_config.group_size;
    if (group_size <= 0 || (group_size % 8) != 0) {
      throw std::runtime_error("SYCL GPTQ INT4 requires a positive group_size divisible by 8");
    }
  }

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int) {
    auto& weight = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& output = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    sycl_int4::gemm_generic(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                            *gate_up_ba_[expert_idx], *weight, *output, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int) {
    sycl_int4::gemm_generic(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                            *down_ba_[expert_idx], *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  bool use_fused_gate_up_decode() const { return true; }
  bool use_fused_down_decode() const { return true; }

  void decode_gate_up_activation(int activated_experts, int qlen) {
    if (qlen != 1 || activated_experts <= 0) return;
    auto* scratch = get_scratch();
    if (scratch->gate_up_pending) {
      for (auto& event : scratch->gate_up_events) event.wait_and_throw();
    }

    scratch->gate_up_events.clear();
    scratch->gate_up_experts.clear();
    scratch->gate_up_events.reserve(static_cast<size_t>(activated_experts));
    scratch->gate_up_experts.reserve(static_cast<size_t>(activated_experts));

    for (int task = 0; task < activated_experts; ++task) {
      const int expert = m_expert_id_map_[task];
      scratch->gate_up_events.push_back(sycl_int4::submit_gate_up_decode(
          m_local_num_[expert], config_.intermediate_size, config_.hidden_size, *gate_up_ba_[expert], *gate_bb_[expert],
          *up_bb_[expert], *down_ba_[expert], config_.swiglu_limit, config_.swiglu_alpha));
      scratch->gate_up_experts.push_back(expert);
    }
    scratch->active_experts = activated_experts;
    scratch->gate_up_pending = true;
  }

  void decode_down_projection(int activated_experts, int qlen) {
    if (qlen != 1 || activated_experts <= 0) return;
    auto* scratch = get_scratch();
    if (!scratch->gate_up_pending || scratch->active_experts != activated_experts ||
        static_cast<int>(scratch->gate_up_events.size()) != activated_experts ||
        static_cast<int>(scratch->gate_up_experts.size()) != activated_experts) {
      throw std::runtime_error("Invalid SYCL GPTQ INT4 decode pipeline state");
    }

    std::vector<sycl::event> down_events;
    down_events.reserve(static_cast<size_t>(activated_experts));
    for (int task = 0; task < activated_experts; ++task) {
      const int expert = m_expert_id_map_[task];
      if (scratch->gate_up_experts[task] != expert) {
        throw std::runtime_error("SYCL GPTQ INT4 expert order changed during decode");
      }
      down_events.push_back(sycl_int4::submit_down_decode(
          m_local_num_[expert], config_.hidden_size, config_.intermediate_size, *down_ba_[expert], *down_bb_[expert],
          *down_bc_[expert], scratch->gate_up_events[task]));
    }
    for (auto& event : down_events) event.wait_and_throw();

    scratch->gate_up_pending = false;
    scratch->active_experts = 0;
    for (int task = 0; task < activated_experts; ++task) {
      const int expert = m_expert_id_map_[task];
      down_bc_[expert]->to_mat(qlen, m_local_down_output_ptr_[expert], 0, 1);
    }
  }

  void load_weights() {
    const int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical = reinterpret_cast<const uint64_t*>(config_.physical_to_logical_map);
    auto pool = config_.pool->get_subpool(tp_part_idx);
    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("SYCL GPTQ INT4 requires scale tensors");
    }

    prepare_contiguous_weights();

    const int gate_up_k = config_.hidden_size;
    const int gate_up_n = config_.intermediate_size;
    const size_t gate_up_qweight_elements = static_cast<size_t>(gate_up_k / 8) * gate_up_n;
    const size_t gate_up_scale_elements = static_cast<size_t>(gate_up_k / group_size) * gate_up_n;
    int nth = T::recommended_nth(gate_up_n);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical, gate_up_qweight_elements, gate_up_scale_elements](int task) {
          const uint64_t expert = static_cast<uint64_t>(task / nth);
          const uint64_t logical = expert_map(physical_to_logical, expert);
          const int ith = task % nth;
          if (config_.should_skip_expert(logical)) return;
          gate_bb_[expert]->from_mat(
              reinterpret_cast<const uint32_t*>(config_.gate_proj) + logical * gate_up_qweight_elements,
              reinterpret_cast<const float*>(config_.gate_scale) + logical * gate_up_scale_elements, ith, nth);
          up_bb_[expert]->from_mat(
              reinterpret_cast<const uint32_t*>(config_.up_proj) + logical * gate_up_qweight_elements,
              reinterpret_cast<const float*>(config_.up_scale) + logical * gate_up_scale_elements, ith, nth);
        },
        nullptr);

    const int down_k = config_.intermediate_size;
    const int down_n = config_.hidden_size;
    const size_t down_qweight_elements = static_cast<size_t>(down_k / 8) * down_n;
    const size_t down_scale_elements = static_cast<size_t>(down_k / group_size) * down_n;
    nth = T::recommended_nth(down_n);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical, down_qweight_elements, down_scale_elements](int task) {
          const uint64_t expert = static_cast<uint64_t>(task / nth);
          const uint64_t logical = expert_map(physical_to_logical, expert);
          const int ith = task % nth;
          if (config_.should_skip_expert(logical)) return;
          down_bb_[expert]->from_mat(
              reinterpret_cast<const uint32_t*>(config_.down_proj) + logical * down_qweight_elements,
              reinterpret_cast<const float*>(config_.down_scale) + logical * down_scale_elements, ith, nth);
        },
        nullptr);
  }

  void write_weights_to_buffer(int, int, int, const GeneralMOEConfig&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&) const {
    throw std::runtime_error("SYCL GPTQ INT4 does not support write_weights_to_buffer");
  }

 private:
  sycl_int4::DecodeScratch* get_scratch() {
    if (!backend_scratch_) {
      backend_scratch_ = std::make_shared<sycl_int4::DecodeScratch>();
    }
    return static_cast<sycl_int4::DecodeScratch*>(backend_scratch_.get());
  }

  void prepare_contiguous_weights() {
    auto* scratch = get_scratch();
    if (scratch->weights_ready || config_.expert_num <= 0) return;

    const size_t experts = static_cast<size_t>(config_.expert_num);
    const size_t gate_up_qweight_stride = gate_bb_[0]->qweight_elements();
    const size_t gate_up_scale_stride = gate_bb_[0]->scale_elements();
    const size_t down_qweight_stride = down_bb_[0]->qweight_elements();
    const size_t down_scale_stride = down_bb_[0]->scale_elements();

    scratch->gate_qweight = sycl_int4::usm_alloc<uint32_t>(experts * gate_up_qweight_stride, "contiguous gate qweight");
    scratch->up_qweight = sycl_int4::usm_alloc<uint32_t>(experts * gate_up_qweight_stride, "contiguous up qweight");
    scratch->down_qweight = sycl_int4::usm_alloc<uint32_t>(experts * down_qweight_stride, "contiguous down qweight");
    scratch->gate_scales = sycl_int4::usm_alloc<float>(experts * gate_up_scale_stride, "contiguous gate scales");
    scratch->up_scales = sycl_int4::usm_alloc<float>(experts * gate_up_scale_stride, "contiguous up scales");
    scratch->down_scales = sycl_int4::usm_alloc<float>(experts * down_scale_stride, "contiguous down scales");

    scratch->gate_up_qweight_stride = gate_up_qweight_stride;
    scratch->gate_up_scale_stride = gate_up_scale_stride;
    scratch->down_qweight_stride = down_qweight_stride;
    scratch->down_scale_stride = down_scale_stride;

    for (size_t expert = 0; expert < experts; ++expert) {
      const bool owns_slab = expert == 0;
      gate_bb_[expert]->bind_external(scratch->gate_qweight + expert * gate_up_qweight_stride,
                                      scratch->gate_scales + expert * gate_up_scale_stride, owns_slab);
      up_bb_[expert]->bind_external(scratch->up_qweight + expert * gate_up_qweight_stride,
                                    scratch->up_scales + expert * gate_up_scale_stride, owns_slab);
      down_bb_[expert]->bind_external(scratch->down_qweight + expert * down_qweight_stride,
                                      scratch->down_scales + expert * down_scale_stride, owns_slab);
    }
    scratch->weights_ready = true;
  }
};

template <typename Kernel>
class TP_MOE<SYCL_GPTQ_INT4_MOE_TP<Kernel>> : public TP_MOE<AVX2_MOE_BASE<Kernel, SYCL_GPTQ_INT4_MOE_TP<Kernel>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<Kernel, SYCL_GPTQ_INT4_MOE_TP<Kernel>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tensor_parallel_backends = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical = reinterpret_cast<const uint64_t*>(config.physical_to_logical_map);
    const int group_size = config.quant_config.group_size;
    if (group_size <= 0) {
      throw std::runtime_error("SYCL GPTQ INT4 requires group_size > 0");
    }
    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("SYCL GPTQ INT4 has no weight source");
    }
    const bool per_expert = !config.gate_projs.empty();

    const int full_intermediate = config.intermediate_size;
    const int full_hidden = config.hidden_size;
    const int gate_up_k_packed = full_hidden / 8;
    const int gate_up_num_groups = full_hidden / group_size;
    const size_t full_gate_up_qweight = static_cast<size_t>(gate_up_k_packed) * full_intermediate;
    const size_t full_gate_up_scales = static_cast<size_t>(gate_up_num_groups) * full_intermediate;
    const int down_k_packed = full_intermediate / 8;
    const int down_num_groups = full_intermediate / group_size;
    const size_t full_down_qweight = static_cast<size_t>(down_k_packed) * full_hidden;
    const size_t full_down_scales = static_cast<size_t>(down_num_groups) * full_hidden;

    pool->dispense_backend()->do_numa_job([&, this](int index) {
      auto& tp_config = tensor_parallel_backends[index]->config_;
      const int tp_intermediate = tp_config.intermediate_size;
      const size_t tp_gate_up_qweight = static_cast<size_t>(gate_up_k_packed) * tp_intermediate;
      const size_t tp_gate_up_scales = static_cast<size_t>(gate_up_num_groups) * tp_intermediate;
      tp_config.gate_proj = new uint32_t[static_cast<size_t>(tp_config.expert_num) * tp_gate_up_qweight];
      tp_config.up_proj = new uint32_t[static_cast<size_t>(tp_config.expert_num) * tp_gate_up_qweight];
      tp_config.gate_scale = new float[static_cast<size_t>(tp_config.expert_num) * tp_gate_up_scales];
      tp_config.up_scale = new float[static_cast<size_t>(tp_config.expert_num) * tp_gate_up_scales];

      const int tp_down_k_packed = tp_intermediate / 8;
      const int tp_down_num_groups = tp_intermediate / group_size;
      const size_t tp_down_qweight = static_cast<size_t>(tp_down_k_packed) * full_hidden;
      const size_t tp_down_scales = static_cast<size_t>(tp_down_num_groups) * full_hidden;
      tp_config.down_proj = new uint32_t[static_cast<size_t>(tp_config.expert_num) * tp_down_qweight];
      tp_config.down_scale = new float[static_cast<size_t>(tp_config.expert_num) * tp_down_scales];

      const int gate_up_column_offset = index * tp_intermediate;
      const int down_packed_offset = index * tp_down_k_packed;
      const int down_group_offset = index * tp_down_num_groups;

      pool->get_subpool(index)->do_work_stealing_job(
          tp_config.expert_num, nullptr,
          [&](int expert_index) {
            const size_t expert = expert_map(physical_to_logical, expert_index);
            const uint32_t* gate_qweight_source;
            const uint32_t* up_qweight_source;
            const uint32_t* down_qweight_source;
            const float* gate_scale_source;
            const float* up_scale_source;
            const float* down_scale_source;

            if (per_expert) {
              gate_qweight_source = reinterpret_cast<const uint32_t*>(config.gate_projs[0][expert]);
              up_qweight_source = reinterpret_cast<const uint32_t*>(config.up_projs[0][expert]);
              down_qweight_source = reinterpret_cast<const uint32_t*>(config.down_projs[0][expert]);
              gate_scale_source = reinterpret_cast<const float*>(config.gate_scales[0][expert]);
              up_scale_source = reinterpret_cast<const float*>(config.up_scales[0][expert]);
              down_scale_source = reinterpret_cast<const float*>(config.down_scales[0][expert]);
            } else {
              gate_qweight_source = reinterpret_cast<const uint32_t*>(config.gate_proj) + expert * full_gate_up_qweight;
              up_qweight_source = reinterpret_cast<const uint32_t*>(config.up_proj) + expert * full_gate_up_qweight;
              down_qweight_source = reinterpret_cast<const uint32_t*>(config.down_proj) + expert * full_down_qweight;
              gate_scale_source = reinterpret_cast<const float*>(config.gate_scale) + expert * full_gate_up_scales;
              up_scale_source = reinterpret_cast<const float*>(config.up_scale) + expert * full_gate_up_scales;
              down_scale_source = reinterpret_cast<const float*>(config.down_scale) + expert * full_down_scales;
            }

            uint32_t* gate_qweight_destination =
                reinterpret_cast<uint32_t*>(tp_config.gate_proj) + expert * tp_gate_up_qweight;
            uint32_t* up_qweight_destination =
                reinterpret_cast<uint32_t*>(tp_config.up_proj) + expert * tp_gate_up_qweight;
            float* gate_scale_destination = reinterpret_cast<float*>(tp_config.gate_scale) + expert * tp_gate_up_scales;
            float* up_scale_destination = reinterpret_cast<float*>(tp_config.up_scale) + expert * tp_gate_up_scales;

            for (int packed_k = 0; packed_k < gate_up_k_packed; ++packed_k) {
              std::memcpy(
                  gate_qweight_destination + static_cast<size_t>(packed_k) * tp_intermediate,
                  gate_qweight_source + static_cast<size_t>(packed_k) * full_intermediate + gate_up_column_offset,
                  static_cast<size_t>(tp_intermediate) * sizeof(uint32_t));
              std::memcpy(up_qweight_destination + static_cast<size_t>(packed_k) * tp_intermediate,
                          up_qweight_source + static_cast<size_t>(packed_k) * full_intermediate + gate_up_column_offset,
                          static_cast<size_t>(tp_intermediate) * sizeof(uint32_t));
            }
            for (int group = 0; group < gate_up_num_groups; ++group) {
              std::memcpy(gate_scale_destination + static_cast<size_t>(group) * tp_intermediate,
                          gate_scale_source + static_cast<size_t>(group) * full_intermediate + gate_up_column_offset,
                          static_cast<size_t>(tp_intermediate) * sizeof(float));
              std::memcpy(up_scale_destination + static_cast<size_t>(group) * tp_intermediate,
                          up_scale_source + static_cast<size_t>(group) * full_intermediate + gate_up_column_offset,
                          static_cast<size_t>(tp_intermediate) * sizeof(float));
            }

            uint32_t* down_qweight_destination =
                reinterpret_cast<uint32_t*>(tp_config.down_proj) + expert * tp_down_qweight;
            float* down_scale_destination = reinterpret_cast<float*>(tp_config.down_scale) + expert * tp_down_scales;
            for (int packed_k = 0; packed_k < tp_down_k_packed; ++packed_k) {
              std::memcpy(down_qweight_destination + static_cast<size_t>(packed_k) * full_hidden,
                          down_qweight_source + static_cast<size_t>(down_packed_offset + packed_k) * full_hidden,
                          static_cast<size_t>(full_hidden) * sizeof(uint32_t));
            }
            for (int group = 0; group < tp_down_num_groups; ++group) {
              std::memcpy(down_scale_destination + static_cast<size_t>(group) * full_hidden,
                          down_scale_source + static_cast<size_t>(down_group_offset + group) * full_hidden,
                          static_cast<size_t>(full_hidden) * sizeof(float));
            }
          },
          nullptr);
    });

    pool->dispense_backend()->do_numa_job([&, this](int index) { tensor_parallel_backends[index]->load_weights(); });
    pool->dispense_backend()->do_numa_job([&, this](int index) {
      auto& tp_config = tensor_parallel_backends[index]->config_;
      delete[] reinterpret_cast<uint32_t*>(tp_config.gate_proj);
      delete[] reinterpret_cast<uint32_t*>(tp_config.up_proj);
      delete[] reinterpret_cast<uint32_t*>(tp_config.down_proj);
      delete[] reinterpret_cast<float*>(tp_config.gate_scale);
      delete[] reinterpret_cast<float*>(tp_config.up_scale);
      delete[] reinterpret_cast<float*>(tp_config.down_scale);
    });
    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int, int, const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                                    const std::vector<uintptr_t>&, const std::vector<uintptr_t>&) {
    throw std::runtime_error("SYCL GPTQ INT4 does not support write_weight_scale_to_buffer");
  }
};

#endif  // CPUINFER_OPERATOR_SYCL_GPTQ_INT4_MOE_H
