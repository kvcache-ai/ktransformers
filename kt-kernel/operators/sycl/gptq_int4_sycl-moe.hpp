/**
 * @Description : SYCL GPTQ INT4 MoE backend for Intel GPUs
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Supports symmetric GPTQ INT4 weights stored as qweight [K/8, N] and
 * scales [K/group_size, N]. Weights are reordered once at load time to an
 * output-major layout. Decode uses packed SG16 gate/up kernels, per-expert
 * asynchronous submission, and a two-subgroup down-projection work-group.
 * Prefill batches routed rows across experts, using row-pair kernels for sparse
 * experts and a W4A8 row8 kernel for dense gate/up projections.
 */
#ifndef CPUINFER_OPERATOR_SYCL_GPTQ_INT4_MOE_H
#define CPUINFER_OPERATOR_SYCL_GPTQ_INT4_MOE_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sycl/ext/oneapi/dot_product.hpp>
#include <utility>
#include <vector>

#include "../avx2/moe_base.hpp"
#include "gptq_int4_sycl-runtime.hpp"

namespace sycl_int4 {

struct BackendScratch {
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

  uint16_t** prefill_inputs = nullptr;
  uint16_t** prefill_activations = nullptr;
  uint32_t** prefill_gate_qweights = nullptr;
  uint32_t** prefill_up_qweights = nullptr;
  uint32_t** prefill_down_qweights = nullptr;
  float** prefill_gate_scales = nullptr;
  float** prefill_up_scales = nullptr;
  float** prefill_down_scales = nullptr;
  float** prefill_q8_scratch = nullptr;
  int* prefill_row_counts = nullptr;
  int* prefill_tile_experts = nullptr;
  int* prefill_tile_rows = nullptr;
  int prefill_expert_capacity = 0;
  int prefill_tile_capacity = 0;

  void reset_weights() noexcept {
    usm_free(gate_qweight);
    usm_free(up_qweight);
    usm_free(down_qweight);
    usm_free(gate_scales);
    usm_free(up_scales);
    usm_free(down_scales);
    gate_qweight = nullptr;
    up_qweight = nullptr;
    down_qweight = nullptr;
    gate_scales = nullptr;
    up_scales = nullptr;
    down_scales = nullptr;
    weights_ready = false;
  }

  ~BackendScratch() {
    // A submit failure can leave a prefix of the decode pipeline in flight.
    // Finish those commands before releasing weights or activation storage.
    for (auto& event : gate_up_events) event.wait();
    reset_weights();
    usm_free(prefill_inputs);
    usm_free(prefill_activations);
    usm_free(prefill_gate_qweights);
    usm_free(prefill_up_qweights);
    usm_free(prefill_down_qweights);
    usm_free(prefill_gate_scales);
    usm_free(prefill_up_scales);
    usm_free(prefill_down_scales);
    usm_free(prefill_q8_scratch);
    usm_free(prefill_row_counts);
    usm_free(prefill_tile_experts);
    usm_free(prefill_tile_rows);
  }
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
      uint16_t* replacement = usm_alloc<uint16_t>(m * k, "GPTQ INT4 activation buffer");
      usm_free(data);
      data = replacement;
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

    void bind_view(uint32_t* qweight_pointer, float* scale_pointer) {
      if (owns_storage) {
        usm_free(qweight);
        usm_free(scales);
      }
      qweight = qweight_pointer;
      scales = scale_pointer;
      owns_storage = false;
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
      float* replacement = usm_alloc<float>(m * n, "GPTQ INT4 output buffer");
      usm_free(data);
      data = replacement;
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

inline float prefill_swiglu(float gate, float up, float swiglu_limit, float swiglu_alpha) {
  if (swiglu_alpha > 0.0f) {
    if (swiglu_limit > 0.0f) {
      gate = sycl::fmin(sycl::fmax(gate, -swiglu_limit), swiglu_limit);
      up = sycl::fmin(sycl::fmax(up, -swiglu_limit), swiglu_limit);
    }
    const float sigmoid = 1.0f / (1.0f + sycl::native::exp(-gate * swiglu_alpha));
    return gate * sigmoid * (up + 1.0f);
  }
  if (swiglu_limit > 0.0f) {
    gate = sycl::fmin(gate, swiglu_limit);
    up = sycl::fmin(sycl::fmax(up, -swiglu_limit), swiglu_limit);
  }
  return gate * (1.0f / (1.0f + sycl::native::exp(-gate))) * up;
}

// Sparse experts use a two-row register tile. One subgroup loads each packed
// weight once and reuses it for both routed rows without local memory.
inline sycl::event submit_prefill_gate_up_sparse(int tile_count, int n, int k, const int* row_counts,
                                                 const int* tile_experts, const int* tile_rows,
                                                 uint16_t** input_pointers, uint32_t** gate_qweight_pointers,
                                                 uint32_t** up_qweight_pointers, float** gate_scale_pointers,
                                                 float** up_scale_pointers, uint16_t** output_pointers, int group_size,
                                                 float swiglu_limit, float swiglu_alpha) {
  if (tile_count <= 0 || n <= 0 || k <= 0) return sycl::event{};
  constexpr int subgroup_size = kSubgroupSize;
  const int num_groups = k / group_size;
  const int k_packed = k / 8;
  const int packed_per_group = group_size / 8;
  const size_t work_groups = static_cast<size_t>(tile_count) * n;

  return queue().submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>(work_groups * subgroup_size, subgroup_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(subgroup_size)]] {
          const size_t group_id = item.get_group(0);
          const int tile = static_cast<int>(group_id / static_cast<size_t>(n));
          const int output = static_cast<int>(group_id - static_cast<size_t>(tile) * n);
          const int active_expert = tile_experts[tile];
          const int row0 = tile_rows[tile];
          const int row1 = row0 + 1;
          const int lane = static_cast<int>(item.get_sub_group().get_local_linear_id());
          const bool valid0 = row0 < row_counts[active_expert];
          const bool valid1 = row1 < row_counts[active_expert];
          const uint16_t* input = input_pointers[active_expert];
          const uint32_t* gate_qweight = gate_qweight_pointers[active_expert];
          const uint32_t* up_qweight = up_qweight_pointers[active_expert];
          const float* gate_scales = gate_scale_pointers[active_expert];
          const float* up_scales = up_scale_pointers[active_expert];
          float gate_accumulator0 = 0.0f;
          float up_accumulator0 = 0.0f;
          float gate_accumulator1 = 0.0f;
          float up_accumulator1 = 0.0f;

          for (int group = 0; group < num_groups; ++group) {
            const int packed_begin = group * packed_per_group;
            float gate_partial0 = 0.0f;
            float up_partial0 = 0.0f;
            float gate_partial1 = 0.0f;
            float up_partial1 = 0.0f;
            for (int packed = lane; packed < packed_per_group; packed += subgroup_size) {
              const int packed_k = packed_begin + packed;
              const uint32_t gate_word = gate_qweight[static_cast<size_t>(output) * k_packed + packed_k];
              const uint32_t up_word = up_qweight[static_cast<size_t>(output) * k_packed + packed_k];
              if (valid0) {
                const uint16_t* values = input + static_cast<size_t>(row0) * k + static_cast<size_t>(packed_k) * 8;
#pragma unroll
                for (int nibble = 0; nibble < 8; ++nibble) {
                  const float value = bf16_to_fp32(values[nibble]);
                  gate_partial0 += value * static_cast<float>(static_cast<int>((gate_word >> (nibble * 4)) & 0xfu) - 8);
                  up_partial0 += value * static_cast<float>(static_cast<int>((up_word >> (nibble * 4)) & 0xfu) - 8);
                }
              }
              if (valid1) {
                const uint16_t* values = input + static_cast<size_t>(row1) * k + static_cast<size_t>(packed_k) * 8;
#pragma unroll
                for (int nibble = 0; nibble < 8; ++nibble) {
                  const float value = bf16_to_fp32(values[nibble]);
                  gate_partial1 += value * static_cast<float>(static_cast<int>((gate_word >> (nibble * 4)) & 0xfu) - 8);
                  up_partial1 += value * static_cast<float>(static_cast<int>((up_word >> (nibble * 4)) & 0xfu) - 8);
                }
              }
            }

            const auto subgroup = item.get_sub_group();
            const float gate_sum0 = sycl::reduce_over_group(subgroup, gate_partial0, sycl::plus<float>());
            const float up_sum0 = sycl::reduce_over_group(subgroup, up_partial0, sycl::plus<float>());
            const float gate_sum1 = sycl::reduce_over_group(subgroup, gate_partial1, sycl::plus<float>());
            const float up_sum1 = sycl::reduce_over_group(subgroup, up_partial1, sycl::plus<float>());
            if (lane == 0) {
              const size_t scale_offset = static_cast<size_t>(output) * num_groups + group;
              gate_accumulator0 += gate_sum0 * gate_scales[scale_offset];
              up_accumulator0 += up_sum0 * up_scales[scale_offset];
              gate_accumulator1 += gate_sum1 * gate_scales[scale_offset];
              up_accumulator1 += up_sum1 * up_scales[scale_offset];
            }
          }

          if (lane == 0) {
            if (valid0) {
              output_pointers[active_expert][static_cast<size_t>(row0) * n + output] =
                  fp32_to_bf16(prefill_swiglu(gate_accumulator0, up_accumulator0, swiglu_limit, swiglu_alpha));
            }
            if (valid1) {
              output_pointers[active_expert][static_cast<size_t>(row1) * n + output] =
                  fp32_to_bf16(prefill_swiglu(gate_accumulator1, up_accumulator1, swiglu_limit, swiglu_alpha));
            }
          }
        });
  });
}

inline sycl::event submit_prefill_dense_quantization(int tile_count, int k, int group_size, const int* row_counts,
                                                     const int* tile_experts, const int* tile_rows,
                                                     uint16_t** input_pointers, float** scratch_pointers) {
  if (tile_count <= 0 || k <= 0 || group_size <= 0) return sycl::event{};
  constexpr int work_group_size = 128;
  const int num_groups = k / group_size;
  const size_t work_groups = static_cast<size_t>(tile_count) * kPrefillDenseRows;

  return queue().submit([&](sycl::handler& handler) {
    handler.parallel_for(sycl::nd_range<1>(work_groups * work_group_size, work_group_size), [=](sycl::nd_item<1> item) {
      const size_t group_id = item.get_group(0);
      const int tile = static_cast<int>(group_id / kPrefillDenseRows);
      const int row_in_tile = static_cast<int>(group_id - static_cast<size_t>(tile) * kPrefillDenseRows);
      const int active_expert = tile_experts[tile];
      const int row = tile_rows[tile] + row_in_tile;
      const int row_count = row_counts[active_expert];
      if (row >= row_count) return;

      const int lane = static_cast<int>(item.get_local_id(0));
      const uint16_t* source = input_pointers[active_expert] + static_cast<size_t>(row) * k;
      int8_t* quantized = reinterpret_cast<int8_t*>(scratch_pointers[active_expert]);
      float* activation_scales = reinterpret_cast<float*>(quantized + static_cast<size_t>(row_count) * k);
      int8_t* destination = quantized + static_cast<size_t>(row) * k;

      for (int group = 0; group < num_groups; ++group) {
        const int group_begin = group * group_size;
        float maximum = 0.0f;
        for (int offset = lane; offset < group_size; offset += work_group_size) {
          maximum = sycl::fmax(maximum, sycl::fabs(bf16_to_fp32(source[group_begin + offset])));
        }
        maximum = sycl::reduce_over_group(item.get_group(), maximum, sycl::maximum<float>());
        const float scale = maximum > 0.0f ? maximum / 127.0f : 0.0f;
        if (lane == 0) {
          activation_scales[static_cast<size_t>(row) * num_groups + group] = scale;
        }
        const float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int offset = lane; offset < group_size; offset += work_group_size) {
          int value = static_cast<int>(sycl::rint(bf16_to_fp32(source[group_begin + offset]) * inverse_scale));
          value = sycl::max(-127, sycl::min(127, value));
          destination[group_begin + offset] = static_cast<int8_t>(value);
        }
      }
    });
  });
}

// Dense experts quantize their input once, then reuse each packed W4 word over
// eight routed rows. DP4A amortizes the quantization over gate and up together.
inline sycl::event submit_prefill_gate_up_dense_q8(int tile_count, int n, int k, const int* row_counts,
                                                   const int* tile_experts, const int* tile_rows,
                                                   float** scratch_pointers, uint32_t** gate_qweight_pointers,
                                                   uint32_t** up_qweight_pointers, float** gate_scale_pointers,
                                                   float** up_scale_pointers, uint16_t** output_pointers,
                                                   int group_size, float swiglu_limit, float swiglu_alpha,
                                                   const sycl::event& dependency) {
  if (tile_count <= 0 || n <= 0 || k <= 0) return sycl::event{};
  constexpr int subgroup_size = kSubgroupSize;
  constexpr int rows = kPrefillDenseRows;
  const int num_groups = k / group_size;
  const int k_packed = k / 8;
  const int packed_per_group = group_size / 8;
  const size_t work_groups = static_cast<size_t>(tile_count) * n;

  return queue().submit([&](sycl::handler& handler) {
    handler.depends_on(dependency);
    handler.parallel_for(
        sycl::nd_range<1>(work_groups * subgroup_size, subgroup_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(subgroup_size)]] {
          const size_t group_id = item.get_group(0);
          const int tile = static_cast<int>(group_id / static_cast<size_t>(n));
          const int output = static_cast<int>(group_id - static_cast<size_t>(tile) * n);
          const int active_expert = tile_experts[tile];
          const int row_begin = tile_rows[tile];
          const int row_count = row_counts[active_expert];
          const int lane = static_cast<int>(item.get_sub_group().get_local_linear_id());
          const int8_t* input = reinterpret_cast<const int8_t*>(scratch_pointers[active_expert]);
          const float* activation_scales = reinterpret_cast<const float*>(input + static_cast<size_t>(row_count) * k);
          const uint32_t* gate_qweight = gate_qweight_pointers[active_expert];
          const uint32_t* up_qweight = up_qweight_pointers[active_expert];
          const float* gate_scales = gate_scale_pointers[active_expert];
          const float* up_scales = up_scale_pointers[active_expert];
          float gate_accumulators[rows] = {};
          float up_accumulators[rows] = {};

          for (int group = 0; group < num_groups; ++group) {
            const int packed_begin = group * packed_per_group;
            int gate_partials[rows] = {};
            int up_partials[rows] = {};
            for (int packed = lane; packed < packed_per_group; packed += subgroup_size) {
              const int packed_k = packed_begin + packed;
              const uint32_t gate_word = gate_qweight[static_cast<size_t>(output) * k_packed + packed_k];
              const uint32_t up_word = up_qweight[static_cast<size_t>(output) * k_packed + packed_k];
              const int32_t gate_low = unpack_i4x4_to_i8x4(gate_word);
              const int32_t gate_high = unpack_i4x4_to_i8x4(gate_word >> 16);
              const int32_t up_low = unpack_i4x4_to_i8x4(up_word);
              const int32_t up_high = unpack_i4x4_to_i8x4(up_word >> 16);
#pragma unroll
              for (int row_offset = 0; row_offset < rows; ++row_offset) {
                const int row = row_begin + row_offset;
                if (row >= row_count) continue;
                const int8_t* values = input + static_cast<size_t>(row) * k + static_cast<size_t>(packed_k) * 8;
                const int32_t input_low = *reinterpret_cast<const int32_t*>(values);
                const int32_t input_high = *reinterpret_cast<const int32_t*>(values + 4);
                gate_partials[row_offset] = sycl::ext::oneapi::dot_acc(input_low, gate_low, gate_partials[row_offset]);
                gate_partials[row_offset] =
                    sycl::ext::oneapi::dot_acc(input_high, gate_high, gate_partials[row_offset]);
                up_partials[row_offset] = sycl::ext::oneapi::dot_acc(input_low, up_low, up_partials[row_offset]);
                up_partials[row_offset] = sycl::ext::oneapi::dot_acc(input_high, up_high, up_partials[row_offset]);
              }
            }

            const auto subgroup = item.get_sub_group();
            const size_t scale_offset = static_cast<size_t>(output) * num_groups + group;
#pragma unroll
            for (int row_offset = 0; row_offset < rows; ++row_offset) {
              const int gate_sum = sycl::reduce_over_group(subgroup, gate_partials[row_offset], sycl::plus<int>());
              const int up_sum = sycl::reduce_over_group(subgroup, up_partials[row_offset], sycl::plus<int>());
              if (lane == 0) {
                const int row = row_begin + row_offset;
                if (row < row_count) {
                  const float activation_scale = activation_scales[static_cast<size_t>(row) * num_groups + group];
                  gate_accumulators[row_offset] +=
                      static_cast<float>(gate_sum) * activation_scale * gate_scales[scale_offset];
                  up_accumulators[row_offset] +=
                      static_cast<float>(up_sum) * activation_scale * up_scales[scale_offset];
                }
              }
            }
          }

          if (lane == 0) {
#pragma unroll
            for (int row_offset = 0; row_offset < rows; ++row_offset) {
              const int row = row_begin + row_offset;
              if (row >= row_count) continue;
              output_pointers[active_expert][static_cast<size_t>(row) * n + output] = fp32_to_bf16(prefill_swiglu(
                  gate_accumulators[row_offset], up_accumulators[row_offset], swiglu_limit, swiglu_alpha));
            }
          }
        });
  });
}

template <int Rows>
inline sycl::event submit_prefill_down(int tile_count, int n, int k, const int* row_counts, const int* tile_experts,
                                       const int* tile_rows, uint16_t** input_pointers, uint32_t** qweight_pointers,
                                       float** scale_pointers, uint16_t** output_pointers, int group_size,
                                       const sycl::event& dependency) {
  static_assert(Rows == kPrefillSparseRows || Rows == kPrefillDenseRows);
  if (tile_count <= 0 || n <= 0 || k <= 0) return sycl::event{};
  constexpr int subgroup_size = kSubgroupSize;
  const int num_groups = k / group_size;
  const int k_packed = k / 8;
  const int packed_per_group = group_size / 8;
  const size_t work_groups = static_cast<size_t>(tile_count) * n;

  return queue().submit([&](sycl::handler& handler) {
    handler.depends_on(dependency);
    handler.parallel_for(
        sycl::nd_range<1>(work_groups * subgroup_size, subgroup_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(subgroup_size)]] {
          const size_t group_id = item.get_group(0);
          const int tile = static_cast<int>(group_id / static_cast<size_t>(n));
          const int output = static_cast<int>(group_id - static_cast<size_t>(tile) * n);
          const int active_expert = tile_experts[tile];
          const int row_begin = tile_rows[tile];
          const int row_count = row_counts[active_expert];
          const int lane = static_cast<int>(item.get_sub_group().get_local_linear_id());
          const uint16_t* input = input_pointers[active_expert];
          const uint32_t* qweight = qweight_pointers[active_expert];
          const float* scales = scale_pointers[active_expert];
          float accumulators[Rows] = {};

          for (int group = 0; group < num_groups; ++group) {
            const int packed_begin = group * packed_per_group;
            float partials[Rows] = {};
            for (int packed = lane; packed < packed_per_group; packed += subgroup_size) {
              const int packed_k = packed_begin + packed;
              const uint32_t word = qweight[static_cast<size_t>(output) * k_packed + packed_k];
#pragma unroll
              for (int row_offset = 0; row_offset < Rows; ++row_offset) {
                const int row = row_begin + row_offset;
                if (row >= row_count) continue;
                const uint16_t* values = input + static_cast<size_t>(row) * k + static_cast<size_t>(packed_k) * 8;
#pragma unroll
                for (int nibble = 0; nibble < 8; ++nibble) {
                  partials[row_offset] += bf16_to_fp32(values[nibble]) *
                                          static_cast<float>(static_cast<int>((word >> (nibble * 4)) & 0xfu) - 8);
                }
              }
            }

            const auto subgroup = item.get_sub_group();
            const float scale = lane == 0 ? scales[static_cast<size_t>(output) * num_groups + group] : 0.0f;
#pragma unroll
            for (int row_offset = 0; row_offset < Rows; ++row_offset) {
              const float sum = sycl::reduce_over_group(subgroup, partials[row_offset], sycl::plus<float>());
              if (lane == 0) accumulators[row_offset] += sum * scale;
            }
          }

          if (lane == 0) {
#pragma unroll
            for (int row_offset = 0; row_offset < Rows; ++row_offset) {
              const int row = row_begin + row_offset;
              if (row < row_count) {
                output_pointers[active_expert][static_cast<size_t>(row) * n + output] =
                    fp32_to_bf16(accumulators[row_offset]);
              }
            }
          }
        });
  });
}

}  // namespace sycl_int4

template <class T = sycl_int4::GemmKernelSYCLGPTQInt4>
class SYCL_GPTQ_INT4_MOE_TP : public AVX2_MOE_BASE<T, SYCL_GPTQ_INT4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, SYCL_GPTQ_INT4_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_up_ba_;
  using Base::m_expert_id_map_;
  using Base::m_local_down_output_ptr_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;

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

  void fused_prefill_experts(int activated_experts, int qlen) {
    if (activated_experts <= 0 || qlen <= 1) return;
    T::config();

    int sparse_tiles = 0;
    int dense_tiles = 0;
    auto* scratch = prepare_prefill_tables(activated_experts, sparse_tiles, dense_tiles);
    if (scratch == nullptr || sparse_tiles + dense_tiles == 0) return;

    sycl::event sparse_gate_up_event;
    sycl::event sparse_down_event;
    if (sparse_tiles > 0) {
      sparse_gate_up_event = sycl_int4::submit_prefill_gate_up_sparse(
          sparse_tiles, config_.intermediate_size, config_.hidden_size, scratch->prefill_row_counts,
          scratch->prefill_tile_experts, scratch->prefill_tile_rows, scratch->prefill_inputs,
          scratch->prefill_gate_qweights, scratch->prefill_up_qweights, scratch->prefill_gate_scales,
          scratch->prefill_up_scales, scratch->prefill_activations, config_.quant_config.group_size,
          config_.swiglu_limit, config_.swiglu_alpha);
      sparse_down_event = sycl_int4::submit_prefill_down<sycl_int4::kPrefillSparseRows>(
          sparse_tiles, config_.hidden_size, config_.intermediate_size, scratch->prefill_row_counts,
          scratch->prefill_tile_experts, scratch->prefill_tile_rows, scratch->prefill_activations,
          scratch->prefill_down_qweights, scratch->prefill_down_scales, scratch->prefill_inputs,
          config_.quant_config.group_size, sparse_gate_up_event);
    }

    sycl::event dense_down_event;
    if (dense_tiles > 0) {
      const int* dense_experts = scratch->prefill_tile_experts + sparse_tiles;
      const int* dense_rows = scratch->prefill_tile_rows + sparse_tiles;
      const sycl::event quantization_event = sycl_int4::submit_prefill_dense_quantization(
          dense_tiles, config_.hidden_size, config_.quant_config.group_size, scratch->prefill_row_counts, dense_experts,
          dense_rows, scratch->prefill_inputs, scratch->prefill_q8_scratch);
      const sycl::event dense_gate_up_event = sycl_int4::submit_prefill_gate_up_dense_q8(
          dense_tiles, config_.intermediate_size, config_.hidden_size, scratch->prefill_row_counts, dense_experts,
          dense_rows, scratch->prefill_q8_scratch, scratch->prefill_gate_qweights, scratch->prefill_up_qweights,
          scratch->prefill_gate_scales, scratch->prefill_up_scales, scratch->prefill_activations,
          config_.quant_config.group_size, config_.swiglu_limit, config_.swiglu_alpha, quantization_event);
      dense_down_event = sycl_int4::submit_prefill_down<sycl_int4::kPrefillDenseRows>(
          dense_tiles, config_.hidden_size, config_.intermediate_size, scratch->prefill_row_counts, dense_experts,
          dense_rows, scratch->prefill_activations, scratch->prefill_down_qweights, scratch->prefill_down_scales,
          scratch->prefill_inputs, config_.quant_config.group_size, dense_gate_up_event);
    }

    if (sparse_tiles > 0) sycl_int4::wait_and_throw(sparse_down_event);
    if (dense_tiles > 0) sycl_int4::wait_and_throw(dense_down_event);

    // Down writes BF16 directly over the no-longer-needed packed input. The
    // base merge can consume that storage without an FP32-to-BF16 host pass.
    for (int task = 0; task < activated_experts; ++task) {
      const int expert = m_expert_id_map_[task];
      m_local_down_output_ptr_[expert] = reinterpret_cast<ggml_bf16_t*>(scratch->prefill_inputs[task]);
    }
  }

  void decode_gate_up_activation(int activated_experts, int qlen) {
    if (qlen != 1 || activated_experts <= 0) return;
    auto* scratch = get_scratch();
    if (scratch->gate_up_pending) {
      for (auto& event : scratch->gate_up_events) sycl_int4::wait_and_throw(event);
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
    for (auto& event : down_events) sycl_int4::wait_and_throw(event);

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
  std::unique_ptr<sycl_int4::BackendScratch> backend_scratch_;

  sycl_int4::BackendScratch* get_scratch() {
    if (!backend_scratch_) {
      backend_scratch_ = std::make_unique<sycl_int4::BackendScratch>();
    }
    return backend_scratch_.get();
  }

  sycl_int4::BackendScratch* prepare_prefill_tables(int activated_experts, int& sparse_tiles, int& dense_tiles) {
    auto* scratch = get_scratch();
    if (scratch->prefill_expert_capacity < activated_experts) {
      sycl_int4::usm_free(scratch->prefill_inputs);
      sycl_int4::usm_free(scratch->prefill_activations);
      sycl_int4::usm_free(scratch->prefill_gate_qweights);
      sycl_int4::usm_free(scratch->prefill_up_qweights);
      sycl_int4::usm_free(scratch->prefill_down_qweights);
      sycl_int4::usm_free(scratch->prefill_gate_scales);
      sycl_int4::usm_free(scratch->prefill_up_scales);
      sycl_int4::usm_free(scratch->prefill_down_scales);
      sycl_int4::usm_free(scratch->prefill_q8_scratch);
      sycl_int4::usm_free(scratch->prefill_row_counts);
      scratch->prefill_inputs = nullptr;
      scratch->prefill_activations = nullptr;
      scratch->prefill_gate_qweights = nullptr;
      scratch->prefill_up_qweights = nullptr;
      scratch->prefill_down_qweights = nullptr;
      scratch->prefill_gate_scales = nullptr;
      scratch->prefill_up_scales = nullptr;
      scratch->prefill_down_scales = nullptr;
      scratch->prefill_q8_scratch = nullptr;
      scratch->prefill_row_counts = nullptr;

      scratch->prefill_inputs = sycl_int4::usm_alloc<uint16_t*>(activated_experts, "prefill input pointer table");
      scratch->prefill_activations =
          sycl_int4::usm_alloc<uint16_t*>(activated_experts, "prefill activation pointer table");
      scratch->prefill_gate_qweights =
          sycl_int4::usm_alloc<uint32_t*>(activated_experts, "prefill gate qweight pointer table");
      scratch->prefill_up_qweights =
          sycl_int4::usm_alloc<uint32_t*>(activated_experts, "prefill up qweight pointer table");
      scratch->prefill_down_qweights =
          sycl_int4::usm_alloc<uint32_t*>(activated_experts, "prefill down qweight pointer table");
      scratch->prefill_gate_scales =
          sycl_int4::usm_alloc<float*>(activated_experts, "prefill gate scale pointer table");
      scratch->prefill_up_scales = sycl_int4::usm_alloc<float*>(activated_experts, "prefill up scale pointer table");
      scratch->prefill_down_scales =
          sycl_int4::usm_alloc<float*>(activated_experts, "prefill down scale pointer table");
      scratch->prefill_q8_scratch = sycl_int4::usm_alloc<float*>(activated_experts, "prefill Q8 scratch pointer table");
      scratch->prefill_row_counts = sycl_int4::usm_alloc<int>(activated_experts, "prefill row counts");
      scratch->prefill_expert_capacity = activated_experts;
    }

    sparse_tiles = 0;
    dense_tiles = 0;
    for (int task = 0; task < activated_experts; ++task) {
      const int expert = m_expert_id_map_[task];
      const int rows = m_local_num_[expert];
      scratch->prefill_inputs[task] = gate_up_ba_[expert]->data;
      scratch->prefill_activations[task] = down_ba_[expert]->data;
      scratch->prefill_gate_qweights[task] = gate_bb_[expert]->qweight;
      scratch->prefill_up_qweights[task] = up_bb_[expert]->qweight;
      scratch->prefill_down_qweights[task] = down_bb_[expert]->qweight;
      scratch->prefill_gate_scales[task] = gate_bb_[expert]->scales;
      scratch->prefill_up_scales[task] = up_bb_[expert]->scales;
      scratch->prefill_down_scales[task] = down_bb_[expert]->scales;
      scratch->prefill_q8_scratch[task] = down_bc_[expert]->data;
      scratch->prefill_row_counts[task] = rows;
      if (rows >= sycl_int4::kPrefillDenseThreshold) {
        dense_tiles += (rows + sycl_int4::kPrefillDenseRows - 1) / sycl_int4::kPrefillDenseRows;
      } else {
        sparse_tiles += (rows + sycl_int4::kPrefillSparseRows - 1) / sycl_int4::kPrefillSparseRows;
      }
    }

    const int tile_count = sparse_tiles + dense_tiles;
    if (scratch->prefill_tile_capacity < tile_count) {
      sycl_int4::usm_free(scratch->prefill_tile_experts);
      sycl_int4::usm_free(scratch->prefill_tile_rows);
      scratch->prefill_tile_experts = nullptr;
      scratch->prefill_tile_rows = nullptr;
      scratch->prefill_tile_experts = sycl_int4::usm_alloc<int>(tile_count, "prefill tile expert table");
      scratch->prefill_tile_rows = sycl_int4::usm_alloc<int>(tile_count, "prefill tile row table");
      scratch->prefill_tile_capacity = tile_count;
    }

    int sparse_tile = 0;
    int dense_tile = sparse_tiles;
    for (int task = 0; task < activated_experts; ++task) {
      const int rows = scratch->prefill_row_counts[task];
      const bool dense = rows >= sycl_int4::kPrefillDenseThreshold;
      const int rows_per_tile = dense ? sycl_int4::kPrefillDenseRows : sycl_int4::kPrefillSparseRows;
      int& tile = dense ? dense_tile : sparse_tile;
      for (int row = 0; row < rows; row += rows_per_tile) {
        scratch->prefill_tile_experts[tile] = task;
        scratch->prefill_tile_rows[tile] = row;
        ++tile;
      }
    }
    return scratch;
  }

  void prepare_contiguous_weights() {
    auto* scratch = get_scratch();
    if (scratch->weights_ready || config_.expert_num <= 0) return;

    const size_t experts = static_cast<size_t>(config_.expert_num);
    const size_t gate_up_qweight_stride = gate_bb_[0]->qweight_elements();
    const size_t gate_up_scale_stride = gate_bb_[0]->scale_elements();
    const size_t down_qweight_stride = down_bb_[0]->qweight_elements();
    const size_t down_scale_stride = down_bb_[0]->scale_elements();

    scratch->reset_weights();
    try {
      scratch->gate_qweight =
          sycl_int4::usm_alloc<uint32_t>(experts * gate_up_qweight_stride, "contiguous gate qweight");
      scratch->up_qweight = sycl_int4::usm_alloc<uint32_t>(experts * gate_up_qweight_stride, "contiguous up qweight");
      scratch->down_qweight = sycl_int4::usm_alloc<uint32_t>(experts * down_qweight_stride, "contiguous down qweight");
      scratch->gate_scales = sycl_int4::usm_alloc<float>(experts * gate_up_scale_stride, "contiguous gate scales");
      scratch->up_scales = sycl_int4::usm_alloc<float>(experts * gate_up_scale_stride, "contiguous up scales");
      scratch->down_scales = sycl_int4::usm_alloc<float>(experts * down_scale_stride, "contiguous down scales");
    } catch (...) {
      scratch->reset_weights();
      throw;
    }

    scratch->gate_up_qweight_stride = gate_up_qweight_stride;
    scratch->gate_up_scale_stride = gate_up_scale_stride;
    scratch->down_qweight_stride = down_qweight_stride;
    scratch->down_scale_stride = down_scale_stride;

    for (size_t expert = 0; expert < experts; ++expert) {
      gate_bb_[expert]->bind_view(scratch->gate_qweight + expert * gate_up_qweight_stride,
                                  scratch->gate_scales + expert * gate_up_scale_stride);
      up_bb_[expert]->bind_view(scratch->up_qweight + expert * gate_up_qweight_stride,
                                scratch->up_scales + expert * gate_up_scale_stride);
      down_bb_[expert]->bind_view(scratch->down_qweight + expert * down_qweight_stride,
                                  scratch->down_scales + expert * down_scale_stride);
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
    const int tensor_parallel_count = this->tp_count;
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

    struct TPWeightStaging {
      std::unique_ptr<uint32_t[]> gate_qweight;
      std::unique_ptr<uint32_t[]> up_qweight;
      std::unique_ptr<uint32_t[]> down_qweight;
      std::unique_ptr<float[]> gate_scales;
      std::unique_ptr<float[]> up_scales;
      std::unique_ptr<float[]> down_scales;
    };
    std::vector<TPWeightStaging> staging(static_cast<size_t>(tensor_parallel_count));
    std::vector<GeneralMOEConfig*> staging_configs(static_cast<size_t>(tensor_parallel_count), nullptr);
    struct TPConfigPointerReset {
      std::vector<GeneralMOEConfig*>& configs;
      ~TPConfigPointerReset() {
        for (auto* config : configs) {
          if (config == nullptr) continue;
          config->gate_proj = nullptr;
          config->up_proj = nullptr;
          config->down_proj = nullptr;
          config->gate_scale = nullptr;
          config->up_scale = nullptr;
          config->down_scale = nullptr;
        }
      }
    } reset_staging_pointers{staging_configs};

    pool->dispense_backend()->do_numa_job([&, this](int index) {
      auto& tp_config = tensor_parallel_backends[index]->config_;
      staging_configs[index] = &tp_config;
      auto& tp_staging = staging[index];
      const int tp_intermediate = tp_config.intermediate_size;
      const size_t tp_gate_up_qweight = static_cast<size_t>(gate_up_k_packed) * tp_intermediate;
      const size_t tp_gate_up_scales = static_cast<size_t>(gate_up_num_groups) * tp_intermediate;
      tp_staging.gate_qweight =
          std::make_unique<uint32_t[]>(static_cast<size_t>(tp_config.expert_num) * tp_gate_up_qweight);
      tp_staging.up_qweight =
          std::make_unique<uint32_t[]>(static_cast<size_t>(tp_config.expert_num) * tp_gate_up_qweight);
      tp_staging.gate_scales = std::make_unique<float[]>(static_cast<size_t>(tp_config.expert_num) * tp_gate_up_scales);
      tp_staging.up_scales = std::make_unique<float[]>(static_cast<size_t>(tp_config.expert_num) * tp_gate_up_scales);

      const int tp_down_k_packed = tp_intermediate / 8;
      const int tp_down_num_groups = tp_intermediate / group_size;
      const size_t tp_down_qweight = static_cast<size_t>(tp_down_k_packed) * full_hidden;
      const size_t tp_down_scales = static_cast<size_t>(tp_down_num_groups) * full_hidden;
      tp_staging.down_qweight =
          std::make_unique<uint32_t[]>(static_cast<size_t>(tp_config.expert_num) * tp_down_qweight);
      tp_staging.down_scales = std::make_unique<float[]>(static_cast<size_t>(tp_config.expert_num) * tp_down_scales);

      tp_config.gate_proj = tp_staging.gate_qweight.get();
      tp_config.up_proj = tp_staging.up_qweight.get();
      tp_config.down_proj = tp_staging.down_qweight.get();
      tp_config.gate_scale = tp_staging.gate_scales.get();
      tp_config.up_scale = tp_staging.up_scales.get();
      tp_config.down_scale = tp_staging.down_scales.get();

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
    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int, int, const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                                    const std::vector<uintptr_t>&, const std::vector<uintptr_t>&) {
    throw std::runtime_error("SYCL GPTQ INT4 does not support write_weight_scale_to_buffer");
  }
};

#endif  // CPUINFER_OPERATOR_SYCL_GPTQ_INT4_MOE_H
