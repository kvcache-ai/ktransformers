#ifndef CPUINFER_OPERATOR_AMX_FP8_TP_STAGING_HPP
#define CPUINFER_OPERATOR_AMX_FP8_TP_STAGING_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "../common.hpp"

namespace amx {

struct BlockFP8TPStaging {
  std::unique_ptr<uint8_t[]> gate;
  std::unique_ptr<uint8_t[]> up;
  std::unique_ptr<uint8_t[]> down;
  std::unique_ptr<float[]> gate_scale;
  std::unique_ptr<float[]> up_scale;
  std::unique_ptr<float[]> down_scale;

  size_t weight_elems = 0;
  size_t scale_elems = 0;

  void allocate(int expert_num, int hidden_size, int local_intermediate_size, int group_size) {
    if (expert_num <= 0 || hidden_size <= 0 || local_intermediate_size <= 0 || group_size <= 0) {
      throw std::invalid_argument("FP8 TP staging dimensions must be positive");
    }
    weight_elems = static_cast<size_t>(hidden_size) * local_intermediate_size;
    scale_elems = static_cast<size_t>(hidden_size / group_size) * (local_intermediate_size / group_size);
    const size_t weights_total = static_cast<size_t>(expert_num) * weight_elems;
    const size_t scales_total = static_cast<size_t>(expert_num) * scale_elems;
    gate = std::make_unique<uint8_t[]>(weights_total);
    up = std::make_unique<uint8_t[]>(weights_total);
    down = std::make_unique<uint8_t[]>(weights_total);
    gate_scale = std::make_unique<float[]>(scales_total);
    up_scale = std::make_unique<float[]>(scales_total);
    down_scale = std::make_unique<float[]>(scales_total);
  }
};

inline void validate_block_fp8_tp_source(const GeneralMOEConfig& config) {
  const int group_size = config.quant_config.group_size;
  if (group_size <= 0 || config.quant_config.zero_point) {
    throw std::invalid_argument("native FP8 TP loading requires block scales and zero_point=false");
  }
  if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.expert_num <= 0 ||
      config.hidden_size % group_size != 0 || config.intermediate_size % group_size != 0) {
    throw std::invalid_argument("native FP8 TP loading requires group-aligned model dimensions");
  }

  const bool per_expert = !config.gate_projs.empty();
  if (!per_expert) {
    if (config.gate_proj == nullptr || config.up_proj == nullptr || config.down_proj == nullptr ||
        config.gate_scale == nullptr || config.up_scale == nullptr || config.down_scale == nullptr) {
      throw std::invalid_argument("native FP8 TP loading requires all weight and scale sources");
    }
    return;
  }

  const auto has_experts = [&](const std::vector<std::vector<void*>>& pointers) {
    return !pointers.empty() && pointers[0].size() >= static_cast<size_t>(config.expert_num);
  };
  if (!has_experts(config.gate_projs) || !has_experts(config.up_projs) || !has_experts(config.down_projs) ||
      !has_experts(config.gate_scales) || !has_experts(config.up_scales) || !has_experts(config.down_scales)) {
    throw std::invalid_argument("native FP8 per-expert TP source is incomplete");
  }
  for (int expert = 0; expert < config.expert_num; ++expert) {
    if (config.gate_projs[0][expert] == nullptr || config.up_projs[0][expert] == nullptr ||
        config.down_projs[0][expert] == nullptr || config.gate_scales[0][expert] == nullptr ||
        config.up_scales[0][expert] == nullptr || config.down_scales[0][expert] == nullptr) {
      throw std::invalid_argument("native FP8 per-expert TP source contains a null pointer at expert " +
                                  std::to_string(expert));
    }
  }
}

inline void stage_block_fp8_tp_expert(const GeneralMOEConfig& full_config, int local_intermediate_size,
                                      int intermediate_offset, int logical_expert, BlockFP8TPStaging& staging) {
  const int group_size = full_config.quant_config.group_size;
  if (local_intermediate_size <= 0 || intermediate_offset < 0 ||
      intermediate_offset + local_intermediate_size > full_config.intermediate_size ||
      local_intermediate_size % group_size != 0 || intermediate_offset % group_size != 0) {
    throw std::invalid_argument("native FP8 TP slice must be group-aligned and within the full intermediate size");
  }
  if (logical_expert < 0 || logical_expert >= full_config.expert_num) {
    throw std::out_of_range("native FP8 TP logical expert is out of range");
  }

  const int hidden_size = full_config.hidden_size;
  const size_t local_weight_elems = static_cast<size_t>(local_intermediate_size) * hidden_size;
  const size_t local_scale_elems =
      static_cast<size_t>(local_intermediate_size / group_size) * (hidden_size / group_size);
  if (staging.weight_elems != local_weight_elems || staging.scale_elems != local_scale_elems ||
      staging.gate == nullptr || staging.up == nullptr || staging.down == nullptr || staging.gate_scale == nullptr ||
      staging.up_scale == nullptr || staging.down_scale == nullptr) {
    throw std::invalid_argument("native FP8 TP staging storage has the wrong shape");
  }

  const bool per_expert = !full_config.gate_projs.empty();
  const size_t full_weight_elems = static_cast<size_t>(full_config.intermediate_size) * hidden_size;
  const size_t full_scale_elems =
      static_cast<size_t>(full_config.intermediate_size / group_size) * (hidden_size / group_size);
  const size_t gate_up_weight_offset = static_cast<size_t>(intermediate_offset) * hidden_size;
  const size_t gate_up_scale_offset =
      static_cast<size_t>(intermediate_offset / group_size) * (hidden_size / group_size);

  const uint8_t* gate_source;
  const uint8_t* up_source;
  const uint8_t* down_source;
  const float* gate_scale_source;
  const float* up_scale_source;
  const float* down_scale_source;
  if (per_expert) {
    gate_source = static_cast<const uint8_t*>(full_config.gate_projs[0][logical_expert]);
    up_source = static_cast<const uint8_t*>(full_config.up_projs[0][logical_expert]);
    down_source = static_cast<const uint8_t*>(full_config.down_projs[0][logical_expert]);
    gate_scale_source = static_cast<const float*>(full_config.gate_scales[0][logical_expert]);
    up_scale_source = static_cast<const float*>(full_config.up_scales[0][logical_expert]);
    down_scale_source = static_cast<const float*>(full_config.down_scales[0][logical_expert]);
  } else {
    gate_source = static_cast<const uint8_t*>(full_config.gate_proj) + logical_expert * full_weight_elems;
    up_source = static_cast<const uint8_t*>(full_config.up_proj) + logical_expert * full_weight_elems;
    down_source = static_cast<const uint8_t*>(full_config.down_proj) + logical_expert * full_weight_elems;
    gate_scale_source = static_cast<const float*>(full_config.gate_scale) + logical_expert * full_scale_elems;
    up_scale_source = static_cast<const float*>(full_config.up_scale) + logical_expert * full_scale_elems;
    down_scale_source = static_cast<const float*>(full_config.down_scale) + logical_expert * full_scale_elems;
  }

  uint8_t* gate_destination = staging.gate.get() + logical_expert * local_weight_elems;
  uint8_t* up_destination = staging.up.get() + logical_expert * local_weight_elems;
  uint8_t* down_destination = staging.down.get() + logical_expert * local_weight_elems;
  float* gate_scale_destination = staging.gate_scale.get() + logical_expert * local_scale_elems;
  float* up_scale_destination = staging.up_scale.get() + logical_expert * local_scale_elems;
  float* down_scale_destination = staging.down_scale.get() + logical_expert * local_scale_elems;

  std::memcpy(gate_destination, gate_source + gate_up_weight_offset, local_weight_elems);
  std::memcpy(up_destination, up_source + gate_up_weight_offset, local_weight_elems);
  std::memcpy(gate_scale_destination, gate_scale_source + gate_up_scale_offset,
              local_scale_elems * sizeof(float));
  std::memcpy(up_scale_destination, up_scale_source + gate_up_scale_offset, local_scale_elems * sizeof(float));

  for (int row = 0; row < hidden_size; ++row) {
    const size_t source_offset = static_cast<size_t>(row) * full_config.intermediate_size + intermediate_offset;
    const size_t destination_offset = static_cast<size_t>(row) * local_intermediate_size;
    std::memcpy(down_destination + destination_offset, down_source + source_offset,
                static_cast<size_t>(local_intermediate_size));
  }

  const int full_down_scale_stride = full_config.intermediate_size / group_size;
  const int local_down_scale_stride = local_intermediate_size / group_size;
  const int down_scale_column_offset = intermediate_offset / group_size;
  const int down_scale_rows = hidden_size / group_size;
  for (int block_row = 0; block_row < down_scale_rows; ++block_row) {
    const float* source =
        down_scale_source + static_cast<size_t>(block_row) * full_down_scale_stride + down_scale_column_offset;
    float* destination = down_scale_destination + static_cast<size_t>(block_row) * local_down_scale_stride;
    std::memcpy(destination, source, static_cast<size_t>(local_down_scale_stride) * sizeof(float));
  }
}

}  // namespace amx

#endif
