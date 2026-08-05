#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../fp8_tp_staging.hpp"

namespace {

constexpr int kExperts = 3;
constexpr int kHidden = 256;
constexpr int kIntermediate = 256;
constexpr int kLocalIntermediate = 128;
constexpr int kGroup = 128;

uint32_t float_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float scale_pattern(int projection, int expert, int index) {
  const uint32_t bits = 0x3f000000u + static_cast<uint32_t>(projection * 0x10000 + expert * 0x100 + index);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint8_t weight_pattern(int projection, int expert, int row, int column) {
  return static_cast<uint8_t>(projection * 61 + expert * 37 + row * 11 + column * 17);
}

bool test_two_way_per_expert_staging() {
  const size_t full_weight_elems = static_cast<size_t>(kIntermediate) * kHidden;
  const size_t full_scale_elems = static_cast<size_t>(kIntermediate / kGroup) * (kHidden / kGroup);
  std::vector<std::vector<uint8_t>> gate(kExperts, std::vector<uint8_t>(full_weight_elems));
  std::vector<std::vector<uint8_t>> up(kExperts, std::vector<uint8_t>(full_weight_elems));
  std::vector<std::vector<uint8_t>> down(kExperts, std::vector<uint8_t>(full_weight_elems));
  std::vector<std::vector<float>> gate_scale(kExperts, std::vector<float>(full_scale_elems));
  std::vector<std::vector<float>> up_scale(kExperts, std::vector<float>(full_scale_elems));
  std::vector<std::vector<float>> down_scale(kExperts, std::vector<float>(full_scale_elems));

  GeneralMOEConfig config(kExperts, 1, kHidden, kIntermediate);
  config.quant_config.group_size = kGroup;
  config.quant_config.zero_point = false;
  config.gate_projs.resize(1);
  config.up_projs.resize(1);
  config.down_projs.resize(1);
  config.gate_scales.resize(1);
  config.up_scales.resize(1);
  config.down_scales.resize(1);

  for (int expert = 0; expert < kExperts; ++expert) {
    for (int row = 0; row < kIntermediate; ++row) {
      for (int column = 0; column < kHidden; ++column) {
        gate[expert][static_cast<size_t>(row) * kHidden + column] = weight_pattern(0, expert, row, column);
        up[expert][static_cast<size_t>(row) * kHidden + column] = weight_pattern(1, expert, row, column);
        down[expert][static_cast<size_t>(row) * kIntermediate + column] =
            weight_pattern(2, expert, row, column);
      }
    }
    for (size_t index = 0; index < full_scale_elems; ++index) {
      gate_scale[expert][index] = scale_pattern(0, expert, static_cast<int>(index));
      up_scale[expert][index] = scale_pattern(1, expert, static_cast<int>(index));
      down_scale[expert][index] = scale_pattern(2, expert, static_cast<int>(index));
    }
    config.gate_projs[0].push_back(gate[expert].data());
    config.up_projs[0].push_back(up[expert].data());
    config.down_projs[0].push_back(down[expert].data());
    config.gate_scales[0].push_back(gate_scale[expert].data());
    config.up_scales[0].push_back(up_scale[expert].data());
    config.down_scales[0].push_back(down_scale[expert].data());
  }

  amx::validate_block_fp8_tp_source(config);
  amx::BlockFP8TPStaging parts[2];
  for (int tp = 0; tp < 2; ++tp) {
    parts[tp].allocate(kExperts, kHidden, kLocalIntermediate, kGroup);
    for (int expert = 0; expert < kExperts; ++expert) {
      amx::stage_block_fp8_tp_expert(config, kLocalIntermediate, tp * kLocalIntermediate, expert, parts[tp]);
    }
  }

  const size_t local_weight_elems = static_cast<size_t>(kLocalIntermediate) * kHidden;
  const size_t local_scale_elems = static_cast<size_t>(kLocalIntermediate / kGroup) * (kHidden / kGroup);
  for (int expert = 0; expert < kExperts; ++expert) {
    for (int tp = 0; tp < 2; ++tp) {
      const size_t local_base = static_cast<size_t>(expert) * local_weight_elems;
      const size_t source_gate_base = static_cast<size_t>(tp * kLocalIntermediate) * kHidden;
      if (std::memcmp(parts[tp].gate.get() + local_base, gate[expert].data() + source_gate_base,
                      local_weight_elems) != 0 ||
          std::memcmp(parts[tp].up.get() + local_base, up[expert].data() + source_gate_base,
                      local_weight_elems) != 0) {
        return false;
      }

      for (int row = 0; row < kHidden; ++row) {
        const uint8_t* expected = down[expert].data() + static_cast<size_t>(row) * kIntermediate +
                                  tp * kLocalIntermediate;
        const uint8_t* actual = parts[tp].down.get() + local_base + static_cast<size_t>(row) * kLocalIntermediate;
        if (std::memcmp(actual, expected, kLocalIntermediate) != 0) return false;
      }

      const size_t local_scale_base = static_cast<size_t>(expert) * local_scale_elems;
      const size_t gate_scale_source = static_cast<size_t>(tp) * local_scale_elems;
      for (size_t index = 0; index < local_scale_elems; ++index) {
        if (float_bits(parts[tp].gate_scale[local_scale_base + index]) !=
                float_bits(gate_scale[expert][gate_scale_source + index]) ||
            float_bits(parts[tp].up_scale[local_scale_base + index]) !=
                float_bits(up_scale[expert][gate_scale_source + index])) {
          return false;
        }
      }

      const int full_down_stride = kIntermediate / kGroup;
      const int local_down_stride = kLocalIntermediate / kGroup;
      for (int block_row = 0; block_row < kHidden / kGroup; ++block_row) {
        const size_t actual_index = local_scale_base + static_cast<size_t>(block_row) * local_down_stride;
        const size_t expected_index = static_cast<size_t>(block_row) * full_down_stride + tp * local_down_stride;
        if (float_bits(parts[tp].down_scale[actual_index]) != float_bits(down_scale[expert][expected_index])) {
          return false;
        }
      }
    }
  }
  return true;
}

bool test_invalid_source_and_slice_rejected() {
  GeneralMOEConfig config(1, 1, kHidden, kIntermediate);
  config.quant_config.group_size = kGroup;
  config.gate_projs = {{reinterpret_cast<void*>(1)}};
  bool incomplete_rejected = false;
  try {
    amx::validate_block_fp8_tp_source(config);
  } catch (const std::invalid_argument&) {
    incomplete_rejected = true;
  }

  std::vector<uint8_t> weight(static_cast<size_t>(kIntermediate) * kHidden);
  std::vector<float> scale(static_cast<size_t>(kIntermediate / kGroup) * (kHidden / kGroup));
  config.gate_projs = {{weight.data()}};
  config.up_projs = {{weight.data()}};
  config.down_projs = {{weight.data()}};
  config.gate_scales = {{scale.data()}};
  config.up_scales = {{scale.data()}};
  config.down_scales = {{scale.data()}};
  amx::BlockFP8TPStaging staging;
  staging.allocate(1, kHidden, kLocalIntermediate, kGroup);
  bool slice_rejected = false;
  try {
    amx::stage_block_fp8_tp_expert(config, kLocalIntermediate, 64, 0, staging);
  } catch (const std::invalid_argument&) {
    slice_rejected = true;
  }
  return incomplete_rejected && slice_rejected;
}

}  // namespace

int main() {
  const bool tp2 = test_two_way_per_expert_staging();
  const bool guards = test_invalid_source_and_slice_rejected();
  std::cout << "native FP8 TP2 per-expert load staging: " << (tp2 ? "PASS" : "FAIL") << '\n';
  std::cout << "native FP8 TP load guards: " << (guards ? "PASS" : "FAIL") << '\n';
  return tp2 && guards ? 0 : 1;
}
