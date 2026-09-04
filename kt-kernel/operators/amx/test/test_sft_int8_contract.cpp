#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../fp8-moe.hpp"
#include "../sft-k2-moe.hpp"
#include "../sft_moe.hpp"
#include "../../moe-sft-tp.hpp"

namespace {

using Bf16SFT = AMX_SFT_MOE_TP<amx::GemmKernel224BF16>;
using Int8SFT = AMX_SFT_MOE_TP<amx::GemmKernel224Int8>;
using Int8SkipLoraSFT = AMX_SFT_MOE_TP<amx::GemmKernel224Int8, AMX_MOE_TP, true>;
using Int4SFT = AMX_SFT_MOE_TP<amx::GemmKernel224Int4>;
using Fp8SFT = AMX_SFT_MOE_TP<amx::GemmKernel224FP8, AMX_FP8_MOE_TP>;
using RawInt4SFT = AMX_K2_SFT_MOE_TP<>;

static_assert(Bf16SFT::kSupportsAuthoritativeBaseGrads);
static_assert(Bf16SFT::kSupportsAuthoritativeLoraGrads);
static_assert(!Int8SFT::kSupportsAuthoritativeBaseGrads);
static_assert(Int8SFT::kSupportsAuthoritativeLoraGrads);
static_assert(!Int8SkipLoraSFT::kSupportsAuthoritativeBaseGrads);
static_assert(!Int8SkipLoraSFT::kSupportsAuthoritativeLoraGrads);
static_assert(!Int4SFT::kSupportsAuthoritativeBaseGrads);
static_assert(!Int4SFT::kSupportsAuthoritativeLoraGrads);
static_assert(Fp8SFT::kIsFP8Backend);
static_assert(!Fp8SFT::kSupportsAuthoritativeBaseGrads);
static_assert(Fp8SFT::kSupportsAuthoritativeLoraGrads);
static_assert(RawInt4SFT::kUsesKGroupPackedBaseWeights);
static_assert(!RawInt4SFT::kSupportsAuthoritativeBaseGrads);
static_assert(RawInt4SFT::kSupportsAuthoritativeLoraGrads);

bool contains(const std::string& value, const char* expected) {
  return value.find(expected) != std::string::npos;
}

bool rejects_unaligned_rawint4_dimensions(int hidden_size, int intermediate_size) {
  MOESFTConfig config(1, 1, hidden_size, intermediate_size);
  config.quant_config.bits = 4;
  config.quant_config.group_size = 32;
  config.quant_config.zero_point = false;
  config.lora_rank = 8;

  try {
    RawInt4SFT invalid(config);
  } catch (const std::runtime_error& error) {
    return contains(error.what(), "group-32 aligned");
  }
  return false;
}

}  // namespace

int main() {
  MOESFTConfig full_sft_config(4, 2, 256, 256);
  full_sft_config.lora_rank = 32;
  full_sft_config.lora_alpha = 48.0f;
  full_sft_config.lora_dropout = 0.125f;
  full_sft_config.max_cache_depth = 7;
  full_sft_config.full_weight_grad = true;
  GeneralMOEConfig local_base = static_cast<const GeneralMOEConfig&>(full_sft_config);
  local_base.intermediate_size = 128;
  const MOESFTConfig local_sft_config = make_tp_sft_config(full_sft_config, local_base);

  int failures = 0;
  if (!rejects_unaligned_rawint4_dimensions(255, 256)) {
    std::fprintf(stderr, "RAWINT4 hidden-size misalignment was not rejected before construction\n");
    ++failures;
  }
  if (!rejects_unaligned_rawint4_dimensions(256, 255)) {
    std::fprintf(stderr, "RAWINT4 intermediate-size misalignment was not rejected before construction\n");
    ++failures;
  }

  if (local_sft_config.intermediate_size != 128 || local_sft_config.lora_rank != 32 ||
      local_sft_config.lora_alpha != 48.0f || local_sft_config.lora_dropout != 0.125f ||
      local_sft_config.max_cache_depth != 7 || !local_sft_config.full_weight_grad) {
    std::fprintf(stderr, "full MOESFTConfig was not preserved while making the TP-local slice\n");
    ++failures;
  }

  const auto unique_suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path =
      std::filesystem::temp_directory_path() / ("kt-int8-file-contract-" + unique_suffix + ".kt");
  const auto missing_path = path.string() + ".missing";
  const std::vector<char> expected = {0, 1, 2, 3, 4, 5, 6, 7};
  {
    std::ofstream output(path, std::ios::binary);
    output.write(expected.data(), static_cast<std::streamsize>(expected.size()));
  }

  std::vector<char> actual(expected.size(), static_cast<char>(0x7f));
  std::string error =
      kt::detail::read_exact_weight_file_slice(path, actual.data(), expected.size(), 1, 0);
  if (!error.empty() || actual != expected) {
    std::fprintf(stderr, "exact read failed: %s\n", error.c_str());
    ++failures;
  }

  std::fill(actual.begin(), actual.end(), static_cast<char>(0x7f));
  error = kt::detail::read_exact_weight_file_slice(path, actual.data(), expected.size(), 2, 1);
  if (!error.empty() ||
      !std::equal(actual.begin() + expected.size() / 2, actual.end(),
                  expected.begin() + expected.size() / 2) ||
      !std::all_of(actual.begin(), actual.begin() + expected.size() / 2,
                   [](char value) { return value == static_cast<char>(0x7f); })) {
    std::fprintf(stderr, "split read failed: %s\n", error.c_str());
    ++failures;
  }

  error = kt::detail::read_exact_weight_file_slice(path, actual.data(), expected.size() - 1, 1, 0);
  if (!contains(error, "size mismatch")) {
    std::fprintf(stderr, "size mismatch was not rejected: %s\n", error.c_str());
    ++failures;
  }

  error = kt::detail::read_exact_weight_file_slice(missing_path, actual.data(), expected.size(), 1, 0);
  if (!contains(error, "missing weight file")) {
    std::fprintf(stderr, "missing file was not rejected: %s\n", error.c_str());
    ++failures;
  }

  error = kt::detail::read_exact_weight_file_slice(path, actual.data(), expected.size(), 3, 0);
  if (!contains(error, "not divisible")) {
    std::fprintf(stderr, "invalid split was not rejected: %s\n", error.c_str());
    ++failures;
  }

  std::filesystem::remove(path);
  if (failures == 0) std::printf("SFT capability, RAWINT4 dimensions, and .kt I/O contract: PASS\n");
  return failures == 0 ? 0 : 1;
}
