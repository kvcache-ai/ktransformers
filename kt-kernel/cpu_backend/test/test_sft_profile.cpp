// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../operators/sft_profile.hpp"

namespace {
void require(bool condition) {
  if (!condition) throw std::runtime_error("SFT profiler contract failed");
}
}  // namespace

int main() {
  const int rows[] = {0, 1, 7, 8, 32, 33, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 8192};
  constexpr size_t count = sizeof(rows) / sizeof(rows[0]);
  SFTProfiler profile(true);
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 4; ++worker) workers.emplace_back([&] { profile.record_expert_rows(rows, count); });
  for (auto& worker : workers) worker.join();
  std::map<std::string, double> result;
  profile.append(result, "", true);
  require(result.at("expert_rows.maximum") == 8192);
  require(result.at("expert_rows.1_7.observations") == 8);
  require(result.at("expert_rows.1_7.rows") == 32);
  require(result.at("expert_rows.8_32.rows") == 160);
  require(result.at("expert_rows.33_63.rows") == 384);
  require(result.at("expert_rows.64_127.rows") == 764);
  require(result.at("expert_rows.128_255.rows") == 1532);
  require(result.at("expert_rows.256_511.rows") == 3068);
  require(result.at("expert_rows.512_1023.rows") == 6140);
  require(result.at("expert_rows.1024_plus.rows") == 36864);
  profile.append(result, "");
  require(result.at("expert_rows.maximum") == 0);
  require(result.at("expert_rows.1_7.observations") == 0);
  require(result.at("expert_rows.1024_plus.rows") == 0);
  profile.record_expert_rows(rows, count);
  profile.reset();
  profile.append(result, "");
  require(result.at("expert_rows.maximum") == 0);

  SFTProfiler disabled(false);
  disabled.record_expert_rows(rows, count);
  disabled.append(result, "");
  require(result.at("expert_rows.maximum") == 0);
  require(result.at("expert_rows.1_7.observations") == 0);
  std::puts("SFT profiler histogram: PASS");
}
