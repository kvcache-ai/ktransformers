// SPDX-License-Identifier: Apache-2.0
// Build: icpx -fsycl -std=c++20 -fp-model=precise -O2 test_prefill_merge.cpp -o test_prefill_merge
// Optional event benchmark: KT_SYCL_PROFILE=1 ./test_prefill_merge 50
#include <algorithm>
#include <bit>
#include <iostream>
#include <random>
#include <vector>

#include "../../operators/sycl/gptq_int4_xmx-kernels.hpp"
using namespace sycl_xmx;
void check(int rows, int topk, int hidden, int samples) {
  const int routed = rows * std::max(topk, 1);
  USM<int> slots;
  USM<float> weights, down, output;
  slots.reserve(routed);
  weights.reserve(routed);
  down.reserve(static_cast<size_t>(routed) * hidden);
  output.reserve(static_cast<size_t>(rows) * hidden);
  std::mt19937 rng(20261015);
  for (int j = 0; j < routed; ++j) {
    slots.data[j] = j % 7 == 0 ? -1 : (routed - j - 1) / 2;  // Invalid and duplicate expert slots.
    weights.data[j] = (int(rng() % 255) - 127) * .001f;
  }
  for (size_t j = 0; j < static_cast<size_t>(routed) * hidden; ++j)
    down.data[j] = (int(rng() % 65535) - 32767) * .0001f;
  auto dep = sycl_int4::queue().single_task([] {});
  auto call = [&](int width) {
#define MERGE(W) merge_vector<W>(slots.data, weights.data, down.data, output.data, rows, topk, hidden, dep)
    switch (width) {
      case 16:
        return MERGE(16);
      case 32:
        return MERGE(32);
      case 64:
        return MERGE(64);
      case 128:
        return MERGE(128);
      case 256:
        return MERGE(256);
      case 512:
        return MERGE(512);
      default:
        return merge(slots.data, weights.data, down.data, output.data, rows, topk, hidden, dep);
    }
#undef MERGE
  };
  call(0).wait_and_throw();
  std::vector<float> reference(output.data, output.data + rows * hidden);
  std::vector<int> widths{0, 16, 32, 64, 128, 256, 512};
  widths.erase(std::remove_if(widths.begin(), widths.end(), [=](int width) { return width > hidden; }), widths.end());
  for (int width : widths) {
    call(width).wait_and_throw();
    for (int j = 0; j < rows * hidden; ++j)
      if (std::bit_cast<uint32_t>(reference[j]) != std::bit_cast<uint32_t>(output.data[j]))
        throw std::runtime_error("Vector merge bitwise mismatch width=" + std::to_string(width));
  }
  std::vector<double> times[7];
  for (int s = -5; s < samples; ++s) {
    if (samples == 0) break;
    std::shuffle(widths.begin(), widths.end(), rng);
    for (int width : widths) {
      auto event = call(width);
      event.wait_and_throw();
      int index = width == 0     ? 0
                  : width == 16  ? 1
                  : width == 32  ? 2
                  : width == 64  ? 3
                  : width == 128 ? 4
                  : width == 256 ? 5
                                 : 6;
      if (s >= 0)
        times[index].push_back((event.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                event.get_profiling_info<sycl::info::event_profiling::command_start>()) /
                               1000.);
    }
  }
  for (int j = 0; j < 7 && samples > 0; ++j) {
    auto& t = times[j];
    std::sort(t.begin(), t.end());
    std::cout << "rows=" << rows << " topk=" << topk << " width=" << (j == 0 ? 0 : (8 << j))
              << " median_us=" << t[t.size() / 2] << '\n';
  }
  std::cout << "Vector merge exact passed rows=" << rows << " topk=" << topk << '\n';
}
int main(int argc, char** argv) {
  try {
    const int samples = argc > 1 ? std::stoi(argv[1]) : 0;
    if (samples < 0) throw std::runtime_error("samples must be nonnegative");
    check(1, 0, 128, 0);
    check(17, 2, 128, 0);
    check(512, 8, 2048, samples);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
