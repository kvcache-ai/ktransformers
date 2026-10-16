// SPDX-License-Identifier: Apache-2.0
#ifndef KT_SYCL_GPTQ_INT4_XMX_PROFILE_HPP
#define KT_SYCL_GPTQ_INT4_XMX_PROFILE_HPP

#include <array>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

#include "gptq_int4_sycl-runtime.hpp"

namespace sycl_xmx {
// Opt-in diagnostics only. Host and device clocks must never be subtracted
// from one another. Event queries are deferred until the whole chain finishes.
class PipelineProfile {
 public:
  using Clock = std::chrono::steady_clock;
  using Time = Clock::time_point;
  static double us(Time start, Time end) { return std::chrono::duration<double, std::micro>(end - start).count(); }
  enum Host { Route, InputCopy, HostQuant, Submit, Wait, OutputCopy, Executor, HostCount };
  enum Device { InputQuant, Gate, MiddleQuant, Down, Merge, DeviceCount };
  struct Frame {
    std::array<double, HostCount> host{};
    std::array<sycl::event, DeviceCount> events;
    std::array<bool, DeviceCount> present{};
    int routed = 0, tiles = 0;
  };
  explicit PipelineProfile(const char* value, int layer = 0) : layer_(layer) {
    const std::string mode = value ? value : "0";
    if (mode != "0" && mode != "host" && mode != "device")
      throw std::runtime_error("KT_SYCL_XMX_TRACE must be 0, host or device");
    enabled = mode != "0";
    device = mode == "device";
    const char* every = std::getenv("KT_SYCL_XMX_TRACE_EVERY");
    if (every) {
      const std::string count(every);
      if (count.empty() || count.find_first_not_of("0123456789") != std::string::npos || count.size() > 9)
        throw std::runtime_error("KT_SYCL_XMX_TRACE_EVERY must be an integer from 0 to 999999999");
      every_ = std::stoul(count);
    }
    if (device && !sycl_int4::queue().has_property<sycl::property::queue::enable_profiling>())
      throw std::runtime_error("KT_SYCL_XMX_TRACE=device requires KT_SYCL_PROFILE=1 before queue construction");
  }
  void record(int rows, const Frame& f) {
    static constexpr const char* host_names[] = {"route", "input_copy",  "host_quant", "submit",
                                                 "wait",  "output_copy", "executor"};
    static constexpr const char* device_names[] = {"input_quant", "gate", "middle_quant", "down", "merge"};
    std::map<std::string, double> sample;
    const std::string prefix = rows == 1 ? "decode." : "prefill.";
    sample["calls"] = 1;
    sample["rows"] = rows;
    sample["routed"] = f.routed;
    sample["tiles"] = f.tiles;
    for (int j = 0; j < HostCount; ++j) sample[std::string("host_") + host_names[j] + "_us"] = f.host[j];
    if (device) {
      uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0;
      double kernels = 0;
      for (int j = 0; j < DeviceCount; ++j) {
        double elapsed = 0;
        if (f.present[j]) {
          const auto begin = f.events[j].get_profiling_info<sycl::info::event_profiling::command_start>();
          const auto end = f.events[j].get_profiling_info<sycl::info::event_profiling::command_end>();
          elapsed = (end - begin) / 1000.0;
          first = std::min(first, begin);
          last = std::max(last, end);
          sample["device_commands"] += 1;
        }
        sample[std::string("device_") + device_names[j] + "_us"] = elapsed;
        kernels += elapsed;
      }
      const double span = last ? (last - first) / 1000.0 : 0;
      sample["device_kernels_us"] = kernels;
      sample["device_span_us"] = span;
      // The executor is a serial dependency chain, so there is no overlap.
      sample["device_gaps_us"] = std::max(0.0, span - kernels);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [key, value] : sample) totals_[prefix + key] += value;
  }
  void record_wrapper(int rows, double wrapper, double conversion) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string prefix = rows == 1 ? "decode." : "prefill.";
    totals_[prefix + "host_wrapper_us"] += wrapper;
    totals_[prefix + "host_bf16_us"] += conversion;
    if (every_ && static_cast<uint64_t>(totals_[prefix + "calls"]) % every_ == 0) {
      std::ostringstream report;
      report.imbue(std::locale::classic());
      report << "KT_SYCL_PIPELINE {\"layer\":" << layer_ << ",\"phase\":\"" << (rows == 1 ? "decode" : "prefill")
             << "\",\"totals\":{";
      bool first = true;
      for (const auto& [key, value] : totals_) {
        if (key.rfind(prefix, 0) != 0) continue;
        if (!first) report << ',';
        first = false;
        report << '\"' << key.substr(prefix.size()) << "\":" << std::setprecision(12) << value;
      }
      report << "}}\n";
      // Logging is deliberately opt-in and outside recorded host timings.
      std::fputs(report.str().c_str(), stderr);
    }
  }
  std::map<std::string, double> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totals_;
  }
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    totals_.clear();
  }
  bool enabled = false, device = false;

 private:
  int layer_ = 0;
  uint64_t every_ = 0;
  mutable std::mutex mutex_;
  std::map<std::string, double> totals_;
};
}  // namespace sycl_xmx
#endif
