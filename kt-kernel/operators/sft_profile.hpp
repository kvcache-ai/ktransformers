// Lightweight staged profiling for KT SFT MoE operators.
// SPDX-License-Identifier: Apache-2.0

#ifndef CPUINFER_OPERATOR_SFT_PROFILE_HPP
#define CPUINFER_OPERATOR_SFT_PROFILE_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

enum class SFTProfileStage : uint8_t {
  // NUMA-local forward stages.
  FwdTotal,
  FwdSetup,
  FwdRoute,
  FwdBufferSetup,
  FwdInputScatter,
  FwdInputPack,
  FwdGateUpBase,
  FwdGateUpLora,
  FwdCacheGateUp,
  FwdActivation,
  FwdCacheIntermediate,
  FwdDownPack,
  FwdDownBase,
  FwdDownLora,
  FwdCacheDown,
  FwdWeightedMerge,

  // NUMA-local backward stages.
  BwdTotal,
  BwdSetup,
  BwdCacheRestore,
  BwdDownTotal,
  BwdDownSetup,
  BwdDownScatter,
  BwdDownBaseDx,
  BwdDownLora,
  BwdActivation,
  BwdGateUpTotal,
  BwdGateUpSetup,
  BwdGateUpBaseDx,
  BwdGateUpLora,
  BwdRouterGrad,
  BwdBaseWeightGrad,

  // TP wrapper and weight-layout stages.
  TpFwdTotal,
  TpFwdNumaCompute,
  TpFwdMerge,
  TpBwdTotal,
  TpBwdBufferClear,
  TpBwdNumaCompute,
  TpBwdGradInputMerge,
  TpBwdLoraMerge,
  TpBwdRouterGradMerge,
  BackwardRepack,
  BaseWeightReload,

  Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(SFTProfileStage::Count)> kSFTProfileStageNames = {
    "forward.total",
    "forward.setup",
    "forward.route",
    "forward.buffer_setup",
    "forward.input_scatter",
    "forward.input_pack",
    "forward.gate_up_base",
    "forward.gate_up_lora",
    "forward.cache_gate_up",
    "forward.activation",
    "forward.cache_intermediate",
    "forward.down_pack",
    "forward.down_base",
    "forward.down_lora",
    "forward.cache_down",
    "forward.weighted_merge",
    "backward.total",
    "backward.setup",
    "backward.cache_restore",
    "backward.down.total",
    "backward.down.setup",
    "backward.down.scatter",
    "backward.down.base_dx",
    "backward.down.lora",
    "backward.activation",
    "backward.gate_up.total",
    "backward.gate_up.setup",
    "backward.gate_up.base_dx",
    "backward.gate_up.lora",
    "backward.router_grad",
    "backward.base_weight_grad",
    "tp.forward.total",
    "tp.forward.numa_compute",
    "tp.forward.merge",
    "tp.backward.total",
    "tp.backward.buffer_clear",
    "tp.backward.numa_compute",
    "tp.backward.grad_input_merge",
    "tp.backward.lora_merge",
    "tp.backward.router_grad_merge",
    "weights.backward_repack",
    "weights.base_reload",
};

inline bool sft_profile_enabled_from_env() {
  const char* value = std::getenv("KT_SFT_PROFILE");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
         std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0;
}

class SFTProfiler {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit SFTProfiler(bool enabled = sft_profile_enabled_from_env()) : enabled_(enabled) {
    reset();
  }

  bool enabled() const { return enabled_; }

  TimePoint start() const { return enabled_ ? Clock::now() : TimePoint{}; }

  void record(SFTProfileStage stage, TimePoint start) {
    if (!enabled_) return;
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    const size_t idx = static_cast<size_t>(stage);
    total_ns_[idx].fetch_add(static_cast<uint64_t>(elapsed), std::memory_order_relaxed);
    calls_[idx].fetch_add(1, std::memory_order_relaxed);
  }

  void record_workload(uint64_t tokens, uint64_t routed_rows, uint64_t active_experts) {
    if (!enabled_) return;
    tokens_.fetch_add(tokens, std::memory_order_relaxed);
    routed_rows_.fetch_add(routed_rows, std::memory_order_relaxed);
    active_experts_.fetch_add(active_experts, std::memory_order_relaxed);
    workloads_.fetch_add(1, std::memory_order_relaxed);
  }

  void append(std::map<std::string, double>& out, const std::string& prefix, bool reset_after = false) {
    out[prefix + "enabled"] = enabled_ ? 1.0 : 0.0;
    out[prefix + "workloads"] = static_cast<double>(load_or_exchange(workloads_, reset_after));
    out[prefix + "tokens"] = static_cast<double>(load_or_exchange(tokens_, reset_after));
    out[prefix + "routed_rows"] = static_cast<double>(load_or_exchange(routed_rows_, reset_after));
    out[prefix + "active_experts"] = static_cast<double>(load_or_exchange(active_experts_, reset_after));
    for (size_t i = 0; i < static_cast<size_t>(SFTProfileStage::Count); ++i) {
      const std::string stage_prefix = prefix + kSFTProfileStageNames[i] + ".";
      out[stage_prefix + "total_ns"] = static_cast<double>(load_or_exchange(total_ns_[i], reset_after));
      out[stage_prefix + "calls"] = static_cast<double>(load_or_exchange(calls_[i], reset_after));
    }
  }

  void reset() {
    for (auto& value : total_ns_) value.store(0, std::memory_order_relaxed);
    for (auto& value : calls_) value.store(0, std::memory_order_relaxed);
    workloads_.store(0, std::memory_order_relaxed);
    tokens_.store(0, std::memory_order_relaxed);
    routed_rows_.store(0, std::memory_order_relaxed);
    active_experts_.store(0, std::memory_order_relaxed);
  }

 private:
  static uint64_t load_or_exchange(std::atomic<uint64_t>& value, bool reset_after) {
    return reset_after ? value.exchange(0, std::memory_order_relaxed) : value.load(std::memory_order_relaxed);
  }

  bool enabled_;
  std::array<std::atomic<uint64_t>, static_cast<size_t>(SFTProfileStage::Count)> total_ns_{};
  std::array<std::atomic<uint64_t>, static_cast<size_t>(SFTProfileStage::Count)> calls_{};
  std::atomic<uint64_t> workloads_{0};
  std::atomic<uint64_t> tokens_{0};
  std::atomic<uint64_t> routed_rows_{0};
  std::atomic<uint64_t> active_experts_{0};
};

class SFTProfileScope {
 public:
  SFTProfileScope(SFTProfiler& profiler, SFTProfileStage stage)
      : profiler_(profiler), stage_(stage), start_(profiler.start()) {}

  ~SFTProfileScope() { profiler_.record(stage_, start_); }

  SFTProfileScope(const SFTProfileScope&) = delete;
  SFTProfileScope& operator=(const SFTProfileScope&) = delete;

 private:
  SFTProfiler& profiler_;
  SFTProfileStage stage_;
  SFTProfiler::TimePoint start_;
};

#endif  // CPUINFER_OPERATOR_SFT_PROFILE_HPP
