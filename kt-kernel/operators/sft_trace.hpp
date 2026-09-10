// SPDX-License-Identifier: Apache-2.0
#ifndef CPUINFER_OPERATOR_SFT_TRACE_HPP
#define CPUINFER_OPERATOR_SFT_TRACE_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(KTRANSFORMERS_USE_CUDA) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define KT_SFT_HAS_NVTX 1
#else
#define KT_SFT_HAS_NVTX 0
#endif

namespace sft {

// Optional diagnostics, independent of numeric format and executor ownership.
// A range is thread-affine and cannot be copied or moved to another worker.
class TraceScope {
 public:
  TraceScope([[maybe_unused]] const char* name, [[maybe_unused]] std::int64_t layer,
             [[maybe_unused]] std::uint32_t category = 0) {
#if KT_SFT_HAS_NVTX
    if (!enabled()) return;
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.category = category;
    event.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    event.payload.llValue = layer;
    event.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    event.message.registered = nvtxDomainRegisterStringA(domain(), name);
    nvtxDomainRangePushEx(domain(), &event);
    active_ = true;
#endif
  }
  ~TraceScope() {
#if KT_SFT_HAS_NVTX
    if (active_) nvtxDomainRangePop(domain());
#endif
  }
  TraceScope(const TraceScope&) = delete;
  TraceScope& operator=(const TraceScope&) = delete;
  TraceScope(TraceScope&&) = delete;
  TraceScope& operator=(TraceScope&&) = delete;
  static constexpr bool supported() { return KT_SFT_HAS_NVTX; }

 private:
#if KT_SFT_HAS_NVTX
  static bool enabled() {
    static const bool value = [] {
      const char* setting = std::getenv("KT_SFT_TRACE");
      return setting && std::strcmp(setting, "1") == 0;
    }();
    return value;
  }
  static nvtxDomainHandle_t domain() {
    // Process-lifetime instrumentation domain: never destroy it ahead of a
    // late draining operator/future during interpreter or shared-library exit.
    static const auto handle = nvtxDomainCreateA("kt.sft");
    return handle;
  }
  bool active_ = false;
#endif
};

}  // namespace sft

#undef KT_SFT_HAS_NVTX
#endif
