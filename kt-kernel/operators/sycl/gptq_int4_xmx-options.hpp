// SPDX-License-Identifier: Apache-2.0
#ifndef KT_SYCL_GPTQ_INT4_XMX_OPTIONS_HPP
#define KT_SYCL_GPTQ_INT4_XMX_OPTIONS_HPP

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace sycl_xmx {
// Defaults are selected once when constructing an executor. Explicit settings
// remain available for diagnostics; selecting a preset never changes global env.
class Options {
 public:
  Options() {
    const char* value = std::getenv("KT_SYCL_XMX_PROFILE");
    if (value && std::string(value) != "panther-lake")
      throw std::runtime_error("KT_SYCL_XMX_PROFILE must be panther-lake or unset");
    panther_lake_ = value != nullptr;
  }
  bool panther_lake() const { return panther_lake_; }
  const char* get(const char* name) const {
    if (const char* value = std::getenv(name)) return value;
    if (!panther_lake_) return nullptr;
    static constexpr std::pair<const char*, const char*> defaults[] = {
        {"KT_SYCL_XMX_MODE", "int8"},
        {"KT_SYCL_XMX_DECODE_GATE_KERNEL", "xmx-split"},
        {"KT_SYCL_XMX_GATE_SPLITS", "2"},
        {"KT_SYCL_XMX_DOWN_SPLITS", "1"},
        {"KT_SYCL_XMX_PREFILL_ROWS", "8"},
        {"KT_SYCL_XMX_PREFILL_DOWN_ROWS", "16"},
        {"KT_SYCL_XMX_PREFILL_N", "32"},
        {"KT_SYCL_XMX_PREFILL_LOAD", "gather"},
        {"KT_SYCL_XMX_PREFILL_MERGE", "vector"},
        {"KT_SYCL_XMX_FUSED_EXPERT", "1"},
        {"KT_SYCL_XMX_FUSED_QUANT", "prefill"},
        {"KT_SYCL_XMX_NUMA_DISPATCH", "1"},
    };
    for (const auto& [key, value] : defaults)
      if (std::strcmp(key, name) == 0) return value;
    return nullptr;
  }

 private:
  bool panther_lake_ = false;
};
}  // namespace sycl_xmx
#endif
