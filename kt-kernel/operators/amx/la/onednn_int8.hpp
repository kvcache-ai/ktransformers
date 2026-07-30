#ifndef KTRANSFORMERS_ONEDNN_INT8_HPP
#define KTRANSFORMERS_ONEDNN_INT8_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(KTRANSFORMERS_USE_ONEDNN_VNNI)
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#endif

namespace amx {

enum class Int8VnniBackend {
  Native,
  OneDnn,
};

inline Int8VnniBackend int8_vnni_backend() {
  static const Int8VnniBackend selected = [] {
    const char* value = std::getenv("KT_INT8_VNNI_BACKEND");
    std::string backend = value == nullptr ? "auto" : value;
    std::transform(backend.begin(), backend.end(), backend.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (backend == "native") return Int8VnniBackend::Native;
    if (backend != "auto" && backend != "onednn") {
      throw std::runtime_error("KT_INT8_VNNI_BACKEND must be one of auto, onednn, or native");
    }

#if defined(HAVE_AMX)
    if (backend == "onednn") {
      throw std::runtime_error("KT_INT8_VNNI_BACKEND=onednn is only valid for non-AMX AVX512 builds");
    }
    return Int8VnniBackend::Native;
#elif defined(KTRANSFORMERS_USE_ONEDNN_VNNI)
    return Int8VnniBackend::OneDnn;
#else
    if (backend == "onednn") {
      throw std::runtime_error(
          "KT_INT8_VNNI_BACKEND=onednn was requested, but kt-kernel was built without oneDNN VNNI support");
    }
    return Int8VnniBackend::Native;
#endif
  }();
  return selected;
}

inline const char* int8_vnni_backend_name() {
  return int8_vnni_backend() == Int8VnniBackend::OneDnn ? "onednn-vnni" : "avx512-vnni";
}

#if defined(KTRANSFORMERS_USE_ONEDNN_VNNI)

class OneDnnInt8Brgemm {
 public:
  static void execute(int m, int batch_size, bool add_c, const int8_t* a, const int8_t* b, int32_t* c) {
    const Kernel* kernel = get_kernel(m, batch_size, add_c);
    thread_local std::vector<uint8_t> scratchpad;
    if (scratchpad.size() < kernel->scratchpad_size) scratchpad.resize(kernel->scratchpad_size);
    kernel->brgemm.execute(a, b, kernel->offsets, c, scratchpad.data());
  }

 private:
  using Offset = std::pair<dnnl::memory::dim, dnnl::memory::dim>;

  struct Kernel {
    Kernel(int m, int batch_size, bool add_c)
        : brgemm(m, kN, kK, batch_size, kLda, kLdb, kLdc, dnnl::memory::data_type::u8, dnnl::memory::data_type::s8,
                 dnnl::memory::data_type::s32, true) {
      if (!brgemm) throw std::runtime_error("oneDNN could not create the INT8 VNNI BRGEMM kernel");
      brgemm.set_add_C(add_c);
      if (!brgemm.finalize()) {
        throw std::runtime_error("oneDNN does not support the requested INT8 VNNI BRGEMM shape");
      }
      brgemm.generate();
      brgemm.set_hw_context();
      scratchpad_size = brgemm.get_scratchpad_size();
      offsets.reserve(batch_size);
      for (int index = 0; index < batch_size; ++index) {
        offsets.emplace_back(index * kABytesPerBatch, index * kBBytesPerBatch);
      }
    }

    dnnl::ukernel::brgemm brgemm;
    std::vector<Offset> offsets;
    size_t scratchpad_size = 0;
  };

  static const Kernel* get_kernel(int m, int batch_size, bool add_c) {
    if (m <= 0 || m > kMaxM || batch_size <= 0 || batch_size > kMaxBatch) {
      throw std::runtime_error("invalid oneDNN INT8 BRGEMM dimensions");
    }

    static std::once_flag initialized[kMaxM + 1][kMaxBatch + 1][2];
    static std::unique_ptr<const Kernel> kernels[kMaxM + 1][kMaxBatch + 1][2];
    const int add_index = add_c ? 1 : 0;
    std::call_once(initialized[m][batch_size][add_index],
                   [=] { kernels[m][batch_size][add_index] = std::make_unique<const Kernel>(m, batch_size, add_c); });
    return kernels[m][batch_size][add_index].get();
  }

  // KT stores each N=32 weight tile as two adjacent oneDNN pack32 N=16
  // panels. A and B advance by one complete M=32/N=32, K=64 KT tile for
  // every BRGEMM batch while C retains the KT row stride of 32.
  static constexpr int kN = 16;
  static constexpr int kK = 64;
  static constexpr int kLda = 64;
  static constexpr int kLdb = 16;
  static constexpr int kLdc = 32;
  static constexpr int kABytesPerBatch = 32 * 64;
  static constexpr int kBBytesPerBatch = 32 * 64;
  static constexpr int kMaxM = 32;
  static constexpr int kMaxBatch = 56;
};

#endif

}  // namespace amx

#endif
