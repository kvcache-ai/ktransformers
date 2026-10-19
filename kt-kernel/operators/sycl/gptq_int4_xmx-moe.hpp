// SPDX-License-Identifier: Apache-2.0
#ifndef KT_SYCL_GPTQ_INT4_XMX_MOE_HPP
#define KT_SYCL_GPTQ_INT4_XMX_MOE_HPP

#include <cstdlib>
#include <limits>
#include <string>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include "../avx2/avx2_bf16_utils.hpp"
#include "../moe-tp.hpp"
#include "gptq_int4_vector-kernels.hpp"
#include "gptq_int4_xmx-fused-expert.hpp"
#include "gptq_int4_xmx-options.hpp"
#include "gptq_int4_xmx-prefill.hpp"
#include "gptq_int4_xmx-profile.hpp"

// Dedicated single-NUMA executor. No CPU expert-input pools or FP16 weight copy.
class SYCL_XMX_GPTQ_INT4_MOE_TP {
 public:
  using input_t = ggml_bf16_t;
  using output_t = float;
  GeneralMOEConfig config;
  explicit SYCL_XMX_GPTQ_INT4_MOE_TP(GeneralMOEConfig c, int part = 0)
      : config(c), profile_(std::getenv("KT_SYCL_XMX_TRACE"), c.layer_idx) {
    if (part != 0 || c.pool->config.subpool_count != 1)
      throw std::runtime_error("SYCL XMX currently requires one CPU threadpool/NUMA part");
    if (c.quant_config.bits != 4 || c.quant_config.zero_point)
      throw std::runtime_error("SYCL XMX requires symmetric GPTQ INT4 weights");
    int gs = c.quant_config.group_size;
    if (gs <= 0 || gs % 32 || c.hidden_size <= 0 || c.intermediate_size <= 0 || c.hidden_size % gs ||
        c.intermediate_size % gs || c.hidden_size % 16 || c.intermediate_size % 16 || c.expert_num <= 0 ||
        c.num_experts_per_tok <= 0 || c.max_possible_qlen() <= 0)
      throw std::runtime_error("SYCL XMX requires positive dimensions, N aligned to 16 and K/group_size aligned to 32");
    if (options_.panther_lake() && (c.hidden_size != 2048 || c.intermediate_size != 512 || gs != 128))
      throw std::runtime_error("KT_SYCL_XMX_PROFILE=panther-lake requires H=2048 I=512 group=128");
    auto device = sycl_int4::queue().get_device();
    if (!device.has(sycl::aspect::ext_intel_esimd) || !device.has(sycl::aspect::ext_intel_matrix))
      throw std::runtime_error("SYCL XMX requires an Intel GPU with ESIMD and XMX");
    const char* mode = options_.get("KT_SYCL_XMX_MODE");
    std::string value = mode ? mode : "fp16";
    if (value != "fp16" && value != "int8") throw std::runtime_error("KT_SYCL_XMX_MODE must be fp16 or int8");
    q8_ = value == "int8";
    gate_kernel_ = parse_kernel("KT_SYCL_XMX_GATE_KERNEL", Kernel::XMX);
    down_kernel_ = parse_kernel("KT_SYCL_XMX_DOWN_KERNEL", Kernel::XMX);
    decode_gate_ = parse_kernel("KT_SYCL_XMX_DECODE_GATE_KERNEL", gate_kernel_);
    decode_down_ = parse_kernel("KT_SYCL_XMX_DECODE_DOWN_KERNEL", down_kernel_);
    vector_splits_ = parse_splits("KT_SYCL_XMX_VECTOR_SPLITS", 4);
    gate_splits_ = parse_splits("KT_SYCL_XMX_GATE_SPLITS", vector_splits_);
    down_splits_ = parse_splits("KT_SYCL_XMX_DOWN_SPLITS", vector_splits_);
    const char* expert = options_.get("KT_SYCL_XMX_FUSED_EXPERT");
    const std::string expert_mode = expert ? expert : "0";
    if (expert_mode != "0" && expert_mode != "1") throw std::runtime_error("KT_SYCL_XMX_FUSED_EXPERT must be 0 or 1");
    fused_expert_ = expert_mode == "1";
    if (fused_expert_ && (!q8_ || c.hidden_size != 2048 || c.intermediate_size != 512 || gs != 128 ||
                          (decode_gate_ != Kernel::XMX && decode_gate_ != Kernel::XMXSplit) ||
                          decode_down_ != Kernel::XMX || (decode_gate_ == Kernel::XMXSplit && gate_splits_ > 2)))
      throw std::runtime_error(
          "KT_SYCL_XMX_FUSED_EXPERT requires INT8 XMX H=2048 I=512 group=128, unsplit down and gate splits<=2");
    const char* merge = options_.get("KT_SYCL_XMX_DECODE_MERGE");
    const std::string merge_mode = merge ? merge : "separate";
    if (merge_mode != "separate" && merge_mode != "fused")
      throw std::runtime_error("KT_SYCL_XMX_DECODE_MERGE must be separate or fused");
    fused_merge_ = merge_mode == "fused";
    if (fused_expert_ && fused_merge_)
      throw std::runtime_error("KT_SYCL_XMX_FUSED_EXPERT does not support fused decode merge");
    if (fused_merge_ && (c.num_experts_per_tok > 8 || decode_down_ != Kernel::XMX))
      throw std::runtime_error("KT_SYCL_XMX_DECODE_MERGE=fused requires topk<=8 and unsplit XMX decode down");
    const char* tile_rows = options_.get("KT_SYCL_XMX_ROWS");
    std::string tile = tile_rows ? tile_rows : "8";
    if (tile != "1" && tile != "2" && tile != "4" && tile != "8")
      throw std::runtime_error("KT_SYCL_XMX_ROWS must be 1, 2, 4 or 8");
    const char* prefill_rows = options_.get("KT_SYCL_XMX_PREFILL_ROWS");
    const std::string prefill_tile = prefill_rows ? std::string(prefill_rows) : tile;
    if (prefill_tile != "1" && prefill_tile != "2" && prefill_tile != "4" && prefill_tile != "8" &&
        prefill_tile != "16" && prefill_tile != "32")
      throw std::runtime_error("KT_SYCL_XMX_PREFILL_ROWS must be 1, 2, 4, 8, 16 or 32");
    prefill_rows_ = std::stoi(prefill_tile);
    if (prefill_rows_ > 8 && (!q8_ || gate_kernel_ != Kernel::XMX || down_kernel_ != Kernel::XMX))
      throw std::runtime_error("KT_SYCL_XMX_PREFILL_ROWS > 8 requires INT8 unsplit XMX prefill projections");
    const char* down_rows = options_.get("KT_SYCL_XMX_PREFILL_DOWN_ROWS");
    const std::string down_tile = down_rows ? down_rows : prefill_tile;
    if (down_tile != "1" && down_tile != "2" && down_tile != "4" && down_tile != "8" && down_tile != "16" &&
        down_tile != "32")
      throw std::runtime_error("KT_SYCL_XMX_PREFILL_DOWN_ROWS must be 1, 2, 4, 8, 16 or 32");
    prefill_down_rows_ = std::stoi(down_tile);
    if (prefill_down_rows_ != prefill_rows_ && (!q8_ || gate_kernel_ != Kernel::XMX || down_kernel_ != Kernel::XMX))
      throw std::runtime_error("Independent prefill down rows require INT8 unsplit XMX projections");
    const char* prefill_n = options_.get("KT_SYCL_XMX_PREFILL_N");
    const std::string prefill_columns = prefill_n ? prefill_n : "16";
    if (prefill_columns != "16" && prefill_columns != "32")
      throw std::runtime_error("KT_SYCL_XMX_PREFILL_N must be 16 or 32");
    prefill_n_ = std::stoi(prefill_columns);
    if (prefill_n_ == 32 && (!q8_ || gate_kernel_ != Kernel::XMX || down_kernel_ != Kernel::XMX || c.hidden_size % 32 ||
                             c.intermediate_size % 32 || prefill_rows_ > 16 || prefill_down_rows_ > 16))
      throw std::runtime_error("KT_SYCL_XMX_PREFILL_N=32 requires INT8 XMX prefill, N%32=0 and rows<=16");
    const char* load = options_.get("KT_SYCL_XMX_PREFILL_LOAD");
    const std::string load_mode = load ? load : "block";
    if (load_mode != "block" && load_mode != "gather")
      throw std::runtime_error("KT_SYCL_XMX_PREFILL_LOAD must be block or gather");
    if (load_mode != "block" && prefill_n_ != 32)
      throw std::runtime_error("KT_SYCL_XMX_PREFILL_LOAD=gather requires PREFILL_N=32");
    prefill_load_ = load_mode == "gather" ? sycl_xmx::PrefillLoad::Gather : sycl_xmx::PrefillLoad::Block;
    const char* merge_prefill = options_.get("KT_SYCL_XMX_PREFILL_MERGE");
    const std::string prefill_merge = merge_prefill ? merge_prefill : "scalar";
    if (prefill_merge != "scalar" && prefill_merge != "vector")
      throw std::runtime_error("KT_SYCL_XMX_PREFILL_MERGE must be scalar or vector");
    vector_prefill_merge_ = prefill_merge == "vector";
    const char* graph = options_.get("KT_SYCL_XMX_GRAPH");
    if (graph && std::string(graph) != "0" && std::string(graph) != "1")
      throw std::runtime_error("KT_SYCL_XMX_GRAPH must be 0 or 1");
    use_graph_ = graph && std::string(graph) == "1";
    const char* hq = options_.get("KT_SYCL_XMX_HOST_QUANT");
    if (hq && std::string(hq) != "0" && std::string(hq) != "1")
      throw std::runtime_error("KT_SYCL_XMX_HOST_QUANT must be 0 or 1");
    host_quant_ = hq && std::string(hq) == "1";
    const char* fq = options_.get("KT_SYCL_XMX_FUSED_QUANT");
    if (fq && std::string(fq) != "0" && std::string(fq) != "1" && std::string(fq) != "prefill")
      throw std::runtime_error("KT_SYCL_XMX_FUSED_QUANT must be 0, 1 or prefill");
    fused_quant_ = fq && std::string(fq) != "0";
    fused_decode_ = fq && std::string(fq) == "1";
    if (fused_quant_ &&
        (!q8_ || gs != 128 || c.intermediate_size % 128 || gate_kernel_ != Kernel::XMX || !uses_q8(down_kernel_) ||
         (fused_decode_ && (decode_gate_ != Kernel::XMX || !uses_q8(decode_down_)))))
      throw std::runtime_error("SYCL XMX fused quantization requires int8 XMX gate, quantized down and group_size=128");
    if (use_graph_ && profile_.device)
      throw std::runtime_error(
          "KT_SYCL_XMX_TRACE=device does not profile recorded graph nodes; use TRACE=host or GRAPH=0");
    if (use_graph_ && !device.has(sycl::aspect::ext_oneapi_limited_graph))
      throw std::runtime_error("SYCL XMX command graph is unavailable on this device");
    const int max_rows = c.max_possible_qlen();
    if (max_rows > std::numeric_limits<int>::max() / c.num_experts_per_tok ||
        max_rows > std::numeric_limits<int>::max() / c.hidden_size)
      throw std::runtime_error("SYCL XMX workspace dimensions exceed 32-bit indexing");
    const int max_routed = max_rows * c.num_experts_per_tok;
    if (max_routed > std::numeric_limits<int>::max() / c.hidden_size ||
        max_routed > std::numeric_limits<int>::max() / c.intermediate_size)
      throw std::runtime_error("SYCL XMX routed workspace exceeds 32-bit indexing");
    if (use_graph_) graphs_.resize(static_cast<size_t>(c.num_experts_per_tok + 1) * (c.num_experts_per_tok + 1));
    counts_.resize(c.expert_num);
    offsets_.resize(c.expert_num + 1);
    cursors_.resize(c.expert_num);
    input_.reserve(static_cast<size_t>(max_rows) * c.hidden_size);
    activated_.reserve(static_cast<size_t>(max_routed) * c.intermediate_size);
    down_.reserve(static_cast<size_t>(max_routed) * c.hidden_size);
    output_.reserve(static_cast<size_t>(max_rows) * c.hidden_size);
    row_tokens_.reserve(max_routed);
    slot_rows_.reserve(max_routed);
    routing_.reserve(max_routed);
    tiles_.reserve(max_routed);
    down_tiles_.reserve(max_routed);
    if (uses_q8(gate_kernel_) || uses_q8(decode_gate_)) {
      qi_.reserve(static_cast<size_t>(max_rows) * c.hidden_size);
      si_.reserve(static_cast<size_t>(max_rows) * c.hidden_size / gs);
    }
    if (uses_q8(down_kernel_) || uses_q8(decode_down_)) {
      qm_.reserve(static_cast<size_t>(max_routed) * c.intermediate_size);
      sm_.reserve(static_cast<size_t>(max_routed) * c.intermediate_size / gs);
    }
    if (options_.panther_lake()) printf("SYCL XMX profile=panther-lake\n");
    printf("Created SYCL_XMX_GPTQ_INT4_MOE mode=%s layer=%d\n", value.c_str(), c.layer_idx);
    printf("SYCL decode fused_expert=%d merge=%s\n", fused_expert_, merge_mode.c_str());
    printf("SYCL prefill tile gate=M%dN%d down=M%dN%d\n", prefill_rows_, prefill_n_, prefill_down_rows_, prefill_n_);
    printf("SYCL prefill load=%s merge=%s\n", load_mode.c_str(), prefill_merge.c_str());
    printf("SYCL projection kernels prefill=%s/%s decode=%s/%s splits=%d/%d\n", kernel_name(gate_kernel_),
           kernel_name(down_kernel_), kernel_name(decode_gate_), kernel_name(decode_down_), gate_splits_, down_splits_);
  }

  void load_weights(const GeneralMOEConfig& source) {
    const int e = config.expert_num, h = config.hidden_size, i = config.intermediate_size;
    const int gs = config.quant_config.group_size;
    gate_.init(e, i, h, gs);
    up_.init(e, i, h, gs);
    down_weight_.init(e, h, i, gs);
    const auto* map = static_cast<const uint64_t*>(source.physical_to_logical_map);
    loaded_.assign(e, false);
    for (int physical = 0; physical < e; ++physical) {
      uint64_t mapped = map ? map[physical] : static_cast<uint64_t>(physical);
      if (mapped >= static_cast<uint64_t>(e)) throw std::runtime_error("Invalid XMX physical_to_logical_map");
      int logical = static_cast<int>(mapped);
      if (source.should_skip_expert(logical)) continue;
      auto load = [&](sycl_xmx::Weight& w, void* p, void* s, const auto& ps, const auto& ss) {
        const uint32_t* qw =
            ps.empty() ? static_cast<const uint32_t*>(p) : static_cast<const uint32_t*>(ps.at(0).at(logical));
        const float* sc = ss.empty() ? static_cast<const float*>(s) : static_cast<const float*>(ss.at(0).at(logical));
        if (ps.empty() && qw) qw += static_cast<size_t>(logical) * w.n * w.k / 8;
        if (ss.empty() && sc) sc += static_cast<size_t>(logical) * w.n * (w.k / w.gs);
        w.from_mat(physical, qw, sc);
      };
      load(gate_, source.gate_proj, source.gate_scale, source.gate_projs, source.gate_scales);
      load(up_, source.up_proj, source.up_scale, source.up_projs, source.up_scales);
      load(down_weight_, source.down_proj, source.down_scale, source.down_projs, source.down_scales);
      loaded_[physical] = true;
    }
  }

  void forward(int rows, int topk, const int64_t* ids, const float* weights, const void* input, void* output) {
    if (rows < 0 || rows > config.max_possible_qlen() || topk < 0 || topk > config.num_experts_per_tok)
      throw std::runtime_error("XMX forward exceeds configured workspace");
    if (rows == 0) return;
    if (loaded_.empty()) throw std::runtime_error("XMX weights not loaded");
    const int h = config.hidden_size, i = config.intermediate_size;
    using P = sycl_xmx::PipelineProfile;
    const auto begin = profile_.enabled ? P::Clock::now() : P::Time{};
    auto checkpoint = begin;
    if (profile_.enabled) frame_ = {};
    std::fill(counts_.begin(), counts_.end(), 0);
    std::fill(slot_rows_.data, slot_rows_.data + rows * topk, -1);
    for (int slot = 0; slot < rows * topk; ++slot) {
      int64_t expert = ids[slot];
      if (expert >= 0 && expert < config.expert_num && loaded_[expert]) ++counts_[expert];
    }
    offsets_[0] = 0;
    for (int e = 0; e < config.expert_num; ++e) {
      offsets_[e + 1] = offsets_[e] + counts_[e];
      cursors_[e] = offsets_[e];
    }
    for (int slot = 0; slot < rows * topk; ++slot) {
      int64_t expert = ids[slot];
      if (expert < 0 || expert >= config.expert_num || !loaded_[expert]) continue;
      const int routed = cursors_[expert]++;
      row_tokens_.data[routed] = slot / topk;
      slot_rows_.data[slot] = routed;
    }
    int routed = offsets_.back(), nt = 0, tile_rows = rows == 1 ? 1 : prefill_rows_;
    for (int e = 0; e < config.expert_num; ++e) {
      for (int r = 0; r < counts_[e]; r += tile_rows)
        tiles_.data[nt++] = {e, offsets_[e] + r, std::min(tile_rows, counts_[e] - r)};
    }
    down_tile_count_ = nt;
    if (rows > 1 && prefill_down_rows_ != prefill_rows_) {
      down_tile_count_ = 0;
      for (int e = 0; e < config.expert_num; ++e)
        for (int r = 0; r < counts_[e]; r += prefill_down_rows_)
          down_tiles_.data[down_tile_count_++] = {e, offsets_[e] + r, std::min(prefill_down_rows_, counts_[e] - r)};
    }
    if (profile_.enabled) {
      const auto now = P::Clock::now();
      frame_.host[P::Route] = P::us(checkpoint, now);
      frame_.routed = routed;
      frame_.tiles = nt;
      checkpoint = now;
    }
    if (nt == 0) {
      std::fill_n(static_cast<float*>(output), static_cast<size_t>(rows) * h, 0.0f);
      if (profile_.enabled) {
        frame_.host[P::Executor] = P::us(begin, P::Clock::now());
        profile_.record(rows, frame_);
      }
      return;
    }
    std::memcpy(input_.data, input, static_cast<size_t>(rows) * h * 2);
    std::memcpy(routing_.data, weights, static_cast<size_t>(rows) * topk * sizeof(float));
    if (profile_.enabled) {
      const auto now = P::Clock::now();
      frame_.host[P::InputCopy] = P::us(checkpoint, now);
      checkpoint = now;
    }
    if (uses_q8(decode_gate_) && host_quant_ && rows == 1) {
      const int gs = config.quant_config.group_size;
      for (int g = 0; g < h / gs; ++g) {
        const auto* src = static_cast<const ggml_bf16_t*>(input) + g * gs;
        float maximum = 0;
        for (int j = 0; j < gs; j += 8) {
          __m256 v = avx2::load_bf16_to_fp32(src + j);
          v = _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
          alignas(32) float values[8];
          _mm256_store_ps(values, v);
          for (float value : values) maximum = std::max(maximum, value);
        }
        const float scale = maximum / 127.0f;
        si_.data[g] = scale;
        const __m256 inv = _mm256_set1_ps(scale > 0 ? 1.0f / scale : 0.0f);
        for (int j = 0; j < gs; j += 8) {
          __m256 v = _mm256_mul_ps(avx2::load_bf16_to_fp32(src + j), inv);
          __m256i iv = _mm256_cvtps_epi32(_mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
          alignas(32) int values[8];
          _mm256_store_si256(reinterpret_cast<__m256i*>(values), iv);
          for (int r = 0; r < 8; ++r) qi_.data[g * gs + j + r] = static_cast<int8_t>(std::clamp(values[r], -127, 127));
        }
      }
    }
    if (profile_.enabled) frame_.host[P::HostQuant] = P::us(checkpoint, P::Clock::now());
    try {
      switch (tile_rows) {
        case 1:
          run<1>(nt, rows, topk, routed);
          break;
        case 2:
          run<2>(nt, rows, topk, routed);
          break;
        case 4:
          run<4>(nt, rows, topk, routed);
          break;
        case 16:
          run<16>(nt, rows, topk, routed);
          break;
        case 32:
          run<32>(nt, rows, topk, routed);
          break;
        default:
          run<8>(nt, rows, topk, routed);
          break;
      }
    } catch (...) {
      // A failed submission may leave earlier stages in flight. Drain before
      // the caller reuses the shared workspace or destroys this executor.
      try {
        sycl_int4::queue().wait_and_throw();
      } catch (...) {
      }
      throw;
    }
    if (profile_.enabled) checkpoint = P::Clock::now();
    std::memcpy(output, output_.data, static_cast<size_t>(rows) * h * sizeof(float));
    if (profile_.enabled) {
      const auto now = P::Clock::now();
      frame_.host[P::OutputCopy] = P::us(checkpoint, now);
      frame_.host[P::Executor] = P::us(begin, now);
      profile_.record(rows, frame_);
    }
  }

  sycl_xmx::PipelineProfile& pipeline_profile() { return profile_; }

 private:
  sycl_xmx::Options options_;
  sycl_xmx::PipelineProfile profile_;
  sycl_xmx::PipelineProfile::Frame frame_;
  void remember(int stage, const sycl::event& event) {
    if (profile_.device) {
      frame_.events[stage] = event;
      frame_.present[stage] = true;
    }
  }
  enum class Kernel { XMX, XMXSplit, Vector, DP4A };
  Kernel parse_kernel(const char* name, Kernel fallback) {
    const char* value = options_.get(name);
    if (!value || std::string(value) == "inherit") return fallback;
    if (std::string(value) == "xmx") return Kernel::XMX;
    if (std::string(value) == "xmx-split") return Kernel::XMXSplit;
    if (std::string(value) == "vector") return Kernel::Vector;
    if (std::string(value) == "dp4a") return Kernel::DP4A;
    throw std::runtime_error(std::string(name) + " must be xmx, xmx-split, vector, dp4a or inherit");
  }
  static const char* kernel_name(Kernel k) {
    return k == Kernel::XMX ? "xmx" : k == Kernel::XMXSplit ? "xmx-split" : k == Kernel::Vector ? "vector" : "dp4a";
  }
  int parse_splits(const char* name, int fallback) {
    const char* value = options_.get(name);
    if (!value) return fallback;
    std::string s(value);
    if (s == "1" || s == "2" || s == "4" || s == "8") return std::stoi(s);
    throw std::runtime_error(std::string(name) + " must be 1, 2, 4 or 8");
  }
  bool uses_q8(Kernel k) const { return k == Kernel::DP4A || ((k == Kernel::XMX || k == Kernel::XMXSplit) && q8_); }
  bool q8_ = false;
  Kernel gate_kernel_ = Kernel::XMX, down_kernel_ = Kernel::XMX;
  Kernel decode_gate_ = Kernel::XMX, decode_down_ = Kernel::XMX;
  int vector_splits_ = 4, gate_splits_ = 4, down_splits_ = 4;
  int prefill_rows_ = 8, prefill_down_rows_ = 8, prefill_n_ = 16, down_tile_count_ = 0;
  sycl_xmx::PrefillLoad prefill_load_ = sycl_xmx::PrefillLoad::Block;
  bool vector_prefill_merge_ = false;
  bool use_graph_ = false;
  bool fused_merge_ = false;
  bool fused_expert_ = false;
  bool host_quant_ = false;
  bool fused_quant_ = false;
  bool fused_decode_ = false;
  using Graph =
      sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
  std::vector<bool> loaded_;
  std::vector<int> counts_, offsets_, cursors_;
  sycl_xmx::Weight gate_, up_, down_weight_;
  sycl_xmx::USM<uint16_t> input_, activated_;
  sycl_xmx::USM<float> down_, output_, routing_, si_, sm_;
  sycl_xmx::USM<int8_t> qi_, qm_;
  sycl_xmx::USM<int> row_tokens_, slot_rows_;
  sycl_xmx::USM<sycl_xmx::Tile> tiles_, down_tiles_;
  // Executable graphs must be destroyed before the USM storage they reference.
  std::vector<std::unique_ptr<Graph>> graphs_;

  template <int Rows, bool GateUp>
  sycl::event submit_projection(Kernel kernel, const sycl::event* dep, int nt, bool fused = false, bool prefill = false,
                                const sycl_xmx::Tile* selected_tiles = nullptr) {
    const auto* tiles = selected_tiles ? selected_tiles : tiles_.data;
    const auto& first = GateUp ? gate_ : down_weight_;
    const auto* second = GateUp ? &up_ : nullptr;
    const auto* input = GateUp ? input_.data : activated_.data;
    const auto* quantized = GateUp ? qi_.data : qm_.data;
    const auto* scales = GateUp ? si_.data : sm_.data;
    const auto* tokens = GateUp ? row_tokens_.data : nullptr;
    auto* activated = GateUp ? activated_.data : nullptr;
    auto* down = GateUp ? nullptr : down_.data;
    const int splits = GateUp ? gate_splits_ : down_splits_;
    if constexpr (Rows <= 16) {
      if (prefill && prefill_n_ == 32) {
        auto n32 = [&]<sycl_xmx::PrefillLoad Load>() {
          if constexpr (GateUp) {
            if (fused)
              return sycl_xmx::prefill_n32<Rows, true, true, Load>(tiles, nt, tokens, quantized, scales, first, second,
                                                                   nullptr, nullptr, config.swiglu_limit,
                                                                   config.swiglu_alpha, dep, qm_.data, sm_.data);
          }
          return sycl_xmx::prefill_n32<Rows, GateUp, false, Load>(tiles, nt, tokens, quantized, scales, first, second,
                                                                  activated, down, config.swiglu_limit,
                                                                  config.swiglu_alpha, dep);
        };
        if (prefill_load_ != sycl_xmx::PrefillLoad::Block)
          return n32.template operator()<sycl_xmx::PrefillLoad::Gather>();
        return n32.template operator()<sycl_xmx::PrefillLoad::Block>();
      }
    }
    if constexpr (Rows > 8) {
      if constexpr (GateUp) {
        if (fused)
          return sycl_xmx::projection<Rows, true, true, true>(tiles, nt, tokens, input, quantized, scales, first,
                                                              second, nullptr, nullptr, config.swiglu_limit,
                                                              config.swiglu_alpha, dep, qm_.data, sm_.data);
      }
      return sycl_xmx::projection<Rows, true, GateUp>(tiles, nt, tokens, input, quantized, scales, first, second,
                                                      activated, down, config.swiglu_limit, config.swiglu_alpha, dep);
    } else {
      if (kernel == Kernel::Vector)
        return sycl_xmx::vector_dispatch<false, GateUp>(splits, tiles, nt, Rows, tokens, input, quantized, scales,
                                                        first, second, activated, down, config.swiglu_limit,
                                                        config.swiglu_alpha, dep);
      if (kernel == Kernel::DP4A)
        return sycl_xmx::vector_dispatch<true, GateUp>(splits, tiles, nt, Rows, tokens, input, quantized, scales, first,
                                                       second, activated, down, config.swiglu_limit,
                                                       config.swiglu_alpha, dep);
      if (kernel == Kernel::XMXSplit) {
#define KT_XMX_SPLIT(S)                                                                                          \
  (q8_ ? sycl_xmx::projection<Rows, true, GateUp, false, S>(tiles, nt, tokens, input, quantized, scales, first,  \
                                                            second, activated, down, config.swiglu_limit,        \
                                                            config.swiglu_alpha, dep)                            \
       : sycl_xmx::projection<Rows, false, GateUp, false, S>(tiles, nt, tokens, input, quantized, scales, first, \
                                                             second, activated, down, config.swiglu_limit,       \
                                                             config.swiglu_alpha, dep))
        switch (splits) {
          case 1:
            return KT_XMX_SPLIT(1);
          case 2:
            return KT_XMX_SPLIT(2);
          case 4:
            return KT_XMX_SPLIT(4);
          default:
            return KT_XMX_SPLIT(8);
        }
#undef KT_XMX_SPLIT
      }
      if constexpr (GateUp) {
        if (fused)
          return sycl_xmx::projection<Rows, true, true, true>(tiles, nt, tokens, input, quantized, scales, first,
                                                              second, nullptr, nullptr, config.swiglu_limit,
                                                              config.swiglu_alpha, dep, qm_.data, sm_.data);
      }
      return q8_ ? sycl_xmx::projection<Rows, true, GateUp>(tiles, nt, tokens, input, quantized, scales, first, second,
                                                            activated, down, config.swiglu_limit, config.swiglu_alpha,
                                                            dep)
                 : sycl_xmx::projection<Rows, false, GateUp>(tiles, nt, tokens, input, quantized, scales, first, second,
                                                             activated, down, config.swiglu_limit, config.swiglu_alpha,
                                                             dep);
    }
  }
  sycl::event submit_prefill_down(Kernel kernel, const sycl::event* dep) {
    const auto* tiles = prefill_down_rows_ == prefill_rows_ ? tiles_.data : down_tiles_.data;
#define KT_PREFILL_DOWN(R) submit_projection<R, false>(kernel, dep, down_tile_count_, false, true, tiles)
    switch (prefill_down_rows_) {
      case 1:
        return KT_PREFILL_DOWN(1);
      case 2:
        return KT_PREFILL_DOWN(2);
      case 4:
        return KT_PREFILL_DOWN(4);
      case 8:
        return KT_PREFILL_DOWN(8);
      case 16:
        return KT_PREFILL_DOWN(16);
      default:
        return KT_PREFILL_DOWN(32);
    }
#undef KT_PREFILL_DOWN
  }
  template <int Rows>
  sycl::event submit(int nt, int rows, int topk, int routed) {
    const Kernel gate = rows == 1 ? decode_gate_ : gate_kernel_;
    const Kernel down = rows == 1 ? decode_down_ : down_kernel_;
    const bool device_quant = uses_q8(gate) && !(host_quant_ && rows == 1);
    const bool use_fused_quant = fused_quant_ && (rows > 1 || fused_decode_);
    sycl::event qi;
    if (device_quant)
      qi =
          sycl_xmx::quantize(input_.data, qi_.data, si_.data, rows, config.hidden_size, config.quant_config.group_size);
    if (device_quant) remember(sycl_xmx::PipelineProfile::InputQuant, qi);
    if constexpr (Rows == 1) {
      if (rows == 1 && fused_expert_) {
        sycl::event experts;
#define KT_FUSED_EXPERT(S)                                                                                    \
  sycl_xmx::fused_expert_decode<S>(tiles_.data, nt, qi_.data, si_.data, gate_, up_, down_weight_, down_.data, \
                                   config.swiglu_limit, config.swiglu_alpha, device_quant ? &qi : nullptr)
        if (gate == Kernel::XMX || gate_splits_ == 1)
          experts = KT_FUSED_EXPERT(1);
        else
          experts = KT_FUSED_EXPERT(2);
#undef KT_FUSED_EXPERT
        remember(sycl_xmx::PipelineProfile::Gate, experts);  // includes quantization and down
        auto result = sycl_xmx::merge(slot_rows_.data, routing_.data, down_.data, output_.data, rows, topk,
                                      config.hidden_size, experts);
        remember(sycl_xmx::PipelineProfile::Merge, result);
        return result;
      }
    }
    auto gu = submit_projection<Rows, true>(gate, device_quant ? &qi : nullptr, nt, use_fused_quant, rows > 1);
    remember(sycl_xmx::PipelineProfile::Gate, gu);
    sycl::event middle = gu;
    if (uses_q8(down) && !use_fused_quant) {
      middle = sycl_xmx::quantize(activated_.data, qm_.data, sm_.data, routed, config.intermediate_size,
                                  config.quant_config.group_size, &gu);
      remember(sycl_xmx::PipelineProfile::MiddleQuant, middle);
    }
    if constexpr (Rows == 1) {
      if (rows == 1 && fused_merge_) {
        auto result =
            q8_ ? sycl_xmx::projection<1, true, false, false, 1, true>(
                      tiles_.data, nt, nullptr, activated_.data, qm_.data, sm_.data, down_weight_, nullptr, nullptr,
                      nullptr, 0, 0, &middle, nullptr, nullptr, slot_rows_.data, routing_.data, output_.data, topk)
                : sycl_xmx::projection<1, false, false, false, 1, true>(
                      tiles_.data, nt, nullptr, activated_.data, qm_.data, sm_.data, down_weight_, nullptr, nullptr,
                      nullptr, 0, 0, &middle, nullptr, nullptr, slot_rows_.data, routing_.data, output_.data, topk);
        // Down includes expert merge; no separate Merge event is recorded.
        remember(sycl_xmx::PipelineProfile::Down, result);
        return result;
      }
    }
    auto d = rows > 1 ? submit_prefill_down(down, &middle) : submit_projection<Rows, false>(down, &middle, nt);
    remember(sycl_xmx::PipelineProfile::Down, d);
    sycl::event result;
    if (rows > 1 && vector_prefill_merge_ && config.hidden_size % 256 == 0)
      result = sycl_xmx::merge_vector<256>(slot_rows_.data, routing_.data, down_.data, output_.data, rows, topk,
                                           config.hidden_size, d);
    else if (rows > 1 && vector_prefill_merge_ && config.hidden_size % 128 == 0)
      result = sycl_xmx::merge_vector<128>(slot_rows_.data, routing_.data, down_.data, output_.data, rows, topk,
                                           config.hidden_size, d);
    else
      result =
          sycl_xmx::merge(slot_rows_.data, routing_.data, down_.data, output_.data, rows, topk, config.hidden_size, d);
    remember(sycl_xmx::PipelineProfile::Merge, result);
    return result;
  }
  template <int Rows>
  void run(int nt, int rows, int topk, int routed) {
    if constexpr (Rows == 1) {
      if (use_graph_ && rows == 1) {
        namespace gx = sycl::ext::oneapi::experimental;
        auto& cached = graphs_[topk * (config.num_experts_per_tok + 1) + nt];
        auto& q = sycl_int4::queue();
        if (!cached) {
          gx::command_graph graph(q, {gx::property::graph::assume_buffer_outlives_graph{}});
          graph.begin_recording(q);
          try {
            (void)submit<Rows>(nt, rows, topk, routed);
          } catch (...) {
            graph.end_recording();
            throw;
          }
          graph.end_recording();
          cached = std::make_unique<Graph>(graph.finalize());
        }
        using P = sycl_xmx::PipelineProfile;
        const auto begin = profile_.enabled ? P::Clock::now() : P::Time{};
        auto result = q.ext_oneapi_graph(*cached);
        const auto submitted = profile_.enabled ? P::Clock::now() : P::Time{};
        sycl_int4::wait_and_throw(result);
        if (profile_.enabled) {
          frame_.host[P::Submit] = P::us(begin, submitted);
          frame_.host[P::Wait] = P::us(submitted, P::Clock::now());
        }
        return;
      }
    }
    using P = sycl_xmx::PipelineProfile;
    const auto begin = profile_.enabled ? P::Clock::now() : P::Time{};
    auto result = submit<Rows>(nt, rows, topk, routed);
    const auto submitted = profile_.enabled ? P::Clock::now() : P::Time{};
    sycl_int4::wait_and_throw(result);
    if (profile_.enabled) {
      frame_.host[P::Submit] = P::us(begin, submitted);
      frame_.host[P::Wait] = P::us(submitted, P::Clock::now());
    }
  }
};

template <>
class TP_MOE<SYCL_XMX_GPTQ_INT4_MOE_TP> : public TP_MOE_Common<SYCL_XMX_GPTQ_INT4_MOE_TP> {
 public:
  using Base = TP_MOE_Common<SYCL_XMX_GPTQ_INT4_MOE_TP>;
  explicit TP_MOE(const GeneralMOEConfig& config) : Base(config) {
    const sycl_xmx::Options options;
    const char* numa = options.get("KT_SYCL_XMX_NUMA_DISPATCH");
    if (numa && std::string(numa) != "0" && std::string(numa) != "1")
      throw std::runtime_error("KT_SYCL_XMX_NUMA_DISPATCH must be 0 or 1");
    use_numa_dispatch_ = numa && std::string(numa) == "1";
  }
  static int implementation_revision() { return 7; }
  void load_weights() override {
    this->weights_loaded = false;
    this->tps[0]->load_weights(this->config);
    this->weights_loaded = true;
  }
  void forward(int rows, int topk, const int64_t* ids, const float* weights, const void* input, void* output,
               bool incremental = false) override {
    if (!this->weights_loaded) throw std::runtime_error("XMX weights not loaded");
    using P = sycl_xmx::PipelineProfile;
    auto& profile = this->tps[0]->pipeline_profile();
    const auto begin = profile.enabled ? P::Clock::now() : P::Time{};
    if (use_numa_dispatch_) {
      // Worker exceptions must be captured here: the generic NUMA distributor
      // does not propagate exceptions from its worker threads.
      std::exception_ptr failure;
      this->config.pool->dispense_backend()->do_numa_job([&](int part) {
        try {
          this->tps[part]->forward(rows, topk, ids, weights, input, this->local_output_numa[part]);
        } catch (...) {
          failure = std::current_exception();
        }
      });
      if (failure) std::rethrow_exception(failure);
    } else {
      this->tps[0]->forward(rows, topk, ids, weights, input, this->local_output_numa[0]);
    }
    const auto converted = profile.enabled ? P::Clock::now() : P::Time{};
    merge_results(rows, output, incremental);
    if (profile.enabled && rows > 0) {
      const auto end = P::Clock::now();
      profile.record_wrapper(rows, P::us(begin, end), P::us(converted, end));
    }
  }
  std::map<std::string, double> profile_snapshot() { return this->tps[0]->pipeline_profile().snapshot(); }
  void reset_profile() { this->tps[0]->pipeline_profile().reset(); }
  void forward_binding(intptr_t rows, int topk, intptr_t ids, intptr_t weights, intptr_t input, intptr_t output,
                       bool incremental) {
    if (!rows) throw std::runtime_error("XMX qlen pointer is null");
    forward(*reinterpret_cast<const int*>(rows), topk, reinterpret_cast<const int64_t*>(ids),
            reinterpret_cast<const float*>(weights), reinterpret_cast<const void*>(input),
            reinterpret_cast<void*>(output), incremental);
  }
  void merge_results(int rows, void* output) override { merge_results(rows, output, false); }
  void merge_results(int rows, void* output, bool incremental) override {
    auto* dst = static_cast<ggml_bf16_t*>(output);
    const float* src = this->local_output_numa[0];
    for (int j = 0; j < rows * this->config.hidden_size; j += 8) {
      __m256 value = _mm256_loadu_ps(src + j);
      if (incremental) value = _mm256_add_ps(value, avx2::load_bf16_to_fp32(dst + j));
      avx2::store_fp32_to_bf16(dst + j, value);
    }
  }

 private:
  bool use_numa_dispatch_ = false;
};
#endif
