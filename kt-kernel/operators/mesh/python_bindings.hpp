#ifndef CPUINFER_OPERATOR_MESH_PYTHON_BINDINGS_HPP
#define CPUINFER_OPERATOR_MESH_PYTHON_BINDINGS_HPP

#include <cstdint>
#include <concepts>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../../cpu_backend/cpuinfer.h"
#include "../common.hpp"
#include "pybind11/pybind11.h"

#ifdef HAVE_LIBURING
#include "async_io.hpp"
#endif

namespace mesh {

#ifdef HAVE_LIBURING
inline std::vector<std::vector<ExpertFileSlot>> convert_iouring_slots(
    const std::vector<std::vector<std::tuple<int, long long, size_t>>>& src) {
  std::vector<std::vector<ExpertFileSlot>> dst;
  dst.reserve(src.size());
  for (const auto& row : src) {
    auto& out_row = dst.emplace_back();
    out_row.reserve(row.size());
    for (const auto& item : row) {
      ExpertFileSlot slot;
      slot.fd = std::get<0>(item);
      slot.offset = static_cast<off_t>(std::get<1>(item));
      slot.size = std::get<2>(item);
      out_row.push_back(slot);
    }
  }
  return dst;
}
#endif

template <class MoeClass>
class ForwardWithScoresBindings {
 public:
  struct Args {
    CPUInfer* cpuinfer;
    MoeClass* moe;
    intptr_t qlen;
    int k;
    intptr_t expert_ids;
    intptr_t weights;
    intptr_t input;
    intptr_t output;
    bool incremental;
    intptr_t router_scores;
    int score_rows;
    int score_cols;
    int score_transform;
  };

  static void inner(void* args) {
    Args* args_ = static_cast<Args*>(args);
    args_->cpuinfer->enqueue(&MoeClass::forward_binding_with_scores,
                             args_->moe,
                             args_->qlen,
                             args_->k,
                             args_->expert_ids,
                             args_->weights,
                             args_->input,
                             args_->output,
                             args_->incremental,
                             args_->router_scores,
                             args_->score_rows,
                             args_->score_cols,
                             args_->score_transform);
  }

  static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<MoeClass> moe,
                                                          intptr_t qlen,
                                                          int k,
                                                          intptr_t expert_ids,
                                                          intptr_t weights,
                                                          intptr_t input,
                                                          intptr_t output,
                                                          bool incremental,
                                                          intptr_t router_scores,
                                                          int score_rows,
                                                          int score_cols,
                                                          int score_transform) {
    Args* args = new Args{nullptr,
                          moe.get(),
                          qlen,
                          k,
                          expert_ids,
                          weights,
                          input,
                          output,
                          incremental,
                          router_scores,
                          score_rows,
                          score_cols,
                          score_transform};
    return std::make_pair(reinterpret_cast<intptr_t>(&inner), reinterpret_cast<intptr_t>(args));
  }
};

template <class MoeClass>
class ObserveRouterScoresBindings {
 public:
  struct Args {
    CPUInfer* cpuinfer;
    MoeClass* moe;
    intptr_t scores;
    int rows;
    int cols;
    int score_transform;
  };

  static void inner(void* args) {
    Args* args_ = static_cast<Args*>(args);
    args_->cpuinfer->enqueue(
        &MoeClass::observe_router_scores_binding, args_->moe, args_->scores, args_->rows, args_->cols, args_->score_transform);
  }

  static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<MoeClass> moe,
                                                          intptr_t scores,
                                                          int rows,
                                                          int cols,
                                                          int score_transform) {
    Args* args = new Args{nullptr, moe.get(), scores, rows, cols, score_transform};
    return std::make_pair(reinterpret_cast<intptr_t>(&inner), reinterpret_cast<intptr_t>(args));
  }
};

template <class MoeClass>
class ObserveRouterScoresBatchBindings {
 public:
  struct Args {
    CPUInfer* cpuinfer;
    MoeClass* moe;
    intptr_t scores;
    int score_stride;
    int cols;
    intptr_t layer_indices;
    intptr_t score_transforms;
    intptr_t gpu_experts_masks;
    int layer_count;
  };

  static void inner(void* args) {
    Args* args_ = static_cast<Args*>(args);
    args_->cpuinfer->enqueue(&MoeClass::observe_router_scores_batch_binding,
                             args_->moe,
                             args_->scores,
                             args_->score_stride,
                             args_->cols,
                             args_->layer_indices,
                             args_->score_transforms,
                             args_->gpu_experts_masks,
                             args_->layer_count);
  }

  static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<MoeClass> moe,
                                                          intptr_t scores,
                                                          int score_stride,
                                                          int cols,
                                                          intptr_t layer_indices,
                                                          intptr_t score_transforms,
                                                          intptr_t gpu_experts_masks,
                                                          int layer_count) {
    Args* args =
        new Args{nullptr, moe.get(), scores, score_stride, cols, layer_indices, score_transforms, gpu_experts_masks,
                 layer_count};
    return std::make_pair(reinterpret_cast<intptr_t>(&inner), reinterpret_cast<intptr_t>(args));
  }
};

template <class MoeClass>
class PrefetchExpertsBindings {
 public:
  struct Args {
    CPUInfer* cpuinfer;
    MoeClass* moe;
    intptr_t expert_ids;
    int count;
    intptr_t protect_ids;
    int protect_count;
    int max_to_submit;
    int prefetch_kind;
  };

  static void inner(void* args) {
    Args* args_ = static_cast<Args*>(args);
    args_->cpuinfer->enqueue(&MoeClass::prefetch_experts_binding,
                             args_->moe,
                             args_->expert_ids,
                             args_->count,
                             args_->protect_ids,
                             args_->protect_count,
                             args_->max_to_submit,
                             args_->prefetch_kind);
  }

  static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<MoeClass> moe,
                                                          intptr_t expert_ids,
                                                          int count,
                                                          intptr_t protect_ids = 0,
                                                          int protect_count = 0,
                                                          int max_to_submit = 0,
                                                          int prefetch_kind = 0) {
    Args* args = new Args{nullptr, moe.get(), expert_ids, count, protect_ids, protect_count, max_to_submit,
                          prefetch_kind};
    return std::make_pair(reinterpret_cast<intptr_t>(&inner), reinterpret_cast<intptr_t>(args));
  }
};

template <class MoeClass>
class SplitDeferredExpertsBindings {
 public:
  struct Args {
    CPUInfer* cpuinfer;
    MoeClass* moe;
    intptr_t source_ids;
    intptr_t immediate_ids;
    intptr_t deferred_ids;
    int count;
    int k;
    int max_deferred_per_token;
  };

  static void inner(void* args) {
    Args* args_ = static_cast<Args*>(args);
    args_->cpuinfer->enqueue(&MoeClass::split_deferred_experts_binding,
                             args_->moe,
                             args_->source_ids,
                             args_->immediate_ids,
                             args_->deferred_ids,
                             args_->count,
                             args_->k,
                             args_->max_deferred_per_token);
  }

  static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<MoeClass> moe,
                                                          intptr_t source_ids,
                                                          intptr_t immediate_ids,
                                                          intptr_t deferred_ids,
                                                          int count,
                                                          int k,
                                                          int max_deferred_per_token) {
    Args* args =
        new Args{nullptr, moe.get(), source_ids, immediate_ids, deferred_ids, count, k, max_deferred_per_token};
    return std::make_pair(reinterpret_cast<intptr_t>(&inner), reinterpret_cast<intptr_t>(args));
  }
};

template <class MoeClass>
class PreparePrefillLayerBindings {
 public:
  struct Args {
    CPUInfer* cpuinfer;
    MoeClass* moe;
  };

  static void inner(void* args) {
    Args* args_ = static_cast<Args*>(args);
    args_->cpuinfer->enqueue(&MoeClass::mesh_prepare_prefill_layer_binding, args_->moe);
  }

  static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<MoeClass> moe) {
    Args* args = new Args{nullptr, moe.get()};
    return std::make_pair(reinterpret_cast<intptr_t>(&inner), reinterpret_cast<intptr_t>(args));
  }
};

template <class MoeClass>
class ReleasePrefillLayerBindings {
 public:
  struct Args {
    CPUInfer* cpuinfer;
    MoeClass* moe;
  };

  static void inner(void* args) {
    Args* args_ = static_cast<Args*>(args);
    args_->cpuinfer->enqueue(&MoeClass::mesh_release_prefill_layer_binding, args_->moe);
  }

  static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<MoeClass> moe) {
    Args* args = new Args{nullptr, moe.get()};
    return std::make_pair(reinterpret_cast<intptr_t>(&inner), reinterpret_cast<intptr_t>(args));
  }
};

template <class MoeClass>
class TransitionDecodeCacheBindings {
 public:
  struct Args {
    CPUInfer* cpuinfer;
    MoeClass* moe;
    int decode_capacity;
    int fill_limit;
  };

  static void inner(void* args) {
    Args* args_ = static_cast<Args*>(args);
    args_->cpuinfer->enqueue(&MoeClass::mesh_transition_decode_cache_binding,
                             args_->moe,
                             args_->decode_capacity,
                             args_->fill_limit);
  }

  static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<MoeClass> moe,
                                                          int decode_capacity,
                                                          int fill_limit) {
    Args* args = new Args{nullptr, moe.get(), decode_capacity, fill_limit};
    return std::make_pair(reinterpret_cast<intptr_t>(&inner), reinterpret_cast<intptr_t>(args));
  }
};

template <class MoeClass, class PyClass>
void bind_moe_runtime_methods(PyClass& moe_cls) {
  if constexpr (requires(MoeClass moe, intptr_t qlen, int k, intptr_t expert_ids, intptr_t weights, intptr_t input,
                         intptr_t output, bool incremental, intptr_t router_scores, int score_rows, int score_cols,
                         int score_transform) {
                  moe.forward_binding_with_scores(qlen,
                                                  k,
                                                  expert_ids,
                                                  weights,
                                                  input,
                                                  output,
                                                  incremental,
                                                  router_scores,
                                                  score_rows,
                                                  score_cols,
                                                  score_transform);
                }) {
    moe_cls.def("forward_task",
                &ForwardWithScoresBindings<MoeClass>::cpuinfer_interface,
                pybind11::arg("qlen"),
                pybind11::arg("k"),
                pybind11::arg("expert_ids"),
                pybind11::arg("weights"),
                pybind11::arg("input"),
                pybind11::arg("output"),
                pybind11::arg("incremental"),
                pybind11::arg("router_scores"),
                pybind11::arg("score_rows"),
                pybind11::arg("score_cols"),
                pybind11::arg("score_transform"));
  }

  if constexpr (requires(MoeClass moe, intptr_t scores, int rows, int cols, int score_transform) {
                  moe.observe_router_scores_binding(scores, rows, cols, score_transform);
                }) {
    moe_cls.def("observe_router_scores_task",
                &ObserveRouterScoresBindings<MoeClass>::cpuinfer_interface,
                pybind11::arg("scores"),
                pybind11::arg("rows"),
                pybind11::arg("cols"),
                pybind11::arg("score_transform"));
    moe_cls.def("observe_router_scores",
                &MoeClass::observe_router_scores_binding,
                pybind11::arg("scores"),
                pybind11::arg("rows"),
                pybind11::arg("cols"),
                pybind11::arg("score_transform"));
  }

  if constexpr (requires(MoeClass moe, intptr_t scores, int score_stride, int cols, intptr_t layer_indices,
                         intptr_t score_transforms, intptr_t gpu_experts_masks, int layer_count) {
                  moe.observe_router_scores_batch_binding(scores,
                                                          score_stride,
                                                          cols,
                                                          layer_indices,
                                                          score_transforms,
                                                          gpu_experts_masks,
                                                          layer_count);
                }) {
    moe_cls.def("observe_router_scores_batch_task",
                &ObserveRouterScoresBatchBindings<MoeClass>::cpuinfer_interface,
                pybind11::arg("scores"),
                pybind11::arg("score_stride"),
                pybind11::arg("cols"),
                pybind11::arg("layer_indices"),
                pybind11::arg("score_transforms"),
                pybind11::arg("gpu_experts_masks"),
                pybind11::arg("layer_count"));
    moe_cls.def("observe_router_scores_batch",
                &MoeClass::observe_router_scores_batch_binding,
                pybind11::arg("scores"),
                pybind11::arg("score_stride"),
                pybind11::arg("cols"),
                pybind11::arg("layer_indices"),
                pybind11::arg("score_transforms"),
                pybind11::arg("gpu_experts_masks"),
                pybind11::arg("layer_count"));
  }

  if constexpr (requires(MoeClass moe) {
                  moe.cache_stats_snapshot();
                  moe.reset_cache_stats();
                }) {
    moe_cls.def("cache_stats_snapshot", &MoeClass::cache_stats_snapshot);
    moe_cls.def("reset_cache_stats", &MoeClass::reset_cache_stats);
  }
}

template <class MoeClass, class PyClass>
void bind_moe_residency_methods(PyClass& moe_cls) {
  if constexpr (requires(MoeClass moe, int expert_id) {
                  moe.promote_expert(expert_id);
                  moe.demote_expert(expert_id);
                  { moe.is_expert_promoted(expert_id) } -> std::convertible_to<bool>;
                }) {
    moe_cls.def("promote_expert", &MoeClass::promote_expert, pybind11::arg("expert_id"),
                "Materialize an expert into the MESH resident CPU cache");
    moe_cls.def("demote_expert", &MoeClass::demote_expert, pybind11::arg("expert_id"),
                "Demote an expert out of the MESH resident CPU cache");
    moe_cls.def("is_expert_promoted", &MoeClass::is_expert_promoted, pybind11::arg("expert_id"),
                "Check if an expert is currently resident in the MESH CPU cache");
  }

  if constexpr (requires(MoeClass moe, intptr_t expert_ids, int count, intptr_t protect_ids, int protect_count,
                         int max_to_submit, int prefetch_kind) {
                  moe.prefetch_experts_binding(expert_ids, count, protect_ids, protect_count, max_to_submit,
                                               prefetch_kind);
                }) {
    moe_cls.def("prefetch_experts_task",
                &PrefetchExpertsBindings<MoeClass>::cpuinfer_interface,
                pybind11::arg("expert_ids"),
                pybind11::arg("count"),
                pybind11::arg("protect_ids") = 0,
                pybind11::arg("protect_count") = 0,
                pybind11::arg("max_to_submit") = 0,
                pybind11::arg("prefetch_kind") = 0,
                "Submit non-blocking io_uring reads for selected experts without running AMX compute");
    moe_cls.def("prefetch_experts",
                &MoeClass::prefetch_experts_binding,
                pybind11::arg("expert_ids"),
                pybind11::arg("count"),
                pybind11::arg("protect_ids") = 0,
                pybind11::arg("protect_count") = 0,
                pybind11::arg("max_to_submit") = 0,
                pybind11::arg("prefetch_kind") = 0,
                "Submit non-blocking io_uring reads for selected experts without running AMX compute");
  }

  if constexpr (requires(MoeClass moe, intptr_t source_ids, intptr_t immediate_ids, intptr_t deferred_ids, int count,
                         int k, int max_deferred_per_token) {
                  moe.split_deferred_experts_binding(source_ids,
                                                     immediate_ids,
                                                     deferred_ids,
                                                     count,
                                                     k,
                                                     max_deferred_per_token);
                }) {
    moe_cls.def("split_deferred_experts_task",
                &SplitDeferredExpertsBindings<MoeClass>::cpuinfer_interface,
                pybind11::arg("source_ids"),
                pybind11::arg("immediate_ids"),
                pybind11::arg("deferred_ids"),
                pybind11::arg("count"),
                pybind11::arg("k"),
                pybind11::arg("max_deferred_per_token"),
                "Split top-k experts by current CPU residency state and prefetch deferred cold misses");
    moe_cls.def("split_deferred_experts",
                &MoeClass::split_deferred_experts_binding,
                pybind11::arg("source_ids"),
                pybind11::arg("immediate_ids"),
                pybind11::arg("deferred_ids"),
                pybind11::arg("count"),
                pybind11::arg("k"),
                pybind11::arg("max_deferred_per_token"),
                "Split top-k experts by current CPU residency state and prefetch deferred cold misses");
  }

  if constexpr (requires(MoeClass moe) {
                  moe.mesh_prepare_prefill_layer_binding();
                  moe.mesh_release_prefill_layer_binding();
                  moe.mesh_transition_decode_cache_binding(0, 0);
                }) {
    moe_cls.def("mesh_prepare_prefill_layer_task",
                &PreparePrefillLayerBindings<MoeClass>::cpuinfer_interface,
                "Prepare MESH prefill static CPU expert slots for this layer");
    moe_cls.def("mesh_release_prefill_layer_task",
                &ReleasePrefillLayerBindings<MoeClass>::cpuinfer_interface,
                "Release this layer's MESH prefill scratch slot buffers");
    moe_cls.def("mesh_transition_decode_cache_task",
                &TransitionDecodeCacheBindings<MoeClass>::cpuinfer_interface,
                pybind11::arg("decode_capacity"),
                pybind11::arg("fill_limit"),
                "Trim this layer to decode hot-cache capacity and submit Heat-based refill prefetches");
    moe_cls.def("mesh_prepare_prefill_layer",
                &MoeClass::mesh_prepare_prefill_layer_binding,
                "Prepare MESH prefill static CPU expert slots for this layer");
    moe_cls.def("mesh_release_prefill_layer",
                &MoeClass::mesh_release_prefill_layer_binding,
                "Release this layer's MESH prefill scratch slot buffers");
    moe_cls.def("mesh_transition_decode_cache",
                &MoeClass::mesh_transition_decode_cache_binding,
                pybind11::arg("decode_capacity"),
                pybind11::arg("fill_limit"),
                "Trim this layer to decode hot-cache capacity and submit Heat-based refill prefetches");
  }
}

template <class PyClass>
void bind_moe_config_extension(PyClass& cls) {
  cls.def_readwrite("max_tier0_experts", &GeneralMOEConfig::max_tier0_experts)
      .def_readwrite("max_resident_experts", &GeneralMOEConfig::max_resident_experts)
      .def_readwrite("resident_cache_policy", &GeneralMOEConfig::resident_cache_policy)
      .def_readwrite("enable_cache_stats", &GeneralMOEConfig::enable_cache_stats)
      .def_readwrite("iouring_direct_io", &GeneralMOEConfig::iouring_direct_io)
      .def_readwrite("mesh_lookahead_enabled", &GeneralMOEConfig::mesh_lookahead_enabled)
      .def_readwrite("mesh_topk_fallback_enabled", &GeneralMOEConfig::mesh_topk_fallback_enabled)
      .def_readwrite("mesh_lookahead_weight", &GeneralMOEConfig::mesh_lookahead_weight)
      .def_readwrite("mesh_heat_gamma", &GeneralMOEConfig::mesh_heat_gamma)
      .def_readwrite("mesh_heat_beta", &GeneralMOEConfig::mesh_heat_beta)
      .def_readwrite("mesh_transition_alpha", &GeneralMOEConfig::mesh_transition_alpha)
      .def_readwrite("mesh_prefetch_budget", &GeneralMOEConfig::mesh_prefetch_budget)
      .def_readwrite("mesh_coldstart_prefill_enabled", &GeneralMOEConfig::mesh_coldstart_prefill_enabled)
      .def_readwrite("mesh_coldstart_prefill_limit", &GeneralMOEConfig::mesh_coldstart_prefill_limit)
      .def_readwrite("mesh_prefill_layer_mode_enabled", &GeneralMOEConfig::mesh_prefill_layer_mode_enabled)
      .def_readwrite("mesh_prefill_static_experts", &GeneralMOEConfig::mesh_prefill_static_experts)
      .def_readwrite("mesh_decode_resident_experts", &GeneralMOEConfig::mesh_decode_resident_experts)
      .def_readwrite("mesh_memory_guard_enabled", &GeneralMOEConfig::mesh_memory_guard_enabled)
      .def_readwrite("mesh_memory_high_watermark", &GeneralMOEConfig::mesh_memory_high_watermark)
      .def_readwrite("mesh_memory_target_watermark", &GeneralMOEConfig::mesh_memory_target_watermark)
      .def_readwrite("mesh_memory_check_interval", &GeneralMOEConfig::mesh_memory_check_interval)
      .def_readwrite("mesh_memory_max_demotes_per_check", &GeneralMOEConfig::mesh_memory_max_demotes_per_check);

#ifdef HAVE_LIBURING
  cls.def("set_io_backend", [](GeneralMOEConfig& self, int backend) {
       self.io_backend = backend == static_cast<int>(IOBackend::IOURING) ? IOBackend::IOURING : IOBackend::MMAP;
     })
      .def("set_iouring_file_slots",
           [](GeneralMOEConfig& self,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& gate,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& gate_scale,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& up,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& up_scale,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& down,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& down_scale,
              ktransformers::AsyncExpertReader& reader) {
             self.io_backend = IOBackend::IOURING;
             self.gate_file_slots = convert_iouring_slots(gate);
             self.gate_scale_file_slots = convert_iouring_slots(gate_scale);
             self.up_file_slots = convert_iouring_slots(up);
             self.up_scale_file_slots = convert_iouring_slots(up_scale);
             self.down_file_slots = convert_iouring_slots(down);
             self.down_scale_file_slots = convert_iouring_slots(down_scale);
             self.async_reader = &reader;
           })
      .def("set_iouring_file_slots_for_readers",
           [](GeneralMOEConfig& self,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& gate,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& gate_scale,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& up,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& up_scale,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& down,
              const std::vector<std::vector<std::tuple<int, long long, size_t>>>& down_scale,
              pybind11::list readers) {
             self.io_backend = IOBackend::IOURING;
             self.gate_file_slots = convert_iouring_slots(gate);
             self.gate_scale_file_slots = convert_iouring_slots(gate_scale);
             self.up_file_slots = convert_iouring_slots(up);
             self.up_scale_file_slots = convert_iouring_slots(up_scale);
             self.down_file_slots = convert_iouring_slots(down);
             self.down_scale_file_slots = convert_iouring_slots(down_scale);
             self.async_readers.clear();
             self.async_readers.reserve(readers.size());
             for (pybind11::handle item : readers) {
               auto& reader = item.cast<ktransformers::AsyncExpertReader&>();
               self.async_readers.push_back(&reader);
             }
             if (self.async_readers.empty()) {
               throw std::runtime_error("io_uring reader list must not be empty");
             }
             self.async_reader = self.async_readers[0];
           });
#endif
}

inline void bind_async_io_python(pybind11::module_& m) {
#ifdef HAVE_LIBURING
  pybind11::class_<ktransformers::AsyncExpertReader>(m, "AsyncExpertReader")
      .def(pybind11::init<int>(), pybind11::arg("queue_depth") = 128,
           "Create AsyncExpertReader with specified queue depth and worker threads")
      .def("submit_read",
           [](ktransformers::AsyncExpertReader& reader, int fd, intptr_t buffer, size_t size, off_t offset,
              int expert_id) { return reader.submit_read(fd, reinterpret_cast<void*>(buffer), size, offset, expert_id); },
           pybind11::arg("fd"), pybind11::arg("buffer"), pybind11::arg("size"), pybind11::arg("offset"),
           pybind11::arg("expert_id"), "Submit an async read request")
      .def("wait_for_expert", &ktransformers::AsyncExpertReader::wait_for_expert, pybind11::arg("expert_id"),
           pybind11::arg("timeout_ms") = 5000, "Wait for a specific expert to be loaded")
      .def("wait_for_request", &ktransformers::AsyncExpertReader::wait_for_request, pybind11::arg("request_id"),
           pybind11::arg("timeout_ms") = 5000, "Wait for a specific request to complete")
      .def("wait_for_requests", &ktransformers::AsyncExpertReader::wait_for_requests, pybind11::arg("request_ids"),
           pybind11::arg("timeout_ms") = 5000, "Wait for all listed requests to complete")
      .def("get_request_result", &ktransformers::AsyncExpertReader::get_request_result, pybind11::arg("request_id"),
           "Return the io_uring result for a request, or INT_MIN if unknown")
      .def("request_succeeded", &ktransformers::AsyncExpertReader::request_succeeded, pybind11::arg("request_id"),
           "Return true only when the request completed as a full-size read")
      .def("describe_requests", &ktransformers::AsyncExpertReader::describe_requests, pybind11::arg("request_ids"),
           "Return a compact status summary for request diagnostics");

  pybind11::enum_<IOBackend>(m, "IOBackend")
      .value("MMAP", IOBackend::MMAP, "Compatibility value for ordinary KT resident loading")
      .value("IOURING", IOBackend::IOURING, "io_uring direct I/O (bypass page cache)")
      .export_values();
#endif
}

}  // namespace mesh

#endif  // CPUINFER_OPERATOR_MESH_PYTHON_BINDINGS_HPP
