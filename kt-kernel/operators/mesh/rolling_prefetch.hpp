// Rolling Layer Prefetch (RLP) — pure state machine for the opt-in MESH prefill
// strategy. Not a default code path. Caller drives I/O; this object only tracks
// per-layer pipeline state and tells the caller which layer to submit next.
//
// Lifecycle of one layer L:
//   Empty -> Reading -> Ready -> Computing -> Released
//
// Driver protocol (single-threaded; one MESH prefill request at a time):
//   1. ctor(depth, total_moe_layers)
//   2. bootstrap() -> list of layer indices [0..min(depth,total)) ; caller submits I/O
//   3. record_layer_requests(layer, ids) for each in-flight layer (optional, used for cancel)
//   4. mark_ready(layer) when caller observes I/O completion for `layer`
//   5. begin_compute(layer) right before compute kernel launch (must equal next_compute)
//   6. on_layer_compute_done(layer) -> next layer index to submit (-1 if none) ; caller submits I/O for it
//   7. drain_and_cancel() at end-of-request, returns per-layer in-flight request IDs to cancel
//
// Misuse (out-of-order, double-bootstrap, wrong state) throws std::logic_error
// rather than corrupting state — the bug should not be silent.

#ifndef CPUINFER_OPERATOR_MESH_ROLLING_PREFETCH_HPP
#define CPUINFER_OPERATOR_MESH_ROLLING_PREFETCH_HPP

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace mesh {

enum class LayerPipelineState : uint8_t {
  Empty = 0,
  Reading = 1,
  Ready = 2,
  Computing = 3,
  Released = 4,
};

inline const char* layer_pipeline_state_name(LayerPipelineState s) {
  switch (s) {
    case LayerPipelineState::Empty: return "Empty";
    case LayerPipelineState::Reading: return "Reading";
    case LayerPipelineState::Ready: return "Ready";
    case LayerPipelineState::Computing: return "Computing";
    case LayerPipelineState::Released: return "Released";
  }
  return "?";
}

class RollingPrefetchScheduler {
 public:
  RollingPrefetchScheduler(int depth, int total_moe_layers)
      : depth_(depth),
        total_moe_layers_(total_moe_layers),
        next_submit_layer_(0),
        next_compute_layer_(0),
        bootstrapped_(false) {
    if (depth <= 0) {
      throw std::invalid_argument("RollingPrefetchScheduler: depth must be >= 1");
    }
    if (total_moe_layers <= 0) {
      throw std::invalid_argument("RollingPrefetchScheduler: total_moe_layers must be >= 1");
    }
    per_layer_state_.assign(static_cast<size_t>(total_moe_layers_), LayerPipelineState::Empty);
    per_layer_request_ids_.assign(static_cast<size_t>(total_moe_layers_), {});
  }

  // Initial fill of the pipeline. Returns layer indices that just transitioned
  // Empty -> Reading; caller is responsible for issuing I/O for each.
  std::vector<int> bootstrap() {
    if (bootstrapped_) {
      throw std::logic_error("RollingPrefetchScheduler::bootstrap called twice");
    }
    bootstrapped_ = true;
    const int initial = std::min(depth_, total_moe_layers_);
    std::vector<int> submitted;
    submitted.reserve(static_cast<size_t>(initial));
    for (int layer = 0; layer < initial; ++layer) {
      per_layer_state_[layer] = LayerPipelineState::Reading;
      submitted.push_back(layer);
    }
    next_submit_layer_ = initial;
    return submitted;
  }

  // Attach io_uring (or other) request IDs to a Reading layer. Stored only so
  // drain_and_cancel can return them; the scheduler does not interpret them.
  void record_layer_requests(int layer, std::vector<uint64_t> request_ids) {
    require_in_range(layer);
    require_state(layer, LayerPipelineState::Reading, "record_layer_requests");
    per_layer_request_ids_[static_cast<size_t>(layer)] = std::move(request_ids);
  }

  // Caller observed I/O completion for `layer`. Reading -> Ready.
  void mark_ready(int layer) {
    require_in_range(layer);
    require_state(layer, LayerPipelineState::Reading, "mark_ready");
    per_layer_state_[static_cast<size_t>(layer)] = LayerPipelineState::Ready;
  }

  // Right before the compute kernel for `layer` runs. Must be the strict
  // in-order next layer (rolling pipeline assumes monotonic compute order).
  void begin_compute(int layer) {
    require_in_range(layer);
    if (layer != next_compute_layer_) {
      std::ostringstream oss;
      oss << "RollingPrefetchScheduler::begin_compute out of order: got layer=" << layer
          << " expected next_compute_layer=" << next_compute_layer_;
      throw std::logic_error(oss.str());
    }
    require_state(layer, LayerPipelineState::Ready, "begin_compute");
    per_layer_state_[static_cast<size_t>(layer)] = LayerPipelineState::Computing;
  }

  // Caller finished compute for `layer`. Computing -> Released, and if there
  // is still untouched tail, transitions Empty -> Reading for next_submit_layer_
  // and returns that index. Returns -1 when the tail is exhausted.
  int on_layer_compute_done(int layer) {
    require_in_range(layer);
    require_state(layer, LayerPipelineState::Computing, "on_layer_compute_done");
    per_layer_state_[static_cast<size_t>(layer)] = LayerPipelineState::Released;
    per_layer_request_ids_[static_cast<size_t>(layer)].clear();
    next_compute_layer_ = layer + 1;

    if (next_submit_layer_ >= total_moe_layers_) {
      return -1;
    }
    const int submitted = next_submit_layer_;
    per_layer_state_[static_cast<size_t>(submitted)] = LayerPipelineState::Reading;
    next_submit_layer_ = submitted + 1;
    return submitted;
  }

  // End-of-request cleanup. For every layer still in Reading, return its
  // request IDs so the caller can issue io_uring async-cancel; transition any
  // non-Released layer to Released so the scheduler can be reset/discarded.
  std::vector<std::pair<int, std::vector<uint64_t>>> drain_and_cancel() {
    std::vector<std::pair<int, std::vector<uint64_t>>> in_flight;
    for (int layer = 0; layer < total_moe_layers_; ++layer) {
      const size_t idx = static_cast<size_t>(layer);
      if (per_layer_state_[idx] == LayerPipelineState::Reading) {
        in_flight.emplace_back(layer, std::move(per_layer_request_ids_[idx]));
        per_layer_request_ids_[idx].clear();
      } else {
        per_layer_request_ids_[idx].clear();
      }
      if (per_layer_state_[idx] != LayerPipelineState::Released) {
        per_layer_state_[idx] = LayerPipelineState::Released;
      }
    }
    return in_flight;
  }

  LayerPipelineState state_of(int layer) const {
    require_in_range(layer);
    return per_layer_state_[static_cast<size_t>(layer)];
  }

  int depth() const { return depth_; }
  int total_layers() const { return total_moe_layers_; }
  int next_submit_layer() const { return next_submit_layer_; }
  int next_compute_layer() const { return next_compute_layer_; }
  bool bootstrapped() const { return bootstrapped_; }

 private:
  void require_in_range(int layer) const {
    if (layer < 0 || layer >= total_moe_layers_) {
      std::ostringstream oss;
      oss << "RollingPrefetchScheduler: layer=" << layer << " out of range [0, "
          << total_moe_layers_ << ")";
      throw std::out_of_range(oss.str());
    }
  }

  void require_state(int layer, LayerPipelineState expected, const char* op) const {
    const auto actual = per_layer_state_[static_cast<size_t>(layer)];
    if (actual != expected) {
      std::ostringstream oss;
      oss << "RollingPrefetchScheduler::" << op << " layer=" << layer
          << " expected state=" << layer_pipeline_state_name(expected)
          << " but was " << layer_pipeline_state_name(actual);
      throw std::logic_error(oss.str());
    }
  }

  int depth_;
  int total_moe_layers_;
  int next_submit_layer_;
  int next_compute_layer_;
  bool bootstrapped_;
  std::vector<LayerPipelineState> per_layer_state_;
  std::vector<std::vector<uint64_t>> per_layer_request_ids_;
};

inline void bind_rolling_prefetch_python(pybind11::module_& m) {
  namespace py = pybind11;
  py::enum_<LayerPipelineState>(m, "LayerPipelineState")
      .value("Empty", LayerPipelineState::Empty)
      .value("Reading", LayerPipelineState::Reading)
      .value("Ready", LayerPipelineState::Ready)
      .value("Computing", LayerPipelineState::Computing)
      .value("Released", LayerPipelineState::Released)
      .export_values();

  py::class_<RollingPrefetchScheduler>(m, "RollingPrefetchScheduler")
      .def(py::init<int, int>(), py::arg("depth"), py::arg("total_moe_layers"))
      .def("bootstrap", &RollingPrefetchScheduler::bootstrap)
      .def("record_layer_requests", &RollingPrefetchScheduler::record_layer_requests,
           py::arg("layer"), py::arg("request_ids"))
      .def("mark_ready", &RollingPrefetchScheduler::mark_ready, py::arg("layer"))
      .def("begin_compute", &RollingPrefetchScheduler::begin_compute, py::arg("layer"))
      .def("on_layer_compute_done", &RollingPrefetchScheduler::on_layer_compute_done,
           py::arg("layer"))
      .def("drain_and_cancel", &RollingPrefetchScheduler::drain_and_cancel)
      .def("state_of", &RollingPrefetchScheduler::state_of, py::arg("layer"))
      .def_property_readonly("depth", &RollingPrefetchScheduler::depth)
      .def_property_readonly("total_layers", &RollingPrefetchScheduler::total_layers)
      .def_property_readonly("next_submit_layer", &RollingPrefetchScheduler::next_submit_layer)
      .def_property_readonly("next_compute_layer", &RollingPrefetchScheduler::next_compute_layer)
      .def_property_readonly("bootstrapped", &RollingPrefetchScheduler::bootstrapped);
}

}  // namespace mesh

#endif  // CPUINFER_OPERATOR_MESH_ROLLING_PREFETCH_HPP
