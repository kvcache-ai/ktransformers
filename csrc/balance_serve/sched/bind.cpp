#include "scheduler.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(sched_ext, m) {
  py::class_<scheduler::ModelSettings>(m, "ModelSettings")
      .def(py::init<>())
      .def_readwrite("model_path", &scheduler::ModelSettings::model_path)
      .def_readwrite("params_count", &scheduler::ModelSettings::params_count)
      .def_readwrite("layer_count", &scheduler::ModelSettings::layer_count)
      .def_readwrite("num_k_heads", &scheduler::ModelSettings::num_k_heads)
      .def_readwrite("k_head_dim", &scheduler::ModelSettings::k_head_dim)
      .def_readwrite("bytes_per_params",
                     &scheduler::ModelSettings::bytes_per_params)
      .def_readwrite("bytes_per_kv_cache_element",
                     &scheduler::ModelSettings::bytes_per_kv_cache_element)
      .def("params_size", &scheduler::ModelSettings::params_nbytes)
      .def("bytes_per_token_kv_cache",
           &scheduler::ModelSettings::bytes_per_token_kv_cache)
      // 添加 pickle 支持
      .def(py::pickle(
          [](const scheduler::ModelSettings &self) { // __getstate__
            return py::make_tuple(self.params_count, self.layer_count,
                                  self.num_k_heads, self.k_head_dim,
                                  self.bytes_per_params,
                                  self.bytes_per_kv_cache_element);
          },
          [](py::tuple t) { // __setstate__
            if (t.size() != 6)
              throw std::runtime_error("Invalid state! t.size() = " +
                                       std::to_string(t.size()));
            scheduler::ModelSettings ms;
            ms.params_count = t[0].cast<size_t>();
            ms.layer_count = t[1].cast<size_t>();
            ms.num_k_heads = t[2].cast<size_t>();
            ms.k_head_dim = t[3].cast<size_t>();
            ms.bytes_per_params = t[4].cast<double>();
            ms.bytes_per_kv_cache_element = t[5].cast<double>();
            return ms;
          }));

  py::class_<scheduler::SampleOptions>(m, "SampleOptions")
      .def(py::init<>())
      .def_readwrite("temperature", &scheduler::SampleOptions::temperature)
      .def_readwrite("top_p",
                     &scheduler::SampleOptions::top_p) // 确保 top_p 也能被访问
      .def(py::pickle(
          [](const scheduler::SampleOptions &self) {
            return py::make_tuple(self.temperature,
                                  self.top_p); // 序列化 temperature 和 top_p
          },
          [](py::tuple t) {
            if (t.size() != 2) // 确保解包时参数数量匹配
              throw std::runtime_error("Invalid state! t.size() = " +
                                       std::to_string(t.size()));
            scheduler::SampleOptions so;
            so.temperature = t[0].cast<double>();
            so.top_p = t[1].cast<double>(); // 反序列化 top_p
            return so;
          }));

  py::class_<scheduler::Settings>(m, "Settings")
      .def(py::init<>())
      .def_readwrite("model_name", &scheduler::Settings::model_name)
      .def_readwrite("quant_type", &scheduler::Settings::quant_type)
      .def_readwrite("model_settings", &scheduler::Settings::model_settings)
      .def_readwrite("page_size", &scheduler::Settings::page_size)
      .def_readwrite("gpu_device_id", &scheduler::Settings::gpu_device_id)
      .def_readwrite("gpu_memory_size", &scheduler::Settings::gpu_memory_size)
      .def_readwrite("memory_utilization_percentage",
                     &scheduler::Settings::memory_utilization_percentage)
      .def_readwrite("max_batch_size", &scheduler::Settings::max_batch_size)
      .def_readwrite(
          "recommended_chunk_prefill_token_count",
          &scheduler::Settings::recommended_chunk_prefill_token_count)
      .def_readwrite("sample_options", &scheduler::Settings::sample_options)
      .def_readwrite("sched_metrics_port",
                     &scheduler::Settings::sched_metrics_port)
      .def_readwrite("gpu_only", &scheduler::Settings::gpu_only)
      .def_readwrite("use_self_defined_head_dim",
                     &scheduler::Settings::use_self_defined_head_dim)
      .def_readwrite("self_defined_head_dim",
                     &scheduler::Settings::self_defined_head_dim)
      .def_readwrite("full_kv_cache_on_each_gpu",
                     &scheduler::Settings::full_kv_cache_on_each_gpu)
      .def_readwrite("k_cache_on", &scheduler::Settings::k_cache_on)
      .def_readwrite("v_cache_on", &scheduler::Settings::v_cache_on)
      .def_readwrite("kvc2_config_path", &scheduler::Settings::kvc2_config_path)
      .def_readwrite("kvc2_root_path", &scheduler::Settings::kvc2_root_path)
      .def_readwrite("memory_pool_size_GB",
                     &scheduler::Settings::memory_pool_size_GB)
      .def_readwrite("evict_count", &scheduler::Settings::evict_count)
      .def_readwrite("strategy_name", &scheduler::Settings::strategy_name)
      .def_readwrite("kvc2_metrics_port",
                     &scheduler::Settings::kvc2_metrics_port)
      .def_readwrite("load_from_disk", &scheduler::Settings::load_from_disk)
      .def_readwrite("save_to_disk", &scheduler::Settings::save_to_disk)
      // derived
      .def_readwrite("gpu_device_count", &scheduler::Settings::gpu_device_count)
      .def_readwrite("total_kvcache_pages",
                     &scheduler::Settings::total_kvcache_pages)
      .def_readwrite("devices", &scheduler::Settings::devices)
      .def("auto_derive", &scheduler::Settings::auto_derive);

  py::class_<scheduler::BatchQueryTodo,
             std::shared_ptr<scheduler::BatchQueryTodo>>(m, "BatchQueryTodo")
      .def(py::init<>())
      .def_readwrite("query_ids", &scheduler::BatchQueryTodo::query_ids)
      .def_readwrite("query_tokens", &scheduler::BatchQueryTodo::query_tokens)
      .def_readwrite("query_lengths", &scheduler::BatchQueryTodo::query_lengths)
      .def_readwrite("block_indexes", &scheduler::BatchQueryTodo::block_indexes)
      .def_readwrite("attn_masks", &scheduler::BatchQueryTodo::attn_masks)
      .def_readwrite("rope_ranges", &scheduler::BatchQueryTodo::rope_ranges)
      .def_readwrite("sample_options",
                     &scheduler::BatchQueryTodo::sample_options)
      .def_readwrite("prefill_mini_batches",
                     &scheduler::BatchQueryTodo::prefill_mini_batches)
      .def_readwrite("decode_mini_batches",
                     &scheduler::BatchQueryTodo::decode_mini_batches)
      .def_readwrite("stop_criteria", &scheduler::BatchQueryTodo::stop_criteria)
      .def("debug", &scheduler::BatchQueryTodo::debug)
      .def(py::pickle(
          [](const scheduler::BatchQueryTodo &self) {
            return py::make_tuple(
                self.query_ids, self.query_tokens, self.query_lengths,
                self.block_indexes, self.attn_masks, self.rope_ranges,
                self.sample_options, self.prefill_mini_batches,
                self.decode_mini_batches, self.stop_criteria);
          },
          [](py::tuple t) {
            if (t.size() != 10)
              throw std::runtime_error("Invalid state! t.size() = " +
                                       std::to_string(t.size()));
            scheduler::BatchQueryTodo bqt;
            bqt.query_ids = t[0].cast<std::vector<scheduler::QueryID>>();
            bqt.query_tokens = t[1].cast<std::vector<torch::Tensor>>();
            bqt.query_lengths =
                t[2].cast<std::vector<scheduler::TokenLength>>();
            bqt.block_indexes = t[3].cast<std::vector<torch::Tensor>>();
            bqt.attn_masks = t[4].cast<std::optional<torch::Tensor>>();
            bqt.rope_ranges = t[5].cast<std::optional<torch::Tensor>>();
            bqt.sample_options =
                t[6].cast<std::vector<scheduler::SampleOptions>>();
            bqt.prefill_mini_batches =
                t[7].cast<std::vector<scheduler::PrefillTask>>();
            bqt.decode_mini_batches =
                t[8].cast<std::vector<std::vector<scheduler::QueryID>>>();
            bqt.stop_criteria =
                t[9].cast<std::vector<std::vector<std::vector<int>>>>();
            return bqt;
          }));

  py::class_<scheduler::QueryUpdate>(m, "QueryUpdate")
      .def(py::init<>())
      .def_readwrite("id", &scheduler::QueryUpdate::id)
      .def_readwrite("ok", &scheduler::QueryUpdate::ok)
      .def_readwrite("is_prefill", &scheduler::QueryUpdate::is_prefill)
      .def_readwrite("decode_done", &scheduler::QueryUpdate::decode_done)
      .def_readwrite("active_position",
                     &scheduler::QueryUpdate::active_position)
      .def_readwrite("generated_token",
                     &scheduler::QueryUpdate::generated_token)
      .def(py::pickle(
          [](const scheduler::QueryUpdate &self) {
            return py::make_tuple(self.id, self.ok, self.is_prefill,
                                  self.decode_done, self.active_position,
                                  self.generated_token);
          },
          [](py::tuple t) {
            if (t.size() != 6)
              throw std::runtime_error("Invalid state! t.size() = " +
                                       std::to_string(t.size()));
            scheduler::QueryUpdate qu;
            qu.id = t[0].cast<scheduler::QueryID>();
            qu.ok = t[1].cast<bool>();
            qu.is_prefill = t[2].cast<bool>();
            qu.decode_done = t[3].cast<bool>();
            qu.active_position = t[4].cast<scheduler::TokenLength>();
            qu.generated_token = t[5].cast<scheduler::Token>();
            return qu;
          }));

  py::class_<scheduler::InferenceContext>(m, "InferenceContext")
      .def(py::init<>())
      .def_readwrite("k_cache", &scheduler::InferenceContext::k_cache)
      .def_readwrite("v_cache", &scheduler::InferenceContext::v_cache);

  py::class_<scheduler::QueryAdd>(m, "QueryAdd")
      .def(py::init<>())
      .def_readwrite("query_token", &scheduler::QueryAdd::query_token)
      // .def_readwrite("attn_mask", &scheduler::QueryAdd::attn_mask)
      .def_readwrite("query_length", &scheduler::QueryAdd::query_length)
      .def_readwrite("estimated_length", &scheduler::QueryAdd::estimated_length)
      .def_readwrite("sample_options", &scheduler::QueryAdd::sample_options)
      .def_readwrite("user_id", &scheduler::QueryAdd::user_id)
      .def_readwrite("SLO_TTFT_ms", &scheduler::QueryAdd::SLO_TTFT_ms)
      .def_readwrite("SLO_TBT_ms", &scheduler::QueryAdd::SLO_TBT_ms)
      .def_readwrite("stop_criteria", &scheduler::QueryAdd::stop_criteria)
      .def("serialize", &scheduler::QueryAdd::serialize)
      .def_static("deserialize", &scheduler::QueryAdd::deserialize)
      .def(py::pickle(
          [](const scheduler::QueryAdd &self) {
            return py::make_tuple(self.query_token,
                                  // self.attn_mask,
                                  self.query_length, self.estimated_length,
                                  self.sample_options, self.user_id,
                                  self.SLO_TTFT_ms, self.SLO_TBT_ms,
                                  self.stop_criteria);
          },
          [](py::tuple t) {
            if (t.size() != 8)
              throw std::runtime_error("Invalid state! t.size() = " +
                                       std::to_string(t.size()));
            scheduler::QueryAdd qa;
            qa.query_token = t[0].cast<std::vector<scheduler::Token>>();
            // qa.attn_mask = t[1].cast<torch::Tensor>();
            qa.query_length = t[1].cast<scheduler::TokenLength>();
            qa.estimated_length = t[2].cast<scheduler::TokenLength>();
            qa.sample_options = t[3].cast<scheduler::SampleOptions>();
            qa.user_id = t[4].cast<scheduler::UserID>();
            qa.SLO_TTFT_ms = t[5].cast<int>();
            qa.SLO_TBT_ms = t[6].cast<int>();
            qa.stop_criteria = t[7].cast<std::vector<std::vector<int>>>();
            return qa;
          }));

  py::class_<scheduler::Scheduler, std::shared_ptr<scheduler::Scheduler>>(
      m, "Scheduler")
      .def("init", &scheduler::Scheduler::init)
      .def("run", &scheduler::Scheduler::run)
      .def("stop", &scheduler::Scheduler::stop)
      .def("add_query", &scheduler::Scheduler::add_query,
           py::call_guard<py::gil_scoped_release>())
      .def("cancel_query", &scheduler::Scheduler::cancel_query,
           py::call_guard<py::gil_scoped_release>())
      .def("update_last_batch", &scheduler::Scheduler::update_last_batch,
           py::call_guard<py::gil_scoped_release>())
      .def("get_inference_context",
           &scheduler::Scheduler::get_inference_context);

  m.def("create_scheduler", &scheduler::create_scheduler,
        "Create a new Scheduler instance");
}
