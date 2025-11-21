/**
 * @Description  :
 * @Author       : chenht2022, Jianwei Dong
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : Jianwei Dong
 * @LastEditTime : 2024-08-26 22:47:06
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
// Python bindings
#include <sys/types.h>

#include <cstddef>

#include "cpu_backend/cpuinfer.h"
#include "cpu_backend/worker_pool.h"
#include "operators/common.hpp"

#if defined(USE_MOE_KERNEL)
#include "operators/moe_kernel/la/kernel.hpp"
#include "operators/moe_kernel/moe.hpp"
#endif

#if defined(__aarch64__) && defined(CPU_USE_KML)
#if defined(KTRANSFORMERS_CPU_MLA)
#include "operators/kml/deepseekv3.hpp"
#include "operators/kml/gate.hpp"
#include "operators/kml/mla.hpp"
#include "operators/kml/mla_int8.hpp"
#endif
#include "operators/kml/moe.hpp"
static const bool _is_plain_ = true;
#else
static const bool _is_plain_ = false;
#endif

#if defined(__x86_64__) && defined(USE_AMX_AVX_KERNEL)
#include "operators/amx/awq-moe.hpp"
#include "operators/amx/k2-moe.hpp"
#include "operators/amx/la/amx_kernels.hpp"
#include "operators/amx/moe.hpp"
#endif
#include <pybind11/stl.h>  // std::vector/std::pair/std::string conversions

#include <cstdint>
#include <memory>
#include <type_traits>

#include "operators/kvcache/kvcache.h"
#include "operators/llamafile/linear.h"
#include "operators/llamafile/mla.hpp"
#include "operators/llamafile/mlp.h"
#include "operators/llamafile/moe.hpp"
#include "pybind11/pybind11.h"

namespace py = pybind11;
using namespace pybind11::literals;

py::object to_float_ptr(uintptr_t input_ptr, int size, ggml_type type) {
  if (type < 0 || type >= GGML_TYPE_COUNT) {
    PyErr_SetString(PyExc_ValueError, "Invalid ggml_type");
    throw py::error_already_set();
  }

  py::module torch = py::module::import("torch");
  py::dict kwargs;
  kwargs["dtype"] = torch.attr("float32");
  py::object tensor = torch.attr("empty")(size, **kwargs);

  uintptr_t output_ptr = tensor.attr("data_ptr")().cast<uintptr_t>();
  float* output_float_ptr = reinterpret_cast<float*>(output_ptr);

  try {
    to_float(reinterpret_cast<void*>(input_ptr), output_float_ptr, size, type);
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    throw py::error_already_set();
  }

  return tensor;
}

py::object from_float_ptr(uintptr_t input_ptr, int size, ggml_type type) {
  if (type < 0 || type >= GGML_TYPE_COUNT) {
    PyErr_SetString(PyExc_ValueError, "Invalid ggml_type");
    throw py::error_already_set();
  }

  py::module torch = py::module::import("torch");

  size_t output_elem_bytes = ggml_type_size(type);
  size_t output_elem_count = (size + ggml_blck_size(type) - 1) / ggml_blck_size(type);
  size_t total_bytes = output_elem_count * output_elem_bytes;

  py::dict kwargs;
  kwargs["dtype"] = torch.attr("uint8");
  py::object tensor = torch.attr("empty")(total_bytes, **kwargs);

  uintptr_t output_ptr = tensor.attr("data_ptr")().cast<uintptr_t>();
  void* output_void_ptr = reinterpret_cast<void*>(output_ptr);

  try {
    from_float(reinterpret_cast<float*>(input_ptr), output_void_ptr, size, type);
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    throw py::error_already_set();
  }

  return tensor;
}

template <typename T>
std::vector<std::vector<uintptr_t>> void_ptr_nested_to_uint(const std::vector<std::vector<T*>>& input) {
  std::vector<std::vector<uintptr_t>> result;
  for (const auto& row : input) {
    std::vector<uintptr_t> new_row;
    for (auto ptr : row) {
      new_row.push_back(reinterpret_cast<uintptr_t>(ptr));
    }
    result.push_back(std::move(new_row));
  }
  return result;
}

template <typename T>
std::vector<std::vector<T*>> uint_to_void_ptr_nested(const std::vector<std::vector<uintptr_t>>& input) {
  std::vector<std::vector<T*>> result;
  for (const auto& row : input) {
    std::vector<T*> new_row;
    for (auto val : row) {
      new_row.push_back(reinterpret_cast<T*>(val));
    }
    result.push_back(std::move(new_row));
  }
  return result;
}

#define DEF_PTR_PROPERTY(cls, name)                                                  \
  def_property(                                                                      \
      #name, [](const cls& self) { return reinterpret_cast<uintptr_t>(self.name); }, \
      [](cls& self, uintptr_t val) { self.name = reinterpret_cast<void*>(val); })

#define DEF_PTR_2D_PROPERTY(cls, name)                                                 \
  def_property(                                                                        \
      #name, [](const cls& self) { return void_ptr_nested_to_uint<void>(self.name); }, \
      [](cls& self, const std::vector<std::vector<uintptr_t>>& val) {                  \
        self.name = uint_to_void_ptr_nested<void>(val);                                \
      })

template <class T>
class MOEBindings {
 public:
  class WarmUpBindings {
   public:
    struct Args {
      CPUInfer* cpuinfer;
      TP_MOE<T>* moe;
    };
    static void inner(void* args) {
      Args* args_ = (Args*)args;
      args_->cpuinfer->enqueue(&TP_MOE<T>::warm_up, args_->moe);
    }
    static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<TP_MOE<T>> moe) {
      Args* args = new Args{nullptr, moe.get()};
      return std::make_pair((intptr_t)&inner, (intptr_t)args);
    }
  };
  class LoadWeightsBindings {
   public:
    struct Args {
      CPUInfer* cpuinfer;
      TP_MOE<T>* moe;
    };
    static void inner(void* args) {
      Args* args_ = (Args*)args;
      args_->cpuinfer->enqueue(&TP_MOE<T>::load_weights, args_->moe);
    }
    static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<TP_MOE<T>> moe,
                                                            const uintptr_t physical_to_logical_map = 0) {
      Args* args = new Args{nullptr, moe.get()};
      if (physical_to_logical_map) {
        // printf("debug physical_to_logical_map in arg:%lu\n", physical_to_logical_map);
        moe->config.physical_to_logical_map = reinterpret_cast<void*>(physical_to_logical_map);
        // printf("moe ptr:%p,confirm: moe->config.physical_to_logical_map:%lu\n", reinterpret_cast<void*>(moe.get()),
        //  reinterpret_cast<uintptr_t>(moe->config.physical_to_logical_map));
      }
      return std::make_pair((intptr_t)&inner, (intptr_t)args);
    }
    static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<TP_MOE<T>> moe) {
      return cpuinfer_interface(moe, 0);
    }
  };
  class ForwardBindings {
   public:
    struct Args {
      CPUInfer* cpuinfer;
      TP_MOE<T>* moe;
      intptr_t qlen;
      int k;
      intptr_t expert_ids;
      intptr_t weights;
      intptr_t input;
      intptr_t output;
      bool incremental;
    };
    static void inner(void* args) {
      Args* args_ = (Args*)args;
      args_->cpuinfer->enqueue(&TP_MOE<T>::forward_binding, args_->moe, args_->qlen, args_->k, args_->expert_ids,
                               args_->weights, args_->input, args_->output, args_->incremental);
    }
    static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<TP_MOE<T>> moe, intptr_t qlen, int k,
                                                            intptr_t expert_ids, intptr_t weights, intptr_t input,
                                                            intptr_t output, bool incremental = false) {
      Args* args = new Args{nullptr, moe.get(), qlen, k, expert_ids, weights, input, output, incremental};
      return std::make_pair((intptr_t)&inner, (intptr_t)args);
    }
    static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<TP_MOE<T>> moe, intptr_t qlen, int k,
                                                            intptr_t expert_ids, intptr_t weights, intptr_t input,
                                                            intptr_t output) {
      return cpuinfer_interface(moe, qlen, k, expert_ids, weights, input, output, false);
    }
  };
};

template <typename MoeTP>
void bind_moe_module(py::module_& moe_module, const char* name) {
  using MoeClass = TP_MOE<MoeTP>;
  using MoeBindings = MOEBindings<MoeTP>;

  auto moe_cls = py::class_<MoeClass, MoE_Interface, std::shared_ptr<MoeClass>>(moe_module, name);

  moe_cls
      .def(py::init<GeneralMOEConfig>())
      .def("warm_up_task", &MoeBindings::WarmUpBindings::cpuinfer_interface)
      .def("load_weights_task",
           py::overload_cast<std::shared_ptr<MoeClass>>(&MoeBindings::LoadWeightsBindings::cpuinfer_interface))
      .def("load_weights_task",
           py::overload_cast<std::shared_ptr<MoeClass>, const uintptr_t>(
               &MoeBindings::LoadWeightsBindings::cpuinfer_interface),
           py::arg("physical_to_logical_map"))
      // .def("forward_task", &MoeBindings::ForwardBindings::cpuinfer_interface)
      .def("forward_task",
           py::overload_cast<std::shared_ptr<MoeClass>, intptr_t, int, intptr_t, intptr_t, intptr_t, intptr_t>(
               &MoeBindings::ForwardBindings::cpuinfer_interface))
      .def("forward_task",
           py::overload_cast<std::shared_ptr<MoeClass>, intptr_t, int, intptr_t, intptr_t, intptr_t, intptr_t, bool>(
               &MoeBindings::ForwardBindings::cpuinfer_interface))
      .def("warm_up", &MoeClass::warm_up)
      .def("load_weights", &MoeClass::load_weights)
      .def("forward", &MoeClass::forward_binding);

#if defined(__x86_64__) && defined(USE_AMX_AVX_KERNEL)
  if constexpr (std::is_same_v<MoeTP, AMX_K2_MOE_TP<amx::GemmKernel224Int4SmallKGroup>>) {
    struct WriteWeightScaleToBufferBindings {
      struct Args {
        CPUInfer* cpuinfer;
        MoeClass* moe;
        int gpu_tp_count;
        int gpu_experts_num;
        std::vector<uintptr_t> w13_weight_ptrs;
        std::vector<uintptr_t> w13_scale_ptrs;
        std::vector<uintptr_t> w2_weight_ptrs;
        std::vector<uintptr_t> w2_scale_ptrs;
      };

      static void inner(void* args) {
        Args* args_ = (Args*)args;
        args_->cpuinfer->enqueue(&MoeClass::write_weight_scale_to_buffer, args_->moe,
                                 args_->gpu_tp_count, args_->gpu_experts_num,
                                 args_->w13_weight_ptrs, args_->w13_scale_ptrs,
                                 args_->w2_weight_ptrs, args_->w2_scale_ptrs);
      }

      static std::pair<intptr_t, intptr_t> cpuinfer_interface(std::shared_ptr<MoeClass> moe,
                                                              int gpu_tp_count, int gpu_experts_num,
                                                              py::list w13_weight_ptrs, py::list w13_scale_ptrs,
                                                              py::list w2_weight_ptrs, py::list w2_scale_ptrs) {
        // Convert Python lists to std::vector<uintptr_t>
        std::vector<uintptr_t> w13_weight_vec, w13_scale_vec, w2_weight_vec, w2_scale_vec;

        for (auto item : w13_weight_ptrs) w13_weight_vec.push_back(py::cast<uintptr_t>(item));
        for (auto item : w13_scale_ptrs) w13_scale_vec.push_back(py::cast<uintptr_t>(item));
        for (auto item : w2_weight_ptrs) w2_weight_vec.push_back(py::cast<uintptr_t>(item));
        for (auto item : w2_scale_ptrs) w2_scale_vec.push_back(py::cast<uintptr_t>(item));

        Args* args = new Args{nullptr, moe.get(), gpu_tp_count, gpu_experts_num,
                              w13_weight_vec, w13_scale_vec, w2_weight_vec, w2_scale_vec};
        return std::make_pair((intptr_t)&inner, (intptr_t)args);
      }
    };

    moe_cls.def("write_weight_scale_to_buffer_task", &WriteWeightScaleToBufferBindings::cpuinfer_interface,
             py::arg("gpu_tp_count"), py::arg("gpu_experts_num"),
             py::arg("w13_weight_ptrs"), py::arg("w13_scale_ptrs"),
             py::arg("w2_weight_ptrs"), py::arg("w2_scale_ptrs"));
  }
#endif
}

PYBIND11_MODULE(kt_kernel_ext, m) {
  py::class_<WorkerPool>(m, "WorkerPool").def(py::init<int>());
  py::class_<WorkerPoolConfig>(m, "WorkerPoolConfig")
      .def(py::init<>())
      .def_readwrite("subpool_count", &WorkerPoolConfig::subpool_count)
      .def_readwrite("subpool_numa_map", &WorkerPoolConfig::subpool_numa_map)
      .def_readwrite("subpool_thread_count", &WorkerPoolConfig::subpool_thread_count);

  py::class_<CPUInfer>(m, "CPUInfer")
      .def(py::init<int>())
      .def(py::init<WorkerPoolConfig>())
      .def("submit", &CPUInfer::submit)
      .def("sync", &CPUInfer::sync, py::arg("allow_n_pending") = 0)
      .def_readwrite("backend_", &CPUInfer::backend_)
#ifndef KTRANSFORMERS_CPU_ONLY
      .def("sync_with_cuda_stream", &CPUInfer::sync_with_cuda_stream, py::arg("user_cuda_stream"),
           py::arg("allow_n_pending") = 0)
      .def("submit_with_cuda_stream", &CPUInfer::submit_with_cuda_stream)
#endif
      ;

  auto linear_module = m.def_submodule("linear");
  py::class_<LinearConfig>(linear_module, "LinearConfig")
      .def(py::init([](int hidden_size, int intermediate_size, int stride, int group_max_len, intptr_t proj,
                       int proj_type, int hidden_type) {
        return LinearConfig(hidden_size, intermediate_size, stride, group_max_len, (void*)proj, (ggml_type)proj_type,
                            (ggml_type)hidden_type);
      }));
  // py::class_<Linear>(linear_module, "Linear")
  //     .def(py::init<LinearConfig>())
  //     .def("warm_up", &LinearBindings::WarmUpBindings::cpuinfer_interface)
  //     .def("forward", &LinearBindings::ForwardBindings::cpuinfer_interface);

  auto mlp_module = m.def_submodule("mlp");
  py::class_<MLPConfig>(mlp_module, "MLPConfig")
      .def(py::init([](int hidden_size, int intermediate_size, int stride, int group_max_len, intptr_t gate_proj,
                       intptr_t up_proj, intptr_t down_proj, int gate_type, int up_type, int down_type,
                       int hidden_type) {
        return MLPConfig(hidden_size, intermediate_size, stride, group_max_len, (void*)gate_proj, (void*)up_proj,
                         (void*)down_proj, (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type,
                         (ggml_type)hidden_type);
      }));
  // py::class_<MLP>(mlp_module, "MLP")
  //     .def(py::init<MLPConfig>())
  //     .def("warm_up", &MLPBindings::WarmUpBindings::cpuinfer_interface)
  //     .def("forward", &MLPBindings::ForwardBindings::cpuinfer_interface);

  py::class_<GeneralConfig>(m, "GeneralConfig")
      .def(py::init<>())
      .def_readwrite("vocab_size", &GeneralConfig::vocab_size)
      .def_readwrite("hidden_size", &GeneralConfig::hidden_size)
      .def_readwrite("num_experts_per_tok", &GeneralConfig::num_experts_per_tok)
      .def_readwrite("n_routed_experts", &GeneralConfig::n_routed_experts)
      .def_readwrite("n_shared_experts", &GeneralConfig::n_shared_experts)
      .def_readwrite("max_qlen", &GeneralConfig::max_qlen)
      .DEF_PTR_PROPERTY(GeneralConfig, lm_heads_ptr)
      .def_readwrite("lm_heads_type", &GeneralConfig::lm_heads_type)
      .DEF_PTR_PROPERTY(GeneralConfig, norm_weights_ptr)
      .def_readwrite("norm_weights_type", &GeneralConfig::norm_weights_type)
      .DEF_PTR_PROPERTY(GeneralConfig, token_embd_ptr)
      .def_readwrite("token_embd_type", &GeneralConfig::token_embd_type)
      .def_readwrite("pool", &GeneralConfig::pool);
#if defined(__aarch64__) && defined(CPU_USE_KML) && defined(KTRANSFORMERS_CPU_MLA)
  py::class_<DeepseekV3ForCausalLM, std::shared_ptr<DeepseekV3ForCausalLM>>(m, "DeepseekV3ForCausalLM")
      .def(py::init([](GeneralConfig config) { return std::make_shared<DeepseekV3ForCausalLM>(config); }))
      .def_readwrite("model", &DeepseekV3ForCausalLM::model)
      .def("forward", &DeepseekV3ForCausalLM::forward_binding);

  py::class_<DeepseekV3Model, std::shared_ptr<DeepseekV3Model>>(m, "DeepseekV3Model")
      .def(py::init([](GeneralConfig config) { return std::make_shared<DeepseekV3Model>(config); }))
      .def_readwrite("layers", &DeepseekV3Model::layers);

  py::class_<DeepseekV3DecoderLayer, std::shared_ptr<DeepseekV3DecoderLayer>>(m, "DeepseekV3DecoderLayer")
      .def(py::init([](GeneralConfig config, size_t layer_idx) {
        return std::make_shared<DeepseekV3DecoderLayer>(config, layer_idx);
      }))
      .def("load_norm", &DeepseekV3DecoderLayer::load_norm_binding)
      .def_readwrite("self_attn", &DeepseekV3DecoderLayer::self_attn)
      .def_readwrite("gate", &DeepseekV3DecoderLayer::gate)
      .def_readwrite("ffn", &DeepseekV3DecoderLayer::ffn);
#endif
  auto mla_module = m.def_submodule("mla");
  py::class_<GeneralMLAConfig>(mla_module, "MLAConfig")
      .def(py::init([](size_t hidden_size, size_t q_lora_rank, size_t num_heads, size_t nope_size, size_t rope_size,
                       size_t kv_lora_rank) {
        return GeneralMLAConfig(hidden_size, q_lora_rank, num_heads, nope_size, rope_size, kv_lora_rank);
      }))
      .def_readwrite("layer_idx", &GeneralMLAConfig::layer_idx)
      .def_readwrite("pool", &GeneralMLAConfig::pool)
      .def_readwrite("token_count_in_page", &GeneralMLAConfig::token_count_in_page)
      .def_readwrite("max_qlen", &GeneralMLAConfig::max_qlen)
      .def_readwrite("max_kvlen", &GeneralMLAConfig::max_kvlen)

      .def_readwrite("max_position_embeddings", &GeneralMLAConfig::max_position_embeddings)
      .def_readwrite("rope_scaling_factor", &GeneralMLAConfig::rope_scaling_factor)
      .def_readwrite("rope_theta", &GeneralMLAConfig::rope_theta)
      .def_readwrite("rope_scaling_beta_fast", &GeneralMLAConfig::rope_scaling_beta_fast)
      .def_readwrite("rope_scaling_beta_slow", &GeneralMLAConfig::rope_scaling_beta_slow)
      .def_readwrite("rope_scaling_mscale", &GeneralMLAConfig::rope_scaling_mscale)
      .def_readwrite("rope_scaling_mscale_all_dim", &GeneralMLAConfig::rope_scaling_mscale_all_dim)
      .def_readwrite("rope_scaling_original_max_position_embeddings",
                     &GeneralMLAConfig::rope_scaling_original_max_position_embeddings)

      .DEF_PTR_PROPERTY(GeneralMLAConfig, q_a_proj)
      .DEF_PTR_PROPERTY(GeneralMLAConfig, q_a_norm)
      .DEF_PTR_PROPERTY(GeneralMLAConfig, q_b_proj)
      .DEF_PTR_PROPERTY(GeneralMLAConfig, kv_a_proj_with_mqa)
      .DEF_PTR_PROPERTY(GeneralMLAConfig, kv_a_norm)
      .DEF_PTR_PROPERTY(GeneralMLAConfig, kv_b_proj)
      .DEF_PTR_PROPERTY(GeneralMLAConfig, o_proj)

      .def_readwrite("q_a_proj_type", &GeneralMLAConfig::q_a_proj_type)
      .def_readwrite("q_a_norm_type", &GeneralMLAConfig::q_a_norm_type)
      .def_readwrite("q_b_proj_type", &GeneralMLAConfig::q_b_proj_type)
      .def_readwrite("kv_a_proj_with_mqa_type", &GeneralMLAConfig::kv_a_proj_with_mqa_type)
      .def_readwrite("kv_a_norm_type", &GeneralMLAConfig::kv_a_norm_type)
      .def_readwrite("kv_b_proj_type", &GeneralMLAConfig::kv_b_proj_type)
      .def_readwrite("w_o_type", &GeneralMLAConfig::w_o_type)
      .def_readwrite("page_count", &GeneralMLAConfig::page_count)

      ;
  py::class_<MLA_Interface, std::shared_ptr<MLA_Interface>>(mla_module, "MLA_Interface");
#if defined(__aarch64__) && defined(CPU_USE_KML) && defined(KTRANSFORMERS_CPU_MLA)
  py::class_<TP_MLA<KML_MLA_TP<float16_t>>, MLA_Interface, std::shared_ptr<TP_MLA<KML_MLA_TP<float16_t>>>>(mla_module,
                                                                                                           "MLA_F16")
      .def(py::init<GeneralMLAConfig>())
      .def("load_weights", &TP_MLA<KML_MLA_TP<float16_t>>::load_weights)
      .def("forward",
           [](TP_MLA<KML_MLA_TP<float16_t>>& op, std::vector<int> qlens, std::vector<std::vector<int>> page_tables,
              std::vector<int> kvlens, intptr_t input,
              intptr_t output) { op.forward(qlens, page_tables, kvlens, (const void*)input, (void*)output); })
      .def("set_local_pages", &TP_MLA<KML_MLA_TP<float16_t>>::set_local_pages)
      .def("set_pages", [](TP_MLA<KML_MLA_TP<float16_t>>& op, std::vector<std::vector<intptr_t>> nope_pages,
                           std::vector<std::vector<intptr_t>> rope_pages) {
        std::vector<std::vector<void*>> nope_pages_ptr;
        std::vector<std::vector<void*>> rope_pages_ptr;
        op.set_pages(nope_pages_ptr, rope_pages_ptr);
      });

  py::class_<TP_MLA<KML_MLA_TP<float>>, MLA_Interface, std::shared_ptr<TP_MLA<KML_MLA_TP<float>>>>(mla_module,
                                                                                                   "MLA_F32")
      .def(py::init<GeneralMLAConfig>())
      .def("load_weights", &TP_MLA<KML_MLA_TP<float>>::load_weights)
      .def("forward",
           [](TP_MLA<KML_MLA_TP<float>>& op, std::vector<int> qlens, std::vector<std::vector<int>> page_tables,
              std::vector<int> kvlens, intptr_t input,
              intptr_t output) { op.forward(qlens, page_tables, kvlens, (const void*)input, (void*)output); })
      .def("set_local_pages", &TP_MLA<KML_MLA_TP<float>>::set_local_pages)
      .def("set_pages", [](TP_MLA<KML_MLA_TP<float>>& op, std::vector<std::vector<intptr_t>> nope_pages,
                           std::vector<std::vector<intptr_t>> rope_pages) {
        std::vector<std::vector<void*>> nope_pages_ptr;
        std::vector<std::vector<void*>> rope_pages_ptr;
        op.set_pages(nope_pages_ptr, rope_pages_ptr);
      });
  py::class_<TP_MLA<KML_MLA_TP_QUAN<float>>, MLA_Interface, std::shared_ptr<TP_MLA<KML_MLA_TP_QUAN<float>>>>(
      mla_module, "MLA_QUAN_F32")
      .def(py::init<GeneralMLAConfig>())
      .def("load_weights", &TP_MLA<KML_MLA_TP_QUAN<float>>::load_weights)
      .def("forward",
           [](TP_MLA<KML_MLA_TP_QUAN<float>>& op, std::vector<int> qlens, std::vector<std::vector<int>> page_tables,
              std::vector<int> kvlens, intptr_t input,
              intptr_t output) { op.forward(qlens, page_tables, kvlens, (const void*)input, (void*)output); })
      .def("set_local_pages", &TP_MLA<KML_MLA_TP_QUAN<float>>::set_local_pages)
      .def("set_pages", [](TP_MLA<KML_MLA_TP_QUAN<float>>& op, std::vector<std::vector<intptr_t>> nope_pages,
                           std::vector<std::vector<intptr_t>> rope_pages) {
        std::vector<std::vector<void*>> nope_pages_ptr;
        std::vector<std::vector<void*>> rope_pages_ptr;
        op.set_pages(nope_pages_ptr, rope_pages_ptr);
      });

  auto gate_module = m.def_submodule("gate");
  py::class_<GeneralGateConfig>(gate_module, "GateConfig")
      .def(py::init([](int hidden_size, int num_experts_per_tok, int n_routed_experts, int n_group, int topk_group) {
        return GeneralGateConfig(hidden_size, num_experts_per_tok, n_routed_experts, n_group, topk_group);
      }))
      .def_readwrite("routed_scaling_factor", &GeneralGateConfig::routed_scaling_factor)

      .def_readwrite("layer_idx", &GeneralGateConfig::layer_idx)
      .def_readwrite("pool", &GeneralGateConfig::pool)
      .DEF_PTR_PROPERTY(GeneralGateConfig, weight)
      .def_readwrite("weight_type", &GeneralGateConfig::weight_type)
      .DEF_PTR_PROPERTY(GeneralGateConfig, e_score_correction_bias)
      .def_readwrite("e_score_correction_bias_type", &GeneralGateConfig::e_score_correction_bias_type)

      ;
  py::class_<MoEGate, std::shared_ptr<MoEGate>>(gate_module, "MoEGate")
      .def(py::init<GeneralGateConfig>())
      .def("forward", &MoEGate::forward_binding);
#endif

  py::class_<QuantConfig>(m, "QuantConfig")
      .def(py::init<>())
      .def_readwrite("quant_method", &QuantConfig::quant_method)
      .def_readwrite("bits", &QuantConfig::bits)
      .def_readwrite("group_size", &QuantConfig::group_size)
      .def_readwrite("zero_point", &QuantConfig::zero_point);

  auto moe_module = m.def_submodule("moe");

  py::class_<GeneralMOEConfig>(moe_module, "MOEConfig")
      .def(py::init([](int expert_num, int routed_expert_num, int hidden_size, int intermediate_size) {
        return GeneralMOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size);
      }))
      .def(py::init(
          [](int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int num_gpu_experts) {
            GeneralMOEConfig cfg(expert_num, routed_expert_num, hidden_size, intermediate_size);
            cfg.num_gpu_experts = num_gpu_experts;
            return cfg;
          }))
      .def_readwrite("layer_idx", &GeneralMOEConfig::layer_idx)
      .def_readwrite("pool", &GeneralMOEConfig::pool)

      .def_readwrite("num_gpu_experts", &GeneralMOEConfig::num_gpu_experts)
      .DEF_PTR_PROPERTY(GeneralMOEConfig, physical_to_logical_map)

      .DEF_PTR_PROPERTY(GeneralMOEConfig, gate_proj)
      .DEF_PTR_PROPERTY(GeneralMOEConfig, up_proj)
      .DEF_PTR_PROPERTY(GeneralMOEConfig, down_proj)

      .DEF_PTR_PROPERTY(GeneralMOEConfig, gate_scale)
      .DEF_PTR_PROPERTY(GeneralMOEConfig, up_scale)
      .DEF_PTR_PROPERTY(GeneralMOEConfig, down_scale)

      .DEF_PTR_PROPERTY(GeneralMOEConfig, gate_zero)
      .DEF_PTR_PROPERTY(GeneralMOEConfig, up_zero)
      .DEF_PTR_PROPERTY(GeneralMOEConfig, down_zero)

      .def_readwrite("quant_config", &GeneralMOEConfig::quant_config)

      .def_readwrite("max_len", &GeneralMOEConfig::max_len)

      .DEF_PTR_2D_PROPERTY(GeneralMOEConfig, gate_projs)
      .DEF_PTR_2D_PROPERTY(GeneralMOEConfig, up_projs)
      .DEF_PTR_2D_PROPERTY(GeneralMOEConfig, down_projs)

      .DEF_PTR_2D_PROPERTY(GeneralMOEConfig, gate_scales)
      .DEF_PTR_2D_PROPERTY(GeneralMOEConfig, up_scales)
      .DEF_PTR_2D_PROPERTY(GeneralMOEConfig, down_scales)

      .DEF_PTR_2D_PROPERTY(GeneralMOEConfig, gate_zeros)
      .DEF_PTR_2D_PROPERTY(GeneralMOEConfig, up_zeros)
      .DEF_PTR_2D_PROPERTY(GeneralMOEConfig, down_zeros)

      .def_readwrite("path", &GeneralMOEConfig::path)
      .def_readwrite("save", &GeneralMOEConfig::save)
      .def_readwrite("load", &GeneralMOEConfig::load)
      .def_readwrite("m_block", &GeneralMOEConfig::m_block)
      .def_readwrite("group_min_len", &GeneralMOEConfig::group_min_len)
      .def_readwrite("group_max_len", &GeneralMOEConfig::group_max_len)

      .def_readwrite("gate_type", &GeneralMOEConfig::gate_type)
      .def_readwrite("up_type", &GeneralMOEConfig::up_type)
      .def_readwrite("down_type", &GeneralMOEConfig::down_type)
      .def_readwrite("hidden_type", &GeneralMOEConfig::hidden_type)

      ;

  py::class_<MoE_Interface, std::shared_ptr<MoE_Interface>>(moe_module, "MoE_Interface");

  bind_moe_module<LLAMA_MOE_TP>(moe_module, "MOE");

#if defined(__x86_64__) && defined(USE_AMX_AVX_KERNEL)
  bind_moe_module<AMX_MOE_TP<amx::GemmKernel224BF>>(moe_module, "AMXBF16_MOE");
  bind_moe_module<AMX_MOE_TP<amx::GemmKernel224Int8>>(moe_module, "AMXInt8_MOE");
  bind_moe_module<AMX_MOE_TP<amx::GemmKernel224Int4>>(moe_module, "AMXInt4_MOE");
  bind_moe_module<AMX_MOE_TP<amx::GemmKernel224Int4_1>>(moe_module, "AMXInt4_1_MOE");
  bind_moe_module<AMX_AWQ_MOE_TP<amx::GemmKernel224Int4_1_LowKGroup>>(moe_module, "AMXInt4_1KGroup_MOE");
  bind_moe_module<AMX_K2_MOE_TP<amx::GemmKernel224Int4SmallKGroup>>(moe_module, "AMXInt4_KGroup_MOE");
#endif
#if defined(USE_MOE_KERNEL)
  bind_moe_module<MOE_KERNEL_TP<moe_kernel::GemmKernelInt8, _is_plain_>>(moe_module, "Int8_KERNEL_MOE");
#if defined(__aarch64__) && defined(CPU_USE_KML)
  // amd have not implemented int4 kernel yet
  bind_moe_module<MOE_KERNEL_TP<moe_kernel::GemmKernelInt4, _is_plain_>>(moe_module, "Int4_KERNEL_MOE");
#endif
#endif

  // Expose kernel tiling/runtime parameters so Python can modify them at runtime
  {
    auto tiling_module = moe_module.def_submodule("tiling");
#if defined(USE_MOE_KERNEL)
    tiling_module.def(
        "get_int8",
        []() {
          auto t = moe_kernel::GemmKernelInt8::get_tiling();
          py::dict d;
          d["n_block_up_gate"] = std::get<0>(t);
          d["n_block_down"] = std::get<1>(t);
          d["n_block"] = std::get<2>(t);
          d["m_block"] = std::get<3>(t);
          d["k_block"] = std::get<4>(t);
          d["n_block_up_gate_prefi"] = std::get<5>(t);
          d["n_block_down_prefi"] = std::get<6>(t);
          return d;
        },
        "Get current tiling parameters for INT8 kernel");
    tiling_module.def(
        "set_int8",
        [](int n_block_up_gate, int n_block_down, int n_block, int m_block, int k_block, int n_block_up_gate_prefi,
           int n_block_down_prefi) {
          moe_kernel::GemmKernelInt8::set_tiling(n_block_up_gate, n_block_down, n_block, m_block, k_block,
                                                 n_block_up_gate_prefi, n_block_down_prefi);
        },
        py::arg("n_block_up_gate"), py::arg("n_block_down"), py::arg("n_block"), py::arg("m_block"), py::arg("k_block"),
        py::arg("n_block_up_gate_prefi"), py::arg("n_block_down_prefi"), "Set tiling parameters for INT8 kernel");

    tiling_module.def(
        "get_int4",
        []() {
          auto t = moe_kernel::GemmKernelInt4::get_tiling();
          py::dict d;
          d["n_block_up_gate"] = std::get<0>(t);
          d["n_block_down"] = std::get<1>(t);
          d["n_block"] = std::get<2>(t);
          d["m_block"] = std::get<3>(t);
          d["k_block"] = std::get<4>(t);
          d["n_block_up_gate_prefi"] = std::get<5>(t);
          d["n_block_down_prefi"] = std::get<6>(t);
          return d;
        },
        "Get current tiling parameters for INT4 kernel");
    tiling_module.def(
        "set_int4",
        [](int n_block_up_gate, int n_block_down, int n_block, int m_block, int k_block, int n_block_up_gate_prefi,
           int n_block_down_prefi) {
          moe_kernel::GemmKernelInt4::set_tiling(n_block_up_gate, n_block_down, n_block, m_block, k_block,
                                                 n_block_up_gate_prefi, n_block_down_prefi);
        },
        py::arg("n_block_up_gate"), py::arg("n_block_down"), py::arg("n_block"), py::arg("m_block"), py::arg("k_block"),
        py::arg("n_block_up_gate_prefi"), py::arg("n_block_down_prefi"), "Set tiling parameters for INT4 kernel");

    // Convenience: set both
    tiling_module.def(
        "set_all",
        [](int n_block_up_gate, int n_block_down, int n_block, int m_block, int k_block, int n_block_up_gate_prefi,
           int n_block_down_prefi) {
          moe_kernel::GemmKernelInt8::set_tiling(n_block_up_gate, n_block_down, n_block, m_block, k_block,
                                                 n_block_up_gate_prefi, n_block_down_prefi);
          moe_kernel::GemmKernelInt4::set_tiling(n_block_up_gate, n_block_down, n_block, m_block, k_block,
                                                 n_block_up_gate_prefi, n_block_down_prefi);
        },
        py::arg("n_block_up_gate"), py::arg("n_block_down"), py::arg("n_block"), py::arg("m_block"), py::arg("k_block"),
        py::arg("n_block_up_gate_prefi"), py::arg("n_block_down_prefi"),
        "Set tiling parameters for both INT8 and INT4 kernels");
#endif
  }

  auto kvcache_module = m.def_submodule("kvcache");

  py::enum_<AnchorType>(kvcache_module, "AnchorType")
      .value("FIXED", AnchorType::FIXED_ANCHOR)
      .value("DYNAMIC", AnchorType::DYNAMIC)
      .value("QUEST", AnchorType::QUEST)
      .value("BLOCK_MAX", AnchorType::BLOCK_MAX)
      .value("BLOCK_MEAN", AnchorType::BLOCK_MEAN);
  py::enum_<ggml_type>(kvcache_module, "ggml_type")
      // .value("FP16", ggml_type::GGML_TYPE_F16)
      // .value("FP32", ggml_type::GGML_TYPE_F32)
      // .value("Q4_0", ggml_type::GGML_TYPE_Q4_0)
      // .value("Q8_0", ggml_type::GGML_TYPE_Q8_0)
      .value("FP32", GGML_TYPE_F32)
      .value("FP16", GGML_TYPE_F16)
      .value("Q4_0", GGML_TYPE_Q4_0)
      .value("Q4_1", GGML_TYPE_Q4_1)
      .value("Q5_0", GGML_TYPE_Q5_0)
      .value("Q5_1", GGML_TYPE_Q5_1)
      .value("Q8_0", GGML_TYPE_Q8_0)
      .value("Q8_1", GGML_TYPE_Q8_1)
      .value("Q2_K", GGML_TYPE_Q2_K)
      .value("Q3_K", GGML_TYPE_Q3_K)
      .value("Q4_K", GGML_TYPE_Q4_K)
      .value("Q5_K", GGML_TYPE_Q5_K)
      .value("Q6_K", GGML_TYPE_Q6_K)
      .value("Q8_K", GGML_TYPE_Q8_K)
      .value("IQ2_XXS", GGML_TYPE_IQ2_XXS)
      .value("IQ2_XS", GGML_TYPE_IQ2_XS)
      .value("IQ3_XXS", GGML_TYPE_IQ3_XXS)
      .value("IQ1_S", GGML_TYPE_IQ1_S)
      .value("IQ4_NL", GGML_TYPE_IQ4_NL)
      .value("IQ3_S", GGML_TYPE_IQ3_S)
      .value("IQ2_S", GGML_TYPE_IQ2_S)
      .value("IQ4_XS", GGML_TYPE_IQ4_XS)
      .value("I8", GGML_TYPE_I8)
      .value("I16", GGML_TYPE_I16)
      .value("I32", GGML_TYPE_I32)
      .value("I64", GGML_TYPE_I64)
      .value("F64", GGML_TYPE_F64)
      .value("IQ1_M", GGML_TYPE_IQ1_M)
      .value("BF16", GGML_TYPE_BF16)
      .export_values();

  py::enum_<RetrievalType>(kvcache_module, "RetrievalType")
      .value("LAYER", RetrievalType::LAYER)
      .value("KVHEAD", RetrievalType::KVHEAD)
      .value("QHEAD", RetrievalType::QHEAD);

  py::class_<KVCacheConfig>(kvcache_module, "KVCacheConfig")
      .def(py::init<int, int, int, int, int, int, AnchorType, ggml_type, RetrievalType, int, int, int, int, int, int>())
      .def_readwrite("layer_num", &KVCacheConfig::layer_num)
      .def_readwrite("kv_head_num", &KVCacheConfig::kv_head_num)
      .def_readwrite("q_head_num", &KVCacheConfig::q_head_num)
      .def_readwrite("head_dim", &KVCacheConfig::head_dim)
      .def_readwrite("block_len", &KVCacheConfig::block_len)
      .def_readwrite("anchor_num", &KVCacheConfig::anchor_num)
      .def_readwrite("anchor_type", &KVCacheConfig::anchor_type)
      .def_readwrite("kv_type", &KVCacheConfig::kv_type)
      .def_readwrite("retrieval_type", &KVCacheConfig::retrieval_type)
      .def_readwrite("layer_step", &KVCacheConfig::layer_step)
      .def_readwrite("token_step", &KVCacheConfig::token_step)
      .def_readwrite("layer_offset", &KVCacheConfig::layer_offset)
      .def_readwrite("max_block_num", &KVCacheConfig::max_block_num)
      .def_readwrite("max_batch_size", &KVCacheConfig::max_batch_size)
      .def_readwrite("max_thread_num", &KVCacheConfig::max_thread_num);
  py::class_<KVCache>(kvcache_module, "KVCache")
      .def(py::init<KVCacheConfig>())
      .def("get_cache_total_len", &KVCache::get_cache_total_len)
      .def("update_cache_total_len",
           [](KVCache& kvcache, int cache_total_len) { kvcache.update_cache_total_len(cache_total_len); });

  auto utils = m.def_submodule("utils");

  // 注册转换函数
  utils.def("to_float", &to_float_ptr, "Convert tensor from any GGML type to float32", py::arg("input"),
            py::arg("size"), py::arg("type"));

  utils.def("from_float", &from_float_ptr, "Convert tensor from float32 to any GGML type", py::arg("input"),
            py::arg("size"), py::arg("type"));
}
