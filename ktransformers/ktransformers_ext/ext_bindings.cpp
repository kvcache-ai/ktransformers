/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 10:39:37
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
// Python bindings
#include <cstdint>
#include <iostream>
#include <memory>
#include "cpu_backend/cpuinfer.h"
#include "device_launch_parameters.h"
#include "llamafile/flags.h"
#include "operators/llamafile/linear.h"
#include "operators/llamafile/mlp.h"
#include "operators/llamafile/moe.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace pybind11::literals;

class LinearBindings {
   public:
    class WarmUpBindinds {
       public:
        struct Args {
            CPUInfer* cpuinfer;
            Linear* linear;
        };
        static void inner(void* args) {
            Args* args_ = (Args*)args;
            args_->cpuinfer->enqueue(&Linear::warm_up, args_->linear);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(Linear& linear) {
            Args* args = new Args{nullptr, &linear};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
       public:
        struct Args {
            CPUInfer* cpuinfer;
            Linear* linear;
            int qlen;
            const void* input;
            void* output;
        };
        static void inner(void* args) {
            Args* args_ = (Args*)args;
            args_->cpuinfer->enqueue(&Linear::forward, args_->linear, args_->qlen, args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(Linear& linear, int qlen, intptr_t input, intptr_t output) {
            Args* args = new Args{nullptr, &linear, qlen, (const void*)input, (void*)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

class MLPBindings {
   public:
    class WarmUpBindinds {
       public:
        struct Args {
            CPUInfer* cpuinfer;
            MLP* mlp;
        };
        static void inner(void* args) {
            Args* args_ = (Args*)args;
            args_->cpuinfer->enqueue(&MLP::warm_up, args_->mlp);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(MLP& mlp) {
            Args* args = new Args{nullptr, &mlp};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
       public:
        struct Args {
            CPUInfer* cpuinfer;
            MLP* mlp;
            int qlen;
            const void* input;
            void* output;
        };
        static void inner(void* args) {
            Args* args_ = (Args*)args;
            args_->cpuinfer->enqueue(&MLP::forward, args_->mlp, args_->qlen, args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(MLP& mlp, int qlen, intptr_t input, intptr_t output) {
            Args* args = new Args{nullptr, &mlp, qlen, (const void*)input, (void*)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

class MOEBindings {
   public:
    class WarmUpBindinds {
       public:
        struct Args {
            CPUInfer* cpuinfer;
            MOE* moe;
        };
        static void inner(void* args) {
            Args* args_ = (Args*)args;
            args_->cpuinfer->enqueue(&MOE::warm_up, args_->moe);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(MOE& moe) {
            Args* args = new Args{nullptr, &moe};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
       public:
        struct Args {
            CPUInfer* cpuinfer;
            MOE* moe;
            int qlen;
            int k;
            const uint64_t* expert_ids;
            const float* weights;
            const void* input;
            void* output;
        };
        static void inner(void* args) {
            Args* args_ = (Args*)args;
            args_->cpuinfer->enqueue(&MOE::forward, args_->moe, args_->qlen, args_->k, args_->expert_ids, args_->weights, args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(MOE& moe, int qlen, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output) {
            Args* args = new Args{nullptr, &moe, qlen, k, (const uint64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

PYBIND11_MODULE(cpuinfer_ext, m) {
    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<int>())
        .def("submit", &CPUInfer::submit)
        .def("submit_with_cuda_stream", &CPUInfer::submit_with_cuda_stream)
        .def("sync", &CPUInfer::sync)
        .def("sync_with_cuda_stream", &CPUInfer::sync_with_cuda_stream);

    auto linear_module = m.def_submodule("linear");
    py::class_<LinearConfig>(linear_module, "LinearConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride, int group_max_len, intptr_t proj, int proj_type, int hidden_type) {
            return LinearConfig(hidden_size, intermediate_size, stride, group_max_len, (void*)proj, (ggml_type)proj_type, (ggml_type)hidden_type);
        }));
    py::class_<Linear>(linear_module, "Linear")
        .def(py::init<LinearConfig>())
        .def("warm_up", &LinearBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &LinearBindings::ForwardBindings::cpuinfer_interface);

    auto mlp_module = m.def_submodule("mlp");
    py::class_<MLPConfig>(mlp_module, "MLPConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride, int group_max_len, intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj, int gate_type, int up_type, int down_type, int hidden_type) {
            return MLPConfig(hidden_size, intermediate_size, stride, group_max_len, (void*)gate_proj, (void*)up_proj, (void*)down_proj, (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type, (ggml_type)hidden_type);
        }));
    py::class_<MLP>(mlp_module, "MLP")
        .def(py::init<MLPConfig>())
        .def("warm_up", &MLPBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &MLPBindings::ForwardBindings::cpuinfer_interface);

    auto moe_module = m.def_submodule("moe");
    py::class_<MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj, int gate_type, int up_type, int down_type, int hidden_type) {
            return MOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size, stride, group_min_len, group_max_len, (void*)gate_proj, (void*)up_proj, (void*)down_proj, (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type, (ggml_type)hidden_type);
        }));
    py::class_<MOE>(moe_module, "MOE")
        .def(py::init<MOEConfig>())
        .def("warm_up", &MOEBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &MOEBindings::ForwardBindings::cpuinfer_interface);
}
