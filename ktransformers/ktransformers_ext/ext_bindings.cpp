/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-07-25 10:34:23
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
// Python bindings
#include <cstdint>
#include <iostream>
#include <memory>
#include "cpu_backend/cpuinfer.h"
#include "cuda_runtime.h"
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

// Binding functions for the Linear class
class LinearBindings {
   public:
    static void bind_forward(CPUInfer& cpuinfer, Linear* linear, py::args args, py::kwargs kwargs) {
        auto input = args[0].cast<intptr_t>();
        auto output = args[1].cast<intptr_t>();
        cpuinfer.submit(&Linear::forward, linear,
                        (const void*)input, (void*)output);
    }

    static void bind_warm_up(CPUInfer& cpuinfer, Linear* linear, py::args args, py::kwargs kwargs) {
        cpuinfer.submit(&Linear::warm_up, linear);
    }

    static void bind_functions(CPUInfer& cpuinfer, py::object func, py::args args, py::kwargs kwargs) {
        auto linear = func.attr("__self__").cast<Linear*>();
        std::string func_name = py::str(func.attr("__func__").attr("__name__"));

        if (func_name == "forward") {
            bind_forward(cpuinfer, linear, args, kwargs);
        } else if (func_name == "warm_up") {
            bind_warm_up(cpuinfer, linear, args, kwargs);
        } else {
            throw py::value_error("Unsupported function: " +
                                  std::string(func_name));
        }
    }
};

// Binding functions for the MLP class
class MLPBindings {
   public:
    static void bind_forward(CPUInfer& cpuinfer, MLP* mlp, py::args args, py::kwargs kwargs) {
        auto input = args[0].cast<intptr_t>();
        auto output = args[1].cast<intptr_t>();
        cpuinfer.submit(&MLP::forward, mlp,
                        (const void*)input, (void*)output);
    }

    static void bind_warm_up(CPUInfer& cpuinfer, MLP* mlp, py::args args, py::kwargs kwargs) {
        cpuinfer.submit(&MLP::warm_up, mlp);
    }

    static void bind_functions(CPUInfer& cpuinfer, py::object func, py::args args, py::kwargs kwargs) {
        auto mlp = func.attr("__self__").cast<MLP*>();
        std::string func_name = py::str(func.attr("__func__").attr("__name__"));

        if (func_name == "forward") {
            bind_forward(cpuinfer, mlp, args, kwargs);
        } else if (func_name == "warm_up") {
            bind_warm_up(cpuinfer, mlp, args, kwargs);
        } else {
            throw py::value_error("Unsupported function: " +
                                  std::string(func_name));
        }
    }
};

// Binding functions for the MOE class
class MOEBindings {
   public:
    static void bind_forward(CPUInfer& cpuinfer, MOE* moe, py::args args, py::kwargs kwargs) {
        int qlen = args[0].cast<int>();
        int k = args[1].cast<int>();
        auto expert_ids = args[2].cast<intptr_t>();
        auto weights = args[3].cast<intptr_t>();
        auto input = args[4].cast<intptr_t>();
        auto output = args[5].cast<intptr_t>();
        cpuinfer.submit(&MOE::forward, moe,
                        qlen, k, (const uint64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output);
    }

    static void bind_warm_up(CPUInfer& cpuinfer, MOE* moe, py::args args, py::kwargs kwargs) {
        cpuinfer.submit(&MOE::warm_up, moe);
    }

    static void bind_functions(CPUInfer& cpuinfer, py::object func, py::args args, py::kwargs kwargs) {
        auto moe = func.attr("__self__").cast<MOE*>();
        std::string func_name = py::str(func.attr("__func__").attr("__name__"));

        if (func_name == "forward") {
            bind_forward(cpuinfer, moe, args, kwargs);
        } else if (func_name == "warm_up") {
            bind_warm_up(cpuinfer, moe, args, kwargs);
        } else {
            throw py::value_error("Unsupported function: " +
                                  std::string(func_name));
        }
    }
};

struct MOEForwardArgs {
    CPUInfer* cpuinfer;
    MOE* moe;
    int qlen;
    int k;
    uint64_t* expert_ids;
    float* weights;
    void* input;
    void* output;
};

void submit_moe_forward_with_host_args_ptr(void* host_args_ptr) {
    MOEForwardArgs* host_args = (MOEForwardArgs*)host_args_ptr;
    host_args->cpuinfer->submit(&MOE::forward, host_args->moe,
                                host_args->qlen, host_args->k, host_args->expert_ids, host_args->weights, host_args->input, host_args->output);
}

void cpuinfer_sync(void* host_args_ptr) {
    CPUInfer* cpuinfer = (CPUInfer*)host_args_ptr;
    cpuinfer->sync();
}

PYBIND11_MODULE(cpuinfer_ext, m) {
    auto linear_module = m.def_submodule("linear");

    py::class_<LinearConfig>(linear_module, "LinearConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride, intptr_t proj, int proj_type, int hidden_type) {
            return LinearConfig(hidden_size, intermediate_size, stride, (void*)proj, (ggml_type)proj_type, (ggml_type)hidden_type);
        }));

    py::class_<Linear>(linear_module, "Linear")
        .def(py::init<LinearConfig>())
        .def("warm_up", [](Linear& linear) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        })
        .def("forward", [](Linear& linear, intptr_t input, intptr_t output) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        });

    auto mlp_module = m.def_submodule("mlp");

    py::class_<MLPConfig>(mlp_module, "MLPConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride, intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj, int gate_type, int up_type, int down_type, int hidden_type) {
            return MLPConfig(hidden_size, intermediate_size, stride, (void*)gate_proj, (void*)up_proj, (void*)down_proj, (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type, (ggml_type)hidden_type);
        }));

    py::class_<MLP>(mlp_module, "MLP")
        .def(py::init<MLPConfig>())
        .def("warm_up", [](MLP& mlp) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        })
        .def("forward", [](MLP& mlp, intptr_t input, intptr_t output) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        });

    auto moe_module = m.def_submodule("moe");

    py::class_<MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj, int gate_type, int up_type, int down_type, int hidden_type) {
            return MOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size, stride, group_min_len, group_max_len, (void*)gate_proj, (void*)up_proj, (void*)down_proj, (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type, (ggml_type)hidden_type);
        }));

    py::class_<MOE>(moe_module, "MOE")
        .def(py::init<MOEConfig>())
        .def("warm_up", [](MOE& moe) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        })
        .def("forward", [](MOE& moe, int k, uint64_t expert_ids, intptr_t weights, intptr_t input, intptr_t output) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        });

    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<int>())
        .def("submit",
             [linear_module, mlp_module, moe_module](CPUInfer& cpuinfer, py::object func, py::args args, py::kwargs kwargs) {
                 if (py::hasattr(func, "__self__") &&
                     py::hasattr(func, "__func__")) {
                     std::string class_name = py::str(func.attr("__self__")
                                                          .attr("__class__")
                                                          .attr("__name__"));
                     if (class_name == "Linear") {
                         LinearBindings::bind_functions(cpuinfer, func,
                                                        args, kwargs);
                     } else if (class_name == "MLP") {
                         MLPBindings::bind_functions(cpuinfer, func,
                                                     args, kwargs);
                     } else if (class_name == "MOE") {
                         MOEBindings::bind_functions(cpuinfer, func,
                                                     args, kwargs);
                     } else {
                         // handle other classes
                         throw py::type_error("Unsupported class type: " +
                                              class_name);
                     }
                 } else {
                     // handle cases where func does not have __self__ or
                     // __func__
                     throw py::type_error(
                         "Invalid function object: missing "
                         "__self__ or __func__ attribute.");
                 }
             })
        .def("submit_with_cuda_stream",
             [linear_module, mlp_module, moe_module](CPUInfer& cpuinfer, intptr_t user_cuda_stream, py::object func, py::args args, py::kwargs kwargs) {
                 if (py::hasattr(func, "__self__") &&
                     py::hasattr(func, "__func__")) {
                     std::string class_name = py::str(func.attr("__self__")
                                                          .attr("__class__")
                                                          .attr("__name__"));
                     if (class_name == "MOE") {
                         std::string func_name = py::str(func.attr("__func__").attr("__name__"));
                         if (func_name == "forward") {
                             auto moe = func.attr("__self__").cast<MOE*>();
                             int qlen = args[0].cast<int>();
                             int k = args[1].cast<int>();
                             auto expert_ids = args[2].cast<intptr_t>();
                             auto weights = args[3].cast<intptr_t>();
                             auto input = args[4].cast<intptr_t>();
                             auto output = args[5].cast<intptr_t>();
                             MOEForwardArgs* moe_forward_args = new MOEForwardArgs{&cpuinfer, moe, qlen, k, (uint64_t*)expert_ids, (float*)weights, (void*)input, (void*)output};
                             // submit_moe_forward_with_host_args_ptr(moe_forward_args);
                             cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)submit_moe_forward_with_host_args_ptr, moe_forward_args);
                         } else {
                             throw py::value_error("Unsupported function: " +
                                                   std::string(func_name));
                         }
                     } else {
                         // handle other classes
                         throw py::type_error("Unsupported class type: " +
                                              class_name);
                     }
                 } else {
                     // handle cases where func does not have __self__ or
                     // __func__
                     throw py::type_error(
                         "Invalid function object: missing "
                         "__self__ or __func__ attribute.");
                 }
             })
        .def("sync_with_cuda_stream", [](CPUInfer& cpuinfer, intptr_t user_cuda_stream) {
            // cpuinfer_sync((void*)(&cpuinfer));
            cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)cpuinfer_sync, (void*)(&cpuinfer));
        })
        .def("sync", &CPUInfer::sync);
}
