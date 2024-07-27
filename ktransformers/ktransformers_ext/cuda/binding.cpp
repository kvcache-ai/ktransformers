/**
 * @Description  :  
 * @Author       : Azure-Tang
 * @Date         : 2024-07-25 13:38:30
 * @Version      : 1.0.0
 * @LastEditors  : Azure 
 * @LastEditTime : 2024-07-26 08:36:03
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
**/

#include "custom_gguf/ops.h"
#include "gptq_marlin/ops.h"
// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
// namespace py = pybind11;

PYBIND11_MODULE(KTransformersOps, m) {
      m.def("dequantize_q8_0", &dequantize_q8_0, "Function to dequantize q8_0 data.",
            py::arg("data"), py::arg("blk_size"), py::arg("device"));
      m.def("dequantize_q6_k", &dequantize_q6_k, "Function to dequantize q6_k data.",
            py::arg("data"), py::arg("blk_size"), py::arg("device"));
      m.def("dequantize_q4_k",  &dequantize_q4_k, "Function to dequantize q4_k data.",
            py::arg("data"), py::arg("blk_size"), py::arg("device"));
      m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "Function to perform GEMM using Marlin quantization.",
            py::arg("a"), py::arg("b_q_weight"), py::arg("b_scales"), py::arg("g_idx"),
            py::arg("perm"), py::arg("workspace"), py::arg("num_bits"), py::arg("size_m"),
            py::arg("size_n"), py::arg("size_k"), py::arg("is_k_full"));
}
