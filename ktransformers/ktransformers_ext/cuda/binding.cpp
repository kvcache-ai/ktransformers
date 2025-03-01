/**
 * @Description  :
 * @Author       : Azure-Tang, Boxin Zhang
 * @Date         : 2024-07-25 13:38:30
 * @Version      : 0.2.2
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
**/

#include "custom_gguf/ops.h"
#ifdef KTRANSFORMERS_USE_CUDA
#include "gptq_marlin/ops.h"
#endif
// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
// namespace py = pybind11;

PYBIND11_MODULE(KTransformersOps, m) {

    m.def("dequantize_q8_0", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q8_0((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q8_0 data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q6_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q6_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q6_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q5_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q5_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q5_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q4_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q4_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q4_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q3_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q3_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q3_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q2_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q2_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q2_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_iq4_xs", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_iq4_xs((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize iq4_xs data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

#ifdef KTRANSFORMERS_USE_CUDA
    m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "Function to perform GEMM using Marlin quantization.",
        py::arg("a"), py::arg("b_q_weight"), py::arg("b_scales"), py::arg("g_idx"),
        py::arg("perm"), py::arg("workspace"), py::arg("num_bits"), py::arg("size_m"),
        py::arg("size_n"), py::arg("size_k"), py::arg("is_k_full"));
#endif
}
