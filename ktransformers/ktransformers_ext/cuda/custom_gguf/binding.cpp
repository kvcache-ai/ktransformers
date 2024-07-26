#include "ops.h"
// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
// namespace py = pybind11;

int test(){
    return 5;
}

torch::Tensor dequantize_q6_k(torch::Tensor data, int blk_size, torch::Device device);

PYBIND11_MODULE(cudaops, m) {
    m.def("dequantize_q8_0", &dequantize_q8_0, "Function to dequantize q8_0 data.",
          py::arg("data"), py::arg("blk_size"), py::arg("device"));
    m.def("dequantize_q6_k", &dequantize_q6_k, "Function to dequantize q6_k data.",
          py::arg("data"), py::arg("blk_size"), py::arg("device"));
    m.def("dequantize_q4_k",  &dequantize_q4_k, "Function to dequantize q4_k data.",
          py::arg("data"), py::arg("blk_size"), py::arg("device"));
    m.def("test", &test, "Function to test.");
    
}
