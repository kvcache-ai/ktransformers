// Install the dense NVFP4 head GEMM into the kt_kernel_ext extension.
//
// This is deliberately NOT routed through CPUInfer/WorkerPool the way the MoE
// kernels are. Those kernels are task-scheduled because they fan work across a
// persistent thread pool per expert; a dense head GEMM is a single blocking
// (m,n,k) call issued once per decode step, so a plain synchronous binding is
// both simpler and adequate. If profiling later shows the head dominating, the
// natural next step is to tile it through WorkerPool like the MoE path.
//
// Signature mirrors the MoE kernels' pointer style so the Python side can pass
// raw addresses from tensors it already holds, avoiding a copy of the packed
// weight (127 MB for the Qwen3.6 head).

#include "nvfp4-head-gemm.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void install_dense_nvfp4_head(py::module_& linear_module) {
  linear_module.def(
      "dense_nvfp4_head_forward",
      [](intptr_t weight_ptr, intptr_t scale_ptr, float weight_scale_2,
         intptr_t x_ptr, intptr_t out_ptr, int m, int n, int k, int block_n) {
        kt::nvfp4_head::DenseNVFP4HeadConfig cfg;
        cfg.m = m;
        cfg.n = n;
        cfg.k = k;
        cfg.block_n = block_n > 0 ? block_n : 4096;
        cfg.weight = reinterpret_cast<const uint8_t*>(weight_ptr);
        cfg.weight_scale = reinterpret_cast<const float*>(scale_ptr);
        cfg.weight_scale_2 = weight_scale_2;

        const float* x = reinterpret_cast<const float*>(x_ptr);
        float* out = reinterpret_cast<float*>(out_ptr);
        kt::nvfp4_head::dense_nvfp4_head_gemm(cfg, x, k, out, n);
      },
      py::arg("weight_ptr"), py::arg("scale_ptr"), py::arg("weight_scale_2"),
      py::arg("x_ptr"), py::arg("out_ptr"), py::arg("m"), py::arg("n"),
      py::arg("k"), py::arg("block_n") = 4096,
      "Dense NVFP4 GEMM: out[m,n] = sum_k x[m,k] * W[n,k].\n\n"
      "W is the checkpoint's packed form -- uint8 (n, k/2), two E2M1 nibbles per\n"
      "byte low-first -- with one float scale per group of 16 along k, and a\n"
      "per-tensor weight_scale_2. Both the packed weight and the scales are read\n"
      "through the supplied pointers, so no copy of the weight is made.");

  linear_module.def(
      "dense_nvfp4_head_dequant_row",
      [](intptr_t weight_ptr, intptr_t scale_ptr, float weight_scale_2, int row,
         intptr_t out_ptr, int k) {
        const uint8_t* packed =
            reinterpret_cast<const uint8_t*>(weight_ptr) + static_cast<size_t>(row) * (k / 2);
        const float* scale =
            reinterpret_cast<const float*>(scale_ptr) + static_cast<size_t>(row) * (k / 16);
        kt::nvfp4_head::dequant_row_scalar(packed, scale, k, weight_scale_2,
                                           reinterpret_cast<float*>(out_ptr));
      },
      py::arg("weight_ptr"), py::arg("scale_ptr"), py::arg("weight_scale_2"),
      py::arg("row"), py::arg("out_ptr"), py::arg("k"),
      "Dequantize one row of the packed head into a float buffer. Exposed so the\n"
      "vectorized path can be checked against the definition of record from Python.");
}

}  // namespace
