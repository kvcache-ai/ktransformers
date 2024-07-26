


#include <cuda_fp16.h>


__device__ float ggml_compute_fp16_to_fp32(uint16_t h) {
    return __uint2float_rd(h);
}

static inline float ggml_compute_fp16_to_fp32(uint16_t h) {
    uint16_t tmp;
    memcpy(&tmp, &h, sizeof(ggml_fp16_t));
    return (float)tmp;
}

// define the global table for fp16 to fp32 conversion
__device__ float ggml_table_f32_f16[1 << 16];

// CUDA Kernel to init the table
__global__ void init_fp16_to_fp32_table() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (auto blk_id = idx; blk_id<(1 << 16); blk_id+=blockDim.x * gridDim.x){
        ggml_table_f32_f16[blk_id] = GGML_COMPUTE_FP16_TO_FP32(blk_id);
    }
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

extern __device__ float ggml_table_f32_f16[1 << 16]; // Declare as __device__ if used within device code

// This version of the function is designed to be called from within a CUDA kernel
#if !defined(GGML_FP16_TO_FP32)
__device__ float ggml_lookup_fp16_to_fp32(uint16_t f) {
    return ggml_table_f32_f16[f];
}

#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#endif