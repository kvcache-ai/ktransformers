/**
 * @Description  :  
 * @Author       : Azure-Tang
 * @Date         : 2024-07-22 09:27:55
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-12 03:48:46
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
**/
#pragma once

#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor dequantize_q8_0(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_q6_k(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_q5_k(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_q4_k(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_q3_k(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_q2_k(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_iq4_xs(torch::Tensor data, int blk_size, torch::Device device);
