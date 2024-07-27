/**
 * @Description  :  
 * @Author       : Azure-Tang
 * @Date         : 2024-07-22 09:27:55
 * @Version      : 1.0.0
 * @LastEditors  : Azure 
 * @LastEditTime : 2024-07-26 08:38:20
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
**/
#pragma once

#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor dequantize_q8_0(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_q6_k(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_q4_k(torch::Tensor data, int blk_size, torch::Device device);