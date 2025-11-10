/**
 * @Description  :  
 * @Author       : Azure
 * @Date         : 2024-07-22 09:27:55
 * @Version      : 1.0.0
 * @LastEditors  : Azure 
 * @LastEditTime : 2024-07-26 08:35:00
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
**/
#pragma once

#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits, int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full);

// torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
//                                  int64_t size_k, int64_t size_n,
//                                  int64_t num_bits);