#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

void topk_softmax(torch::Tensor& topk_weights,          // [num_tokens, topk]
                  torch::Tensor& topk_indices,          // [num_tokens, topk]
                  torch::Tensor& token_expert_indices,  // [num_tokens, topk]
                  torch::Tensor& gating_output);