#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "../operators/rope.hpp"

std::vector<float> create_random_vector(size_t total_size, std::vector<size_t> shape, unsigned int seed = 0) {
  std::vector<float> vec(total_size);
  std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  // for (size_t i = 0; i < total_size; ++i) {
  //   vec[i] = 1; // dist(gen);
  // }
  for (size_t i = 0; i < shape[0]; ++i) {
    size_t offset_i = i * shape[1] * shape[2] * shape[3];
    for (size_t j = 0; j < shape[1]; ++j) {
      size_t offset_j = j * shape[2] * shape[3];
      for (size_t k = 0; k < shape[2]; ++k) {
        size_t offset_k = k * shape[3];
        for (size_t a = 0; a < shape[3]; ++a) {
          vec[offset_i + offset_j + offset_k + a] = a;
        }
      }
    }
  }
  return vec;
}

void print_vector_to_file(const std::vector<float>& vec, const char* filename) {
  FILE* fp = fopen(filename, "w");
  for (auto x : vec) {
    fprintf(fp, "%.2f ", x);
  }
  fclose(fp);
}

std::pair<std::vector<float>, std::vector<float>> cpp_torch_rope_with_apply_single(
    const std::vector<float>& q_in_const, const std::vector<float>& k_in_const,
    DeepseekV3YarnRotaryEmbedding<float>& rotary_emb, size_t B, size_t H, size_t S, size_t D_rope) {
  rotary_emb.init(S);

  const float* full_cos_cache_ptr = rotary_emb.cos();
  const float* full_sin_cache_ptr = rotary_emb.sin();

  std::vector<float> q_out = q_in_const;
  std::vector<float> k_out = k_in_const;

  size_t stride_head = S * D_rope;
  size_t stride_batch = H * stride_head;

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      float* current_k_head_ptr = k_out.data() + b * stride_batch + h * stride_head;
      Rope<DeepseekV3YarnRotaryEmbedding<float>, float>::apply_multiple(rotary_emb, current_k_head_ptr,
                                                                        static_cast<int>(D_rope), 0, S);
      for (size_t s = 0; s < S; ++s) {
        float* current_q_head_ptr = q_out.data() + b * stride_batch + h * stride_head + s * D_rope;

        Rope<DeepseekV3YarnRotaryEmbedding<float>, float>::apply_single(rotary_emb, current_q_head_ptr,
                                                                        static_cast<int>(D_rope), s);
      }
    }
  }

  return {q_out, k_out};
}

int main() {
  size_t batch_size = 2;
  size_t num_heads = 16;
  size_t seq_len = 32;
  size_t rope_size = 16;
  float theta = 10000.0f;

  float beta_fast_cfg = 32.0f;
  float beta_slow_cfg = 1.0f;
  float factor_cfg = 40.0f;
  float mscale_cfg = 1.0f;
  float mscale_all_dim_cfg = 1.0f;
  size_t original_max_pos_embeddings_cfg = 4096;

  std::cout << "--- Test Parameters ---" << std::endl;
  std::cout << "Batch Size: " << batch_size << std::endl;
  std::cout << "Num Heads: " << num_heads << std::endl;
  std::cout << "Seq Len: " << seq_len << std::endl;
  std::cout << "Rope Size (dim): " << rope_size << std::endl;
  std::cout << "Theta (base): " << theta << std::endl;
  std::cout << "Scaling Factor: " << factor_cfg << std::endl;
  std::cout << "Original Max Pos Embeddings: " << original_max_pos_embeddings_cfg << std::endl;
  std::cout << "-----------------------" << std::endl << std::endl;

  DeepseekV3YarnRotaryEmbedding<float> rotary_emb(rope_size, original_max_pos_embeddings_cfg, theta, factor_cfg,
                                                  original_max_pos_embeddings_cfg, beta_fast_cfg, beta_slow_cfg,
                                                  mscale_cfg, mscale_all_dim_cfg);
  std::cout << "DeepseekV3YarnRotaryEmbedding instantiated." << std::endl;

  size_t total_elements_per_tensor = batch_size * num_heads * seq_len * rope_size;

  unsigned int q_seed = 123;
  unsigned int k_seed = 456;
  std::vector<float> q_pe_vec =
      create_random_vector(total_elements_per_tensor, {batch_size, num_heads, seq_len, rope_size}, q_seed);
  std::vector<float> k_pe_vec =
      create_random_vector(total_elements_per_tensor, {batch_size, num_heads, seq_len, rope_size}, k_seed);

  std::cout << "Input Q_PE and K_PE vectors created. Total elements per tensor: " << total_elements_per_tensor
            << std::endl;

  std::cout << std::endl;

  std::cout << "Applying RoPE using cpp_torch_rope_with_apply_single..." << std::endl;
  auto [q2_vec, k2_vec] =
      cpp_torch_rope_with_apply_single(q_pe_vec, k_pe_vec, rotary_emb, batch_size, num_heads, seq_len, rope_size);
  std::cout << "RoPE application finished." << std::endl << std::endl;

  std::cout << std::endl << "test_rope.cpp finished successfully." << std::endl;

  print_vector_to_file(q2_vec, "q_cpp.out");
  print_vector_to_file(k2_vec, "k_cpp.out");

  return 0;
}