#ifndef __MODEL_CONFIG_HPP_
#define __MODEL_CONFIG_HPP_

#include "nlohmann/json.hpp"
#include <iostream>

#include <filesystem>
#include <fstream>

using DimSize = size_t;
using URL = std::string;
using ModelName = std::string;

// We must assure this can be load by config.json
class ModelConfig {
public:
  DimSize hidden_size;
  DimSize intermediate_size;
  size_t max_position_embeddings;
  std::string model_type;
  size_t num_attention_heads;
  size_t num_hidden_layers;
  size_t num_key_value_heads;
  size_t vocab_size;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ModelConfig, hidden_size, intermediate_size,
                                 max_position_embeddings, model_type,
                                 num_attention_heads, num_hidden_layers,
                                 num_key_value_heads, vocab_size);

  void load_from(std::filesystem::path path) {
    std::cout << "Load from " << path << std::endl;
    std::ifstream i(path);
    nlohmann::json j;
    i >> j;
    *this = j.get<ModelConfig>();
  }
};

using QuantType = std::string;
static const QuantType NoQuantType = "";

class QuantConfig {
public:
  QuantType name;

  // For GEMV
  QuantType type_of_dot_vector = NoQuantType;
  inline bool can_be_used_as_matrix() {
    return type_of_dot_vector != NoQuantType;
  }

  bool can_be_used_as_vector;

  double bytes_per_element;
  bool has_scale;
  bool has_min;

  size_t block_element_count;
  size_t block_element_size;

  URL reference = "";

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(QuantConfig, name,
                                              type_of_dot_vector,
                                              can_be_used_as_vector,
                                              bytes_per_element, has_scale,
                                              has_min, block_element_count,
                                              block_element_size, reference);
};

inline std::map<QuantType, QuantConfig> quant_configs;
inline std::map<ModelName, ModelConfig> model_configs;

inline void load_quant_configs(std::filesystem::path path) {
  nlohmann::json j;
  if (std::filesystem::exists(path)) {
    std::cout << __FUNCTION__ << " from " << path << std::endl;
    std::ifstream i(path);
    i >> j;
    quant_configs = j.get<std::map<QuantType, QuantConfig>>();
    std::cout << "Loaded Quant Configs" << std::endl;
    for (auto &[k, v] : quant_configs) {
      std::cout << " - " << k << std::endl;
    }
  } else {
    std::cout << __FUNCTION__ << " no file at " << path << std::endl;
  }
}

inline void dump_quant_configs(std::filesystem::path path) {
  std::ofstream o(path);
  nlohmann::json j = quant_configs;
  o << j.dump(4);
}

inline void load_model_configs(std::filesystem::path path) {
  nlohmann::json j;
  if (std::filesystem::exists(path)) {
    std::cout << __FUNCTION__ << " from " << path << std::endl;
    std::ifstream i(path);
    i >> j;
    model_configs = j.get<std::map<ModelName, ModelConfig>>();
    std::cout << "Loaded Model Configs" << std::endl;
    for (auto &[k, v] : model_configs) {
      std::cout << " - " << k << std::endl;
    }
  } else {
    std::cout << __FUNCTION__ << " no file at " << path << std::endl;
  }
}

inline void dump_model_configs(std::filesystem::path path) {
  std::ofstream o(path);
  nlohmann::json j = model_configs;
  o << j.dump(4);
}

#endif