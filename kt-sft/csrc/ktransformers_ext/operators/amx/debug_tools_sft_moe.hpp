#ifndef SFT_DEBUG_HPP
#define SFT_DEBUG_HPP

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>

inline std::string get_env_or_default(const char *var_name, const std::string &default_value) {
	const char *value = std::getenv(var_name);
	return (value != nullptr) ? std::string(value) : default_value;
}

/* use example:  
	for (uint64_t expert_idx = 0; expert_idx < (uint64_t)config_.expert_num; ++expert_idx) {
		dump_grad_bin("layer0_E_End"+std::to_string(expert_idx)+"_gate_proj_out_trans_", (uint8_t*)gate_proj_t_ + expert_idx * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type), config_.hidden_size * config_.intermediate_size, config_.grad_type);
		std::cout << "gate_proj_t_:" << static_cast<const void*>((uint8_t*)gate_proj_t_ + expert_idx * config_.hidden_size * config_.intermediate_size) << ", grad_type: " << config_.grad_type << std::endl;
	}
*/
inline void dump_grad_bin(const std::string &file_name,
                          const void       *data,
                          size_t            elem_cnt,
                          ggml_type         dtype,
						  std::streamoff    offset_bytes = 0)
{
    std::string path = get_env_or_default("SFT_DEBUG_PATH","debug") + "/" + file_name;
    switch (dtype) {
        case GGML_TYPE_F32:  path += ".f32";  break;
        case GGML_TYPE_F16:  path += ".f16";  break;
        case GGML_TYPE_BF16: path += ".bf16"; break;
		case GGML_TYPE_I8: path += ".int8"; break;
        default:             path += ".raw";  break;
    }
	std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f.is_open()) {
        std::ofstream tmp(path, std::ios::out | std::ios::binary);
        tmp.close();
        f.open(path, std::ios::in | std::ios::out | std::ios::binary);
    }

    f.seekp(offset_bytes * ggml_type_size(dtype));
	// std::cout << "seekp: " << offset_bytes * ggml_type_size(dtype) << std::endl;

    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(elem_cnt * ggml_type_size(dtype)));
    f.close();
}

// inline void dump_bin(std::string file_name, float16_t *data, size_t count) {
//   file_name = get_env_or_default("SFT_DEBUG_PATH", "debug") + "/" + file_name + ".f16";
//   std::ofstream f(file_name, std::ios::binary);
//   f.write(reinterpret_cast<const char *>(data), count * sizeof(*data));
//   f.close();
// }
inline void dump_bin(std::string file_name, float *data, size_t count) {
	file_name = get_env_or_default("SFT_DEBUG_PATH", "debug") + "/" + file_name + ".f32";
	std::cout << file_name << std::endl;
	std::ofstream f(file_name, std::ios::binary);
	f.write(reinterpret_cast<const char *>(data), count * sizeof(*data));
	f.close();
}
inline void dump_bin(std::string file_name, int64_t *data, size_t count) {
	file_name = get_env_or_default("SFT_DEBUG_PATH", "debug") + "/" + file_name + ".int64";
	std::cout << file_name << std::endl;
	std::ofstream f(file_name, std::ios::binary);
	f.write(reinterpret_cast<const char *>(data), count * sizeof(*data));
	f.close();
}
inline void dump_bin(std::string file_name, uint8_t *data, size_t count) {
	file_name = get_env_or_default("SFT_DEBUG_PATH", "debug") + "/" + file_name + ".uint8";
	std::cout << file_name << std::endl;
	std::ofstream f(file_name, std::ios::binary);
	f.write(reinterpret_cast<const char *>(data), count * sizeof(*data));
	f.close();
}

#endif