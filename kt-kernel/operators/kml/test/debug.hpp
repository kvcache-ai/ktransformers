#ifndef KML_DEBUG_HPP
#define KML_DEBUG_HPP

#include <arm_sve.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>

inline std::string get_env_or_default(const char* var_name, const std::string& default_value) {
  const char* value = std::getenv(var_name);
  return (value != nullptr) ? std::string(value) : default_value;
}

inline void dump_bin(std::string file_name, float16_t* data, size_t count) {
  file_name = get_env_or_default("KML_DEBUG_PATH", "debug") + "/" + file_name + ".f16";
  std::ofstream f(file_name, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data), count * sizeof(*data));
  f.close();
}
inline void dump_bin(std::string file_name, float* data, size_t count) {
  file_name = get_env_or_default("KML_DEBUG_PATH", "debug") + "/" + file_name + ".f32";
  std::ofstream f(file_name, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data), count * sizeof(*data));
  f.close();
}
inline void dump_bin(std::string file_name, int64_t* data, size_t count) {
  file_name = get_env_or_default("KML_DEBUG_PATH", "debug") + "/" + file_name + ".int64";
  std::ofstream f(file_name, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data), count * sizeof(*data));
  f.close();
}

inline void dump_bin(std::string file_name, int8_t* data, size_t count) {
  file_name = get_env_or_default("KML_DEBUG_PATH", "debug") + "/" + file_name + ".int8";
  std::ofstream f(file_name, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data), count * sizeof(*data));
  f.close();
}

inline void dump_bin(std::string file_name, int32_t* data, size_t count) {
  file_name = get_env_or_default("KML_DEBUG_PATH", "debug") + "/" + file_name + ".int32";
  std::ofstream f(file_name, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data), count * sizeof(*data));
  f.close();
}

inline void load_bin(std::string file_name, float* data, size_t count) {
  file_name = get_env_or_default("KML_DEBUG_PATH", "debug") + "/" + file_name + ".f32";
  std::ifstream f(file_name, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_name);
  }
  f.read(reinterpret_cast<char*>(data), count * sizeof(*data));
  f.close();
}

#endif
