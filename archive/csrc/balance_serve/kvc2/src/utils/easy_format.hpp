#ifndef __EASY_FORMAT_HPP_
#define __EASY_FORMAT_HPP_
#include <array>
#include <iomanip>
#include <sstream>
#include <string>

#include <vector>

template <typename T>
inline std::string format_vector(const std::vector<T>& v) {
  std::ostringstream oss;
  if (v.empty())
    return "[]";
  for (size_t i = 0; i < v.size(); ++i) {
    oss << v[i];
    if (i < v.size() - 1)
      oss << ", ";  // 逗号分隔
  }
  return oss.str();
}

inline std::array<std::string, 7> units = {"", "K", "M", "G", "T", "P", "E"};

inline std::string readable_number(size_t size) {
  size_t unit_index = 0;
  double readable_size = size;
  while (readable_size >= 1000 && unit_index < units.size() - 1) {
    readable_size /= 1000;
    unit_index++;
  }
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2) << readable_size;
  std::string str = ss.str();
  return str + "" + units[unit_index];
}
#endif