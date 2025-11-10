#pragma once
#include <array>
#include <iomanip>
#include <sstream>
#include <string>

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