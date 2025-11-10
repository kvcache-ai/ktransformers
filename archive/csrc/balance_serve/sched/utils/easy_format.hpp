#include <sstream>
#include <string>
#include <vector>

template <typename T> std::string format_vector(const std::vector<T> &v) {
  std::ostringstream oss;
  if (v.empty())
    return "[]";
  for (size_t i = 0; i < v.size(); ++i) {
    oss << v[i];
    if (i < v.size() - 1)
      oss << ", "; // 逗号分隔
  }
  return oss.str();
}
