#include <memory>
#include <type_traits>

template <typename T, typename U>
T div_up(T x, U by) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_integral_v<U>);
  return (x + by - 1) / by;
}

template <typename T>
T* offset_by_bytes(T* t, size_t n) {
  return reinterpret_cast<T*>(reinterpret_cast<size_t>(t) + n);
}
