#include <type_traits>

template <typename T, typename U> T div_up(T x, U by) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_integral_v<U>);
  return (x + by - 1) / by;
}