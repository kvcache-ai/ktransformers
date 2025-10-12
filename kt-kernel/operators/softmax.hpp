#ifndef CPUINFER_OPERATOR_SOFTMAX_HPP
#define CPUINFER_OPERATOR_SOFTMAX_HPP

#include <algorithm>  // max_element
#include <cmath>      // exp
#include <cstddef>
#ifdef __aarch64__
#include <arm_sve.h>
#endif

#include <type_traits>

template <typename T, typename A>
concept SOFTMAX_APPLIER = requires(T t, A* v, size_t size, size_t count, size_t ld) {
  { T::apply_single(v, size) } -> std::same_as<void>;
  { T::apply_multiple(count, v, size, ld) } -> std::same_as<void>;
};

template <typename A>
class Softmax {
 public:
  /// 对单个向量做 softmax，就地写回
  static void apply_single(A* v, size_t size) {
    static thread_local std::vector<float> v2(100000);
    if (size == 0 || v == nullptr) return;
    if (size > v2.size()) {
      v2.resize(size);
    }

    for (int i = 0; i < size; i++) {
      v2[i] = v[i];
    }

    const float max_val = *std::max_element(v2.begin(), v2.begin() + size);

    float sum = 0;
    for (size_t i = 0; i < size; ++i) {
      v2[i] = std::exp(v2[i] - max_val);
      sum += v2[i];
    }
    if (sum == 0) return;  // 理论上不会发生，但防御一下
    const float inv_sum = 1.0 / sum;
    for (size_t i = 0; i < size; ++i) {
      v[i] = v2[i] * inv_sum;
    }
  }

  static void apply_multiple(size_t count, A* v, size_t size, size_t ld) {
    for (size_t i = 0; i < count; ++i) {
      apply_single(v + i * ld, size);
    }
  }
};

#endif  // CPUINFER_OPERATOR_SOFTMAX_HPP
