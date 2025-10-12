#ifndef CPUINFER_RMS_NORM_HPP
#define CPUINFER_RMS_NORM_HPP

#include <cmath>

template <typename T, typename A>
concept RMS_NORM = requires(T t, int size, int hidden_size, int qlen, A* weights, A* input) {
  { T::rms_norm(hidden_size, qlen, input) } -> std::same_as<void>;
  { T::rms_norm_single(size, input) } -> std::same_as<void>;
  { T::rms_norm_with_weights(hidden_size, qlen, weights, input) } -> std::same_as<void>;
  { T::rms_norm_single_with_weights(size, weights, input) } -> std::same_as<void>;
};

template <typename A>
struct RMSNorm {
  static void rms_norm_single(int size, A* input) {
    const float epsilon = 1e-6;
    float sum = 0;
    for (int i = 0; i < size; i++) {
      sum += (float)input[i] * (float)input[i];
    }
    sum = sqrt(sum / size + epsilon);
    for (int i = 0; i < size; i++) {
      input[i] = (float)input[i] / sum;
    }
  }

  static void rms_norm(int hidden_size, int qlen, A* input) {
    const A epsilon = 1e-6;
    for (int t = 0; t < qlen; t++) {
      rms_norm_single(hidden_size, input + t * hidden_size);
    }
  }

  static void rms_norm_with_weights(int hidden_size, int qlen, A* weights, A* input) {
    const A epsilon = 1e-6;
    for (int t = 0; t < qlen; t++) {
      rms_norm_single_with_weights(hidden_size, input + t * hidden_size);
    }
  }
  static void rms_norm_single_with_weights(int size, A* weights, A* input) {
    const float epsilon = 1e-6;
    float sum = 0;
    for (int i = 0; i < size; i++) {
      sum += (float)input[i] * (float)input[i];
    }
    sum = sqrt(sum / size + epsilon);
    for (int i = 0; i < size; i++) {
      input[i] = (float)weights[i] * (float)input[i] / sum;
    }
  }
};

#endif