#include <cstdint>
#include <cstdio>
#include <vector>

#include "../la/onednn_int8.hpp"

int main() {
#if !defined(KTRANSFORMERS_USE_ONEDNN_VNNI)
  return 0;
#else
  constexpr int M = 7;
  constexpr int N = 16;
  constexpr int K = 64;
  constexpr int LDC = 32;
  alignas(64) uint8_t a[M * K];
  alignas(64) int8_t b[K * N];
  alignas(64) int8_t packed_b[K * N];
  alignas(64) int32_t c[M * LDC];

  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      const int8_t value = static_cast<int8_t>((m * 31 + k * 17) % 255 - 127);
      a[m * K + k] = static_cast<uint8_t>(static_cast<int>(value) + 128);
    }
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      b[k * N + n] = static_cast<int8_t>((k * 13 + n * 7) % 255 - 127);
    }
  }
  for (int group_k = 0; group_k < K / 4; ++group_k) {
    for (int n = 0; n < N; ++n) {
      for (int byte = 0; byte < 4; ++byte) {
        packed_b[(group_k * N + n) * 4 + byte] = b[(group_k * 4 + byte) * N + n];
      }
    }
  }

  int failures = 0;
  amx::OneDnnInt8Brgemm::execute(M, 1, false, reinterpret_cast<const int8_t*>(a), packed_b, c);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t expected = 0;
      int32_t sum_b = 0;
      for (int k = 0; k < K; ++k) {
        expected += (static_cast<int32_t>(a[m * K + k]) - 128) * static_cast<int32_t>(b[k * N + n]);
        sum_b += b[k * N + n];
      }
      c[m * LDC + n] -= 128 * sum_b;
      if (c[m * LDC + n] != expected) ++failures;
    }
  }

  amx::OneDnnInt8Brgemm::execute(M, 1, true, reinterpret_cast<const int8_t*>(a), packed_b, c);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t expected = 0;
      int32_t sum_b = 0;
      for (int k = 0; k < K; ++k) {
        expected += (static_cast<int32_t>(a[m * K + k]) - 128) * static_cast<int32_t>(b[k * N + n]);
        sum_b += b[k * N + n];
      }
      c[m * LDC + n] -= 128 * sum_b;
      if (c[m * LDC + n] != 2 * expected) ++failures;
    }
  }

  if (failures == 0) std::puts("oneDNN INT8 BRGEMM signed compensation: PASS");
  return failures == 0 ? 0 : 1;
#endif
}
