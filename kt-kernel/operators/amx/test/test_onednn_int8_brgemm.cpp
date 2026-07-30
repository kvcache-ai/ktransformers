#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../la/amx_kernels.hpp"

static std::vector<int32_t> reference_compensation(amx::GemmKernel224Int8::BufferB& buffer) {
  using Kernel = amx::GemmKernel224Int8;
  std::vector<int32_t> result(buffer.n, 0);
  for (int tile_n = 0; tile_n < buffer.n; tile_n += Kernel::N_STEP) {
    for (int block_k = 0; block_k < buffer.k; block_k += Kernel::K_BLOCK) {
      const int block_size = std::min(Kernel::K_BLOCK, buffer.k - block_k);
      for (int tile_k = 0; tile_k < block_size; tile_k += Kernel::K_STEP) {
        const int8_t* tile = buffer.get_submat(buffer.n, buffer.k, tile_n, block_k + tile_k);
        for (int half = 0; half < 2; ++half) {
          const int8_t* panel = tile + half * Kernel::TILE_N * Kernel::K_STEP;
          for (int group_k = 0; group_k < Kernel::K_STEP / Kernel::VNNI_BLK; ++group_k) {
            for (int column = 0; column < Kernel::TILE_N; ++column) {
              const int8_t* values = panel + (group_k * Kernel::TILE_N + column) * Kernel::VNNI_BLK;
              for (int byte = 0; byte < Kernel::VNNI_BLK; ++byte) {
                result[tile_n + half * Kernel::TILE_N + column] -= 128 * values[byte];
              }
            }
          }
        }
      }
    }
  }
  return result;
}

static int test_repack_compensation() {
  using Kernel = amx::GemmKernel224Int8;
  using BufferB = Kernel::BufferB;
  constexpr int FWD_N = 64;
  constexpr int FWD_K = 128;

  void* forward_memory = std::aligned_alloc(64, BufferB::required_size(FWD_N, FWD_K));
  void* backward_memory = std::aligned_alloc(64, BufferB::required_size(FWD_K, FWD_N));
  if (forward_memory == nullptr || backward_memory == nullptr) return 1;

  BufferB forward(FWD_N, FWD_K, forward_memory);
  BufferB backward(FWD_K, FWD_N, backward_memory);
  std::vector<ggml_bf16_t> source(static_cast<size_t>(FWD_N) * FWD_K);

  auto pack_forward = [&](int salt) {
    for (int row = 0; row < FWD_N; ++row) {
      for (int column = 0; column < FWD_K; ++column) {
        const float value = static_cast<float>(((row * 37 + column * 19 + salt) % 251) - 125) / 32.0f;
        source[static_cast<size_t>(row) * FWD_K + column] = GGML_FP32_TO_BF16(value);
      }
    }
    const int nth = Kernel::recommended_nth(FWD_N);
    for (int ith = 0; ith < nth; ++ith) forward.from_mat(source.data(), ith, nth);
  };

  int failures = 0;
  pack_forward(7);
  backward.repack_from_bb_transposed(forward);
  if (!backward.has_ready_compensation()) ++failures;
  const std::vector<int32_t> expected_first = reference_compensation(backward);
  const int32_t* actual_first = backward.get_onednn_compensation(0);
  if (!std::equal(expected_first.begin(), expected_first.end(), actual_first)) ++failures;
  const std::vector<int32_t> saved_first(actual_first, actual_first + FWD_K);

  pack_forward(83);
  backward.repack_from_bb_transposed(forward);
  if (!backward.has_ready_compensation()) ++failures;
  const std::vector<int32_t> expected_second = reference_compensation(backward);
  const int32_t* actual_second = backward.get_onednn_compensation(0);
  if (!std::equal(expected_second.begin(), expected_second.end(), actual_second)) ++failures;
  if (std::equal(saved_first.begin(), saved_first.end(), actual_second)) ++failures;

  std::free(forward_memory);
  std::free(backward_memory);
  if (failures == 0) std::puts("oneDNN INT8 transposed repack compensation: PASS");
  return failures;
}

int main() {
#if !defined(KTRANSFORMERS_USE_ONEDNN_VNNI)
  return 0;
#else
  setenv("KT_INT8_VNNI_BACKEND", "onednn", 1);
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
  failures += test_repack_compensation();
  return failures == 0 ? 0 : 1;
#endif
}
