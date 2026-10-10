#include "bf16_sumsq.h"

#include <bit>
#include <cstdint>
#include <stdexcept>

double bf16_sumsq_ptr(std::uintptr_t data_ptr, std::int64_t numel) {
  if (numel < 0 || (numel != 0 && data_ptr == 0)) {
    throw std::invalid_argument("BF16 sum of squares requires a valid pointer and non-negative length");
  }

  const auto* values = reinterpret_cast<const std::uint16_t*>(data_ptr);
  double total = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total)
  for (std::int64_t index = 0; index < numel; ++index) {
    const auto bits = static_cast<std::uint32_t>(values[index]) << 16;
    const float value = std::bit_cast<float>(bits);
    total += static_cast<double>(value) * static_cast<double>(value);
  }
  return total;
}
