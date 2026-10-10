#pragma once

#include <cstdint>

// The caller must keep the contiguous CPU BF16 tensor alive for this call.
double bf16_sumsq_ptr(std::uintptr_t data_ptr, std::int64_t numel);
