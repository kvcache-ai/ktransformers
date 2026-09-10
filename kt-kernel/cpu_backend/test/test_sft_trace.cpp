// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <type_traits>

#include "../../operators/sft_trace.hpp"

static_assert(!std::is_copy_constructible_v<sft::TraceScope>);
static_assert(!std::is_move_constructible_v<sft::TraceScope>);
static_assert(std::is_nothrow_destructible_v<sft::TraceScope>);

int main() {
  sft::TraceScope outer("test.outer", 7);
  { sft::TraceScope inner("test.inner", 7, 2); }
  std::printf("PASS SFT trace scope (NVTX supported=%d)\n", sft::TraceScope::supported());
}
