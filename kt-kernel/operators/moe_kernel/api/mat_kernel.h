#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common.h"

using GemmFn = void (*)(const KERNEL_CBLAS_LAYOUT layout, const KERNEL_CBLAS_TRANSPOSE transa,
                        const KERNEL_CBLAS_TRANSPOSE transb, const KERNEL_CBLAS_OFFSET offsetc, const size_t m,
                        const size_t n, const size_t k, const float alpha, const void* a, const size_t lda,
                        const int8_t oa, const void* b, const size_t ldb, const int8_t ob, const float beta, int32_t* c,
                        const size_t ldc, const int32_t* oc);

struct MatKernelSelection {
  GemmFn fn;
  int divide_elements_size;
};

MatKernelSelection select_kernel_for_int4(MatKernelVariant variant);
MatKernelSelection select_kernel_for_int8(MatKernelVariant variant);

template <typename T>
MatKernelSelection select_mat_kernel(MatKernelVariant variant) {
  if constexpr (std::is_same_v<typename T::dt, int4_2_t>) {
    return select_kernel_for_int4(variant);
  } else {
    return select_kernel_for_int8(variant);
  }
}
