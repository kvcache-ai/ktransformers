// BOOST_STRONG_TYPEDEF(int8_t, int4_2_t);
#pragma once
#include <cstdint>

#include "llama.cpp/ggml.h"
#if !defined(CPUINFER_HAS_FLOAT16_T)
// using float16_t = ggml_fp16_t;
#define CPUINFER_HAS_FLOAT16_T 1
#endif

#if !defined(CPUINFER_HAS_BFLOAT16_T)
// using bfloat16_t = ggml_bf16_t;
#define CPUINFER_HAS_BFLOAT16_T 1
#endif  // CPUINFER_HAS_BFLOAT16_T
const bool PACKED = true;
#if defined(__aarch64__) || defined(__arm__) || defined(CPU_USE_KML)
#ifndef CPU_USE_KML
#define CPU_USE_KML
#endif
#endif  // USE_MOE_KERNEL_AMD or CPU_USE_KML

#define STRONG_TYPEDEF(T, D)                                   \
  struct D {                                                   \
    T t;                                                       \
    explicit D(const T &v) : t(v) {}                           \
    D() = default;                                             \
    D(const D &) = default;                                    \
    D &operator=(const D &) = default;                         \
    D &operator=(const T &rhs) {                               \
      t = rhs;                                                 \
      return *this;                                            \
    }                                                          \
    operator const T &() const { return t; }                   \
    operator T &() { return t; }                               \
    bool operator==(const D &rhs) const { return t == rhs.t; } \
    bool operator!=(const D &rhs) const { return t != rhs.t; } \
    bool operator<(const D &rhs) const { return t < rhs.t; }   \
  };
STRONG_TYPEDEF(int8_t, int4_2_t)
typedef int8_t BLASINT8;

/* matrix transpose or conjugate transpose */
typedef enum KERNEL_CBLAS_TRANSPOSE {
  KernelCblasNoTrans = 111,
  KernelCblasTrans = 112,
  KernelCblasConjTrans = 113,
  KernelCblasConjNoTrans = 114
} KERNEL_CBLAS_TRANSPOSE;
/* matrix stored in rows or cols */
typedef enum KERNEL_CBLAS_ORDER { KernelCblasRowMajor = 101, KernelCblasColMajor = 102 } KERNEL_CBLAS_ORDER;
/* matrix position is left or right */
typedef enum KERNEL_CBLAS_SIDE { KernelCblasLeft = 141, KernelCblasRight = 142 } KERNEL_CBLAS_SIDE;
typedef KERNEL_CBLAS_ORDER KERNEL_CBLAS_LAYOUT;
typedef enum KERNEL_CBLAS_OFFSET {
  KernelCblasRowOffset = 171,
  KernelCblasColOffset = 172,
  KernelCblasFixOffset = 173
} KERNEL_CBLAS_OFFSET;

enum class MatKernelVariant {
  Decode,
  Prefill,
};