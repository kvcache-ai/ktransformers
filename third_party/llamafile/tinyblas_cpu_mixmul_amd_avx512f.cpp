// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/tinyblas_cpu_mixmul_amd_avx512f.cpp
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

#ifdef __x86_64__
#define llamafile_mixmul llamafile_mixmul_amd_avx512f
#include "tinyblas_cpu_mixmul.inc"
#endif  // __x86_64__
