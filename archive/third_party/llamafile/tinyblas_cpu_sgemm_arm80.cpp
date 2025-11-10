// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/tinyblas_cpu_sgemm_arm80.cpp
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

#ifdef __aarch64__
#define llamafile_sgemm llamafile_sgemm_arm80
#include "tinyblas_cpu_sgemm.inc"
#endif  // __aarch64__
