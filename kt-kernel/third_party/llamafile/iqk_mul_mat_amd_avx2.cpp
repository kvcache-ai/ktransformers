// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/iqk_mul_mat_amd_avx2.cpp
// Copyrigth 2024 Iwan Kawrakow.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

#if defined(__x86_64__) || defined(_M_X64)
#include "iqk_mul_mat.inc"
#endif  // __x86_64__
