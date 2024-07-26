// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/iqk_mul_mat_amd_zen4.cpp
// Copyrigth 2024 Iwan Kawrakow.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

#ifdef __x86_64__
#define iqk_mul_mat iqk_mul_mat_zen4
#define iqk_mul_mat_moe iqk_mul_mat_moe_zen4
#include "iqk_mul_mat.inc"
#endif  // __x86_64__
