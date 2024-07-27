// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/iqk_mul_mat_arm82.cpp
// Copyrigth 2024 Iwan Kawrakow.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

#ifdef __aarch64__
#define iqk_mul_mat iqk_mul_mat_arm82
#define iqk_mul_mat_moe iqk_mul_mat_moe_arm82
#include "iqk_mul_mat.inc"
#endif  // __aarch64__
