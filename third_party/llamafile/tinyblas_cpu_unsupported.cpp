// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/tinyblas_cpu_unsupported.cpp
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sgemm.h"

bool llamafile_sgemm_unsupported(long m, long n, long k, const void* A, long lda, const void* B, long ldb, void* C, long ldc, int ith, int nth, int task, int Atype, int Btype, int Ctype, int precision) {
    return false;
}

bool llamafile_mixmul_unsupported(const struct ggml_compute_params* params,
                                  const struct ggml_tensor* weights,
                                  const struct ggml_tensor* thought,
                                  const struct ggml_tensor* plan,
                                  struct ggml_tensor* result) {
    return false;
}

bool iqk_mul_mat_moe_unsupported(long, long, long, int, int, const void*, const void*, float*, long, long, const void*, int, int) {
    return false;
}
