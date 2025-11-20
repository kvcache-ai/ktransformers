// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/iqk_mul_mat.inc
// Copyrigth 2024 Iwan Kawrakow - Apache 2.0 Licens
// with additions from
// https://github.com/ikawrakow/ik_llama.cpp/blob/main/ggml/src/iqk/iqk_mul_mat.cpp
// Copyrigth 2024-2025 Iwan Kawrakow - MIT Licens
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp fenc=utf-8 :vi
//
// Copyright 2024 Iwan Kawrakow
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
//
//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#if defined(KTRANSFORMERS_USE_NPU) && KTRANSFORMERS_USE_NPU
        // use ARM version
        #include "iqk_mul_mat_arm.inc"
#else
        // use x86 version
        #include "iqk_mul_mat_x86.inc"
#endif