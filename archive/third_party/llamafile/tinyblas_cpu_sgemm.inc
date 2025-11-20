// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/tinyblas_cpu_sgemm.inc
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

//
//
//                                ██████╗ ██╗   █████╗ ██████╗
//         ██████╗██╗██╗ ██╗██═██╗██╔══██╗██║  ██╔══██╗██╔═══╝
//         ╚═██╔═╝██║███▄██║██ ██║██████╔╝██║  ███████║██████╗
//           ██║  ██║██▀███║╚███╔╝██╔══██╗██║  ██╔══██║╔═══██║
//           ██║  ██║██║ ██║ ███║ ██████╔╝████╗██║  ██║██████║
//           ╚═╝  ╚═╝╚═╝ ╚═╝ ╚══╝ ╚═════╝ ╚═══╝╚═╝  ╚═╝╚═════╝
//
//                   BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#if defined(KTRANSFORMERS_USE_NPU) && KTRANSFORMERS_USE_NPU
        // use ARM version
        #include "tinyblas_cpu_sgemm_arm.inc"
#else
        // use x86 version
        #include "tinyblas_cpu_sgemm_x86.inc"
#endif