// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/bench.h
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#include <stdio.h>

#include "micros.h"

// Architecture-agnostic compiler barrier. The original x86 inline asm
// ("" ::: "memory") only expresses a memory clobber, which the standard
// [atomic_thread_fence] sequence point / asm volatile with memory clobber
// below reproduces on every target, including RISC-V.
#if defined(__GNUC__) || defined(__clang__)
#define KT_BENCH_BARRIER() __asm__ volatile("" ::: "memory")
#else
#define KT_BENCH_BARRIER()                                   \
    do {                                                     \
        static char _kt_barrier_sink;                        \
        asm volatile("" : : "r"(&_kt_barrier_sink) : "memory"); \
    } while (0)
#endif

#define BENCH(x)                                                                       \
    do {                                                                               \
        x;                                                                             \
        KT_BENCH_BARRIER();                                                            \
        long long start = micros();                                                    \
        for (int i = 0; i < ITERATIONS; ++i) {                                         \
            KT_BENCH_BARRIER();                                                        \
            x;                                                                         \
            KT_BENCH_BARRIER();                                                        \
        }                                                                              \
        printf("%9lld us %s\n", (micros() - start + ITERATIONS - 1) / ITERATIONS, #x); \
    } while (0)
