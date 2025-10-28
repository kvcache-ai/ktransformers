// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/bench.h
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#include <stdio.h>

#include "micros.h"

#define BENCH(x)                                                                       \
    do {                                                                               \
        x;                                                                             \
        __asm__ volatile("" ::: "memory");                                             \
        long long start = micros();                                                    \
        for (int i = 0; i < ITERATIONS; ++i) {                                         \
            __asm__ volatile("" ::: "memory");                                         \
            x;                                                                         \
            __asm__ volatile("" ::: "memory");                                         \
        }                                                                              \
        printf("%9lld us %s\n", (micros() - start + ITERATIONS - 1) / ITERATIONS, #x); \
    } while (0)
