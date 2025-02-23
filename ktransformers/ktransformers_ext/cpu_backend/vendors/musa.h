#pragma once

#include <musa_runtime.h>
#include <musa_bf16.h>

#define cudaLaunchHostFunc musaLaunchHostFunc
#define cudaStream_t musaStream_t
#define cudaHostFn_t musaHostFn_t
#define nv_bfloat16 mt_bfloat16