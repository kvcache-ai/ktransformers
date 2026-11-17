// kt-kernel <-> llama.cpp compatibility shim (RISC-V build).
// Copyright(c) 2026 by KVCache.AI, All Rights Reserved.
//
// The deployed third_party/llama.cpp tree (pre-patched b3173-lineage, flat
// layout) already declares the task-type enum, ggml_compute_params, the fp16
// lookup table and ggml_internal_get_type_traits; this shim only fills the
// gaps that kt-kernel sources rely on but that tree does not provide.
#ifndef KT_GGML_COMPAT_H
#define KT_GGML_COMPAT_H

#include <stddef.h>
#include <stdint.h>
#include "llama.cpp/ggml.h"
#include "llama.cpp/ggml-impl.h"

// task-type enum / ggml_compute_params / ggml_table_f32_f16 /
// ggml_internal_get_type_traits: all present in this llama.cpp tree.
// quantize_row_q4_0/q8_0: declared in ggml-quants.h with C linkage.

#endif  // KT_GGML_COMPAT_H
