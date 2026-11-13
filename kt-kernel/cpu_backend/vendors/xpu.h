#pragma once

// ============================================================================
// Intel XPU vendor adapter for cpu_backend.
//
// Unlike vendors/hip.h, musa.h and ascend_npu.h this header does not alias the
// CUDA runtime API: torch.xpu streams are in-order sycl::queue objects and the
// bridge that emulates cudaLaunchHostFunc on them lives in
// cpu_backend/xpu_stream_bridge.cpp, the only translation unit of the build
// that includes SYCL / Level Zero headers (and the only one that needs their
// include paths). This header must therefore stay free of SYCL and Level Zero
// includes so that cpuinfer.h and ext_bindings.cpp keep compiling with the
// plain host compiler.
//
// Semantics, environment knobs (KT_XPU_BRIDGE, KT_XPU_BRIDGE_EVENTS) and the
// differences from CUDA are documented at the top of xpu_stream_bridge.cpp.
// ============================================================================

#include <functional>

namespace kt::xpu {

// queue is a sycl::queue* (torch.xpu.Stream.sycl_queue). cudaLaunchHostFunc semantics: fn(arg) runs after all
// work already enqueued on the queue; work enqueued afterwards does not wait for it (l0 mode) / waits for it
// (host_task mode) — callers must not rely on either.
void launch_host_func(void* queue, void (*fn)(void*), void* arg);

// Blocks the queue until drain() returns. drain runs after all previously enqueued work on the queue and after
// every earlier launch_host_func on the same queue.
void launch_sync(void* queue, std::function<void()> drain);

const char* bridge_mode();     // configured default: "l0" or "host_task" (recording queues always use host_task)
unsigned int bridge_events();  // configured Level Zero event ring size per queue (l0 mode)
void shutdown();               // stop workers, release Level Zero objects, forget cached config and errors; idempotent

}  // namespace kt::xpu
