#pragma once

// ============================================================================
// Ascend NPU vendor adapter for cpu_backend.
//
// Provides CUDA-shaped wrappers around CANN aclrt* runtime APIs so the rest of
// cpu_backend (cpuinfer.h, ext_bindings.cpp etc.) can keep using cudaStream_t /
// cudaHostFn_t / cudaLaunchHostFunc names. Mirrors the pattern used by
// vendors/hip.h and vendors/musa.h.
//
// IMPORTANT semantic difference from CUDA:
//   * aclrtLaunchCallback() inserts a callback into the stream's report queue.
//     The callback is dispatched by a dedicated host "callback thread" which
//     must (a) have been registered with aclrtSubscribeReport(threadId, stream)
//     and (b) be continuously calling aclrtProcessReport(timeout) in a loop.
//   * If no such subscriber thread exists, queued callbacks NEVER fire and any
//     code waiting on the callback (e.g. CPUInfer::sync_with_cuda_stream) will
//     hang forever.
//   * The submit_with_cuda_stream / sync_with_cuda_stream path therefore
//     requires the host process to also spin up a callback worker (see
//     cpu_backend/ascend_callback_worker.cpp). The synchronous CPUInfer::submit()
//     / sync() path works without it
//     and is what the Phase 1 PoC uses.
// ============================================================================

#include <acl/acl_base.h>
#include <acl/acl_rt.h>

#include <cstdint>

#if defined(KTRANSFORMERS_USE_ASCEND_NPU)
#include "../ascend_callback_worker.h"
#endif

// ---- types -----------------------------------------------------------------
using cudaStream_t = aclrtStream;
using cudaError_t  = aclError;
using cudaHostFn_t = aclrtCallback;  // both are void (*)(void *)

// ACL_SUCCESS is `static const int = 0;` in acl_base_rt.h. Re-expose with the
// CUDA name. Using an inline constexpr avoids ODR issues across TUs.
inline constexpr cudaError_t cudaSuccess = ACL_SUCCESS;

// ---- callbacks -------------------------------------------------------------
// CUDA:  cudaLaunchHostFunc(stream, fn, userData)
// ACL :  aclrtLaunchCallback(fn, userData, blockType, stream)
//
// We pick ACL_CALLBACK_NO_BLOCK to match CUDA's fire-and-forget enqueue
// semantics. (ACL_CALLBACK_BLOCK can block the host if the device queue
// is full.) See header note above re: the required subscriber thread.
static inline cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
#if defined(KTRANSFORMERS_USE_ASCEND_NPU)
  kt::ascend::ensure_stream_subscribed(stream);
#endif
  return aclrtLaunchCallback(fn, userData, ACL_CALLBACK_NO_BLOCK, stream);
}

// ---- error reporting -------------------------------------------------------
// CUDA gives a static string per error code; ACL only exposes the *most recent*
// error message on the current thread. Best-effort emulation.
static inline const char* cudaGetErrorString(cudaError_t /*err*/) {
  const char* m = aclGetRecentErrMsg();
  return (m && *m) ? m : "ACL error";
}
