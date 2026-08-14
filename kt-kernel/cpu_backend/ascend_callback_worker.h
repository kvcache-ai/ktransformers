#pragma once

// Ascend ACL stream callback subscriber for kt-kernel.
//
// CANN dispatches aclrtLaunchCallback tasks only on a host thread that has called
// aclrtSubscribeReport(threadId, stream) and is running aclrtProcessReport() in a
// loop.  This worker mirrors torch_npu's pattern (see NPUGraph.cpp / Graph.cpp).

#if defined(KTRANSFORMERS_USE_ASCEND_NPU)

#include <acl/acl_rt.h>

#include <cstdint>

namespace kt::ascend {

// Start the global callback worker (idempotent).  Call after ACL/torch.npu init.
// If ctx is null, uses aclrtGetCurrentContext().
void ensure_callback_worker(aclrtContext ctx = nullptr);

// Register ``stream`` with the worker so enqueued callbacks are dispatched.
void ensure_stream_subscribed(aclrtStream stream);

// Optional shutdown (process exit).
void shutdown_callback_worker();

// True iff the worker thread was started and has entered its aclrtProcessReport
// loop.  When this is false, stream callbacks are never dispatched and callers
// must fall back to the synchronous submit/sync path.
bool callback_worker_running();

}  // namespace kt::ascend

#endif  // KTRANSFORMERS_USE_ASCEND_NPU
