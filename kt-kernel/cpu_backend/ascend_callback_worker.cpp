#if defined(KTRANSFORMERS_USE_ASCEND_NPU)

#include "ascend_callback_worker.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace kt::ascend {

namespace {

constexpr int kProcessReportTimeoutMs = 100;

std::mutex g_mu;
std::set<aclrtStream> g_subscribed_streams;
aclrtContext g_context = nullptr;
std::thread g_worker;
std::atomic<bool> g_stop{false};
std::atomic<bool> g_started{false};
// Set by the worker thread once it has a usable ACL context and is about to
// enter the aclrtProcessReport() loop.  aclrtSubscribeReport() must not run
// before that, otherwise the very first callbacks can be dropped.
std::atomic<bool> g_worker_ready{false};
uint64_t g_worker_thread_id = 0;

void worker_main(aclrtContext ctx) {
  if (ctx != nullptr) {
    aclError err = aclrtSetCurrentContext(ctx);
    if (err != ACL_SUCCESS) {
      std::fprintf(stderr,
                   "[kt-kernel] ascend_callback_worker: aclrtSetCurrentContext failed (%d)\n",
                   static_cast<int>(err));
    }
  }
  g_worker_ready.store(true, std::memory_order_release);

  while (!g_stop.load(std::memory_order_acquire)) {
    (void)aclrtProcessReport(kProcessReportTimeoutMs);
  }

  // Drain callbacks that were enqueued between the last loop iteration and the
  // stop request; without this they are silently dropped and their waiters hang.
  (void)aclrtProcessReport(kProcessReportTimeoutMs);

  // ACL requires the *subscribing* thread to unsubscribe.  Doing it from
  // shutdown_callback_worker() after join() would target a dead thread id and
  // silently do nothing.
  for (aclrtStream stream : g_subscribed_streams) {
    (void)aclrtUnSubscribeReport(g_worker_thread_id, stream);
  }
}

void start_worker_locked(aclrtContext ctx) {
  if (g_started.load(std::memory_order_acquire)) {
    return;
  }
  g_context = ctx;
  g_stop.store(false, std::memory_order_release);
  g_worker_ready.store(false, std::memory_order_release);
  g_worker = std::thread([ctx]() { worker_main(ctx); });
  g_worker_thread_id = static_cast<uint64_t>(g_worker.native_handle());
  g_started.store(true, std::memory_order_release);

  // Barrier: no aclrtSubscribeReport() before the worker is in its report loop.
  while (!g_worker_ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  for (aclrtStream stream : g_subscribed_streams) {
    aclError err = aclrtSubscribeReport(g_worker_thread_id, stream);
    if (err != ACL_SUCCESS) {
      std::fprintf(stderr,
                   "[kt-kernel] ascend_callback_worker: aclrtSubscribeReport failed (%d)\n",
                   static_cast<int>(err));
    }
  }
}

void subscribe_stream_locked(aclrtStream stream) {
  if (stream == nullptr) {
    return;
  }
  if (g_subscribed_streams.count(stream) != 0) {
    return;
  }
  g_subscribed_streams.insert(stream);
  if (g_started.load(std::memory_order_acquire)) {
    aclError err = aclrtSubscribeReport(g_worker_thread_id, stream);
    if (err != ACL_SUCCESS) {
      std::fprintf(stderr,
                   "[kt-kernel] ascend_callback_worker: aclrtSubscribeReport failed (%d)\n",
                   static_cast<int>(err));
    }
  }
}

}  // namespace

void ensure_callback_worker(aclrtContext ctx) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_started.load(std::memory_order_acquire)) {
    return;
  }
  aclrtContext use_ctx = ctx;
  if (use_ctx == nullptr) {
    aclError err = aclrtGetCurrentContext(&use_ctx);
    if (err != ACL_SUCCESS || use_ctx == nullptr) {
      std::fprintf(stderr,
                   "[kt-kernel] ascend_callback_worker: no ACL context; call after torch.npu init\n");
      return;
    }
  }
  start_worker_locked(use_ctx);
}

void ensure_stream_subscribed(aclrtStream stream) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_started.load(std::memory_order_acquire)) {
    aclrtContext ctx = nullptr;
    (void)aclrtGetCurrentContext(&ctx);
    start_worker_locked(ctx);
  }
  subscribe_stream_locked(stream);
}

void shutdown_callback_worker() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_started.load(std::memory_order_acquire)) {
    return;
  }
  g_stop.store(true, std::memory_order_release);
  if (g_worker.joinable()) {
    // worker_main() drains the report queue and unsubscribes before returning.
    g_worker.join();
  }
  g_subscribed_streams.clear();
  g_started.store(false, std::memory_order_release);
  g_worker_ready.store(false, std::memory_order_release);
  g_worker_thread_id = 0;
}

bool callback_worker_running() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_started.load(std::memory_order_acquire) && g_worker_ready.load(std::memory_order_acquire);
}

}  // namespace kt::ascend

#endif  // KTRANSFORMERS_USE_ASCEND_NPU
