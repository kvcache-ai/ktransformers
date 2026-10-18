#if defined(KTRANSFORMERS_USE_XPU)

// ============================================================================
// Intel XPU stream bridge for cpu_backend (implements vendors/xpu.h).
//
// torch.xpu streams are in-order sycl::queue objects; torch.xpu.Stream.sycl_queue
// is the raw sycl::queue* that the Python side hands to
// CPUInfer::submit_with_cuda_stream / sync_with_cuda_stream as an integer. This
// file emulates cudaLaunchHostFunc on such a queue so the rest of cpu_backend
// keeps its CUDA-shaped flow:
//
//   submit edge (launch_host_func): GPU -> CPU. fn(arg) runs once everything
//                                   enqueued on the queue before it has finished.
//   sync edge   (launch_sync):      CPU -> GPU. Work enqueued after it waits
//                                   until drain() has returned.
//
// Two implementations are available, selected with KT_XPU_BRIDGE:
//
//   l0 (default)  submit = q.ext_oneapi_submit_barrier(); a per-queue worker
//                 thread waits on the returned event, then calls fn. sync = the
//                 same barrier (so drain() is ordered after preceding GPU work)
//                 plus a second barrier on a host-visible Level Zero event that
//                 the worker signals (zeEventHostSignal) once drain() and every
//                 earlier item have run. About 12 us per edge on Arc Pro B60.
//                 The events come from a per-queue ring (KT_XPU_BRIDGE_EVENTS).
//   host_task     q.submit(h.host_task(...)) for both edges: exactly
//                 cudaLaunchHostFunc's semantics and works under XPUGraph
//                 capture, but costs 60-77 us per edge.
//
// Semantic differences from CUDA:
//   * In l0 mode the stream is NOT held while fn runs: work enqueued after
//     launch_host_func may execute concurrently with fn. In host_task mode (as
//     on CUDA) it waits. Callers must not rely on either; the sync edge is the
//     only CPU -> GPU ordering guarantee, and it holds in both modes.
//   * Items are processed strictly FIFO per queue, so drain() runs after every
//     earlier launch_host_func on the same queue, mirroring the in-stream order
//     of CUDA host functions, and no worker ever waits on another queue's
//     worker (no cross-queue deadlock). Host tasks bypass that FIFO, so when a
//     queue switches to the host_task path (it started recording) its worker's
//     FIFO is drained first -- otherwise a recorded drain could overtake an l0
//     item still waiting on the GPU. That happens once per capture.
//   * Queues that are recording (torch.xpu.XPUGraph capture) and queues whose
//     backend is not Level Zero always use host_task, whatever KT_XPU_BRIDGE
//     says: host tasks are recorded and replayed by the graph, whereas the Level
//     Zero event path is unverified under graph recording.
//   * Exceptions thrown by fn / drain (e.g. TaskQueue::sync rethrowing a failed
//     CPU task) cannot propagate through the stream. They are printed to stderr,
//     the GPU-side event is still signalled (otherwise the queue would wait
//     forever), and the first one is rethrown as std::runtime_error by the next
//     launch_* call (on any queue). shutdown() drops it, so a restart does not
//     fail with a stale message from before.
//
// Environment:
//   KT_XPU_BRIDGE         l0 (default) | host_task. Anything else raises
//                         std::runtime_error on first use.
//   KT_XPU_BRIDGE_EVENTS  Level Zero event ring size per queue in l0 mode
//                         (default 1024, minimum 2).
// Both are read lazily on first use; shutdown() forgets them so tests can switch
// modes in-process.
//
// Lifetime: shutdown() excludes in-flight launch_* calls on the queue it tears down (submit_mu_)
// and makes later ones fail with "worker is shutting down" rather than touch freed events.
// shutdown() (bound as kt_kernel_ext.shutdown_xpu_stream_bridge and
// registered with atexit by the Python side) joins the workers and releases the
// Level Zero objects. The static destructor only joins threads and leaks the
// rest on purpose: the order of static destruction relative to the SYCL runtime
// is undefined.
// ============================================================================

#include <level_zero/ze_api.h>
#include <pthread.h>

#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vendors/xpu.h"

namespace kt::xpu {

namespace {

constexpr const char* kLogPrefix = "[kt-kernel] xpu_stream_bridge: ";  // stderr diagnostics
constexpr const char* kErrPrefix = "kt-kernel XPU stream bridge: ";    // exception messages
constexpr uint32_t kDefaultEvents = 1024;
constexpr uint32_t kMinEvents = 2;

using HostFn = void (*)(void*);

enum class Mode { kL0, kHostTask };

struct Config {
  Mode mode = Mode::kL0;
  uint32_t num_events = kDefaultEvents;
};

// ---- diagnostics -----------------------------------------------------------

std::string describe(std::exception_ptr ep) {
  try {
    std::rethrow_exception(ep);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "non-standard exception";
  }
}

void log_error(const std::string& message) {
  std::fprintf(stderr, "%s%s\n", kLogPrefix, message.c_str());
  std::fflush(stderr);
}

std::string ze_message(const char* what, ze_result_t result) {
  char code[32];
  std::snprintf(code, sizeof(code), "0x%08x", static_cast<unsigned>(result));
  return std::string(what) + " failed (ze_result_t " + code + ")";
}

void ze_check(ze_result_t result, const char* what) {
  if (result != ZE_RESULT_SUCCESS) throw std::runtime_error(kErrPrefix + ze_message(what, result));
}

[[noreturn]] void rethrow_sycl(const char* what, const sycl::exception& e) {
  throw std::runtime_error(std::string(kErrPrefix) + what + ": " + e.what());
}

// ---- global state ----------------------------------------------------------

class Worker;

// First exception raised by fn / drain on a bridge thread. It cannot propagate
// through the stream, so it is parked here and rethrown once by the next
// launch_* call.
struct ErrorSlot {
  std::mutex mu;
  bool set = false;
  std::string message;
};

struct Registry {
  std::mutex mu;  // guards config and workers; held for the whole of shutdown()
  std::optional<Config> config;
  std::unordered_map<sycl::queue*, std::shared_ptr<Worker>> workers;
  ErrorSlot error;
};

// Never deleted: Level Zero / SYCL objects must not be released during static
// destruction (see ExitGuard at the bottom).
Registry& registry() {
  static Registry* const instance = new Registry();
  return *instance;
}

void record_error(const std::string& message) {
  log_error(message);
  ErrorSlot& slot = registry().error;
  std::lock_guard<std::mutex> lock(slot.mu);
  if (!slot.set) {
    slot.set = true;
    slot.message = message;
  }
}

void rethrow_pending_error() {
  ErrorSlot& slot = registry().error;
  std::string message;
  {
    std::lock_guard<std::mutex> lock(slot.mu);
    if (!slot.set) return;
    slot.set = false;
    message = std::move(slot.message);
    slot.message.clear();
  }
  throw std::runtime_error(std::string(kErrPrefix) + "a bridged host function failed earlier: " + message);
}

// ---- l0 mode: one worker per sycl::queue -----------------------------------

class Worker {
 public:
  Worker(sycl::queue* q, uint32_t num_events);
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;
  // release() must have run first. At process teardown Worker objects are
  // leaked on purpose instead (see ExitGuard).
  ~Worker() = default;

  void launch_host_func(HostFn fn, void* arg);
  void launch_sync(std::function<void()> drain);
  // Block until the worker has run everything queued so far. Used when a queue switches to the
  // host_task path (graph recording): host tasks bypass this FIFO, so without draining it first a
  // recorded drain could overtake an l0 item still waiting on the GPU.
  void drain_fifo();
  const sycl::context& context() const { return ctx_; }
  sycl::device device() const { return q_->get_device(); }

  // Stop accepting items, let the thread finish what is queued (pending sync
  // events get signalled) and join it.
  void stop_and_join();
  // Destroy the Level Zero events / pool and drop the SYCL wrappers. Only after
  // stop_and_join(), never at process teardown.
  void release();

 private:
  struct Item {
    std::optional<sycl::event> gpu_ready;  // complete once everything enqueued before this item is done
    HostFn fn = nullptr;
    void* arg = nullptr;
    std::function<void()> drain;         // sync
    ze_event_handle_t signal = nullptr;  // sync: signalled after drain(); the queue waits on it
    // fence: carries no work, just reports that the worker has reached this point in the FIFO.
    std::shared_ptr<std::promise<void>> reached;
    bool is_sync() const { return signal != nullptr; }
  };
  struct Slot {
    ze_event_handle_t ze_event = nullptr;
    std::optional<sycl::event> wrapper;  // interop wrapper of ze_event used by the current barrier
    std::optional<sycl::event> barrier;  // barrier waiting on ze_event; complete => ze_event no longer in use
  };

  void create_events(uint32_t num_events);
  void throw_if_dead();  // submit_mu_ must be held
  void push(Item item);
  void run();

  sycl::queue* const q_;
  const sycl::context ctx_;

  // Serialises "enqueue on q_, then push" so that the FIFO order equals the
  // queue order. Without it a sync barrier could precede, on the queue, a
  // submit barrier whose item precedes the sync item in the FIFO, and the
  // worker would wait on the GPU while the GPU waits on the worker.
  // It also excludes stop_and_join() / release() from the event ring below, so
  // a concurrent shutdown() cannot free the slots a launch is working on.
  std::mutex submit_mu_;
  bool dead_ = false;  // guarded by submit_mu_: set by stop_and_join(), rejects later launches
  ze_event_pool_handle_t pool_ = nullptr;
  std::vector<Slot> slots_;
  uint32_t next_slot_ = 0;

  std::mutex mu_;  // guards items_ and stop_
  std::condition_variable cv_;
  std::deque<Item> items_;
  bool stop_ = false;
  std::thread thread_;
};

Worker::Worker(sycl::queue* q, uint32_t num_events) : q_(q), ctx_(q->get_context()) {
  create_events(num_events);
  thread_ = std::thread([this] { run(); });
}

void Worker::create_events(uint32_t num_events) {
  ze_context_handle_t ze_ctx = nullptr;
  ze_device_handle_t ze_dev = nullptr;
  try {
    ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx_);
    ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_->get_device());
  } catch (const sycl::exception& e) {
    rethrow_sycl("get_native(Level Zero context / device)", e);
  }
  try {
    ze_event_pool_desc_t pool_desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
                                   num_events};
    ze_check(zeEventPoolCreate(ze_ctx, &pool_desc, 1, &ze_dev, &pool_), "zeEventPoolCreate");
    slots_.resize(num_events);
    for (uint32_t i = 0; i < num_events; ++i) {
      ze_event_desc_t event_desc{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, i, ZE_EVENT_SCOPE_FLAG_HOST,
                                 ZE_EVENT_SCOPE_FLAG_HOST};
      ze_check(zeEventCreate(pool_, &event_desc, &slots_[i].ze_event), "zeEventCreate");
    }
  } catch (...) {
    release();
    throw;
  }
}

void Worker::run() {
  (void)pthread_setname_np(pthread_self(), "kt-xpu-bridge");
  for (;;) {
    Item item;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this] { return stop_ || !items_.empty(); });
      // stop_ requested and nothing left: every pending sync event has been signalled.
      if (items_.empty()) return;
      item = std::move(items_.front());
      items_.pop_front();
    }
    if (item.reached) {
      item.reached->set_value();
      continue;
    }
    if (item.is_sync()) {
      if (item.gpu_ready) {
        // Order drain() after everything the queue had already been given, like a CUDA host
        // function. Unlike the submit edge below, a failed wait does NOT skip drain(): the GPU
        // is released by the signal at the end either way, and skipping the drain would let the
        // work behind the barrier read buffers whose CPU tasks never completed.
        try {
          item.gpu_ready->wait();
        } catch (const sycl::exception& e) {
          record_error(std::string("waiting for the queue before drain(): ") + e.what());
        }
      }
      try {
        item.drain();
      } catch (...) {
        record_error("drain() threw: " + describe(std::current_exception()));
      }
      // Signal even after a failure, otherwise the queue (and everything behind it) waits forever.
      ze_result_t result = zeEventHostSignal(item.signal);
      if (result != ZE_RESULT_SUCCESS) record_error(ze_message("zeEventHostSignal", result));
    } else {
      bool ready = false;
      try {
        if (item.gpu_ready) item.gpu_ready->wait();
        ready = true;
      } catch (const sycl::exception& e) {
        record_error(std::string("waiting for the queue before a host function: ") + e.what());
      }
      // Like cudaLaunchHostFunc, the host function is not called when the preceding work failed.
      if (ready) {
        try {
          item.fn(item.arg);
        } catch (...) {
          record_error("host function threw: " + describe(std::current_exception()));
        }
      }
    }
  }
}

void Worker::throw_if_dead() {
  if (dead_) {
    throw std::runtime_error(std::string(kErrPrefix) + "worker is shutting down (shutdown() raced with launch_*)");
  }
}

void Worker::push(Item item) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_) {
      throw std::runtime_error(std::string(kErrPrefix) + "worker is shutting down (shutdown() raced with launch_*)");
    }
    items_.push_back(std::move(item));
  }
  cv_.notify_one();
}

void Worker::launch_host_func(HostFn fn, void* arg) {
  std::lock_guard<std::mutex> lock(submit_mu_);
  throw_if_dead();
  Item item;
  try {
    item.gpu_ready = q_->ext_oneapi_submit_barrier();
  } catch (const sycl::exception& e) {
    rethrow_sycl("ext_oneapi_submit_barrier", e);
  }
  item.fn = fn;
  item.arg = arg;
  push(std::move(item));
}

void Worker::launch_sync(std::function<void()> drain) {
  std::lock_guard<std::mutex> lock(submit_mu_);
  throw_if_dead();
  Slot& slot = slots_[next_slot_];
  next_slot_ = (next_slot_ + 1) % static_cast<uint32_t>(slots_.size());
  if (slot.barrier) {
    // Reuse. The barrier that consumed the slot's previous use completes only
    // after the worker signalled the event (in practice it is long done);
    // afterwards nothing references the event and it can be reset. The wrapper
    // is dropped and recreated rather than reused: the runtime may cache the
    // completion of an event it has already seen complete.
    try {
      slot.barrier->wait();
    } catch (const sycl::exception& e) {
      rethrow_sycl("waiting for the previous barrier of a sync slot", e);
    }
    slot.barrier.reset();
    slot.wrapper.reset();
    ze_check(zeEventHostReset(slot.ze_event), "zeEventHostReset");
  }
  Item item;
  try {
    // Completes once everything enqueued before this sync has run; the worker waits on it so
    // drain() is ordered after preceding GPU work, as vendors/xpu.h promises.
    item.gpu_ready = q_->ext_oneapi_submit_barrier();
    slot.wrapper = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
        {slot.ze_event, sycl::ext::oneapi::level_zero::ownership::keep}, ctx_);
    slot.barrier = q_->ext_oneapi_submit_barrier({*slot.wrapper});
  } catch (const sycl::exception& e) {
    rethrow_sycl("make_event / ext_oneapi_submit_barrier", e);
  }
  item.drain = std::move(drain);
  item.signal = slot.ze_event;
  try {
    push(std::move(item));
  } catch (...) {
    // The barrier is already enqueued; signal so the queue is not stuck forever.
    (void)zeEventHostSignal(slot.ze_event);
    throw;
  }
}

void Worker::drain_fifo() {
  std::lock_guard<std::mutex> lock(submit_mu_);
  if (dead_) return;  // nothing left to drain: shutdown() already joined the worker
  auto reached = std::make_shared<std::promise<void>>();
  std::future<void> done = reached->get_future();
  Item item;
  item.reached = reached;
  push(std::move(item));
  // The items ahead of the fence only wait on GPU work and on CPU tasks, never on this thread, so
  // this cannot deadlock. shutdown() blocks on submit_mu_ meanwhile, but the worker keeps running.
  done.wait();
}

void Worker::stop_and_join() {
  if (thread_.joinable() && std::this_thread::get_id() == thread_.get_id()) {
    throw std::runtime_error(std::string(kErrPrefix) + "shutdown() called from inside a bridged host function");
  }
  // submit_mu_ keeps a concurrent launch_* out of the event ring for the whole stop; dead_ makes
  // every later launch fail fast instead of touching slots release() is about to free.
  std::lock_guard<std::mutex> submit_lock(submit_mu_);
  dead_ = true;
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Worker::release() {
  std::lock_guard<std::mutex> submit_lock(submit_mu_);
  for (Slot& slot : slots_) {
    if (slot.barrier) {
      // Make sure no enqueued barrier still references the event before destroying it.
      try {
        slot.barrier->wait();
      } catch (const sycl::exception& e) {
        log_error(std::string("waiting for a sync barrier during shutdown: ") + e.what());
      }
      slot.barrier.reset();
    }
    slot.wrapper.reset();
    if (slot.ze_event != nullptr) {
      ze_result_t result = zeEventDestroy(slot.ze_event);
      if (result != ZE_RESULT_SUCCESS) log_error(ze_message("zeEventDestroy", result));
      slot.ze_event = nullptr;
    }
  }
  slots_.clear();
  if (pool_ != nullptr) {
    ze_result_t result = zeEventPoolDestroy(pool_);
    if (result != ZE_RESULT_SUCCESS) log_error(ze_message("zeEventPoolDestroy", result));
    pool_ = nullptr;
  }
}

// ---- configuration and routing ---------------------------------------------

Config read_config_from_env() {
  Config cfg;
  const char* mode = std::getenv("KT_XPU_BRIDGE");
  if (mode == nullptr || *mode == '\0' || std::strcmp(mode, "l0") == 0) {
    cfg.mode = Mode::kL0;
  } else if (std::strcmp(mode, "host_task") == 0) {
    cfg.mode = Mode::kHostTask;
  } else {
    throw std::runtime_error(std::string(kErrPrefix) + "unknown KT_XPU_BRIDGE value '" + mode +
                             "' (expected l0 or host_task)");
  }
  const char* events = std::getenv("KT_XPU_BRIDGE_EVENTS");
  if (events != nullptr && *events != '\0') {
    char* end = nullptr;
    errno = 0;
    unsigned long value = std::strtoul(events, &end, 10);
    if (errno != 0 || end == events || *end != '\0' || std::strchr(events, '-') != nullptr || value > UINT32_MAX) {
      throw std::runtime_error(std::string(kErrPrefix) + "invalid KT_XPU_BRIDGE_EVENTS value '" + events +
                               "' (expected an integer >= " + std::to_string(kMinEvents) + ")");
    }
    if (value < kMinEvents) {
      std::fprintf(stderr, "%sKT_XPU_BRIDGE_EVENTS=%lu is below the minimum of %u, using %u\n", kLogPrefix, value,
                   kMinEvents, kMinEvents);
      value = kMinEvents;
    }
    cfg.num_events = static_cast<uint32_t>(value);
  }
  return cfg;
}

const Config& config_locked(Registry& reg) {
  if (!reg.config) reg.config = read_config_from_env();
  return *reg.config;
}

bool use_host_task(const Config& cfg, sycl::queue* q) {
  if (cfg.mode == Mode::kHostTask) return true;
  try {
    if (q->get_backend() != sycl::backend::ext_oneapi_level_zero) return true;
    return q->ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording;
  } catch (const sycl::exception& e) {
    rethrow_sycl("querying the queue backend / state", e);
  }
}

std::shared_ptr<Worker> worker_locked(Registry& reg, sycl::queue* q, const Config& cfg) {
  auto it = reg.workers.find(q);
  if (it != reg.workers.end()) {
    // The map is keyed on the caller's sycl::queue*, which the bridge does not own. torch keeps its
    // stream-pool queues for the process lifetime, but another caller could free a queue and have a
    // new one land on the same address -- whose context differs, so the cached worker's event pool
    // would belong to the wrong context. Re-create when the identity does not match.
    if (it->second->context() == q->get_context() && it->second->device() == q->get_device()) {
      return it->second;
    }
    it->second->stop_and_join();
    it->second->release();
    reg.workers.erase(it);
  }
  auto worker = std::make_shared<Worker>(q, cfg.num_events);
  reg.workers.emplace(q, worker);
  return worker;
}

struct Route {
  sycl::queue* q = nullptr;
  std::shared_ptr<Worker> worker;  // non-null => l0 path
  // host_task path, but this queue still has a live l0 worker whose FIFO must be drained first.
  std::shared_ptr<Worker> fence_worker;
};

// Surfaces a pending error, resolves the configuration and decides the mode for this call.
Route route(void* queue) {
  if (queue == nullptr) throw std::invalid_argument(std::string(kErrPrefix) + "queue must not be null");
  rethrow_pending_error();
  Route r;
  r.q = static_cast<sycl::queue*>(queue);
  Registry& reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);
  const Config& cfg = config_locked(reg);
  if (!use_host_task(cfg, r.q)) {
    r.worker = worker_locked(reg, r.q, cfg);
  } else {
    // Do not create one here; only adopt a worker this queue already has, so its queued items keep
    // their order relative to the host task we are about to submit.
    auto it = reg.workers.find(r.q);
    if (it != reg.workers.end()) r.fence_worker = it->second;
  }
  return r;
}

// host_task mode: cudaLaunchHostFunc semantics straight from the SYCL runtime.
// Errors are parked like in l0 mode; the SYCL runtime would otherwise swallow
// them (torch's queues have no async handler).
template <typename Body>
void submit_host_task(sycl::queue* q, Body body, const char* what) {
  try {
    q->submit([&](sycl::handler& h) {
      h.host_task([body = std::move(body), what]() mutable {
        try {
          body();
        } catch (...) {
          record_error(std::string(what) + " threw: " + describe(std::current_exception()));
        }
      });
    });
  } catch (const sycl::exception& e) {
    rethrow_sycl("submitting a host_task", e);
  }
}

// Process teardown: only stop and join the worker threads. Their Level Zero /
// SYCL objects are leaked on purpose, because the order of static destruction
// relative to the SYCL runtime is undefined. The Python side registers
// shutdown() with atexit, so normally there is nothing left to join here.
struct ExitGuard {
  ~ExitGuard() {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    for (auto& entry : reg.workers) {
      try {
        entry.second->stop_and_join();
      } catch (...) {
      }
    }
  }
};
ExitGuard g_exit_guard;

}  // namespace

void launch_host_func(void* queue, void (*fn)(void*), void* arg) {
  if (fn == nullptr) throw std::invalid_argument(std::string(kErrPrefix) + "fn must not be null");
  Route r = route(queue);
  if (r.worker) {
    r.worker->launch_host_func(fn, arg);
  } else {
    if (r.fence_worker) r.fence_worker->drain_fifo();
    submit_host_task(r.q, [fn, arg] { fn(arg); }, "host function");
  }
}

void launch_sync(void* queue, std::function<void()> drain) {
  if (!drain) throw std::invalid_argument(std::string(kErrPrefix) + "drain must not be empty");
  Route r = route(queue);
  if (r.worker) {
    r.worker->launch_sync(std::move(drain));
  } else {
    if (r.fence_worker) r.fence_worker->drain_fifo();
    submit_host_task(r.q, std::move(drain), "drain()");
  }
}

const char* bridge_mode() {
  Registry& reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);
  return config_locked(reg).mode == Mode::kHostTask ? "host_task" : "l0";
}

unsigned int bridge_events() {
  Registry& reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);
  return config_locked(reg).num_events;
}

void shutdown() {
  Registry& reg = registry();
  std::lock_guard<std::mutex> lock(reg.mu);
  for (auto& entry : reg.workers) entry.second->stop_and_join();
  for (auto& entry : reg.workers) entry.second->release();
  reg.workers.clear();
  reg.config.reset();
  // Drop any parked error too. It was already reported on stderr, and carrying it across a
  // restart would make the next unrelated launch_* fail with a stale message.
  {
    std::lock_guard<std::mutex> error_lock(reg.error.mu);
    reg.error.set = false;
    reg.error.message.clear();
  }
}

}  // namespace kt::xpu

#endif  // KTRANSFORMERS_USE_XPU
