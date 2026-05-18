/**
 * @Description  : Asynchronous I/O wrapper using io_uring for expert weight loading
 * @Author       : RaQiu
 * @Date         : 2026-05-06
 * @Version      : 1.0.0
 * @Copyright (c) 2026 by KVCache.AI, All Rights Reserved.
 **/

#include "async_io.hpp"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <sstream>

#ifdef HAVE_LIBURING
#include <sys/time.h>
#include <errno.h>
#endif

namespace ktransformers {

namespace {

int configured_max_read_retries() {
    constexpr int kDefaultMaxReadRetries = 3;
    const char* raw = std::getenv("KT_IOURING_MAX_READ_RETRIES");
    if (raw == nullptr || raw[0] == '\0') {
        return kDefaultMaxReadRetries;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw) {
        return kDefaultMaxReadRetries;
    }
    return static_cast<int>(std::max<long>(0, std::min<long>(parsed, 16)));
}

void log_read_status(const char* tag,
                     uint64_t request_id,
                     int expert_id,
                     int retry_count,
                     int max_retries,
                     int fd,
                     off_t offset,
                     size_t expected_size,
                     int result) {
    if (result < 0) {
        std::fprintf(stderr,
                     "[MESHIO][%s] request=%llu expert=%d retry=%d/%d fd=%d offset=%lld size=%zu "
                     "result=%d(%s)\n",
                     tag,
                     static_cast<unsigned long long>(request_id),
                     expert_id,
                     retry_count,
                     max_retries,
                     fd,
                     static_cast<long long>(offset),
                     expected_size,
                     result,
                     std::strerror(-result));
    } else {
        std::fprintf(stderr,
                     "[MESHIO][%s] request=%llu expert=%d retry=%d/%d fd=%d offset=%lld size=%zu "
                     "result=%d(short-read)\n",
                     tag,
                     static_cast<unsigned long long>(request_id),
                     expert_id,
                     retry_count,
                     max_retries,
                     fd,
                     static_cast<long long>(offset),
                     expected_size,
                     result);
    }
}

}  // namespace

AsyncExpertReader::AsyncExpertReader(int queue_depth)
    : queue_depth_(queue_depth), next_user_data_(1), max_read_retries_(configured_max_read_retries()) {
#ifdef HAVE_LIBURING
    int ret = io_uring_queue_init(queue_depth_, &ring_, 0);
    if (ret < 0) {
        throw std::runtime_error("Failed to initialize io_uring: " + std::string(strerror(-ret)));
    }
    io_thread_ = std::thread(&AsyncExpertReader::io_thread_main, this);
#else
    [[maybe_unused]] int unused_queue_depth = queue_depth_;
#endif
}

AsyncExpertReader::~AsyncExpertReader() {
#ifdef HAVE_LIBURING
    stop_requested_.store(true, std::memory_order_release);
    queue_cv_.notify_all();
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
    io_uring_queue_exit(&ring_);
#endif
}

uint64_t AsyncExpertReader::submit_read(int fd, void* buf, size_t size, off_t offset, int expert_id) {
#ifdef HAVE_LIBURING
    uint64_t user_data = next_user_data_.fetch_add(1, std::memory_order_acq_rel);

    auto request = std::make_shared<RequestInfo>();
    request->expert_id = expert_id;
    request->fd = fd;
    request->buffer = buf;
    request->expected_size = size;
    request->offset = offset;
    request->state.store(RequestState::Inflight, std::memory_order_release);
    request->result.store(0, std::memory_order_release);
    {
        std::lock_guard<std::mutex> request_lock(request_map_mutex_);
        requests_[user_data] = request;
    }
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        pending_jobs_.push_back(ReadJob{user_data, request});
    }
    queue_cv_.notify_one();

    return user_data;
#else
    throw std::runtime_error("io_uring not available on this platform");
#endif
}

std::vector<uint64_t> AsyncExpertReader::submit_reads(const std::vector<ReadRequest>& requests, SubmitStats* stats) {
#ifdef HAVE_LIBURING
    const auto total_start = std::chrono::steady_clock::now();
    if (requests.empty()) {
        return {};
    }

    std::vector<uint64_t> request_ids;
    request_ids.reserve(requests.size());
    std::vector<ReadJob> jobs;
    jobs.reserve(requests.size());

    auto elapsed_us = [](std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end) -> uint64_t {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    };
    if (stats != nullptr) {
        stats->request_count += static_cast<uint64_t>(requests.size());
    }

    const auto bookkeeping_start = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> request_lock(request_map_mutex_);
        for (const auto& req : requests) {
            uint64_t user_data = next_user_data_.fetch_add(1, std::memory_order_acq_rel);
            auto request = std::make_shared<RequestInfo>();
            request->expert_id = req.expert_id;
            request->fd = req.fd;
            request->buffer = req.buffer;
            request->expected_size = req.size;
            request->offset = req.offset;
            request->state.store(RequestState::Inflight, std::memory_order_release);
            request->result.store(0, std::memory_order_release);
            requests_[user_data] = request;
            request_ids.push_back(user_data);
            jobs.push_back(ReadJob{user_data, std::move(request)});
        }
    }
    const auto bookkeeping_end = std::chrono::steady_clock::now();
    if (stats != nullptr) {
        stats->request_bookkeeping_us += elapsed_us(bookkeeping_start, bookkeeping_end);
    }

    const auto lock_start = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        const auto lock_end = std::chrono::steady_clock::now();
        if (stats != nullptr) {
            stats->lock_wait_us += elapsed_us(lock_start, lock_end);
        }
        pending_jobs_.insert(pending_jobs_.end(), jobs.begin(), jobs.end());
    }
    queue_cv_.notify_one();
    if (stats != nullptr) {
        stats->total_us += elapsed_us(total_start, std::chrono::steady_clock::now());
    }
    return request_ids;
#else
    [[maybe_unused]] const std::vector<ReadRequest>& unused_requests = requests;
    [[maybe_unused]] SubmitStats* unused_stats = stats;
    throw std::runtime_error("io_uring not available on this platform");
#endif
}

int AsyncExpertReader::wait_one_completion() {
#ifdef HAVE_LIBURING
    std::unique_lock<std::mutex> lock(completion_events_mutex_);
    completion_events_cv_.wait(lock, [this]() { return !completion_events_.empty() || stop_requested_.load(); });
    if (completion_events_.empty()) {
        return -1;
    }
    CompletionEvent event = completion_events_.front();
    completion_events_.pop_front();
    return event.ok ? event.expert_id : -1;
#else
    return -1;
#endif
}

std::vector<int> AsyncExpertReader::poll_completions() {
    std::vector<int> completed;

#ifdef HAVE_LIBURING
    std::lock_guard<std::mutex> lock(completion_events_mutex_);
    while (!completion_events_.empty()) {
        CompletionEvent event = completion_events_.front();
        completion_events_.pop_front();
        if (event.ok) {
            completed.push_back(event.expert_id);
        }
    }
#endif

    return completed;
}

bool AsyncExpertReader::wait_for_expert(int expert_id, int timeout_ms) {
#ifdef HAVE_LIBURING
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        bool found = false;
        bool all_completed = true;
        bool any_failed = false;
        {
            std::lock_guard<std::mutex> lock(request_map_mutex_);
            if (completed_experts_.count(expert_id) > 0) {
                found = true;
            }
            for (const auto& pair : requests_) {
                const auto& request = pair.second;
                if (request == nullptr || request->expert_id != expert_id) {
                    continue;
                }
                found = true;
                const RequestState state = request->state.load(std::memory_order_acquire);
                if (state == RequestState::Inflight) {
                    all_completed = false;
                } else if (state == RequestState::Failed) {
                    any_failed = true;
                }
            }
        }
        if (!found) {
            return false;
        }
        if (any_failed) {
            return false;
        }
        if (all_completed) {
            return true;
        }

        std::unique_lock<std::mutex> lock(completion_events_mutex_);
        if (completion_events_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            bool found_after_timeout = false;
            bool all_completed_after_timeout = true;
            bool any_failed_after_timeout = false;
            {
                std::lock_guard<std::mutex> request_lock(request_map_mutex_);
                if (completed_experts_.count(expert_id) > 0) {
                    found_after_timeout = true;
                }
                for (const auto& pair : requests_) {
                    const auto& request = pair.second;
                    if (request == nullptr || request->expert_id != expert_id) {
                        continue;
                    }
                    found_after_timeout = true;
                    const RequestState state = request->state.load(std::memory_order_acquire);
                    if (state == RequestState::Inflight) {
                        all_completed_after_timeout = false;
                    } else if (state == RequestState::Failed) {
                        any_failed_after_timeout = true;
                    }
                }
            }
            return found_after_timeout && all_completed_after_timeout && !any_failed_after_timeout;
        }
    }
#else
    [[maybe_unused]] int unused_expert_id = expert_id;
    [[maybe_unused]] int unused_timeout_ms = timeout_ms;
    return false;
#endif
}

bool AsyncExpertReader::wait_for_request(uint64_t request_id, int timeout_ms) {
#ifdef HAVE_LIBURING
    auto request = get_request(request_id);
    if (request == nullptr) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(request->mutex);
    while (true) {
        const RequestState state = request->state.load(std::memory_order_acquire);
        if (state == RequestState::Completed) {
            return true;
        }
        if (state == RequestState::Failed) {
            return false;
        }
        const bool changed = request->cv.wait_until(lock, deadline, [&request]() {
            return request->state.load(std::memory_order_acquire) != RequestState::Inflight;
        });
        if (!changed) {
            const RequestState final_state = request->state.load(std::memory_order_acquire);
            return final_state == RequestState::Completed;
        }
    }
#else
    [[maybe_unused]] uint64_t unused_request_id = request_id;
    [[maybe_unused]] int unused_timeout_ms = timeout_ms;
    return false;
#endif
}

bool AsyncExpertReader::wait_for_requests(const std::vector<uint64_t>& request_ids, int timeout_ms) {
#ifdef HAVE_LIBURING
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::vector<std::shared_ptr<RequestInfo>> requests;
    requests.reserve(request_ids.size());
    for (uint64_t request_id : request_ids) {
        auto request = get_request(request_id);
        if (request == nullptr) {
            return false;
        }
        requests.push_back(std::move(request));
    }

    for (const auto& request : requests) {
        std::unique_lock<std::mutex> lock(request->mutex);
        while (true) {
            const RequestState state = request->state.load(std::memory_order_acquire);
            if (state == RequestState::Completed) {
                break;
            }
            if (state == RequestState::Failed) {
                return false;
            }
            const bool changed = request->cv.wait_until(lock, deadline, [&request]() {
                return request->state.load(std::memory_order_acquire) != RequestState::Inflight;
            });
            if (!changed) {
                const RequestState final_state = request->state.load(std::memory_order_acquire);
                if (final_state == RequestState::Completed) {
                    break;
                }
                return false;
            }
        }
    }
    return true;
#else
    [[maybe_unused]] const std::vector<uint64_t>& unused_request_ids = request_ids;
    [[maybe_unused]] int unused_timeout_ms = timeout_ms;
    return false;
#endif
}

void AsyncExpertReader::submit_batch(const std::vector<ReadRequest>& requests) {
#ifdef HAVE_LIBURING
    (void)submit_reads(requests);
#else
    [[maybe_unused]] const std::vector<ReadRequest>& unused_requests = requests;
    throw std::runtime_error("io_uring not available on this platform");
#endif
}

int AsyncExpertReader::get_request_result(uint64_t request_id) const {
#ifdef HAVE_LIBURING
    auto request = get_request(request_id);
    if (request == nullptr || request->state.load(std::memory_order_acquire) == RequestState::Inflight) {
        return INT_MIN;
    }
    return request->result.load(std::memory_order_acquire);
#else
    [[maybe_unused]] uint64_t unused_request_id = request_id;
    return INT_MIN;
#endif
}

bool AsyncExpertReader::request_succeeded(uint64_t request_id) const {
#ifdef HAVE_LIBURING
    auto request = get_request(request_id);
    return request != nullptr && request->state.load(std::memory_order_acquire) == RequestState::Completed;
#else
    [[maybe_unused]] uint64_t unused_request_id = request_id;
    return false;
#endif
}

std::string AsyncExpertReader::describe_requests(const std::vector<uint64_t>& request_ids) const {
#ifdef HAVE_LIBURING
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (uint64_t request_id : request_ids) {
        if (!first) {
            oss << ",";
        }
        first = false;
        oss << request_id << ":";
        auto request = get_request(request_id);
        if (request == nullptr) {
            oss << "missing";
        } else {
            const RequestState state = request->state.load(std::memory_order_acquire);
            const int result = request->result.load(std::memory_order_acquire);
            if (state == RequestState::Completed) {
                oss << "ok";
            } else if (state == RequestState::Failed) {
                oss << "fail";
            } else {
                oss << "inflight";
            }
            oss << "/res=" << result;
            oss << "/expected=" << request->expected_size;
            oss << "/retry=" << request->retry_count.load(std::memory_order_acquire) << "/" << max_read_retries_;
            if (result < 0) {
                oss << "(" << std::strerror(-result) << ")";
            }
        }
    }
    oss << "]";
    return oss.str();
#else
    [[maybe_unused]] const std::vector<uint64_t>& unused_request_ids = request_ids;
    return "[]";
#endif
}

int AsyncExpertReader::get_inflight_count() const {
#ifdef HAVE_LIBURING
    int queued = 0;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queued = static_cast<int>(pending_jobs_.size());
    }
    return inflight_count_.load(std::memory_order_acquire) + queued;
#else
    return 0;
#endif
}

#ifdef HAVE_LIBURING
std::shared_ptr<AsyncExpertReader::RequestInfo> AsyncExpertReader::get_request(uint64_t request_id) const {
    std::lock_guard<std::mutex> lock(request_map_mutex_);
    auto it = requests_.find(request_id);
    return it == requests_.end() ? nullptr : it->second;
}

void AsyncExpertReader::push_completion_event(int expert_id, bool ok) {
    {
        std::lock_guard<std::mutex> lock(completion_events_mutex_);
        completion_events_.push_back(CompletionEvent{expert_id, ok});
    }
    completion_events_cv_.notify_all();
}

void AsyncExpertReader::complete_request(uint64_t request_id,
                                         const std::shared_ptr<RequestInfo>& request,
                                         bool ok,
                                         int result) {
    if (request == nullptr) return;
    {
        std::lock_guard<std::mutex> request_lock(request->mutex);
        request->result.store(result, std::memory_order_release);
        request->state.store(ok ? RequestState::Completed : RequestState::Failed, std::memory_order_release);
    }
    if (ok) {
        std::lock_guard<std::mutex> lock(request_map_mutex_);
        completed_experts_.insert(request->expert_id);
    }
    request->cv.notify_all();
    push_completion_event(request->expert_id, ok);
    (void)request_id;
}

bool AsyncExpertReader::prep_read(uint64_t request_id, const std::shared_ptr<RequestInfo>& request) {
    if (request == nullptr) return false;
    while (true) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (sqe != nullptr) {
            io_uring_prep_read(sqe,
                               request->fd,
                               request->buffer,
                               request->expected_size,
                               request->offset);
            io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(request_id));
            request->state.store(RequestState::Inflight, std::memory_order_release);
            request->result.store(0, std::memory_order_release);
            inflight_count_.fetch_add(1, std::memory_order_acq_rel);
            return true;
        }
        if (!flush_submissions()) {
            return false;
        }
        (void)process_one_completion(0);
    }
}

bool AsyncExpertReader::flush_submissions() {
    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
        std::fprintf(stderr, "[MESHIO][submit_failed] result=%d(%s)\n", ret, std::strerror(-ret));
        return false;
    }
    return true;
}

bool AsyncExpertReader::resubmit_read(uint64_t request_id,
                                      const std::shared_ptr<RequestInfo>& request,
                                      int failed_result) {
    if (request == nullptr) return false;
    const int retry = request->retry_count.load(std::memory_order_acquire);
    if (retry >= max_read_retries_) {
        return false;
    }
    request->retry_count.store(retry + 1, std::memory_order_release);
    log_read_status("read_retry",
                    request_id,
                    request->expert_id,
                    retry + 1,
                    max_read_retries_,
                    request->fd,
                    request->offset,
                    request->expected_size,
                    failed_result);
    if (!prep_read(request_id, request)) {
        request->result.store(-EIO, std::memory_order_release);
        return false;
    }
    if (!flush_submissions()) {
        inflight_count_.fetch_sub(1, std::memory_order_acq_rel);
        request->result.store(-EIO, std::memory_order_release);
        return false;
    }
    return true;
}

bool AsyncExpertReader::process_one_completion(int timeout_ms) {
    struct io_uring_cqe* cqe = nullptr;
    int ret = 0;
    if (timeout_ms < 0) {
        ret = io_uring_wait_cqe(&ring_, &cqe);
    } else if (timeout_ms == 0) {
        ret = io_uring_peek_cqe(&ring_, &cqe);
    } else {
        struct __kernel_timespec ts;
        ts.tv_sec = timeout_ms / 1000;
        ts.tv_nsec = (timeout_ms % 1000) * 1000000L;
        ret = io_uring_wait_cqe_timeout(&ring_, &cqe, &ts);
    }
    if (ret < 0) {
        return false;
    }

    const uint64_t request_id = reinterpret_cast<uint64_t>(io_uring_cqe_get_data(cqe));
    const int result = cqe->res;
    io_uring_cqe_seen(&ring_, cqe);
    auto request = get_request(request_id);
    if (request == nullptr) {
        inflight_count_.fetch_sub(1, std::memory_order_acq_rel);
        return true;
    }

    const bool ok = result >= 0 && static_cast<size_t>(result) == request->expected_size;
    if (ok) {
        inflight_count_.fetch_sub(1, std::memory_order_acq_rel);
        complete_request(request_id, request, true, result);
        return true;
    }

    inflight_count_.fetch_sub(1, std::memory_order_acq_rel);
    if (resubmit_read(request_id, request, result)) {
        return true;
    }

    request->retry_count.store(std::min(request->retry_count.load(std::memory_order_acquire), max_read_retries_),
                               std::memory_order_release);
    log_read_status("read_failed",
                    request_id,
                    request->expert_id,
                    request->retry_count.load(std::memory_order_acquire),
                    max_read_retries_,
                    request->fd,
                    request->offset,
                    request->expected_size,
                    result);
    complete_request(request_id, request, false, result);
    return true;
}

void AsyncExpertReader::fail_queued_jobs() {
    std::deque<ReadJob> jobs;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        jobs.swap(pending_jobs_);
    }
    for (auto& job : jobs) {
        complete_request(job.request_id, job.request, false, -ECANCELED);
    }
}

void AsyncExpertReader::io_thread_main() {
    while (true) {
        std::deque<ReadJob> jobs;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (pending_jobs_.empty() && inflight_count_.load(std::memory_order_acquire) == 0 &&
                !stop_requested_.load(std::memory_order_acquire)) {
                queue_cv_.wait(lock, [this]() {
                    return stop_requested_.load(std::memory_order_acquire) || !pending_jobs_.empty();
                });
            }
            jobs.swap(pending_jobs_);
        }

        if (!jobs.empty()) {
            std::vector<ReadJob> submitted_jobs;
            submitted_jobs.reserve(jobs.size());
            bool submit_ok = true;
            for (auto& job : jobs) {
                if (!prep_read(job.request_id, job.request)) {
                    complete_request(job.request_id, job.request, false, -EBUSY);
                    submit_ok = false;
                    continue;
                }
                submitted_jobs.push_back(job);
            }
            if (!flush_submissions()) {
                submit_ok = false;
                for (auto& job : submitted_jobs) {
                    inflight_count_.fetch_sub(1, std::memory_order_acq_rel);
                    complete_request(job.request_id, job.request, false, -EIO);
                }
            }
            if (submit_ok) {
                while (process_one_completion(0)) {
                }
            }
        } else if (inflight_count_.load(std::memory_order_acquire) > 0) {
            (void)process_one_completion(1);
            while (process_one_completion(0)) {
            }
        }

        if (stop_requested_.load(std::memory_order_acquire)) {
            bool no_pending_jobs = false;
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                no_pending_jobs = pending_jobs_.empty();
            }
            if (no_pending_jobs && inflight_count_.load(std::memory_order_acquire) == 0) {
                break;
            }
        }
    }
    fail_queued_jobs();
}

#endif

}  // namespace ktransformers
