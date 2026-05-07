/**
 * @Description  : Asynchronous I/O wrapper using io_uring for expert weight loading
 * @Author       : RaQiu
 * @Date         : 2026-05-06
 * @Version      : 1.0.0
 * @Copyright (c) 2026 by KVCache.AI, All Rights Reserved.
 **/

#include "async_io.hpp"
#include <cstring>
#include <stdexcept>
#include <chrono>

#ifdef HAVE_LIBURING
#include <sys/time.h>
#include <errno.h>
#endif

namespace ktransformers {

AsyncExpertReader::AsyncExpertReader(int queue_depth)
    : queue_depth_(queue_depth), next_user_data_(1) {
#ifdef HAVE_LIBURING
    int ret = io_uring_queue_init(queue_depth_, &ring_, 0);
    if (ret < 0) {
        throw std::runtime_error("Failed to initialize io_uring: " + std::string(strerror(-ret)));
    }
#else
    // Non-Linux platforms: no-op constructor
    (void)queue_depth_;
#endif
}

AsyncExpertReader::~AsyncExpertReader() {
#ifdef HAVE_LIBURING
    io_uring_queue_exit(&ring_);
#endif
}

uint64_t AsyncExpertReader::submit_read(int fd, void* buf, size_t size, off_t offset, int expert_id) {
#ifdef HAVE_LIBURING
    std::lock_guard<std::mutex> lock(mutex_);

    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
        throw std::runtime_error("io_uring submission queue full");
    }

    uint64_t user_data = next_user_data_++;
    io_uring_prep_read(sqe, fd, buf, size, offset);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(user_data));

    inflight_requests_[user_data] = expert_id;

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
        inflight_requests_.erase(user_data);
        throw std::runtime_error("io_uring_submit failed: " + std::string(strerror(-ret)));
    }

    return user_data;
#else
    throw std::runtime_error("io_uring not available on this platform");
#endif
}

int AsyncExpertReader::wait_one_completion() {
#ifdef HAVE_LIBURING
    struct io_uring_cqe* cqe = nullptr;
    int ret = wait_cqe_internal(&cqe, -1);  // Infinite timeout
    if (ret < 0) {
        return -1;
    }

    uint64_t user_data = reinterpret_cast<uint64_t>(io_uring_cqe_get_data(cqe));
    int result = cqe->res;
    io_uring_cqe_seen(&ring_, cqe);

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = inflight_requests_.find(user_data);
    if (it == inflight_requests_.end()) {
        return -1;  // Unknown request
    }

    int expert_id = it->second;
    inflight_requests_.erase(it);

    if (result < 0) {
        failed_requests_.insert(user_data);
        // Read error
        return -1;
    }

    completed_experts_.insert(expert_id);
    completed_requests_.insert(user_data);
    return expert_id;
#else
    return -1;
#endif
}

std::vector<int> AsyncExpertReader::poll_completions() {
    std::vector<int> completed;

#ifdef HAVE_LIBURING
    struct io_uring_cqe* cqe = nullptr;
    while (true) {
        int ret = wait_cqe_internal(&cqe, 0);  // Non-blocking
        if (ret < 0) {
            break;  // No more completions
        }

        uint64_t user_data = reinterpret_cast<uint64_t>(io_uring_cqe_get_data(cqe));
        int result = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = inflight_requests_.find(user_data);
    if (it != inflight_requests_.end()) {
      if (result >= 0) {
        completed.push_back(it->second);
        completed_experts_.insert(it->second);
        completed_requests_.insert(user_data);
      } else {
        failed_requests_.insert(user_data);
      }
      inflight_requests_.erase(it);
    }
    }
#endif

    return completed;
}

bool AsyncExpertReader::wait_for_expert(int expert_id, int timeout_ms) {
#ifdef HAVE_LIBURING
    auto start = std::chrono::steady_clock::now();

    while (true) {
        // Check if already completed
        {
            std::lock_guard<std::mutex> lock(mutex_);
            bool found = false;
            for (const auto& pair : inflight_requests_) {
                if (pair.second == expert_id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Check if it was completed before
                if (completed_experts_.count(expert_id) > 0) {
                    return true;  // Already completed
                }
                return false;  // Never submitted or timed out
            }
        }

        // Check timeout
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed >= timeout_ms) {
            return false;  // Timeout
        }

        // Wait for next completion
        int remaining_ms = timeout_ms - static_cast<int>(elapsed);
        struct io_uring_cqe* cqe = nullptr;
        int ret = wait_cqe_internal(&cqe, remaining_ms);
        if (ret < 0) {
            continue;  // Timeout or error, retry
        }

        uint64_t user_data = reinterpret_cast<uint64_t>(io_uring_cqe_get_data(cqe));
        int result = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = inflight_requests_.find(user_data);
        if (it != inflight_requests_.end()) {
            int completed_id = it->second;
            inflight_requests_.erase(it);

            if (result >= 0) {
                completed_experts_.insert(completed_id);
                completed_requests_.insert(user_data);
                if (completed_id == expert_id) {
                    return true;  // Target expert completed
                }
            } else {
                failed_requests_.insert(user_data);
                if (completed_id == expert_id) {
                    return false;
                }
            }
        }
    }
#else
    (void)expert_id;
    (void)timeout_ms;
    return false;
#endif
}

bool AsyncExpertReader::wait_for_request(uint64_t request_id, int timeout_ms) {
#ifdef HAVE_LIBURING
    auto start = std::chrono::steady_clock::now();

    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (completed_requests_.count(request_id) > 0) {
                return true;
            }
            if (failed_requests_.count(request_id) > 0) {
                return false;
            }
            if (inflight_requests_.count(request_id) == 0) {
                return false;
            }
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed >= timeout_ms) {
            return false;
        }

        int remaining_ms = timeout_ms - static_cast<int>(elapsed);
        struct io_uring_cqe* cqe = nullptr;
        int ret = wait_cqe_internal(&cqe, remaining_ms);
        if (ret < 0) {
            continue;
        }

        uint64_t user_data = reinterpret_cast<uint64_t>(io_uring_cqe_get_data(cqe));
        int result = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = inflight_requests_.find(user_data);
        if (it == inflight_requests_.end()) {
            continue;
        }
        int completed_id = it->second;
        inflight_requests_.erase(it);
        if (result >= 0) {
            completed_experts_.insert(completed_id);
            completed_requests_.insert(user_data);
        } else {
            failed_requests_.insert(user_data);
        }
    }
#else
    (void)request_id;
    (void)timeout_ms;
    return false;
#endif
}

bool AsyncExpertReader::wait_for_requests(const std::vector<uint64_t>& request_ids, int timeout_ms) {
#ifdef HAVE_LIBURING
    auto start = std::chrono::steady_clock::now();
    for (uint64_t request_id : request_ids) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed >= timeout_ms) {
            return false;
        }
        int remaining_ms = timeout_ms - static_cast<int>(elapsed);
        if (!wait_for_request(request_id, remaining_ms)) {
            return false;
        }
    }
    return true;
#else
    (void)request_ids;
    (void)timeout_ms;
    return false;
#endif
}

void AsyncExpertReader::submit_batch(const std::vector<ReadRequest>& requests) {
#ifdef HAVE_LIBURING
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& req : requests) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            throw std::runtime_error("io_uring submission queue full during batch submit");
        }

        uint64_t user_data = next_user_data_++;
        io_uring_prep_read(sqe, req.fd, req.buffer, req.size, req.offset);
        io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(user_data));

        inflight_requests_[user_data] = req.expert_id;
    }

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
        throw std::runtime_error("io_uring_submit failed in batch: " + std::string(strerror(-ret)));
    }
#else
    (void)requests;
    throw std::runtime_error("io_uring not available on this platform");
#endif
}

int AsyncExpertReader::get_inflight_count() const {
#ifdef HAVE_LIBURING
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(inflight_requests_.size());
#else
    return 0;
#endif
}

#ifdef HAVE_LIBURING
int AsyncExpertReader::wait_cqe_internal(struct io_uring_cqe** cqe_out, int timeout_ms) {
    if (timeout_ms < 0) {
        // Infinite wait
        return io_uring_wait_cqe(&ring_, cqe_out);
    } else if (timeout_ms == 0) {
        // Non-blocking peek
        return io_uring_peek_cqe(&ring_, cqe_out);
    } else {
        // Timed wait
        struct __kernel_timespec ts;
        ts.tv_sec = timeout_ms / 1000;
        ts.tv_nsec = (timeout_ms % 1000) * 1000000L;
        return io_uring_wait_cqe_timeout(&ring_, cqe_out, &ts);
    }
}
#endif

}  // namespace ktransformers
