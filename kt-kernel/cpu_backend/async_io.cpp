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
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <sstream>

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
    std::lock_guard<std::mutex> ring_lock(ring_mutex_);

    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
        throw std::runtime_error("io_uring submission queue full");
    }

    uint64_t user_data = next_user_data_++;
    io_uring_prep_read(sqe, fd, buf, size, offset);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(user_data));

    {
        std::lock_guard<std::mutex> lock(mutex_);
        inflight_requests_[user_data] = PendingRequest{expert_id, size};
        completed_requests_.erase(user_data);
        failed_requests_.erase(user_data);
        request_results_.erase(user_data);
    }

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        inflight_requests_.erase(user_data);
        request_results_[user_data] = ret;
        throw std::runtime_error("io_uring_submit failed: " + std::string(strerror(-ret)));
    }

    return user_data;
#else
    throw std::runtime_error("io_uring not available on this platform");
#endif
}

int AsyncExpertReader::wait_one_completion() {
#ifdef HAVE_LIBURING
    uint64_t request_id = 0;
    int expert_id = -1;
    bool ok = false;
    int ret = wait_and_record_completion(-1, &request_id, &expert_id, &ok);
    if (ret < 0) {
        return -1;
    }

    (void)request_id;
    return ok ? expert_id : -1;
#else
    return -1;
#endif
}

std::vector<int> AsyncExpertReader::poll_completions() {
    std::vector<int> completed;

#ifdef HAVE_LIBURING
    while (true) {
        uint64_t request_id = 0;
        int expert_id = -1;
        bool ok = false;
        int ret = wait_and_record_completion(0, &request_id, &expert_id, &ok);
        if (ret < 0) {
            break;  // No more completions
        }
        if (ok) {
            completed.push_back(expert_id);
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
                if (pair.second.expert_id == expert_id) {
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
        int wait_ms = std::min(remaining_ms, 10);
        uint64_t request_id = 0;
        int completed_id = -1;
        bool ok = false;
        int ret = wait_and_record_completion(wait_ms, &request_id, &completed_id, &ok);
        if (ret < 0) {
            continue;  // Timeout or error, retry
        }

        (void)request_id;
        if (completed_id == expert_id) {
            return ok;
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
        int wait_ms = std::min(remaining_ms, 10);
        uint64_t completed_request_id = 0;
        int expert_id = -1;
        bool ok = false;
        int ret = wait_and_record_completion(wait_ms, &completed_request_id, &expert_id, &ok);
        if (ret < 0) {
            continue;
        }
        (void)completed_request_id;
        (void)expert_id;
        (void)ok;
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
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            bool all_completed = true;
            for (uint64_t request_id : request_ids) {
                if (completed_requests_.count(request_id) > 0) {
                    continue;
                }
                if (failed_requests_.count(request_id) > 0) {
                    return false;
                }
                if (inflight_requests_.count(request_id) == 0) {
                    return false;
                }
                all_completed = false;
            }
            if (all_completed) {
                return true;
            }
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed >= timeout_ms) {
            return false;
        }

        int remaining_ms = timeout_ms - static_cast<int>(elapsed);
        int wait_ms = std::min(remaining_ms, 10);
        uint64_t completed_request_id = 0;
        int expert_id = -1;
        bool ok = false;
        int ret = wait_and_record_completion(wait_ms, &completed_request_id, &expert_id, &ok);
        (void)completed_request_id;
        (void)expert_id;
        (void)ok;
        if (ret < 0) {
            continue;
        }
    }
#else
    (void)request_ids;
    (void)timeout_ms;
    return false;
#endif
}

void AsyncExpertReader::submit_batch(const std::vector<ReadRequest>& requests) {
#ifdef HAVE_LIBURING
    std::lock_guard<std::mutex> ring_lock(ring_mutex_);

    for (const auto& req : requests) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            throw std::runtime_error("io_uring submission queue full during batch submit");
        }

        uint64_t user_data = next_user_data_++;
        io_uring_prep_read(sqe, req.fd, req.buffer, req.size, req.offset);
        io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(user_data));

        {
            std::lock_guard<std::mutex> lock(mutex_);
            inflight_requests_[user_data] = PendingRequest{req.expert_id, req.size};
            completed_requests_.erase(user_data);
            failed_requests_.erase(user_data);
            request_results_.erase(user_data);
        }
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

int AsyncExpertReader::get_request_result(uint64_t request_id) const {
#ifdef HAVE_LIBURING
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = request_results_.find(request_id);
    return it == request_results_.end() ? INT_MIN : it->second;
#else
    (void)request_id;
    return INT_MIN;
#endif
}

std::string AsyncExpertReader::describe_requests(const std::vector<uint64_t>& request_ids) const {
#ifdef HAVE_LIBURING
    std::ostringstream oss;
    std::lock_guard<std::mutex> lock(mutex_);
    oss << "[";
    bool first = true;
    for (uint64_t request_id : request_ids) {
        if (!first) {
            oss << ",";
        }
        first = false;
        oss << request_id << ":";
        if (completed_requests_.count(request_id) > 0) {
            oss << "ok";
        } else if (failed_requests_.count(request_id) > 0) {
            oss << "fail";
        } else if (inflight_requests_.count(request_id) > 0) {
            oss << "inflight";
        } else {
            oss << "missing";
        }
        auto result_it = request_results_.find(request_id);
        if (result_it != request_results_.end()) {
            const int result = result_it->second;
            oss << "/res=" << result;
            if (result < 0) {
                oss << "(" << std::strerror(-result) << ")";
            }
        }
    }
    oss << "]";
    return oss.str();
#else
    (void)request_ids;
    return "[]";
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
int AsyncExpertReader::wait_and_record_completion(int timeout_ms,
                                                  uint64_t* request_id_out,
                                                  int* expert_id_out,
                                                  bool* ok_out) {
    struct io_uring_cqe* cqe = nullptr;
    uint64_t user_data = 0;
    int result = 0;
    {
        std::lock_guard<std::mutex> ring_lock(ring_mutex_);
        int ret = wait_cqe_internal(&cqe, timeout_ms);
        if (ret < 0) {
            return ret;
        }

        user_data = reinterpret_cast<uint64_t>(io_uring_cqe_get_data(cqe));
        result = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);
    }

    int expert_id = -1;
    bool ok = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = inflight_requests_.find(user_data);
        if (it != inflight_requests_.end()) {
            expert_id = it->second.expert_id;
            const size_t expected_size = it->second.size;
            inflight_requests_.erase(it);
            request_results_[user_data] = result;

            ok = result >= 0 && static_cast<size_t>(result) == expected_size;
            if (ok) {
                completed_experts_.insert(expert_id);
                completed_requests_.insert(user_data);
            } else {
                failed_requests_.insert(user_data);
            }
        } else {
            request_results_[user_data] = result;
        }
    }

    if (request_id_out != nullptr) {
        *request_id_out = user_data;
    }
    if (expert_id_out != nullptr) {
        *expert_id_out = expert_id;
    }
    if (ok_out != nullptr) {
        *ok_out = ok;
    }
    return 0;
}

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
