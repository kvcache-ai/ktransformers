/**
 * @Description  : Asynchronous I/O wrapper using io_uring for expert weight loading
 * @Author       : RaQiu
 * @Date         : 2026-05-06
 * @Version      : 1.0.0
 * @Copyright (c) 2026 by KVCache.AI, All Rights Reserved.
 **/

#ifndef CPUINFER_ASYNC_IO_HPP
#define CPUINFER_ASYNC_IO_HPP

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef HAVE_LIBURING
#include <liburing.h>
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif
#endif

namespace ktransformers {

/**
 * @brief Asynchronous expert reader using io_uring
 *
 * Provides zero-copy, async I/O for loading expert weights from disk.
 * Eliminates dependency on OS page cache and provides completion notifications.
 */
class AsyncExpertReader {
public:
    struct ReadRequest {
        int expert_id;
        int fd;
        void* buffer;
        size_t size;
        off_t offset;
        uint64_t user_data;  // For completion matching
    };

    struct SubmitStats {
        uint64_t total_us = 0;
        uint64_t lock_wait_us = 0;
        uint64_t sqe_prep_us = 0;
        uint64_t request_bookkeeping_us = 0;
        uint64_t ring_submit_us = 0;
        uint64_t request_count = 0;
        uint64_t flush_count = 0;
    };

    /**
     * @brief Construct AsyncExpertReader
     * @param queue_depth Maximum number of in-flight requests (default 128)
     */
    explicit AsyncExpertReader(int queue_depth = 128);

    ~AsyncExpertReader();

    // Disable copy
    AsyncExpertReader(const AsyncExpertReader&) = delete;
    AsyncExpertReader& operator=(const AsyncExpertReader&) = delete;

    /**
     * @brief Submit an async read request (non-blocking)
     * @param fd File descriptor
     * @param buf Target buffer (must be aligned for O_DIRECT)
     * @param size Number of bytes to read
     * @param offset File offset
     * @param expert_id Expert ID for tracking
     * @return Request ID for later querying
     */
    uint64_t submit_read(int fd, void* buf, size_t size, off_t offset, int expert_id);

    /**
     * @brief Submit multiple async read requests with one batched ring flush.
     * @return Request IDs in the same order as the input requests.
     */
    std::vector<uint64_t> submit_reads(const std::vector<ReadRequest>& requests, SubmitStats* stats = nullptr);

    /**
     * @brief Wait for at least one request to complete (blocking)
     * @return Expert ID of completed request, or -1 on error
     */
    int wait_one_completion();

    /**
     * @brief Non-blocking check for completed requests
     * @return List of completed expert IDs
     */
    std::vector<int> poll_completions();

    /**
     * @brief Wait for a specific expert to complete (blocking with timeout)
     * @param expert_id Expert ID to wait for
     * @param timeout_ms Timeout in milliseconds (default 5000)
     * @return true if completed, false if timeout or error
     */
    bool wait_for_expert(int expert_id, int timeout_ms = 5000);

    /**
     * @brief Wait for a specific request to complete (blocking with timeout)
     * @param request_id Request ID returned by submit_read
     * @param timeout_ms Timeout in milliseconds (default 5000)
     * @return true if completed successfully, false if timeout or error
     */
    bool wait_for_request(uint64_t request_id, int timeout_ms = 5000);

    /**
     * @brief Wait for all requests in a batch to complete
     * @param request_ids Request IDs returned by submit_read
     * @param timeout_ms Timeout in milliseconds for the whole batch
     * @return true if all completed successfully, false if any timeout or error
     */
    bool wait_for_requests(const std::vector<uint64_t>& request_ids, int timeout_ms = 5000);

    /**
     * @brief Return the last io_uring result for a request.
     *
     * A successful full read returns the byte count. Failed requests return the
     * negative errno reported by io_uring. Unknown or still-in-flight requests
     * return INT_MIN.
     */
    int get_request_result(uint64_t request_id) const;

    /**
     * @brief Return true only when a request completed as a full-size read.
     */
    bool request_succeeded(uint64_t request_id) const;

    /**
     * @brief Produce a compact status summary for diagnostics.
     */
    std::string describe_requests(const std::vector<uint64_t>& request_ids) const;

    /**
     * @brief Submit multiple read requests in batch
     * @param requests Vector of read requests
     */
    void submit_batch(const std::vector<ReadRequest>& requests);

    /**
     * @brief Get number of in-flight requests
     */
    int get_inflight_count() const;

private:
#ifdef HAVE_LIBURING
    enum class RequestState : uint8_t {
        Inflight,
        Completed,
        Failed
    };

    struct RequestInfo {
        int expert_id = -1;
        int fd = -1;
        void* buffer = nullptr;
        size_t expected_size = 0;
        off_t offset = 0;
        std::atomic<RequestState> state{RequestState::Inflight};
        std::atomic<int> result{0};  // errno or byte count
        std::atomic<int> retry_count{0};
        mutable std::mutex mutex;
        std::condition_variable cv;
    };

    struct ReadJob {
        uint64_t request_id = 0;
        std::shared_ptr<RequestInfo> request;
    };

    struct CompletionEvent {
        int expert_id = -1;
        bool ok = false;
    };

    struct io_uring ring_;
    int queue_depth_;
    std::atomic<uint64_t> next_user_data_;
    int max_read_retries_;

    std::unordered_map<uint64_t, std::shared_ptr<RequestInfo>> requests_;
    std::unordered_set<int> completed_experts_;  // Track completed expert IDs
    mutable std::mutex request_map_mutex_;

    std::deque<ReadJob> pending_jobs_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::deque<CompletionEvent> completion_events_;
    mutable std::mutex completion_events_mutex_;
    std::condition_variable completion_events_cv_;

    std::thread io_thread_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<int> inflight_count_{0};

    std::shared_ptr<RequestInfo> get_request(uint64_t request_id) const;
    void io_thread_main();
    bool prep_read(uint64_t request_id, const std::shared_ptr<RequestInfo>& request);
    bool flush_submissions();
    bool process_one_completion(int timeout_ms);
    bool resubmit_read(uint64_t request_id, const std::shared_ptr<RequestInfo>& request, int failed_result);
    void complete_request(uint64_t request_id, const std::shared_ptr<RequestInfo>& request, bool ok, int result);
    void fail_queued_jobs();
    void push_completion_event(int expert_id, bool ok);
#else
    // Stub members for non-Linux platforms
    int queue_depth_;
    uint64_t next_user_data_;
    int max_read_retries_;
#endif
};

}  // namespace ktransformers

#endif  // CPUINFER_ASYNC_IO_HPP
