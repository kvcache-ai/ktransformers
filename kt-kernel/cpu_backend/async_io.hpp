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
#include <mutex>
#include <string>
#include <sys/types.h>
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
        RequestState state = RequestState::Inflight;
        int result = 0;  // errno or byte count
        int retry_count = 0;
    };

    struct io_uring ring_;
    int queue_depth_;
    uint64_t next_user_data_;
    int max_read_retries_;

    // Unified request tracking
    std::unordered_map<uint64_t, RequestInfo> requests_;
    std::unordered_set<int> completed_experts_;  // Track completed expert IDs
    mutable std::mutex ring_mutex_;

    // Helper: wait for completion events
    int wait_cqe_internal(struct io_uring_cqe** cqe_out, int timeout_ms);
    int wait_and_record_completion(int timeout_ms, uint64_t* request_id_out, int* expert_id_out, bool* ok_out);
    bool resubmit_read_locked(uint64_t request_id, RequestInfo& info, int failed_result);
#else
    // Stub members for non-Linux platforms
    int queue_depth_;
    uint64_t next_user_data_;
    int max_read_retries_;
#endif
};

}  // namespace ktransformers

#endif  // CPUINFER_ASYNC_IO_HPP
