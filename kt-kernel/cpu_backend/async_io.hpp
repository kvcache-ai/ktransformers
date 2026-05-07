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
    struct io_uring ring_;
    int queue_depth_;
    uint64_t next_user_data_;

    // Track in-flight requests: user_data -> expert_id
    std::unordered_map<uint64_t, int> inflight_requests_;
    std::unordered_set<int> completed_experts_;  // Track completed expert IDs
    std::unordered_set<uint64_t> completed_requests_;
    std::unordered_set<uint64_t> failed_requests_;
    mutable std::mutex mutex_;

    // Helper: wait for completion events
    int wait_cqe_internal(struct io_uring_cqe** cqe_out, int timeout_ms);
#else
    // Stub members for non-Linux platforms
    int queue_depth_;
    uint64_t next_user_data_;
#endif
};

}  // namespace ktransformers

#endif  // CPUINFER_ASYNC_IO_HPP
