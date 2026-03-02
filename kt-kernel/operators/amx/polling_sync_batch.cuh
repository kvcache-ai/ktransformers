/**
 * @Description  : Polling-based CPU-GPU synchronization for batch expert loading
 * @Date         : 2025-01-27
 *
 * Features:
 * - Persistent GPU kernel polls shared CPU memory for data readiness
 * - Thread-cooperative memcpy for high bandwidth
 * - Single sync slot per rank to minimize PCIe polling overhead
 * - Designed for TP>1 expert weight loading without CUDA IPC
 *
 * Note: This header can be included in both CUDA (.cu) and C++ (.cpp) files.
 * The CUDA kernel code is guarded with __CUDACC__ and only compiled by nvcc.
 */

#ifndef POLLING_SYNC_BATCH_CUH
#define POLLING_SYNC_BATCH_CUH

#include <cstdint>
#include <atomic>

// Always include cuda_runtime.h for CUDA types when available
#ifndef KTRANSFORMERS_CPU_ONLY
#include <cuda_runtime.h>
#endif

namespace polling_sync_batch {

// Signal states for CPU-GPU synchronization
enum SignalState : int32_t {
    SIGNAL_IDLE = 0,        // No data ready, waiting
    SIGNAL_DATA_READY = 1,  // CPU has written data, GPU should copy
    SIGNAL_GPU_DONE = 2,    // GPU copy complete, CPU can proceed
    SIGNAL_SHUTDOWN = 3,    // Terminate polling kernel
};

// Sync slot structure - must be in pinned memory for GPU polling
// Cache line aligned to avoid false sharing
struct alignas(64) BatchSyncSlot {
    volatile int32_t signal;      // Signal state
    volatile int32_t expert_id;   // Current expert ID to load
    volatile int32_t slot_idx;    // Double buffer slot index (0 or 1)
    volatile int32_t processed_count;  // Number of experts processed (for debugging)
    char _padding[48];            // Pad to 64 bytes (cache line)
};

// ============================================================================
// CPU-side API (works in both CUDA and non-CUDA builds)
// ============================================================================

/**
 * CPU side: Signal that data is ready for GPU to copy
 *
 * @param slot       Sync slot
 * @param expert_id  Expert ID to load
 * @param slot_idx   Double buffer slot index (0 or 1)
 */
inline void cpu_signal_data_ready(BatchSyncSlot* slot, int expert_id, int slot_idx) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    slot->expert_id = expert_id;
    slot->slot_idx = slot_idx;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    slot->signal = SIGNAL_DATA_READY;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

/**
 * CPU side: Wait for GPU to complete copy
 *
 * @param slot  Sync slot
 * @return true if GPU signaled done, false if timeout/error
 */
inline bool cpu_wait_gpu_done(BatchSyncSlot* slot) {
    while (true) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int32_t sig = slot->signal;

        if (sig == SIGNAL_GPU_DONE) {
            // Reset to idle for next iteration
            std::atomic_thread_fence(std::memory_order_seq_cst);
            slot->signal = SIGNAL_IDLE;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            return true;
        }

        // Could add timeout here if needed
    }
}

/**
 * CPU side: Signal kernel to shutdown
 */
inline void cpu_signal_shutdown(BatchSyncSlot* slot) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    slot->signal = SIGNAL_SHUTDOWN;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

// ============================================================================
// CUDA-specific code (only compiled by nvcc)
// ============================================================================
#ifdef __CUDACC__

// Device-side memory fence
__device__ __forceinline__ void gpu_memory_fence() {
    __threadfence_system();
}

// Thread-cooperative memcpy kernel helper
__device__ __forceinline__ void cooperative_memcpy(
    void* __restrict__ dst,
    const void* __restrict__ src,
    size_t nbytes,
    int thread_id,
    int num_threads)
{
    // Copy as uint4 (16 bytes) for maximum throughput
    const size_t num_uint4 = nbytes / sizeof(uint4);
    const size_t remainder = nbytes % sizeof(uint4);

    uint4* dst4 = (uint4*)dst;
    const uint4* src4 = (const uint4*)src;

    // Each thread copies a strided portion
    for (size_t i = thread_id; i < num_uint4; i += num_threads) {
        dst4[i] = src4[i];
    }

    // Handle remainder bytes (only thread 0)
    if (remainder > 0 && thread_id == 0) {
        char* dst_end = (char*)dst + num_uint4 * sizeof(uint4);
        const char* src_end = (const char*)src + num_uint4 * sizeof(uint4);
        for (size_t i = 0; i < remainder; i++) {
            dst_end[i] = src_end[i];
        }
    }
}

/**
 * Persistent polling kernel for batch expert loading.
 *
 * Each rank launches this kernel on its own GPU. The kernel:
 * 1. Polls sync_slot->signal for SIGNAL_DATA_READY
 * 2. Reads expert_id and slot_idx
 * 3. Copies weight data from pinned CPU buffer to GPU memory
 * 4. Sets signal = SIGNAL_GPU_DONE
 * 5. Repeats until total_experts reached or SIGNAL_SHUTDOWN
 */
__global__ void polling_copy_kernel(
    BatchSyncSlot* sync_slot,
    // Source: double-buffered pinned CPU memory [slot0, slot1]
    void* src_w13_weight_slot0, void* src_w13_weight_slot1,
    void* src_w13_scale_slot0, void* src_w13_scale_slot1,
    void* src_w2_weight_slot0, void* src_w2_weight_slot1,
    void* src_w2_scale_slot0, void* src_w2_scale_slot1,
    // Destination: GPU memory (base pointers)
    void* dst_w13_weight, void* dst_w13_scale,
    void* dst_w2_weight, void* dst_w2_scale,
    // Sizes per expert
    size_t w13_weight_size, size_t w13_scale_size,
    size_t w2_weight_size, size_t w2_scale_size,
    // Number of experts to process
    int total_experts)
{
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    int processed = 0;

    // Source buffer arrays for easy indexing
    void* src_w13_weight[2] = {src_w13_weight_slot0, src_w13_weight_slot1};
    void* src_w13_scale[2] = {src_w13_scale_slot0, src_w13_scale_slot1};
    void* src_w2_weight[2] = {src_w2_weight_slot0, src_w2_weight_slot1};
    void* src_w2_scale[2] = {src_w2_scale_slot0, src_w2_scale_slot1};

    while (true) {
        // Check termination condition
        if (total_experts >= 0 && processed >= total_experts) {
            break;
        }

        // Poll for data ready signal
        int32_t sig;
        do {
            gpu_memory_fence();
            sig = sync_slot->signal;

            if (sig == SIGNAL_SHUTDOWN) {
                return;  // Exit kernel
            }
        } while (sig != SIGNAL_DATA_READY);

        // Read expert_id and slot_idx
        gpu_memory_fence();
        const int expert_id = sync_slot->expert_id;
        const int slot_idx = sync_slot->slot_idx;

        // Compute destination offsets based on expert_id
        char* dst_w13_w = (char*)dst_w13_weight + (size_t)expert_id * w13_weight_size;
        char* dst_w13_s = (char*)dst_w13_scale + (size_t)expert_id * w13_scale_size;
        char* dst_w2_w = (char*)dst_w2_weight + (size_t)expert_id * w2_weight_size;
        char* dst_w2_s = (char*)dst_w2_scale + (size_t)expert_id * w2_scale_size;

        // Thread-cooperative copy from pinned CPU buffer to GPU
        // All threads participate in the copy for maximum bandwidth
        cooperative_memcpy(dst_w13_w, src_w13_weight[slot_idx], w13_weight_size, tid, num_threads);
        cooperative_memcpy(dst_w13_s, src_w13_scale[slot_idx], w13_scale_size, tid, num_threads);
        cooperative_memcpy(dst_w2_w, src_w2_weight[slot_idx], w2_weight_size, tid, num_threads);
        cooperative_memcpy(dst_w2_s, src_w2_scale[slot_idx], w2_scale_size, tid, num_threads);

        // Ensure all copies are visible
        __syncthreads();
        gpu_memory_fence();

        // Signal completion (only thread 0)
        if (tid == 0) {
            sync_slot->processed_count = processed + 1;
            gpu_memory_fence();
            sync_slot->signal = SIGNAL_GPU_DONE;
            gpu_memory_fence();
        }

        processed++;
        __syncthreads();  // Ensure all threads see the incremented count
    }
}

/**
 * Create a sync slot in pinned memory
 */
inline cudaError_t create_batch_sync_slot(BatchSyncSlot** slot) {
    cudaError_t err = cudaHostAlloc(slot, sizeof(BatchSyncSlot),
                                     cudaHostAllocMapped | cudaHostAllocWriteCombined);
    if (err == cudaSuccess) {
        (*slot)->signal = SIGNAL_IDLE;
        (*slot)->expert_id = -1;
        (*slot)->slot_idx = 0;
        (*slot)->processed_count = 0;
    }
    return err;
}

/**
 * Destroy a sync slot
 */
inline cudaError_t destroy_batch_sync_slot(BatchSyncSlot* slot) {
    return cudaFreeHost(slot);
}

/**
 * Launch the persistent polling kernel
 */
inline void launch_polling_copy_kernel(
    cudaStream_t stream,
    BatchSyncSlot* sync_slot,
    void** src_buffers,  // [8] pointers
    void* dst_w13_weight, void* dst_w13_scale,
    void* dst_w2_weight, void* dst_w2_scale,
    size_t w13_weight_size, size_t w13_scale_size,
    size_t w2_weight_size, size_t w2_scale_size,
    int total_experts,
    int num_threads = 256)
{
    polling_copy_kernel<<<1, num_threads, 0, stream>>>(
        sync_slot,
        src_buffers[0], src_buffers[1],  // w13_weight slot0, slot1
        src_buffers[2], src_buffers[3],  // w13_scale slot0, slot1
        src_buffers[4], src_buffers[5],  // w2_weight slot0, slot1
        src_buffers[6], src_buffers[7],  // w2_scale slot0, slot1
        dst_w13_weight, dst_w13_scale,
        dst_w2_weight, dst_w2_scale,
        w13_weight_size, w13_scale_size,
        w2_weight_size, w2_scale_size,
        total_experts
    );
}

#endif  // __CUDACC__

} // namespace polling_sync_batch

#endif // POLLING_SYNC_BATCH_CUH
