/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-08-05 04:49:08
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-08-05 09:21:29
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "shared_mem_buffer.h"
#include <cstdio>

SharedMemBuffer::SharedMemBuffer() {
    buffer_ = nullptr;
    size_ = 0;
}

SharedMemBuffer::~SharedMemBuffer() {
    if (buffer_) {
        free(buffer_);
    }
}

void SharedMemBuffer::alloc(void* object, std::vector<std::pair<void**, uint64_t>> requests) {
    printf("[DEBUG SharedMemBuffer::alloc] object=%p, num_requests=%zu\n",
           object, requests.size());

    // Calculate total size with 64-byte alignment for each request
    uint64_t size = 0;
    for (size_t i = 0; i < requests.size(); i++) {
        // Align each buffer size to 64 bytes
        uint64_t aligned_size = (requests[i].second + 63) & ~63ULL;
        printf("  Request %zu: size=%llu, aligned=%llu\n",
               i, (unsigned long long)requests[i].second, (unsigned long long)aligned_size);
        size += aligned_size;
    }

    printf("  Total size for this object: %llu bytes (%.2f MB)\n",
           (unsigned long long)size, size / 1024.0 / 1024.0);
    printf("  Current buffer size: %llu bytes (%.2f MB)\n",
           (unsigned long long)size_, size_ / 1024.0 / 1024.0);
    printf("  Number of historical objects: %zu\n", hist_requests_.size());

    if (size > size_) {
        printf("  [DEBUG] Reallocating buffer: old_size=%llu, new_size=%llu\n",
               (unsigned long long)size_, (unsigned long long)size);
        if (buffer_) {
            free(buffer_);
        }
        // Now size is guaranteed to be a multiple of 64
        buffer_ = std::aligned_alloc(64, size);
        if (!buffer_) {
            throw std::bad_alloc();
        }

        size_ = size;
        printf("  [DEBUG] Re-arranging %zu historical objects...\n", hist_requests_.size());
        for (auto& obj_requests : hist_requests_) {
            for (auto& requests : obj_requests.second) {
                arrange(requests);
            }
        }
    } else {
        printf("  [DEBUG] No reallocation needed (size <= size_)\n");
    }

    printf("  [DEBUG] Arranging current object...\n");
    arrange(requests);
    hist_requests_[object].push_back(requests);
    printf("[DEBUG SharedMemBuffer::alloc] Completed for object=%p\n\n", object);
}

void SharedMemBuffer::dealloc(void* object) {
    hist_requests_.erase(object);
}

void SharedMemBuffer::arrange(std::vector<std::pair<void**, uint64_t>> requests) {
    printf("[DEBUG SharedMemBuffer::arrange] Starting from offset=0, buffer_=%p\n", buffer_);
    uint64_t offset = 0;
    for (size_t i = 0; i < requests.size(); i++) {
        void** ptr_location = requests[i].first;
        void* assigned_ptr = (uint8_t*)buffer_ + offset;
        *(ptr_location) = assigned_ptr;

        // Align offset to 64-byte boundary for next buffer
        uint64_t aligned_size = (requests[i].second + 63) & ~63ULL;
        printf("  Buffer %zu: ptr=%p, offset=%llu, size=%llu, aligned=%llu\n",
               i, assigned_ptr, (unsigned long long)offset,
               (unsigned long long)requests[i].second, (unsigned long long)aligned_size);
        offset += aligned_size;
    }
    printf("[DEBUG SharedMemBuffer::arrange] Final offset=%llu (%.2f MB)\n\n",
           (unsigned long long)offset, offset / 1024.0 / 1024.0);
}