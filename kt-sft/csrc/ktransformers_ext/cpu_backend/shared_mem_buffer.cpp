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
    // Calculate total size with 64-byte alignment for each request
    uint64_t size = 0;
    for (size_t i = 0; i < requests.size(); i++) {
        // Align each buffer size to 64 bytes
        uint64_t aligned_size = (requests[i].second + 63) & ~63ULL;
        size += aligned_size;
    }

    if (size > size_) {
        if (buffer_) {
            free(buffer_);
        }
        // Now size is guaranteed to be a multiple of 64
        buffer_ = std::aligned_alloc(64, size);
        if (!buffer_) {
            throw std::bad_alloc();
        }

        size_ = size;
        for (auto& obj_requests : hist_requests_) {
            for (auto& requests : obj_requests.second) {
                arrange(requests);
            }
        }
    }

    arrange(requests);
    hist_requests_[object].push_back(requests);
}

void SharedMemBuffer::dealloc(void* object) {
    hist_requests_.erase(object);
}

void SharedMemBuffer::arrange(std::vector<std::pair<void**, uint64_t>> requests) {
    uint64_t offset = 0;
    for (size_t i = 0; i < requests.size(); i++) {
        void** ptr_location = requests[i].first;
        void* assigned_ptr = (uint8_t*)buffer_ + offset;
        *(ptr_location) = assigned_ptr;

        // Align offset to 64-byte boundary for next buffer
        uint64_t aligned_size = (requests[i].second + 63) & ~63ULL;
        offset += aligned_size;
    }
}