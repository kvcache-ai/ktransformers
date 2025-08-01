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
#include <numa.h>

SharedMemBuffer::SharedMemBuffer() {
  buffer_ = nullptr;
  size_ = 0;
}

SharedMemBuffer::~SharedMemBuffer() {
  if (buffer_) {
    free(buffer_);
  }
}

void SharedMemBuffer::alloc(void *object, std::vector<std::pair<void **, uint64_t>> requests) {
  uint64_t size = 0;
  for (auto &request : requests) {
    size += request.second;
  }
  // printf("alloc %lx, request count %ld, total_size %ld\n", reinterpret_cast<intptr_t>(object), requests.size(),
  // size);
  if (size > size_) {
    if (buffer_) {
      free(buffer_);
    }
    // printf("new alloc %ld bytes for %lx\n", size, reinterpret_cast<intptr_t>(object));
    buffer_ = std::aligned_alloc(64, size); // 或  std::nothrow_t{} 版本的 new
    if (!buffer_) {
      printf("cannot aligned alloc %ld bytes\n", size);
      perror("aligned_alloc"); // errno == ENOMEM/EINVAL
      exit(1);
    }

    size_ = size;
    for (auto &obj_requests : hist_requests_) {
      for (auto &requests : obj_requests.second) {
        arrange(requests);
      }
    }
  }
  arrange(requests);
  hist_requests_[object].push_back(requests);
}

void SharedMemBuffer::dealloc(void *object) { hist_requests_.erase(object); }

void SharedMemBuffer::arrange(std::vector<std::pair<void **, uint64_t>> requests) {
  uint64_t offset = 0;
  for (auto &request : requests) {
    *(request.first) = (uint8_t *)buffer_ + offset;
    offset += request.second;
  }
}

void SharedMemBufferNuma::alloc(int numa, void *object, std::vector<std::pair<void **, uint64_t>> requests) {
  std::lock_guard<std::mutex> guard(lock);
  if (numa != numa_node_of_cpu(sched_getcpu())) {
    printf("alloc %d from other numa for %lx\n", numa, reinterpret_cast<intptr_t>(object));
  }
  if (numa_mem.count(numa) == 0) {
    numa_mem[numa] = std::unique_ptr<SharedMemBuffer>(new SharedMemBuffer());
  }
  // printf("numa %d alloc for %lx\n", numa,reinterpret_cast<intptr_t> (object));
  numa_mem.at(numa)->alloc(object, requests);
}

void SharedMemBufferNuma::dealloc(void *object) {
  auto numa = numa_node_of_cpu(sched_getcpu());
  numa_mem.at(numa)->dealloc(object);
}
