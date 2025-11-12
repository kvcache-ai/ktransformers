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

#include <numa.h>

#include <cstdio>
#include <errno.h>

size_t MemoryRequest::total_size() {
  size_t total = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    total += sizes[i];
  }
  return total;
}

void MemoryRequest::update_base_ptr(void* base) {
  size_t total_offset = 0;
  for (size_t i = 0; i < funcs.size(); ++i) {
    funcs[i]((uint8_t*)base + total_offset);
    total_offset += sizes[i];
  }
}

void MemoryRequest::append_function(std::function<void(void*)> func, size_t size) {
  funcs.push_back(func);
  sizes.push_back(size);
}

SharedMemBuffer::SharedMemBuffer() {
  buffer = nullptr;
  size = 0;
}

SharedMemBuffer::~SharedMemBuffer() {
  if (buffer) {
    free(buffer);
  }
}

void SharedMemBuffer::alloc(void* object, MemoryRequest requests) {
  size_t total_size = requests.total_size();
  object_requests.push_back(requests);

  if (total_size > size) {
    if (buffer) {
      free(buffer);
    }
    void* newbuf = nullptr;
    int rc = posix_memalign(&newbuf, 64, total_size);
    if (rc != 0 || !newbuf) {
      errno = rc;  // posix_memalign returns error code instead of setting errno
      printf("cannot aligned alloc %zu bytes (align=%d)\n", (size_t)total_size, 64);
      perror("posix_memalign");  // ENOMEM/EINVAL
      exit(1);
    }
    buffer = newbuf;
    size = total_size;
    for (auto& req : object_requests) {
      req.update_base_ptr(buffer);
    }
  } else {
    requests.update_base_ptr(buffer);
  }
}

void SharedMemBufferNuma::alloc(int numa, void* object, MemoryRequest requests) {
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
