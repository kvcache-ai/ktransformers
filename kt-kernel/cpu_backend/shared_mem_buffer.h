/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-08-05 04:49:08
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-05 06:36:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#ifndef CPUINFER_SHAREDMEMBUFFER_H
#define CPUINFER_SHAREDMEMBUFFER_H

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

struct MemoryRequest {
  std::vector<std::function<void(void*)>> funcs;
  std::vector<size_t> sizes;

  size_t total_size();
  void update_base_ptr(void* base);

  template <typename T>
  void append_pointer(T** ptr, size_t size) {
    append_function([ptr](void* base) { *ptr = reinterpret_cast<T*>(base); }, size);
  }
  void append_function(std::function<void(void*)> func, size_t size);
};

class SharedMemBuffer {
 public:
  SharedMemBuffer();
  ~SharedMemBuffer();

  void alloc(void* object, MemoryRequest requests);

 private:
  void* buffer;
  uint64_t size;
  std::vector<MemoryRequest> object_requests;
};

static SharedMemBuffer shared_mem_buffer;
static SharedMemBuffer shared_mem_buffer_for_decoder_layer;

class SharedMemBufferNuma {
  std::mutex lock;
  std::map<size_t, std::unique_ptr<SharedMemBuffer>> numa_mem;

 public:
  void alloc(int numa, void* object, MemoryRequest requests);
};

static SharedMemBufferNuma shared_mem_buffer_numa;

#endif