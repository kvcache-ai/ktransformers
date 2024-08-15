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
#include <map>
#include <vector>

class SharedMemBuffer {
   public:
    SharedMemBuffer();
    ~SharedMemBuffer();

    void alloc(void* object, std::vector<std::pair<void**, uint64_t>> requests);
    void dealloc(void* object);

   private:
    void* buffer_;
    uint64_t size_;
    std::map<void*, std::vector<std::vector<std::pair<void**, uint64_t>>>> hist_requests_;

    void arrange(std::vector<std::pair<void**, uint64_t>> requests);
};

static SharedMemBuffer shared_mem_buffer;

#endif