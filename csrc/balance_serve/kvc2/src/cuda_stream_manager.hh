/*
 * @Author: Xie Weiyu ervinxie@qq.com
 * @Date: 2024-11-19 09:24:47
 * @LastEditors: Xie Weiyu ervinxie@qq.com
 * @LastEditTime: 2024-11-20 02:55:49
 * @FilePath: /kvc2/src/cuda_stream_manager.hh
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include "utils/mpsc.hpp"

class CudaStreamManager {
 public:
  // 构造函数，接受要使用的设备 ID 列表和每个设备的流数量
  CudaStreamManager(const std::vector<size_t>& device_ids, int num_streams_per_device);
  ~CudaStreamManager();

  // 请求结构体
  struct Request {
    bool should_exit = false;
    int device_id;
    std::vector<void*> host_mem_addresses;
    std::vector<void*> device_mem_addresses;
    std::vector<size_t> sizes;
    cudaMemcpyKind direction;
    std::function<void()> callback;
  };

  void submitRequest(std::shared_ptr<Request> request);

 private:
  // 每个设备的信息
  struct DeviceInfo {
    int device_id;
    std::thread worker_thread;
    std::vector<cudaStream_t> streams;
    int next_stream_index;
    MPSCQueueConsumerLock<std::shared_ptr<Request>> request_queue;
    std::atomic_bool stop_flag;
  };

  // 设备 ID 到 DeviceInfo 的映射
  std::vector<std::unique_ptr<DeviceInfo>> devices_;

  // 私有方法
  void deviceWorker(DeviceInfo& device_info);
};
