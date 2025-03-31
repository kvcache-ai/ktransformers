#include "cuda_stream_manager.hh"
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
// #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#define FMT_HEADER_ONLY
#include "spdlog/spdlog.h"

CudaStreamManager::CudaStreamManager(const std::vector<size_t>& device_ids, int num_streams_per_device) {
  for (int device_id : device_ids) {
    auto x = std::unique_ptr<DeviceInfo>(new DeviceInfo);
    DeviceInfo& device_info = *x;
    device_info.device_id = device_id;
    device_info.next_stream_index = 0;
    device_info.stop_flag = false;

    // 设置设备
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      SPDLOG_WARN("cudaSetDevice failed on device {}: {}", device_id, cudaGetErrorString(err));
      throw std::runtime_error("cudaSetDevice failed");
    }

    // 创建 CUDA 流
    device_info.streams.resize(num_streams_per_device);
    for (int i = 0; i < num_streams_per_device; ++i) {
      err = cudaStreamCreate(&device_info.streams[i]);
      if (err != cudaSuccess) {
        SPDLOG_WARN("Failed to create CUDA stream on device {}: {}", device_id, cudaGetErrorString(err));
        throw std::runtime_error("Failed to create CUDA stream");
      }
    }

    // 启动设备工作线程
    device_info.worker_thread = std::thread(&CudaStreamManager::deviceWorker, this, std::ref(device_info));

    devices_.push_back(std::move(x));
  }
}

CudaStreamManager::~CudaStreamManager() {
  // 通知所有设备线程停止
  for (auto& device_info : devices_) {
    device_info->stop_flag.store(true);
    auto request = std::shared_ptr<Request>(new Request);
    request->should_exit = true;
    device_info->request_queue.enqueue(std::move(request));
  }

  // 等待所有线程结束
  for (auto& device_info : devices_) {
    if (device_info->worker_thread.joinable()) {
      device_info->worker_thread.join();
    }

    // 销毁 CUDA 流
    cudaSetDevice(device_info->device_id);
    for (auto& stream : device_info->streams) {
      cudaStreamDestroy(stream);
    }
  }
}

void CudaStreamManager::submitRequest(std::shared_ptr<Request> request) {
  // 找到对应的设备
  for (auto& device_info : devices_) {
    if (device_info->device_id == request->device_id) {
      device_info->request_queue.enqueue(request);
      return;
    }
  }
  throw std::runtime_error("Invalid device ID in request");
}

void CudaStreamManager::deviceWorker(DeviceInfo& device_info) {
  // 设置设备
  cudaError_t err = cudaSetDevice(device_info.device_id);
  if (err != cudaSuccess) {
    SPDLOG_WARN("cudaSetDevice failed in worker thread for device {}: {}", device_info.device_id,
                cudaGetErrorString(err));
    return;
  }

  while (device_info.stop_flag.load() == false) {
    auto request = device_info.request_queue.dequeue();
    if (request->should_exit) {
      return;
    }
    // 处理请求
    SPDLOG_DEBUG("Getting request on device {}, count {}", device_info.device_id, request->host_mem_addresses.size());
    int stream_index = device_info.next_stream_index;
    cudaStream_t stream = device_info.streams[stream_index];
    device_info.next_stream_index = (device_info.next_stream_index + 1) % device_info.streams.size();

    size_t num_transfers = request->host_mem_addresses.size();
    for (size_t i = 0; i < num_transfers; ++i) {
      void* dst = request->device_mem_addresses[i];
      void* src = request->host_mem_addresses[i];
      if (request->direction == cudaMemcpyDeviceToHost) {
        std::swap(dst, src);
      }

      cudaError_t err = cudaMemcpyAsync(dst, src, request->sizes[i], request->direction, stream);
      if (err != cudaSuccess) {
        SPDLOG_WARN("cudaMemcpyAsync failed on device {}: {}", device_info.device_id, cudaGetErrorString(err));
        // 可以根据需要处理错误，这里简单地继续
        continue;
      }
    }

    // 添加回调函数，因为是异步，所以需要包起来
    struct CallbackData {
      std::function<void()> callback;
    };
    CallbackData* cb_data = new CallbackData{request->callback};

    err = cudaLaunchHostFunc(
        stream,
        [](void* data) {
          // SPDLOG_DEBUG("Callback function called");
          CallbackData* cb_data = static_cast<CallbackData*>(data);
          cb_data->callback();
          delete cb_data;
        },
        cb_data);

    if (err != cudaSuccess) {
      SPDLOG_WARN("cudaLaunchHostFunc failed on device {}: {}", device_info.device_id, cudaGetErrorString(err));
      // 根据需要处理错误
    }
  }
}
