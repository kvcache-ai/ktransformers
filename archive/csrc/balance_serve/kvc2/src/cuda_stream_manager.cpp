#include "cuda_stream_manager.hh"
#include <functional>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
// #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#define FMT_HEADER_ONLY
#include "spdlog/spdlog.h"

#ifdef KTRANSFORMERS_USE_NPU
#include "acl/acl_rt.h"
#else
#include <cuda_runtime.h>
#endif

CudaStreamManager::CudaStreamManager(const std::vector<size_t>& device_ids, int num_streams_per_device) {
  for (int device_id : device_ids) {
    auto x = std::unique_ptr<DeviceInfo>(new DeviceInfo);
    DeviceInfo& device_info = *x;
    device_info.device_id = device_id;
    device_info.next_stream_index = 0;
    device_info.stop_flag = false;

    // 设置设备
#ifdef KTRANSFORMERS_USE_NPU
    aclError acl_err = aclrtSetDevice(device_id);
    if (acl_err != ACL_SUCCESS) {
      SPDLOG_WARN("aclrtSetDevice failed on device {}: {}", device_id, acl_err);
      throw std::runtime_error("aclrtSetDevice failed");
    }
#else
    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      SPDLOG_WARN("cudaSetDevice failed on device {}: {}", device_id, cudaGetErrorString(cuda_err));
      throw std::runtime_error("cudaSetDevice failed");
    }
#endif

    // 创建流
    device_info.streams.resize(num_streams_per_device);
    for (int i = 0; i < num_streams_per_device; ++i) {
#ifdef KTRANSFORMERS_USE_NPU
      acl_err = aclrtCreateStream(&device_info.streams[i]);
      if (acl_err != ACL_SUCCESS) {
        SPDLOG_WARN("Failed to create NPU stream on device {}: {}", device_id, acl_err);
        throw std::runtime_error("Failed to create NPU stream");
      }
#else
      cuda_err = cudaStreamCreate(&device_info.streams[i]);
      if (cuda_err != cudaSuccess) {
        SPDLOG_WARN("Failed to create CUDA stream on device {}: {}", device_id, cudaGetErrorString(cuda_err));
        throw std::runtime_error("Failed to create CUDA stream");
      }
#endif
    }

    // 启动工作线程
    device_info.worker_thread = std::thread(&CudaStreamManager::deviceWorker, this, std::ref(device_info));

#ifdef KTRANSFORMERS_USE_NPU
    // NPU需要额外的回调线程
    device_info.callback_thread = std::thread(&CudaStreamManager::deviceCallback, this, std::ref(device_info));

    // 绑定回调线程
    for (int i = 0; i < num_streams_per_device; ++i) {
      std::ostringstream oss;
      oss << device_info.callback_thread.get_id();
      uint64_t tid = std::stoull(oss.str());
      acl_err = aclrtSubscribeReport(tid, device_info.streams[i]);
      SPDLOG_DEBUG("subscribe stream callback report on device {} with tid {} in idx {}", device_id, tid, i);
      if (acl_err != ACL_SUCCESS) {
        SPDLOG_WARN("Failed to subscribe stream callback report on device {}: {}", device_id, acl_err);
        throw std::runtime_error("Failed to create stream callback job");
      }
    }
#endif

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
    // 等待工作线程结束
    if (device_info->worker_thread.joinable()) {
      device_info->worker_thread.join();
    }

#ifdef KTRANSFORMERS_USE_NPU
    // NPU需要额外的回调线程处理
    aclrtSetDevice(device_info->device_id);

    // 解绑callback任务并等待callback线程结束
    std::ostringstream oss;
    oss << device_info->callback_thread.get_id();
    uint64_t tid = std::stoull(oss.str());
    for (auto& stream: device_info->streams) {
      aclrtUnSubscribeReport(tid, stream);
    }
    if (device_info->callback_thread.joinable()) {
      device_info->callback_thread.join();
    }
#endif

    // 销毁流
#ifdef KTRANSFORMERS_USE_NPU
    for (auto& stream : device_info->streams) {
      aclrtDestroyStream(stream);
      aclrtResetDevice(device_info->device_id);
    }
#else
    cudaSetDevice(device_info->device_id);
    for (auto& stream : device_info->streams) {
      cudaStreamDestroy(stream);
    }
#endif
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
#ifdef KTRANSFORMERS_USE_NPU
  aclError acl_err = aclrtSetDevice(device_info.device_id);
  if (acl_err != ACL_SUCCESS) {
    SPDLOG_WARN("aclrtSetDevice failed in worker thread for device {}: {}", device_info.device_id, acl_err);
    return;
  }
#else
  cudaError_t cuda_err = cudaSetDevice(device_info.device_id);
  if (cuda_err != cudaSuccess) {
    SPDLOG_WARN("cudaSetDevice failed in worker thread for device {}: {}", device_info.device_id,
                cudaGetErrorString(cuda_err));
    return;
  }
#endif

  while (device_info.stop_flag.load() == false) {
    auto request = device_info.request_queue.dequeue();
    if (request->should_exit) {
      return;
    }
    
    SPDLOG_DEBUG("Getting request on device {}, count {}", device_info.device_id, request->host_mem_addresses.size());
    int stream_index = device_info.next_stream_index;
    auto stream = device_info.streams[stream_index];
    device_info.next_stream_index = (device_info.next_stream_index + 1) % device_info.streams.size();

    size_t num_transfers = request->host_mem_addresses.size();
    for (size_t i = 0; i < num_transfers; ++i) {
      void* dst = request->device_mem_addresses[i];
      void* src = request->host_mem_addresses[i];
      
#ifdef KTRANSFORMERS_USE_NPU
      if (request->direction == ACL_MEMCPY_DEVICE_TO_HOST) {
        std::swap(dst, src);
      }
      aclError err = aclrtMemcpyAsync(dst, request->sizes[i], src, request->sizes[i], request->direction, stream);
      if (err != ACL_SUCCESS) {
        SPDLOG_WARN("aclrtMemcpyAsync failed on device {}: {}", device_info.device_id, err);
        continue;
      }
#else
      if (request->direction == cudaMemcpyDeviceToHost) {
        std::swap(dst, src);
      }

      cudaError_t err = cudaMemcpyAsync(dst, src, request->sizes[i], request->direction, stream);
      if (err != cudaSuccess) {
        SPDLOG_WARN("cudaMemcpyAsync failed on device {}: {}", device_info.device_id, cudaGetErrorString(err));
        // 可以根据需要处理错误，这里简单地继续
        continue;
      }
#endif
    }

    // 添加回调函数
    struct CallbackData {
      std::function<void()> callback;
    };
    CallbackData* cb_data = new CallbackData{request->callback};

#ifdef KTRANSFORMERS_USE_NPU
    aclError err = aclrtLaunchCallback(
          [](void* data) {
            CallbackData* cb_data = static_cast<CallbackData*>(data);
            cb_data->callback();
            delete cb_data;
          },
          cb_data,
          ACL_CALLBACK_BLOCK,
          stream);

    if (err != ACL_SUCCESS) {
      SPDLOG_WARN("aclrtLaunchCallback failed on device {}: {}", device_info.device_id, err);
    }
#else
    cudaError_t err = cudaLaunchHostFunc(
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
    }
#endif
  }
}

#ifdef KTRANSFORMERS_USE_NPU
void CudaStreamManager::deviceCallback(DeviceInfo& device_info) {
  aclError err = aclrtSetDevice(device_info.device_id);
  if (err != ACL_SUCCESS) {
    SPDLOG_WARN("aclrtSetDevice failed in callback thread for device {}: {}", device_info.device_id, err);
    return;
  }
  

  int timeout = 60 * 1000;  // ms
  while (device_info.stop_flag.load() == false) {
    err = aclrtProcessReport(timeout);
    if (err != ACL_SUCCESS) {
      if (err == ACL_ERROR_RT_THREAD_SUBSCRIBE) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      SPDLOG_WARN("aclrtProcessReport failed in callback thread for device {}: {}", device_info.device_id, err);
    }
  }
}
#endif