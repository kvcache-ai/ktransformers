#include "cuda_stream_manager.hh"

#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  try {
    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess) {
      std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
      return 1;
    }

    if (num_devices < 1) {
      std::cerr << "未找到 CUDA 设备。" << std::endl;
      return 1;
    }

    std::vector<size_t> device_ids;
    for (int i = 0; i < num_devices; ++i) {
      device_ids.push_back(i);
    }

    const size_t num_pages = 10;
    const size_t page_size = 4096;  // 每页 4KB

    // 创建 CudaStreamManager 实例，管理所有设备
    CudaStreamManager stream_manager(device_ids, 4);

    // 准备主机内存和设备内存映射
    std::vector<std::vector<void*>> host_mem_addresses(num_devices);
    std::vector<std::vector<void*>> device_mem_addresses(num_devices);

    // 分配主机内存
    for (size_t i = 0; i < num_pages; ++i) {
      void* host_ptr = malloc(page_size);
      if (!host_ptr) {
        throw std::runtime_error("Failed to allocate host memory");
      }
      // 如果需要，初始化数据

      // 将相同的主机内存添加到每个设备的列表中
      for (int device_id = 0; device_id < num_devices; ++device_id) {
        host_mem_addresses[device_id].push_back(host_ptr);
      }
    }

    // 为每个设备分配设备内存
    for (int device_id = 0; device_id < num_devices; ++device_id) {
      err = cudaSetDevice(device_id);
      if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaSetDevice failed");
      }

      for (size_t i = 0; i < num_pages; ++i) {
        void* device_ptr;
        err = cudaMalloc(&device_ptr, page_size);
        if (err != cudaSuccess) {
          std::cerr << "cudaMalloc failed on device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
          throw std::runtime_error("cudaMalloc failed");
        }
        device_mem_addresses[device_id].push_back(device_ptr);
      }
    }

    // 为每个设备创建并提交请求
    for (int device_id = 0; device_id < num_devices; ++device_id) {
      auto request = std::shared_ptr<CudaStreamManager::Request>(new CudaStreamManager::Request);
      request->device_id = device_id;
      request->host_mem_addresses = host_mem_addresses[device_id];
      request->device_mem_addresses = device_mem_addresses[device_id];
      request->sizes = std::vector<size_t>(num_pages, page_size);
      request->direction = cudaMemcpyHostToDevice;
      request->callback = [device_id]() {
        std::cout << "Device " << device_id << " data transfer completed!" << std::endl;
      };

      stream_manager.submitRequest(request);
    }

    // 等待一段时间，确保所有请求都被处理
    // 在实际应用中，可以使用更好的同步机制
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // 清理主机内存
    for (size_t i = 0; i < num_pages; ++i) {
      free(host_mem_addresses[0][i]);  // 所有设备共享相同的主机内存，只需释放一次
    }

    // 清理设备内存
    for (int device_id = 0; device_id < num_devices; ++device_id) {
      err = cudaSetDevice(device_id);
      if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed during cleanup: " << cudaGetErrorString(err) << std::endl;
        continue;
      }
      for (void* ptr : device_mem_addresses[device_id]) {
        cudaFree(ptr);
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "异常: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
