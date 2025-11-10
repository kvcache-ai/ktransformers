#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

class CudaStreamManager {
 public:
  CudaStreamManager(int num_streams);
  ~CudaStreamManager();

  // Request structure
  struct Request {
    std::vector<void*> host_mem_addresses;
    std::vector<void*> device_mem_addresses;
    std::vector<size_t> sizes;
    cudaMemcpyKind direction;
    std::function<void()> callback;
  };

  void submitRequest(const Request& request);

 private:
  int num_streams_;
  std::vector<cudaStream_t> streams_;
  int next_stream_index_;
};

CudaStreamManager::CudaStreamManager(int num_streams) : num_streams_(num_streams), next_stream_index_(0) {
  streams_.resize(num_streams_);
  for (int i = 0; i < num_streams_; ++i) {
    cudaError_t err = cudaStreamCreate(&streams_[i]);
    if (err != cudaSuccess) {
      std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
      for (int j = 0; j < i; ++j) {
        cudaStreamDestroy(streams_[j]);
      }
      throw std::runtime_error("Failed to create CUDA stream");
    }
  }
}

CudaStreamManager::~CudaStreamManager() {
  for (int i = 0; i < num_streams_; ++i) {
    cudaStreamDestroy(streams_[i]);
  }
}

void CudaStreamManager::submitRequest(const Request& request) {
  int stream_index = next_stream_index_;
  cudaStream_t stream = streams_[stream_index];
  next_stream_index_ = (next_stream_index_ + 1) % num_streams_;

  size_t num_transfers = request.host_mem_addresses.size();
  for (size_t i = 0; i < num_transfers; ++i) {
    cudaError_t err = cudaMemcpyAsync(request.device_mem_addresses[i], request.host_mem_addresses[i], request.sizes[i],
                                      request.direction, stream);
    if (err != cudaSuccess) {
      std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("cudaMemcpyAsync failed");
    }
  }

  // Enqueue the callback function
  struct CallbackData {
    std::function<void()> callback;
  };

  CallbackData* cb_data = new CallbackData{request.callback};

  cudaError_t err = cudaLaunchHostFunc(
      stream,
      [](void* data) {
        CallbackData* cb_data = static_cast<CallbackData*>(data);
        cb_data->callback();
        delete cb_data;
      },
      cb_data);

  if (err != cudaSuccess) {
    std::cerr << "cudaLaunchHostFunc failed: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("cudaLaunchHostFunc failed");
  }
}

// Example usage
int main() {
  try {
    CudaStreamManager stream_manager(4);  // Create a manager with 4 streams

    // Prepare host and device memory
    const size_t num_pages = 10;
    std::vector<void*> host_mem_addresses(num_pages);
    std::vector<void*> device_mem_addresses(num_pages);
    std::vector<size_t> sizes(num_pages, 4096);  // 4KB pages

    // Allocate host memory
    for (size_t i = 0; i < num_pages; ++i) {
      host_mem_addresses[i] = malloc(4096);
      if (!host_mem_addresses[i]) {
        throw std::runtime_error("Failed to allocate host memory");
      }
      // Initialize data if necessary
    }

    // Allocate device memory
    for (size_t i = 0; i < num_pages; ++i) {
      cudaError_t err = cudaMalloc(&device_mem_addresses[i], 4096);
      if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMalloc failed");
      }
    }

    // Create a request
    CudaStreamManager::Request request;
    request.host_mem_addresses = host_mem_addresses;
    request.device_mem_addresses = device_mem_addresses;
    request.sizes = sizes;
    request.direction = cudaMemcpyHostToDevice;
    request.callback = []() { std::cout << "Data transfer completed!" << std::endl; };

    // Submit the request
    stream_manager.submitRequest(request);

    // Wait for all streams to complete
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("cudaDeviceSynchronize failed");
    }

    // Clean up
    for (size_t i = 0; i < num_pages; ++i) {
      free(host_mem_addresses[i]);
      cudaFree(device_mem_addresses[i]);
    }

  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
