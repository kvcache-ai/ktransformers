#ifndef __GPU_CACHE_HH_
#define __GPU_CACHE_HH_

#include <torch/torch.h>
#include "cache_entry.hh"
#include "cuda_stream_manager.hh"
#include "defs.h"
#include "kvc2.h"
#include "metrics.h"
#include "utils/periodic_task.hpp"

namespace kvc2 {

class GPUPageCache {
  std::vector<torch::Device> gpu_devices;

  std::vector<int64_t> shape;
  size_t tensor_size;
  std::vector<size_t> tp_offset;
  std::vector<size_t> tp_size;



  // met
  std::shared_ptr<Metrics> met;

  // states
  std::mutex lock;
  size_t num_free_pages;
  std::vector<bool> gpu_only_occupations;
  std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>> occupations,v_occupations;
  size_t _col_idx = 0;


  // cuda stream manager
  std::optional<size_t> next_empty_col();

 public:
  GPUPageCacheConfig config;
  std::unique_ptr<CudaStreamManager> stream_manager;
  std::vector<torch::Tensor> k_cache;
  std::vector<torch::Tensor> v_cache;
  std::unique_ptr<periodic::PeriodicTask> background_flush_back =nullptr;

  GPUPageCache(GPUPageCacheConfig& config);

  std::vector<size_t> gpu_only_alloc_col(size_t count);
  void gpu_only_free_cols(std::vector<size_t> cols);
  

  void gpu_background_flush();


  bool alloc_col(std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& k_entries,
                 std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& v_entries, size_t at);
  void evict_cols();
  void flush_col(size_t at);
  std::vector<std::unique_lock<CacheBlockEntry::MutexT>> try_lock_col(size_t at);

  void free_col(size_t at);

  std::vector<std::shared_ptr<CudaStreamManager::Request>> basic_request(cudaMemcpyKind direction,
                                                                         std::function<void()> callback);

  void submit_requests(std::vector<std::shared_ptr<CudaStreamManager::Request>> reqs);

  void append_col_to_request(std::vector<std::shared_ptr<CudaStreamManager::Request>>& reqs,
                             std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& k_handles,
                             std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& v_handles, size_t at);

  void debug();
};
}  // namespace kvc2
#endif