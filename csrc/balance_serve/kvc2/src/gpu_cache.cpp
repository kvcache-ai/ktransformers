#include "gpu_cache.hh"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#define FMT_HEADER_ONLY
#include "spdlog/spdlog.h"

#include "cache_entry.hh"
#include "utils/arithmetic.hpp"

namespace kvc2 {

GPUPageCache::GPUPageCache(GPUPageCacheConfig& config) : config(config) {
  if (torch::cuda::is_available()) {
    size_t gpu_count = torch::cuda::device_count();
    SPDLOG_INFO("Number of available GPUs: {}, want {}", gpu_count, config.gpu_devices_id.size());
    if (gpu_count < config.gpu_devices_id.size()) {
      SPDLOG_ERROR("Not enough GPUs available.");
      exit(0);
    }
    for (auto x : config.gpu_devices_id) {
      gpu_devices.push_back(torch::Device(torch::kCUDA, x));
    }
  } else {
    SPDLOG_ERROR("CUDA is not available on this system.");
    exit(0);
  }

  SPDLOG_WARN("Creating GPU Cache");
  shape.push_back(config.layer_count);
  shape.push_back(config.total_kvcache_pages);
  shape.push_back(config.num_token_per_page);
  if (config.full_kv_cache_on_each_gpu) {
    if (config.gpu_devices_id.size() > 1) {
      SPDLOG_WARN("Replicated KVCache on multiple gpu");
    }
    shape.push_back(config.num_k_heads);
  } else {
    shape.push_back(config.num_k_heads / config.gpu_devices_id.size());
  }
  shape.push_back(config.k_head_dim);
  tensor_size = torch::elementSize(config.tensor_type);
  for (auto& s : shape) {
    tensor_size *= s;
  }
  SPDLOG_INFO("Creating KV Page Cache, Shape ({},{},{},{},{}), Size {} MiB", shape[0], shape[1], shape[2], shape[3],
              shape[4], tensor_size / (1 << 20));
  if (config.k_cache_on) {
    for (size_t i = 0; i < config.gpu_devices_id.size(); i++) {
      auto k = torch::zeros(shape, torch::TensorOptions().dtype(config.tensor_type));
      k = k.to(gpu_devices[i]);

      k_cache.push_back(k);

      SPDLOG_INFO("K Page Cache of GPU {} is created", config.gpu_devices_id[i]);
    }
    occupations.resize(config.layer_count);
  } else {
    SPDLOG_WARN("Disalbe K Cache");
    assert(config.gpu_only);
  }

  if (config.v_cache_on) {
    for (size_t i = 0; i < config.gpu_devices_id.size(); i++) {
      auto v = torch::zeros(shape, torch::TensorOptions().dtype(config.tensor_type));
      v = v.to(gpu_devices[i]);
      v_cache.push_back(v);

      SPDLOG_INFO("V Page Cache of GPU {} is created", config.gpu_devices_id[i]);
    }
    v_occupations.resize(config.layer_count);
  } else {
    SPDLOG_WARN("Disalbe V Cache");
    // assert(config.gpu_only); // should not assert
  }

  if (config.gpu_only) {
    gpu_only_occupations.resize(config.total_kvcache_pages, false);
  }

  num_free_pages = config.total_kvcache_pages;
  for (size_t i = 0; i < config.layer_count; i++) {
    if (config.k_cache_on)
      occupations[i].resize(config.total_kvcache_pages, nullptr);
    if (config.v_cache_on)
      v_occupations[i].resize(config.total_kvcache_pages, nullptr);
  }

  tp_size.resize(config.gpu_devices_id.size(), shape[2] * shape[3] * shape[4] * c10::elementSize(config.tensor_type));
  tp_offset.resize(config.gpu_devices_id.size(), 0);
  for (size_t i = 1; i < tp_offset.size(); i++) {
    tp_offset[i] = tp_offset[i - 1] + tp_size[i - 1];
  }

  stream_manager =
      std::unique_ptr<CudaStreamManager>(new CudaStreamManager(config.gpu_devices_id, config.num_streams_per_device));
}

bool GPUPageCache::alloc_col(std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& k_entries,
                             std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& v_entries, size_t at) {
  std::lock_guard<std::mutex> lg(lock);
  auto idx = next_empty_col();
  if (idx.has_value()) {
    // must have entry lock
    auto& k0_entry = k_entries[0][at];
    k0_entry->gpu_block_idx = idx;

    for (size_t l = 0; l < config.layer_count; l++) {
      if (config.k_cache_on) {
        assert(k_entries[l][at]->data != nullptr);
        occupations[l][idx.value()] = k_entries[l][at];
      }
      if (config.v_cache_on) {
        assert(v_entries[l][at]->data != nullptr);
        v_occupations[l][idx.value()] = v_entries[l][at];
      }
    }
    return true;
  } else {
    return false;
  }
}

std::vector<size_t> GPUPageCache::gpu_only_alloc_col(size_t count) {
  assert(config.gpu_only);
  std::lock_guard<std::mutex> lg(lock);
  std::vector<size_t> re;

  for (size_t i = 0; i < config.total_kvcache_pages; i++) {
    if (gpu_only_occupations[i] == false) {
      re.push_back(i);
      if (re.size() == count) {
        break;
      }
    }
  }

  if (re.size() == count) {
    for (auto at : re) {
      gpu_only_occupations[at] = true;
    }
  } else {
    SPDLOG_WARN("GPU ONLY: Cannot allocate {} cols", count);
    re.clear();
  }
  return re;
}

void GPUPageCache::gpu_only_free_cols(std::vector<size_t> cols) {
  assert(config.gpu_only);
  std::lock_guard<std::mutex> lg(lock);
  for (auto at : cols) {
    assert(gpu_only_occupations[at]);
    gpu_only_occupations[at] = false;
  }
}

std::optional<size_t> GPUPageCache::next_empty_col() {
  if (num_free_pages == 0) {
    evict_cols();
    if (num_free_pages == 0) {
      return std::nullopt;
    }
  }
  while (occupations[0][_col_idx] != nullptr) {
    _col_idx = (_col_idx + 1) % config.total_kvcache_pages;
  }
  num_free_pages -= 1;
  return _col_idx;
}

void GPUPageCache::evict_cols() {
  auto evicted_count = 0;
  for (size_t i = 0; i < config.total_kvcache_pages; i++) {
    auto& h = occupations[0][i];
    if (h == nullptr) {
      continue;
    }
    auto lg = h->lock_guard();
    if (h->gpu_cc.can_desert()) {
      h->gpu_cc.tc.reset();
      h = nullptr;
      num_free_pages += 1;
      evicted_count += 1;
    }
  }
  if (evicted_count > 0)
    SPDLOG_INFO("GPU: Evicted {} GPU pages", evicted_count);
}

std::vector<std::unique_lock<CacheBlockEntry::MutexT>> GPUPageCache::try_lock_col(size_t at) {
  std::vector<std::unique_lock<CacheBlockEntry::MutexT>> re;
  if (config.k_cache_on) {
    for (size_t l = 0; l < config.layer_count; l++) {
      if (occupations[l][at] == nullptr) {
        return {};
      }
      auto ul = occupations[l][at]->try_lock();
      if (ul.owns_lock()) {
        re.push_back(std::move(ul));
      } else {
        return {};
      }
    }
  }
  if (config.v_cache_on) {
    for (size_t l = 0; l < config.layer_count; l++) {
      if (v_occupations[l][at] == nullptr) {
        return {};
      }
      auto ul = v_occupations[l][at]->try_lock();
      if (ul.owns_lock()) {
        re.push_back(std::move(ul));
      } else {
        return {};
      }
    }
  }
  return re;
}

std::vector<std::shared_ptr<CudaStreamManager::Request>> GPUPageCache::basic_request(cudaMemcpyKind direction,
                                                                                     std::function<void()> callback) {
  std::vector<std::shared_ptr<CudaStreamManager::Request>> re;
  re.resize(config.gpu_devices_id.size(), nullptr);
  for (size_t i = 0; i < re.size(); i++) {
    re[i] = std::shared_ptr<CudaStreamManager::Request>(new CudaStreamManager::Request);
    re[i]->direction = direction;
    re[i]->device_id = config.gpu_devices_id[i];
    re[i]->callback = callback;
  }
  return re;
}

void GPUPageCache::submit_requests(std::vector<std::shared_ptr<CudaStreamManager::Request>> reqs) {
  for (auto& r : reqs) {
    stream_manager->submitRequest(r);
  }
}

void GPUPageCache::append_col_to_request(std::vector<std::shared_ptr<CudaStreamManager::Request>>& reqs,
                                         std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& k_handles,
                                         std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& v_handles,
                                         size_t at) {
  if (config.k_cache_on == false && config.v_cache_on == false) {
    return;
  }
  auto gpu_block_idx = k_handles[0][at]->gpu_block_idx.value();
  for (size_t layer = 0; layer < config.layer_count; layer++) {
    for (size_t which_gpu = 0; which_gpu < config.gpu_devices_id.size(); which_gpu++) {
      if (config.k_cache_on) {
        assert(k_handles[layer][at]->data != nullptr);
        reqs[which_gpu]->sizes.push_back(tp_size[which_gpu]);
        reqs[which_gpu]->host_mem_addresses.push_back(
            offset_by_bytes(k_handles[layer][at]->data, tp_offset[which_gpu]));
        reqs[which_gpu]->device_mem_addresses.push_back(k_cache[which_gpu][layer][gpu_block_idx].data_ptr());
      }

      if (config.v_cache_on) {
        assert(v_handles[layer][at]->data != nullptr);
        reqs[which_gpu]->sizes.push_back(tp_size[which_gpu]);
        reqs[which_gpu]->host_mem_addresses.push_back(
            offset_by_bytes(v_handles[layer][at]->data, tp_offset[which_gpu]));
        reqs[which_gpu]->device_mem_addresses.push_back(v_cache[which_gpu][layer][gpu_block_idx].data_ptr());
      }
    }
  }
  // SPDLOG_DEBUG("GPU: Appended Vertical Handle to Request, count {}", reqs[0]->sizes.size());
}

void GPUPageCache::debug() {
  size_t count = 0;
  for (size_t i = 0; i < config.total_kvcache_pages; i++) {
    if (occupations[0][i] == nullptr) {
      count += 1;
    } else {
      // occupations[0][i]->gpu_cc.debug();
    }
  }
  SPDLOG_DEBUG("Free Page: {}/{}", count, config.total_kvcache_pages);
}

}  // namespace kvc2
