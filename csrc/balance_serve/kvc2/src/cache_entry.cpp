#include "cache_entry.hh"
#include <mutex>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#define FMT_HEADER_ONLY
#include "spdlog/spdlog.h"

#include "gpu_cache.hh"

namespace kvc2 {

bool ConcurrentControlUnit::can_desert() {
  if (ref_count.load() == 0 && dirty.load() == false) {
    tc.reset();
    return true;
  } else {
    return false;
  }
}
void ConcurrentControlUnit::debug() {
  SPDLOG_DEBUG("ref count {}, dirty {}, {}", ref_count.load(), dirty.load(), tc.debug());
}

CacheBlockEntry::~CacheBlockEntry() {
  if (data != nullptr && manager && manager->pool) {
    SPDLOG_WARN("Free {} when destruct", data);
    free_on_cpu();
  }
}

bool CacheBlockEntry::alloc_on_cpu() {
  assert(data == nullptr);
  data = manager->pool->alloc(size);
  if (data == nullptr) {
    manager->evict_for_cpu_cache();
    data = manager->pool->alloc(size);
    if (data == nullptr) {
      SPDLOG_ERROR("Not enough memory for Block Cache");
      return false;
    }
  }
  return true;
}

void CacheBlockEntry::free_on_cpu() {
  manager->pool->free(data, size);
  data = nullptr;
}

bool CacheBlockEntry::alloc_on_cpu_no_lock() {
  if (data == nullptr) {
    if (alloc_on_cpu() == false) {
      return false;
    }
  }
  return true;
}

bool CacheBlockEntry::inc_ref_or_alloc_on_cpu() {
  std::lock_guard<CacheBlockEntry::MutexT> lg(lock);
  if (data == nullptr) {
    if (alloc_on_cpu()) {
      cpu_cc.ref_count.fetch_add(1);
      return true;
    } else {
      return false;
    }
  } else {
    cpu_cc.ref_count.fetch_add(1);
    return true;
  }
}

std::unique_lock<CacheBlockEntry::MutexT> CacheBlockEntry::try_lock() {
  return std::unique_lock<CacheBlockEntry::MutexT>(lock, std::try_to_lock);
}

std::lock_guard<CacheBlockEntry::MutexT> CacheBlockEntry::lock_guard() {
  return std::lock_guard<CacheBlockEntry::MutexT>(lock);
}

void CacheBlockEntry::debug() {
  SPDLOG_DEBUG(
      "CacheBlockEntry: disk[{:4},{:7}], with key {}, hash {:016x}, data: {}, ref_count: {}, size: {}, cpu tc: {}, "
      "in page cache: {}, gpu ref count:{}, gpu tc: {}",
      layer, idx, with_key, hash, data, cpu_cc.ref_count.load(), size, cpu_cc.tc.debug(), manager != nullptr,
      gpu_cc.ref_count.load(), gpu_cc.tc.debug());
}

CacheBlockEntryCollector::CacheBlockEntryCollector(std::function<void(CacheBlockEntry*)> exit_fn) : exit_fn(exit_fn) {}

CacheBlockEntryCollector::~CacheBlockEntryCollector() {
  // SPDLOG_DEBUG("Collector Destruct");
  for (auto& e : entries) {
    exit_fn(e);
  }
}

void CacheBlockEntry::io_with(async_store::IODealer* dealer, IO_Helper<CacheBlockEntry>& io_helper,
                              async_store::ArrayStore* store, size_t layer, size_t index, IOOption option) {
  bool write;

  auto& batch_promise = io_helper.batch_promise;

  switch (option) {
    case IO_Read: {
      write = false;
      if (io_helper.absorb_tc(this, cpu_cc.tc)) {
        // need read
      } else {
        return;
      }
      break;
    }
    case IO_ForceRead: {
      // Not change
      write = false;
      break;
    }
    case IO_ForceWrite: {
      // Not change
      write = true;
      break;
    }
    case IO_Write: {
      write = true;
      break;
    }
    default: {
      assert(0);
    }
  }
  io_helper.new_task();
  this->layer = layer;
  this->idx = index;

  auto req = std::make_shared<async_store::IORequest>();
  req->store = store;
  req->data = data;
  req->index = index;
  req->write = write;
  req->need_promise = true;
  req->promise = &batch_promise;

  SPDLOG_TRACE("Submitting {}", async_store::request_to_string(req.get()));
  dealer->enqueue(std::move(req));
}

CacheEntryManager::CacheEntryManager(CacheEntryManagerConfig config) : config(config) {}

void CacheEntryManager::evict_for_cpu_cache() {
  size_t count = 0;
  evict(
      [&count](const BlockPtr& block) {
        // here we assume each with gpu must resides on cpu
        if (block->data != nullptr && block->cpu_cc.can_desert() &&
            block->gpu_cc.can_desert() /*For now If A Cache Entry Block is on GPU, it must on cpu. */) {
          block->free_on_cpu();
          count += 1;
          return true;
        } else {
          return false;
        }
      },
      [&count, this]() {
        return false;
        // return count == this->config.evict_count;
      });
}

void CacheEntryManager::insert(BlockPtr entry) {
  assert(entry->with_key);
  assert(key_entry_map.count(entry->hash) == 0);
  usage_list.push_front(entry);
  key_entry_map[entry->hash] = usage_list.begin();
}

CacheEntryManager::BlockPtr CacheEntryManager::access(const Key& key) {
  auto it = key_entry_map.at(key);
  auto entry = *it;
  usage_list.erase(it);
  usage_list.push_front(entry);
  key_entry_map[key] = usage_list.begin();
  return entry;
}

// void CacheEntryManager::remove(const Key& key) {
//   auto it = key_entry_map[key];
//   usage_list.erase(it);
//   key_entry_map.erase(key);
// }

void CacheEntryManager::evict(std::function<bool(const BlockPtr&)> filter, std::function<bool()> stop_condition) {
  auto evict_count = 0;
  auto inspect_count = 0;

  std::lock_guard<std::mutex> lg(lock);
  for (auto it = usage_list.rbegin(); it != usage_list.rend();) {
    inspect_count += 1;
    // SPDLOG_DEBUG("Map Size {}, List Size {}, Evicted {} blocks, Inspected {}, {}", key_entry_map.size(),
    //              usage_list.size(), evict_count, inspect_count, pool->debug());
    // (*it)->debug();
    if (stop_condition())
      break;
    auto entry_ul = (*it)->try_lock();
    if (entry_ul.owns_lock() == false) {
      ++it;  // Ensure iterator advances when locking fails
      continue;
    }
    if (filter(*it)) {
      // SPDLOG_DEBUG("Evicting {}", fmt::ptr(it->get()));
      evict_count++;
      if ((*it)->with_key)
        key_entry_map.erase((*it)->hash);
      it = decltype(it)(usage_list.erase(std::next(it).base()));  // Use base() to adjust for reverse iterator
    } else {
      ++it;  // Ensure iterator advances when filter fails
    }
  }

  if (evict_count > 0) {
    SPDLOG_DEBUG("Map Size {}, List Size {}, Evicted {} blocks, Inspected {}, {}", key_entry_map.size(),
                 usage_list.size(), evict_count, inspect_count, pool->debug());
  }
}

CacheEntryManager::BlockPtr CacheEntryManager::get(bool& is_new, size_t size, std::optional<Key> key) {
  std::unique_lock<std::mutex> ul(lock);
  if (key.has_value()) {
    if (key_entry_map.count(key.value())) {
      is_new = false;
      return access(key.value());
    } else {
      auto entry = std::make_shared<CacheBlockEntry>();
      entry->with_key = true;
      entry->hash = key.value();
      entry->size = size;
      entry->manager = this;
      insert(entry);
      is_new = true;
      return entry;
    }
  } else {
    auto entry = std::make_shared<CacheBlockEntry>();
    entry->with_key = false;
    entry->size = size;
    entry->manager = this;
    is_new = true;
    return entry;
  }
}

void CacheEntryManager::debug() {
  fmt::print("Cache Manager: {} entries\n", key_entry_map.size());
  pool->debug();
  fmt::print("Layer 0 Entries in Order\n", key_entry_map.size());
  for (auto& it : usage_list) {
    if (it->layer == 0)
      it->debug();
  }
}

};  // namespace kvc2
