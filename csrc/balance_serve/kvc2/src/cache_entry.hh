#ifndef __CACHE_ENTRY_HH_
#define __CACHE_ENTRY_HH_
#include "async_store.hh"
#include "cuda_stream_manager.hh"
#include "defs.h"
#include "hasher.hpp"
#include "io_helper.hpp"
#include "page_aligned_memory_pool.h"
#include "utils/periodic_task.hpp"

#include <atomic>
#include <list>
#include <memory>
#include "utils/mutex_extend.hpp"

namespace kvc2 {
using CacheBlockKey = TokensHash;

class CacheEntryManager;
struct DoubleVerticalBlocksHandle;
class GPUPageCache;

struct ConcurrentControlUnit {
  std::atomic_size_t ref_count = 0;
  std::atomic_bool dirty = false;
  TransferControl<std::mutex> tc;

  bool can_desert();
  void debug();
};

enum IOOption {
  IO_ForceRead,
  IO_ForceWrite,
  IO_Read,
  IO_Write,
};

inline std::string to_string(IOOption op) {
  switch (op) {
    case IO_ForceRead:
      return "IO_ForceRead";
    case IO_ForceWrite:
      return "IO_ForceWrite";
    case IO_Read:
      return "IO_Read";
    case IO_Write:
      return "IO_Write";
    default:
      return "Unknown";
  }
}

struct CacheBlockEntry {
  friend CacheEntryManager;
  using MutexT = non_recursive_mutex;
  // using MutexT = std::mutex;
  MutexT lock;

  // for cache
  bool with_key = true;
  CacheBlockKey hash = 0;
  CacheBlockKey hash_check = 0;

  CacheInfo cache_info;
  CacheEntryManager* manager = nullptr;

  // for memory pool
  void* data = nullptr;
  size_t size = 0;

  ConcurrentControlUnit cpu_cc;

  // for disk
  size_t layer = -1;
  size_t idx = -1;

  // for gpu

  std::optional<size_t> gpu_block_idx = std::nullopt;
  ConcurrentControlUnit gpu_cc;

  CacheBlockEntry() =default;
  CacheBlockEntry(const CacheBlockEntry& other) = delete;
  CacheBlockEntry& operator=(const CacheBlockEntry& other) = delete;
  CacheBlockEntry(CacheBlockEntry&& other) = delete;
  CacheBlockEntry& operator=(CacheBlockEntry&& other) = delete;
  ~CacheBlockEntry();

 private:
  bool alloc_on_cpu();


 public:
  void free_on_cpu();
  bool alloc_on_cpu_no_lock();

  bool inc_ref_or_alloc_on_cpu();
  void set_key(TokensHash key, std::shared_ptr<CacheBlockEntry> me);

  std::unique_lock<MutexT> try_lock();
  std::lock_guard<MutexT> lock_guard();

  // will not get lock
  void io_with(async_store::IODealer* dealer, IO_Helper<CacheBlockEntry>& io_helper, async_store::ArrayStore* store,
               size_t layer, size_t index, IOOption option);
  void flush_back_async(IO_Helper<CacheBlockEntry>& helper, std::vector<std::atomic_bool*>& dirty_flags);

  void debug();
};

struct CacheBlockEntryCollector{

  std::vector<CacheBlockEntry*> entries;
  std::function<void(CacheBlockEntry*)> exit_fn;

  CacheBlockEntryCollector(std::function<void(CacheBlockEntry*)> exit_fn);
  ~CacheBlockEntryCollector();
  
  CacheBlockEntryCollector(const CacheBlockEntryCollector& other) = delete;
  CacheBlockEntryCollector(CacheBlockEntryCollector&& other) = delete;
  CacheBlockEntryCollector& operator=(const CacheBlockEntryCollector& other) = delete;
  CacheBlockEntryCollector& operator=(CacheBlockEntryCollector&& other) = delete;



};


struct KVC2;
struct CacheEntryManagerConfig {
  size_t evict_count = 100;
  KVC2* kvc2_top = nullptr;
};

class CacheEntryManager {
 public:
  using Key = CacheBlockKey;
  using BlockPtr = std::shared_ptr<CacheBlockEntry>;

 private:
  friend CacheBlockEntry;

  CacheEntryManagerConfig config;

  std::mutex lock;
  std::list<BlockPtr> usage_list;
  std::unordered_map<Key, std::list<BlockPtr>::iterator> key_entry_map;

  void insert(BlockPtr entry);
  BlockPtr access(const Key& key);

  // void remove(const Key& key);
  void evict(std::function<bool(const BlockPtr&)> filter, std::function<bool()> stop_condition);


 public:
  std::unique_ptr<periodic::PeriodicTask> background_flush_back=nullptr;
  std::shared_ptr<PageAlignedMemoryPool> pool;
  std::shared_ptr<GPUPageCache> gpu_cache;

  CacheEntryManager(CacheEntryManagerConfig config);

  // disable all move and copy
  CacheEntryManager(const CacheEntryManager& other) = delete;
  CacheEntryManager& operator=(const CacheEntryManager& other) = delete;
  CacheEntryManager(CacheEntryManager&& other) = delete;
  CacheEntryManager& operator=(CacheEntryManager&& other) = delete;

  void cpu_background_flush();

  void evict_for_cpu_cache();

  // just get block pointers, not allocate them, will not return nullptr
  BlockPtr get(bool& is_new,size_t size, std::optional<Key> key = std::nullopt);

  void debug();
};

}  // namespace kvc2

#endif