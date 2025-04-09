#include <immintrin.h>
#include <tbb/concurrent_hash_map.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#define FMT_HEADER_ONLY
#include "spdlog/spdlog.h"

#include "async_store.hh"
#include "cuda_stream_manager.hh"
#include "kvc2.h"
#include "metrics.h"

#include "cache_entry.hh"
#include "gpu_cache.hh"
#include "hasher.hpp"
#include "io_helper.hpp"
#include "page_aligned_memory_pool.h"

#include "utils/arithmetic.hpp"
#include "utils/easy_format.hpp"
#include "utils/periodic_task.hpp"
namespace kvc2 {
struct KVC2;

// will be set when init
TokenLength NumTokenPerBlock;
int EvictCount;

using Layer = size_t;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CacheInfo, model_name, is_key_cache, quant_type);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(KVC2Config, gpu_only, load_from_disk, save_to_disk, path, config_path,
                                   num_token_per_page, memory_pool_size, evict_count, metrics_port, recompute_ratio);

size_t CacheInfo::hidden_layer_count() {
  return model_configs.at(model_name).num_hidden_layers;
}

std::filesystem::path CacheInfo::path(std::optional<size_t> which_layer) {
  auto folder = std::filesystem::path(model_name) / quant_type / (is_key_cache ? "key" : "value");
  if (which_layer.has_value()) {
    folder /= fmt::format("layer-{}.kvc", which_layer.value());
  }
  return folder;
}

bool CacheInfo::operator==(const CacheInfo& other) const {
  return model_name == other.model_name && is_key_cache == other.is_key_cache && quant_type == other.quant_type;
}

size_t CacheInfo::element_size(size_t block_length) {
  size_t count = model_configs[model_name].hidden_size * block_length;
  auto& q = quant_configs[quant_type];
  return count / q.block_element_count * q.block_element_size;
}

size_t CacheInfo::hash_value() const {
  size_t x = hash_seed;
  x = XXH64(model_name.data(), model_name.size(), x);
  x = XXH64("quant_type", 10, x);
  x = XXH64(quant_type.data(), quant_type.size(), x);
  if (is_key_cache) {
    x = XXH64("key", 3, x);
  } else {
    x = XXH64("value", 5, x);
  }
  return x;
}

}  // namespace kvc2

template <>
struct std::hash<kvc2::CacheInfo> {
  std::size_t operator()(const kvc2::CacheInfo& s) const noexcept { return s.hash_value(); }
};
namespace kvc2 {
struct Location {
  size_t start_idx;  // start block index
  size_t length;     // length of blocks
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Location, start_idx, length);

  Location cut_tail(size_t offset_from_tail) {
    Location re;
    size_t offset = length - offset_from_tail;
    re.start_idx = start_idx + offset;
    re.length = offset_from_tail;
    length = offset;
    return re;
  }
};

struct SegmentLocations {
  std::vector<std::optional<size_t>> offsets;

  void add_location(size_t start_block, Location location) {
    if (location.length + start_block > offsets.size()) {
      offsets.resize(location.length + start_block, std::nullopt);
    }

    for (size_t i = start_block; i < start_block + location.length; i++) {
      offsets[i] = location.start_idx + i - start_block;
    }
  }

  void set_location(size_t start_block, size_t disk_location) {
    if (start_block >= offsets.size()) {
      offsets.resize(start_block + 1, std::nullopt);
    }
    offsets[start_block] = disk_location;
  }

  std::optional<size_t> get_idx(size_t block_idx) const {
    if (block_idx >= offsets.size()) {
      return std::nullopt;
    } else {
      return offsets[block_idx];
    }
  }

  bool has_location(size_t block_idx, size_t length) {
    for (size_t i = block_idx; i < block_idx + length; i++) {
      if (get_idx(i).has_value() == false) {
        return false;
      }
    }
    return true;
  }

  void debug() {
    for (size_t i = 0; i < offsets.size(); ++i) {
      if (offsets[i].has_value()) {
        SPDLOG_DEBUG("Block {} -> Disk Location {}", i, offsets[i].value());
      } else {
        SPDLOG_DEBUG("Block {} -> No Disk Location", i);
      }
    }
  }
};

struct CacheDiskLocations {
  std::unordered_map<CacheInfo, Location> location_map;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(CacheDiskLocations, location_map);

  std::optional<Location> get_location(CacheInfo cache_info, TokenLength local_ids_length) {
    size_t blocks_length = div_up(local_ids_length, NumTokenPerBlock);
    if (location_map.count(cache_info) == 0) {
      return std::nullopt;
    }
    Location re = location_map[cache_info];
    re.length = blocks_length;
    return re;
  }

  std::optional<size_t> get_location_of_a_block(CacheInfo info, size_t local_at) {
    if (location_map.count(info) == 0) {
      return std::nullopt;
    }
    auto loc = location_map[info];
    if (local_at >= loc.length) {
      return std::nullopt;
    }
    return loc.start_idx + local_at;
  }
};

struct DiskCacheAllocator {
 private:
  // metadata
  std::filesystem::path path;
  CacheInfo info;
  std::mutex lock;
  size_t now_idx;

  // store
  size_t capacity;
  std::vector<async_store::ArrayStore*> stores;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DiskCacheAllocator, now_idx);

  void update_capacity() {
    capacity = std::numeric_limits<size_t>::max();
    for (auto& store : stores) {
      capacity = std::min(capacity, async_store::capacity(store));
    }
  }

  void extend(size_t to) {
    for (size_t i = 0; i < info.hidden_layer_count(); i++) {
      async_store::extend(stores[i], to);
    }
    update_capacity();
  }

 public:
  async_store::ArrayStore* get_store(int i) { return stores[i]; }
  Location alloc(size_t block_count) {
    std::lock_guard<std::mutex> lg(lock);
    Location re;
    re.start_idx = now_idx;
    re.length = block_count;
    now_idx += block_count;
    if (now_idx >= capacity) {
      extend(capacity * 2);
    }
    return re;
  }

  DiskCacheAllocator(std::filesystem::path path, CacheInfo info) : path(path), info(info) {
    // SPDLOG_DEBUG("Create DiskCacheAllocator {}", path.c_str());
    auto allocator_path = path / info.path();
    if (std::filesystem::exists(allocator_path) == false) {
      std::filesystem::create_directories(allocator_path);
    }
    // restore metadata later in json load
    now_idx = 0;

    for (size_t i = 0; i < info.hidden_layer_count(); i++) {
      // SPDLOG_DEBUG("Create store {} for {}", (path / info.path(i)).c_str(),i);
      auto store = async_store::create_or_open_store(info.element_size(NumTokenPerBlock), 1000, path / info.path(i));
      stores.push_back(store);
    }
    update_capacity();
  }

  ~DiskCacheAllocator() {
    for (auto store : stores) {
      async_store::close_store(store);
    }
  }
};

struct DiskCacheManager {
  KVC2Config config;
  std::mutex lock;
  std::unordered_map<CacheInfo, std::shared_ptr<DiskCacheAllocator>> allocators;

  friend void to_json(nlohmann ::json& nlohmann_json_j, const DiskCacheManager& nlohmann_json_t) {
    nlohmann_json_j["config"] = nlohmann_json_t.config;
    nlohmann_json_j["allocators"] = nlohmann::json::array();
    for (auto& [info, allocator] : nlohmann_json_t.allocators) {
      nlohmann_json_j["allocators"].push_back({{"info", info}, {"allocator", *allocator}});
    }
  }
  friend void from_json(const nlohmann ::json& nlohmann_json_j, DiskCacheManager& nlohmann_json_t) {
    // SPDLOG_DEBUG("Load DiskCacheManager Json");
    nlohmann_json_j.at("config").get_to(nlohmann_json_t.config);
    for (const auto& allocator_json : nlohmann_json_j.at("allocators")) {
      // SPDLOG_DEBUG("Make Allocator {}",allocator_json.dump());
      CacheInfo info;
      allocator_json.at("info").get_to(info);
      auto allocator = std::make_shared<DiskCacheAllocator>(nlohmann_json_t.config.path, info);
      allocator_json.at("allocator").get_to(*allocator);
      nlohmann_json_t.allocators[info] = allocator;
    }
  };

  DiskCacheManager(KVC2Config config) : config(config) {
    SPDLOG_INFO("DiskCacheManager root path: {}", config.path.c_str());
    if (!std::filesystem::exists(config.path)) {
      std::filesystem::create_directories(config.path);
    }
  }

  std::shared_ptr<DiskCacheAllocator> get_allocator(CacheInfo info) {
    {
      std::lock_guard<std::mutex> lg(lock);
      if (allocators.count(info) == 0) {
        allocators.emplace(info, std::make_shared<DiskCacheAllocator>(config.path, info));
      }
    }
    return allocators.at(info);
  }

  Location allocate(CacheInfo info, size_t cache_block_count) {
    auto allocator = get_allocator(info);
    return allocator->alloc(cache_block_count);
  }
};

struct Prefix {
  uint64_t prefix_id;  // 0 for nullptr, started from 1
  TokenLength start_length;
  Tokens ids;
  CacheDiskLocations locations;
  Prefix* prev = nullptr;

  // No serialization
  bool prev_set = false;

  friend void to_json(nlohmann ::json& nlohmann_json_j, const Prefix& nlohmann_json_t) {
    nlohmann_json_j["prefix_id"] = nlohmann_json_t.prefix_id;
    nlohmann_json_j["start_length"] = nlohmann_json_t.start_length;
    nlohmann_json_j["ids"] = nlohmann_json_t.ids;
    if (nlohmann_json_t.prev) {
      nlohmann_json_j["prev"] = nlohmann_json_t.prev->prefix_id;
    } else {
      nlohmann_json_j["prev"] = 0;
    }
    nlohmann_json_j["locations"] = nlohmann_json_t.locations;
  }
  friend void from_json(const nlohmann ::json& nlohmann_json_j, Prefix& nlohmann_json_t) {
    nlohmann_json_j.at("prefix_id").get_to(nlohmann_json_t.prefix_id);
    nlohmann_json_j.at("start_length").get_to(nlohmann_json_t.start_length);
    nlohmann_json_j.at("ids").get_to(nlohmann_json_t.ids);
    nlohmann_json_j.at("locations").get_to(nlohmann_json_t.locations);

    auto prev_id = nlohmann_json_j.at("prev").get<uint64_t>();
    nlohmann_json_t.prev = reinterpret_cast<Prefix*>(prev_id);
    nlohmann_json_t.prev_set = false;
  };

  TokenLength local_length() { return ids.size(); }
  TokenLength length() { return start_length + local_length(); }
  Tokens prefix_to(TokenLength length) {
    TokenLength local_length = length - start_length;
    Tokens re;
    if (prev) {
      re = prev->prefix_to(start_length);
    }
    re.insert(re.end(), ids.begin(), ids.begin() + local_length);
    return re;
  }
  Tokens full() { return prefix_to(length()); }

  void update_location(CacheInfo info, Location location) { locations.location_map[info] = location; }

  Prefix* to_first_prefix_without_disk_locations(CacheInfo k_info /*, CacheInfo v_info*/) {  // just k_info
    auto now_prefix = this;
    while (now_prefix->prev != nullptr) {
      auto& prev = now_prefix->prev;
      auto k_location = prev->locations.get_location(k_info, prev->local_length());
      // auto v_location = prev->locations.get_location(v_info, prev->local_length());
      if (k_location.has_value()) {
        // assert(v_location.has_value());
        // after now_prefix, we need to insert new kv cache.
        break;
      }
      now_prefix = prev;
    }
    return now_prefix;
  }

  void hash_to_with(TokenLength length, TokensHasher& hasher) {
    TokenLength local_length = length - start_length;
    if (prev) {
      prev->hash_to_with(start_length, hasher);
    }
    hasher.update(ids.data(), local_length);
  }

  void debug() {
    fmt::print("Prefix {}, start_length: {}, local_length: {}, prev: {}, \n", prefix_id, start_length, local_length(),
               (void*)prev);
  }
};
struct PrefixMatch {
  Prefix* prefix;
  TokenLength match_length;

  std::vector<TokensHash> matched_hashes(CacheInfo info, Layer layer) {
    std::vector<TokensHash> re;
    if (prefix == nullptr)
      return re;
    TokensHasher hasher;
    hasher.reset(info.hash_value());
    hasher.update_raw(&layer, sizeof(layer));
    auto ids = prefix->prefix_to(match_length);
    for (TokenLength i = 0; i < ids.size(); i += NumTokenPerBlock) {
      TokenLength len = std::min(NumTokenPerBlock, ids.size() - i);
      re.push_back(hasher.update(ids.data() + i, len));
    }
    return re;
  }

  void collect_locations(CacheInfo info, SegmentLocations& seg_locs) {
    auto now_prefix = prefix;
    size_t length = match_length;
    while (now_prefix != nullptr) {
      TokenLength local_length = length - now_prefix->start_length;
      auto loc = now_prefix->locations.get_location(info, local_length);
      if (loc.has_value()) {
        seg_locs.add_location(now_prefix->start_length / NumTokenPerBlock, loc.value());
      }
      length = now_prefix->start_length;
      now_prefix = now_prefix->prev;
    }
  }
};

std::string to_string(const MatchStatus& status) {
  switch (status) {
    case Exact:
      return "Exact";
    case Partial:
      return "Partial";
    case NotMatchExact:
      return "NotMatchExact";
    case NotMatchPartial:
      return "NotMatchPartial";
    default:
      return "Unknown";
  }
}

struct MatchByBlock {
  // prefix, block idx at prefix, status
  std::vector<std::tuple<Prefix*, BlockLength, MatchStatus>> matches;

  bool any_match() {
    for (auto& [p, l, m] : matches) {
      if (p) {
        return true;
      }
    }
    return false;
  }

  size_t partial_count() {
    size_t re = 0;
    for (auto& [p, l, m] : matches) {
      if (m == Partial) {
        re++;
      }
    }
    return re;
  }

  bool has_partial() { return partial_count() > 0; }

  std::vector<std::optional<TokensHash>> matched_hashes(CacheInfo info, Layer layer) {
    // TODO: This function might be slow
    std::vector<std::optional<TokensHash>> re(matches.size(), std::nullopt);

    for (size_t i = 0; i < matches.size(); i++) {
      TokensHasher hasher;
      hasher.reset(info.hash_value());
      hasher.update_raw(&layer, sizeof(layer));
      auto& [p, idx, status] = matches[i];
      if (p) {
        p->hash_to_with((idx + 1) * NumTokenPerBlock, hasher);
        re[i] = hasher.get();
      }
    }
    return re;
  }

  void collect_locations(CacheInfo info, SegmentLocations& seg_locs) {
    for (size_t i = 0; i < matches.size(); i++) {
      auto& [p, idx, status] = matches[i];
      if (p) {
        auto local_at = idx - p->start_length / NumTokenPerBlock;
        seg_locs.set_location(i, p->locations.get_location_of_a_block(info, local_at).value());
      }
    }
  }

  std::string debug_string() {
    std::string re = fmt::format("{} Match: ", matches.size());
    for (auto& [p, idx, status] : matches) {
      switch (status) {
        case Exact:
          re += "E";
          break;
        case Partial:
          re += "P";
          break;
        case NotMatchExact:
          re += "N";
          break;
        case NotMatchPartial:
          re += "n";
          break;
        default:
          assert(0);
      }
    }
    return re;
  }
};

struct PrefixTree {
  std::shared_mutex rw_lock;

  std::atomic_uint64_t prefix_id_counter = 1;
  using MapT =
      std::unordered_map<TokensHash, std::pair<std::shared_ptr<Prefix>, BlockLength>>;  // Prefix, start_block_idx
  MapT prefix_map;

  std::shared_ptr<Metrics> met;

  std::vector<std::shared_ptr<Prefix>> prefix_refs = {nullptr};  // 0 is nullptr

  friend void to_json(nlohmann ::json& nlohmann_json_j, const PrefixTree& nlohmann_json_t) {
    nlohmann_json_j["prefix_id_counter"] = nlohmann_json_t.prefix_id_counter.load();
    nlohmann_json_j["prefix_refs"] = nlohmann::json::array();
    for (auto prefix : nlohmann_json_t.prefix_refs) {
      if (prefix == nullptr)
        continue;
      nlohmann_json_j["prefix_refs"].push_back(*prefix);
    }
  }
  friend void from_json(const nlohmann ::json& nlohmann_json_j, PrefixTree& nlohmann_json_t) {
    nlohmann_json_t.prefix_id_counter = nlohmann_json_j.at("prefix_id_counter").get<uint64_t>();

    nlohmann_json_t.prefix_refs.resize(nlohmann_json_t.prefix_id_counter);
    for (size_t i = 1; i < nlohmann_json_t.prefix_id_counter; ++i) {
      auto prefix = std::make_shared<Prefix>();
      nlohmann_json_j.at("prefix_refs")[i - 1].get_to(*prefix);
      nlohmann_json_t.prefix_refs[i] = prefix;
    }
    nlohmann_json_t.init_prevs();
    nlohmann_json_t.init_map();
  };

  void init_prevs() {
    for (auto p : prefix_refs) {
      if (p) {
        if (p->prev_set == false) {
          p->prev = prefix_refs[reinterpret_cast<uint64_t>(p->prev)].get();
          p->prev_set = true;
        }
      }
    }
  }

  void init_map() {
    assert(prefix_map.empty());
    for (auto p : prefix_refs) {
      if (p == nullptr)
        continue;

      auto ids = p->full();
      for (TokenLength i = p->start_length; i < p->length(); i += NumTokenPerBlock) {
        TokenLength end = std::min(i + NumTokenPerBlock, p->length());
        assert(end % NumTokenPerBlock == 0);
        auto hash = TokensHasher::hash(ids.data(), end);
        prefix_map[hash] = {p, end / NumTokenPerBlock - 1};
      }
    }
  }

  // Look up prefix from the map, return the matched prefix and length.
  // If the prefix is not found, match contains nullptr and 0.
  PrefixMatch look_up(Token* data, TokenLength length, bool need_lock = true) {
    std::shared_lock<std::shared_mutex> sl;
    if (need_lock) {
      sl = std::shared_lock<std::shared_mutex>(rw_lock);
    }
    // TODO: prefix cache
  }

  PrefixMatch look_up_or_insert(Token* data, TokenLength length) {
    std::unique_lock<std::shared_mutex> ul(rw_lock);

    auto match = look_up(data, length, false);
    if (match.match_length == length) {
      return match;
    }
    auto new_prefix = new_prefix_node(match.prefix, match.match_length, data, length, false);

    PrefixMatch re;
    re.prefix = new_prefix.get();
    re.match_length = length;
    return re;
  }

  std::shared_ptr<Prefix> new_prefix_node(Prefix* prev, TokenLength prev_match_length, Token* data, TokenLength length,
                                          bool need_lock = true) {
    std::unique_lock<std::shared_mutex> ul;
    if (need_lock)
      ul = std::unique_lock<std::shared_mutex>(rw_lock);
    auto new_prefix = std::make_shared<Prefix>();
    new_prefix->prefix_id = prefix_id_counter.fetch_add(1);
    new_prefix->start_length = prev_match_length;
    new_prefix->ids = Tokens(data + prev_match_length, data + length);
    new_prefix->prev = prev;
    new_prefix->prev_set = true;
    prefix_refs.push_back(new_prefix);
    met->prefix_nodes->Increment();
    met->prefix_block_count->Increment(div_up(length - prev_match_length, NumTokenPerBlock));

    assert(prefix_refs.size() == prefix_id_counter.load());

    TokensHasher hasher;
    hasher.update(data, prev_match_length);

    for (TokenLength i = prev_match_length; i < length; i += NumTokenPerBlock) {
      TokenLength len = std::min(NumTokenPerBlock, length - i);
      auto hash = hasher.update(data + i, len);
      prefix_map[hash] = {new_prefix, i / NumTokenPerBlock};
    }

    return new_prefix;
  }

  void debug() {
    fmt::print("PrefixTree with {} prefixes, prefix counter: {}\n", prefix_map.size(), prefix_id_counter.load());
    for (auto& [hash, prefix] : prefix_map) {
      fmt::print("Hash: {:016x}, start block {}\n", hash, prefix.second);
      prefix.first->debug();
    }
  }
};

size_t locations_blocks_count(const std::vector<Location>& locations) {
  auto re = 0;
  for (auto& loc : locations) {
    re += loc.length;
  }
  return re;
}

struct DoubleCacheHandle : public DoubleCacheHandleInterface {
  ModelName model_name;
  QuantType quant_type;
  bool is_k_cache_on;
  bool is_v_cache_on;
  CacheInfo k_info() {
    if (is_k_cache_on == false) {
      SPDLOG_WARN("Get K CacheInfo, but K Cache is off");
    }
    return CacheInfo{
        .model_name = model_name,
        .is_key_cache = true,
        .quant_type = quant_type,
    };
  };

  CacheInfo v_info() {
    if (is_v_cache_on == false) {
      SPDLOG_WARN("Get V CacheInfo, but K Cache is off");
    }
    return CacheInfo{
        .model_name = model_name,
        .is_key_cache = false,
        .quant_type = quant_type,
    };
  };

  Tokens ids;
  TokenLength estimated_length;

  bool enable_alt = false;
  PrefixMatch match;
  // MatchByBlock match_by_blocks;

  std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>> k_cache_handles;
  std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>> v_cache_handles;

  SegmentLocations k_seg_locs;
  SegmentLocations v_seg_locs;

  KVC2* kvc2_top;

  // for Cache Fusion
  std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>> attatched_cache_handles;

  std::unique_ptr<CacheBlockEntryCollector> cpu_releaser = nullptr, gpu_releaser = nullptr;

  std::vector<size_t> gpu_only_block_idx;

  virtual ~DoubleCacheHandle();
  // interface
  TokenLength matched_length() override {
    if (enable_alt) {
      assert(0);
    } else {
      return match.match_length;
    }
  }
  MatchStatus status_at(BlockLength i) {
    assert(i < div_up(estimated_length, NumTokenPerBlock));
    if (enable_alt) {
      assert(false);
      // if (i >= match_by_blocks.matches.size()) {
      //   return match_by_blocks.has_partial() ? MatchStatus::NotMatchPartial : MatchStatus::NotMatchExact;
      // }
      // return std::get<2>(match_by_blocks.matches[i]);
    } else {
      if (i < match.match_length / NumTokenPerBlock) {
        return MatchStatus::Exact;
      } else {
        return MatchStatus::NotMatchExact;
      }
    }
  }
  std::vector<MatchStatus> matched_status() override { assert(false); }

  bool any_match() {
    if (enable_alt) {
      assert(false);
      // return match_by_blocks.any_match();
    } else {
      return match.prefix != nullptr;
    }
  }

  BlockLength match_range_length() {
    if (enable_alt) {
      assert(false);
      // return match_by_blocks.matches.size();
    } else {
      return div_up(match.match_length, NumTokenPerBlock);
    }
  }

  std::vector<layer_data> handle_data(bool is_key_cache) override { return export_raw_pointers(is_key_cache); }
  bool to_gpu() override;
  void to_gpu_async(std::function<void(bool)> call_back) override;

  std::vector<size_t> get_gpu_block_idx() override;

  bool alloc_attached_blocks(BlockLength count);
  std::vector<size_t> get_gpu_attached_block_idx() override;

  void append_tokens(Token* tokens, TokenLength length) override;

  void debug() override {}

  void set_cache_info(ModelName model_name, QuantType quant_type, bool turn_on_k_cache, bool turn_on_v_cache) {
    this->model_name = model_name;
    this->quant_type = quant_type;
    if (turn_on_k_cache) {
      is_k_cache_on = true;
      k_cache_handles.resize(k_info().hidden_layer_count());
    } else {
      is_k_cache_on = false;
      k_cache_handles.clear();
    }
    if (turn_on_v_cache) {
      is_v_cache_on = true;
      v_cache_handles.resize(v_info().hidden_layer_count());
    } else {
      is_v_cache_on = false;
      v_cache_handles.clear();
    }
  }

  void check_before_insert() {
    std::optional<size_t> blocks_count = std::nullopt;

    auto check_single_cache = [&blocks_count](CacheInfo cache_info,
                                              std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& layers,
                                              Tokens& ids) {
      for (size_t i = 0; i < cache_info.hidden_layer_count(); i++) {
        auto& layer = layers[i];
        if (blocks_count.has_value() == false) {
          blocks_count = layer.size();
        } else {
          if (blocks_count.value() != layer.size()) {
            SPDLOG_ERROR("Layer {} has different block count", i);
            throw std::runtime_error("Layer has different block count");
          }
        }
      }
      if (blocks_count.has_value()) {
        if (blocks_count.value() != div_up(ids.size(), NumTokenPerBlock)) {
          SPDLOG_ERROR("Block count not match, ids: {}, blocks: {}", ids.size(), blocks_count.value());
          throw std::runtime_error("Block count not match");
        }
      }
    };

    if (is_k_cache_on)
      check_single_cache(k_info(), k_cache_handles, ids);
    if (is_v_cache_on)
      check_single_cache(v_info(), v_cache_handles, ids);
  }

  template <typename Fn>
  void for_all_cache_block_entry(Fn f) {
    if (is_k_cache_on) {
      for (auto& layer : k_cache_handles) {
        for (auto& block : layer) {
          if (f(block) == false)
            return;
        }
      }
    }
    if (is_v_cache_on) {
      for (auto& layer : v_cache_handles) {
        for (auto& block : layer) {
          if (f(block) == false)
            return;
        }
      }
    }
  }

  // concurrent check ok
  bool alloc_on_cpu() {
    assert(cpu_releaser == nullptr);
    std::unique_ptr<CacheBlockEntryCollector> releaser =
        std::make_unique<CacheBlockEntryCollector>([](CacheBlockEntry* entry) {
          auto lg = entry->lock_guard();
          entry->cpu_cc.ref_count.fetch_sub(1);
        });
    bool ok = true;

    for_all_cache_block_entry([&ok, &releaser](std::shared_ptr<CacheBlockEntry>& block_entry) {
      if (block_entry->inc_ref_or_alloc_on_cpu() == false) {
        ok = false;
        return false;
      } else {
        releaser->entries.push_back(block_entry.get());
      }
      return true;
    });

    if (ok) {
      cpu_releaser = std::move(releaser);
    }
    return ok;
  }

  bool alloc_on_gpu_cols() {
    assert(is_k_cache_on);
    assert(gpu_releaser == nullptr);
    std::unique_ptr<CacheBlockEntryCollector> releaser =
        std::make_unique<CacheBlockEntryCollector>([](CacheBlockEntry* entry) {
          auto lg = entry->lock_guard();
          entry->gpu_cc.ref_count.fetch_sub(1);
        });

    GPUPageCache* gpu_cache = k_cache_handles[0][0]->manager->gpu_cache.get();
    gpu_cache->background_flush_back->wakeUpWait();

    bool ok = true;
    size_t want_count = 0;
    for (size_t i = 0; i < k_cache_handles[0].size(); i++) {
      auto lg = k_cache_handles[0][i]->lock_guard();
      if (k_cache_handles[0][i]->gpu_block_idx.has_value() == false) {
        want_count += 1;
        if (gpu_cache->alloc_col(k_cache_handles, v_cache_handles, i) == false) {
          ok = false;
          break;
        }
      }
      k_cache_handles[0][i]->gpu_cc.ref_count.fetch_add(1);
      releaser->entries.push_back(k_cache_handles[0][i].get());
    }
    if (ok == false) {
      SPDLOG_WARN("Handle cannot allocate {} gpu pages", want_count);
    } else {
      gpu_releaser = std::move(releaser);
    }
    return ok;
  }

  static void segment_io_layer(async_store::IODealer* dealer, IO_Helper<CacheBlockEntry>& io_helper,
                               async_store::ArrayStore* store,
                               std::vector<std::shared_ptr<CacheBlockEntry>>& layer_entries, size_t block_start,
                               size_t length, Layer layer, const SegmentLocations& locations, IOOption option) {
    SPDLOG_TRACE("{} [{}:{}) blocks to/from disk", to_string(option), block_start, block_start + length);
    for (size_t i = block_start; i < block_start + length; i++) {
      if (locations.get_idx(i).has_value()) {
        SPDLOG_TRACE("Location for block {}, {}", i, locations.get_idx(i).value());
        layer_entries[i]->io_with(dealer, io_helper, store, layer, locations.get_idx(i).value(), option);
      }
    }
  }

  std::shared_ptr<IO_Helper<CacheBlockEntry>> segment_io(async_store::IODealer* dealer, DiskCacheManager* manager,
                                                         BlockLength block_start, BlockLength length, IOOption option) {
    auto io_helper = std::make_shared<IO_Helper<CacheBlockEntry>>([option](CacheBlockEntry* b) {
      switch (option) {
        case IO_ForceRead:
          break;
        case IO_ForceWrite:
          break;
        case IO_Read: {
          b->cpu_cc.tc.set_has_data();
          break;
        }
        case IO_Write:
          break;
        default:
          assert(0);
      }
    });

    auto single_segment_io = [dealer, manager, block_start, length, option, io_helper](
                                 CacheInfo info, SegmentLocations& seg_locs,
                                 std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& layers) {
      assert(layers[0].size() >= block_start + length);

      auto allocator = manager->get_allocator(info);

      for (size_t l = 0; l < info.hidden_layer_count(); l++) {
        segment_io_layer(dealer, *io_helper, allocator->get_store(l), layers[l], block_start, length, l, seg_locs,
                         option);
      }
    };

    if (is_k_cache_on)
      single_segment_io(k_info(), k_seg_locs, k_cache_handles);
    if (is_v_cache_on)
      single_segment_io(v_info(), v_seg_locs, v_cache_handles);

    io_helper->finish_add_taks();
    SPDLOG_DEBUG("Segment IO Submitted, total task count {}", io_helper->total_task_count);
    return io_helper;
  }

  std::shared_ptr<IO_Helper<CacheBlockEntry>> gpu_io(GPUPageCache* gpu_cache, BlockLength block_start,
                                                     BlockLength length, IOOption option) {
    auto io_helper = std::make_shared<IO_Helper<CacheBlockEntry>>([option](CacheBlockEntry* b) {
      switch (option) {
        case IO_ForceRead:
          break;
        case IO_ForceWrite:
          break;
        case IO_Read: {
          b->gpu_cc.tc.set_has_data();
          break;
        }
        case IO_Write:
          break;
        default:
          assert(0);
      }
    });

    cudaMemcpyKind direction;
    if (option == IO_Read || option == IO_ForceRead) {
      direction = cudaMemcpyHostToDevice;
    }
    if (option == IO_Write || option == IO_ForceWrite) {
      direction = cudaMemcpyDeviceToHost;
    }

    auto reqs = gpu_cache->basic_request(direction, [io_helper]() { io_helper->batch_promise.set(); });

    for (size_t i = block_start; i < length; i++) {
      auto status = status_at(i);
      if (status == NotMatchExact || status == NotMatchPartial) {
        SPDLOG_DEBUG("GPU: Col Handle not match (Skipped by Alt Match)");
        continue;
      }
      auto ptr = k_cache_handles[0][i].get();

      switch (option) {
        case IO_Read: {
          if (io_helper->absorb_tc(ptr, ptr->gpu_cc.tc) == false) {
            // SPDLOG_DEBUG("GPU: Col Handle need me to wait");
            continue;
          }
          break;
        }
        case IO_ForceRead: {
          break;
        }
        case IO_ForceWrite: {
          break;
        }
        case IO_Write: {
          break;
        }
        default: {
          assert(0);
        }
      }
      SPDLOG_DEBUG("GPU: Col Handle needs me to transfer");
      gpu_cache->append_col_to_request(reqs, k_cache_handles, v_cache_handles, i);
    }
    io_helper->new_task(reqs.size());
    gpu_cache->submit_requests(reqs);
    io_helper->finish_add_taks();
    return io_helper;
  }

  // void set_raw_handles(const std::vector<layer_data>& k, const std::vector<layer_data>& v) {
  //   set_raw_handles(true, k);
  //   set_raw_handles(false, v);
  // }
  void set_raw_handles(bool is_key_cache, const std::vector<layer_data>& layer_data) {
    auto single_set_raw_handles = [layer_data](CacheInfo info,
                                               std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& handles) {
      handles.resize(layer_data.size());
      for (size_t i = 0; i < info.hidden_layer_count(); i++) {
        auto& layer = layer_data[i];
        handles[i].clear();
        for (auto& block_data : layer) {
          auto handle = std::make_shared<CacheBlockEntry>();
          handle->data = reinterpret_cast<void*>(block_data);
          handle->size = info.element_size(NumTokenPerBlock);
          handles[i].push_back(handle);
        }
      }
    };

    if (is_key_cache) {
      is_k_cache_on = true;
      single_set_raw_handles(k_info(), k_cache_handles);
    } else {
      is_v_cache_on = true;
      single_set_raw_handles(v_info(), v_cache_handles);
    }
  }

  std::vector<layer_data> export_raw_pointers(bool is_key_cache) {
    std::vector<layer_data> re;

    auto single_export_raw_pointers = [&re](std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& layers) {
      for (auto& layer_handle : layers) {
        layer_data layer;
        for (size_t i = 0; i < layer_handle.size(); i++) {
          auto block = layer_handle.at(i);
          layer.push_back(reinterpret_cast<data_block_ptr>(block->data));
        }
        re.push_back(layer);
      }
    };

    if (is_key_cache) {
      if (is_k_cache_on == false) {
        SPDLOG_WARN("Export K Cache, but K Cache is off");
      }
      single_export_raw_pointers(k_cache_handles);
    } else {
      if (is_v_cache_on == false) {
        SPDLOG_WARN("Export V Cache, but V Cache is off");
      }
      single_export_raw_pointers(v_cache_handles);
    }

    return re;
  }

  void get_handles();
  void get_empty_handles();

  void collect_locations() {
    if (enable_alt) {
      assert(false);
      // match_by_blocks.collect_locations(k_info(), k_seg_locs);
      // match_by_blocks.collect_locations(v_info(), v_seg_locs);
    } else {
      if (is_k_cache_on)
        match.collect_locations(k_info(), k_seg_locs);
      if (is_v_cache_on)
        match.collect_locations(v_info(), v_seg_locs);
    }
    if (is_k_cache_on)
      k_seg_locs.debug();
    // v_seg_locs.debug();
  }
};

struct KVC2 : KVC2Interface {
  KVC2Config config;
  std::shared_ptr<Metrics> met;

  std::filesystem::path root;
  std::unique_ptr<PrefixTree> tree;
  std::unique_ptr<DiskCacheManager> disk_cache;
  std::shared_ptr<PageAlignedMemoryPool> memory_pool;
  std::unique_ptr<CacheEntryManager> cache_manager;
  std::unique_ptr<async_store::IODealer> io_dealer;

  std::shared_ptr<GPUPageCache> gpu_cache;

 public:
  void load() override {
    load_quant_configs(root / "quant_configs.json");
    load_model_configs(root / "model_configs.json");
    {
      auto where = root / "tree.json";
      if (std::filesystem::exists(where)) {
        nlohmann::json j;
        std::ifstream i(where);
        i >> j;
        j.get_to(*tree);
        SPDLOG_WARN("Loaded from {}", where.c_str());
      }
    }
    {
      auto where = root / "disk_cache.json";
      if (std::filesystem::exists(where)) {
        nlohmann::json j;
        std::ifstream i(where);
        i >> j;
        j.get_to(*disk_cache);
        SPDLOG_WARN("Loaded from {}", where.c_str());
      }
    }
    {
      auto where = root / "config.json";
      if (std::filesystem::exists(where)) {
        nlohmann::json j;
        std::ifstream i(where);
        i >> j;
        j.get_to(config);
        SPDLOG_WARN("Loaded from {}", where.c_str());
      }
    }
  }

  void save() override {
    if (config.save_to_disk == false) {
      return;
    }
    flush_back();
    {
      nlohmann::json j;
      j = *tree;
      auto where = root / "tree.json";
      std::ofstream o(where);
      o << j;
      SPDLOG_WARN("Serialized to {}", where.c_str());
    }
    {
      nlohmann::json j;
      j = *disk_cache;
      auto where = root / "disk_cache.json";
      std::ofstream o(where);
      o << j;
      SPDLOG_WARN("Serialized to {}", where.c_str());
    }
    {
      nlohmann::json j;
      j = config;
      auto where = root / "config.json";
      std::ofstream o(where);
      o << j;
      SPDLOG_WARN("Serialized to {}", where.c_str());
    }
    dump_quant_configs(root / "quant_configs.json");
    dump_model_configs(root / "model_configs.json");
  }

  void raw_insert(ModelName model_name, QuantType quant_type, Token* id, TokenLength length,
                  const std::vector<layer_data>& k_cache, const std::vector<layer_data>& v_cache) override {
    TimeObserver time_observer(met->raw_insert_time_ms);

    SPDLOG_INFO("Raw Insert");
    if (length % NumTokenPerBlock != 0) {
      SPDLOG_WARN("Try to insert tokens with length {}, which is not a multiple of NumTokenPerBlock({}), getting floor",
                  length, NumTokenPerBlock);
      length = length / NumTokenPerBlock * NumTokenPerBlock;
    }

    auto h = std::make_shared<DoubleCacheHandle>();
    h->kvc2_top = this;
    h->set_cache_info(model_name, quant_type, config.k_cache_on, config.v_cache_on);
    h->ids = Tokens(id, id + length);

    if (config.k_cache_on)
      h->set_raw_handles(true, k_cache);
    if (config.v_cache_on)
      h->set_raw_handles(false, v_cache);

    h->check_before_insert();

    h->match = tree->look_up_or_insert(id, length);

    auto now_prefix = h->match.prefix;
    assert(config.k_cache_on);

    if (now_prefix->locations.get_location(h->k_info(), length - now_prefix->start_length).has_value()) {
      assert(now_prefix->locations.get_location(h->v_info(), length - now_prefix->start_length).has_value());
      SPDLOG_INFO("KV Cache Already on disk");
      // already on disk
    } else {
      now_prefix = now_prefix->to_first_prefix_without_disk_locations(h->k_info());

      // insert new kv cache locations
      TokenLength new_length = length - now_prefix->start_length;
      SPDLOG_DEBUG("Inserting new kv cache, length: {}", new_length);
      assert(new_length > 0);

      if (config.v_cache_on) {
        // allocate a big space on disk
        auto k_loc = disk_cache->allocate(h->k_info(), div_up(new_length, NumTokenPerBlock));
        auto v_loc = disk_cache->allocate(h->v_info(), div_up(new_length, NumTokenPerBlock));
        h->k_seg_locs.add_location(now_prefix->start_length / NumTokenPerBlock, k_loc);
        h->v_seg_locs.add_location(now_prefix->start_length / NumTokenPerBlock, v_loc);

        // split it to prefix trees
        for (auto tail = h->match.prefix; tail != now_prefix->prev; tail = tail->prev) {
          TokenLength local_ids_length = tail->local_length();
          tail->update_location(h->k_info(), k_loc.cut_tail(div_up(local_ids_length, NumTokenPerBlock)));
          tail->update_location(h->v_info(), v_loc.cut_tail(div_up(local_ids_length, NumTokenPerBlock)));
        }
        assert(k_loc.length == 0);
        assert(v_loc.length == 0);
      } else {
        // allocate a big space on disk
        auto k_loc = disk_cache->allocate(h->k_info(), div_up(new_length, NumTokenPerBlock));
        h->k_seg_locs.add_location(now_prefix->start_length / NumTokenPerBlock, k_loc);

        // split it to prefix trees
        for (auto tail = h->match.prefix; tail != now_prefix->prev; tail = tail->prev) {
          TokenLength local_ids_length = tail->local_length();
          tail->update_location(h->k_info(), k_loc.cut_tail(div_up(local_ids_length, NumTokenPerBlock)));
        }
        assert(k_loc.length == 0);
      }

      // write new kv cache
      auto disk_io_helper =
          h->segment_io(io_dealer.get(), disk_cache.get(), now_prefix->start_length / NumTokenPerBlock,
                        div_up(new_length, NumTokenPerBlock), IO_ForceWrite);
      disk_io_helper->wait();
    }
  }

  TokenLength raw_read(ModelName model_name, QuantType quant_type, Token* id, TokenLength length,
                       const std::vector<layer_data>& k_cache, const std::vector<layer_data>& v_cache) override {
    SPDLOG_INFO("Raw Read");
    auto h = std::make_shared<DoubleCacheHandle>();
    h->kvc2_top = this;
    h->set_cache_info(model_name, quant_type, config.k_cache_on, config.v_cache_on);
    h->ids = Tokens(id, id + length);

    if (config.k_cache_on)
      h->set_raw_handles(true, k_cache);
    if (config.v_cache_on)
      h->set_raw_handles(false, v_cache);

    h->match = tree->look_up(id, length);
    if (h->match.prefix == nullptr) {
      SPDLOG_INFO("Not Found");
      return 0;
    }
    SPDLOG_DEBUG("Found {}", h->match.match_length);
    h->collect_locations();
    auto disk_io_helper = h->segment_io(io_dealer.get(), disk_cache.get(), 0,
                                        div_up(h->match.match_length, NumTokenPerBlock), IO_ForceRead);

    disk_io_helper->wait();
    return h->match.match_length;
  }

  std::shared_ptr<DoubleCacheHandleInterface> lookup(ModelName model_name, QuantType quant_type, Token* id,
                                                     TokenLength length, TokenLength estimated_length) override {
    TimeObserver time_observer(met->lookup_time_ms);
    auto re = std::make_shared<DoubleCacheHandle>();
    re->set_cache_info(model_name, quant_type, config.k_cache_on, config.v_cache_on);
    re->ids = Tokens(id, id + length);
    re->estimated_length = estimated_length;
    re->kvc2_top = this;
    SPDLOG_DEBUG("Lookup TokenLength {}", length);
    if (config.gpu_only == false) {
      // TODO:
    }
    return re;
  };

  std::shared_ptr<DoubleCacheHandleInterface> lookup_to_gpu(ModelName model_name, QuantType quant_type, Token* id,
                                                            size_t length, size_t estimated_length) override {
    std::promise<std::shared_ptr<DoubleCacheHandleInterface>> p;
    lookup_to_gpu_async(model_name, quant_type, id, length, estimated_length, [&p](auto re) { p.set_value(re); });
    return p.get_future().get();
  }

  void lookup_to_gpu_async(ModelName model_name, QuantType quant_type, Token* id, TokenLength length,
                           TokenLength estimated_length,
                           std::function<void(std::shared_ptr<DoubleCacheHandleInterface>)> call_back) override {
    auto re = lookup(model_name, quant_type, id, length, estimated_length);
    if (re == nullptr) {
      call_back(nullptr);
      return;
    }
    auto h = static_cast<DoubleCacheHandle*>(re.get());
    if (config.gpu_only) {
      auto total_block_count = div_up(estimated_length, NumTokenPerBlock);
      h->gpu_only_block_idx = gpu_cache->gpu_only_alloc_col(total_block_count);
      if (h->gpu_only_block_idx.empty()) {
        call_back(nullptr);
      } else {
        call_back(re);
      }

    } else {
      if (h->k_info().hidden_layer_count() != gpu_cache->config.layer_count) {
        SPDLOG_ERROR("GPU Cache Layer Count not match");
        assert(false);
      }

      if (h->alloc_on_gpu_cols() == false) {
        call_back(nullptr);
        return;
      }

      h->to_gpu_async([call_back, re](bool ok) {
        if (ok) {
          call_back(re);
        } else {
          call_back(nullptr);
        }
      });
    }
  }

  std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_kvcache() override {
    return {gpu_cache->k_cache, gpu_cache->v_cache};
  }

  void flush_back() {
    gpu_cache->background_flush_back->wakeUpWait();
    cache_manager->background_flush_back->wakeUpWait();
  }

  void debug() override {
    cache_manager->debug();
    tree->debug();
  }

  virtual ~KVC2() { flush_back(); };

  KVC2(KVC2Config config) : config(config) {
    SPDLOG_INFO("Creating KVC2 using these config");
    SPDLOG_INFO("    GPU Only: {}", config.gpu_only);
    SPDLOG_INFO("    Load: {}, Save: {}", config.load_from_disk, config.save_to_disk);
    SPDLOG_INFO("    Path: {}", config.path);
    SPDLOG_INFO("    Config Path: {}", config.config_path);
    SPDLOG_INFO("    Num Token/Page: {}, Memory Pool Size: {}", config.num_token_per_page,
                readable_number(config.memory_pool_size));
    SPDLOG_INFO("    Evict Count: {}, Metrics Port: {}", config.evict_count, config.metrics_port);
    SPDLOG_INFO("    Recompute Ratio: {:.2f}", config.recompute_ratio);

    if (config.gpu_cache_config) {
      const auto& gpu_config = *config.gpu_cache_config;
      SPDLOG_INFO("    GPU Devices: {}", format_vector(gpu_config.gpu_devices_id));
      SPDLOG_INFO("    Layer Count: {}, Total KVCache Pages: {}", gpu_config.layer_count,
                  gpu_config.total_kvcache_pages);
      SPDLOG_INFO("    Num Token/Page: {}, Num K Heads: {}", gpu_config.num_token_per_page, gpu_config.num_k_heads);
      SPDLOG_INFO("    K Head Dim: {}, Tensor Type: {}", gpu_config.k_head_dim,
                  static_cast<int>(gpu_config.tensor_type));
      SPDLOG_INFO("    MemcpyCudaStreams/Device: {}", gpu_config.num_streams_per_device);
    } else {
      SPDLOG_INFO("    GPU Cache Config: None");
    }

    load_model_configs(config.config_path + "/model_configs.json");
    load_quant_configs(config.config_path + "/quant_configs.json");

    // met
    MetricsConfig met_conf;
    met_conf.endpoint = "0.0.0.0:" + std::to_string(config.metrics_port);
    SPDLOG_INFO("Creating kvc2 metrics exporter on {}", met_conf.endpoint);
    met = std::make_shared<Metrics>(met_conf);

    if (config.gpu_only == false) {
      if (config.k_cache_on == false) {
        SPDLOG_ERROR("if k_cache_on is false, gpu_only must be true");
        assert(false);
      }
      root = config.path;
      tree = std::make_unique<PrefixTree>();
      disk_cache = std::make_unique<DiskCacheManager>(config);
      memory_pool = std::make_shared<PageAlignedMemoryPool>(config.memory_pool_size);
      cache_manager = std::unique_ptr<CacheEntryManager>(
          new CacheEntryManager(CacheEntryManagerConfig{.evict_count = config.evict_count, .kvc2_top = this}));
      cache_manager->pool = memory_pool;

      io_dealer = std::make_unique<async_store::IODealer>();
      io_dealer->start_io_thread().detach();

      tree->met = met;
      if (config.gpu_cache_config.has_value()) {
        gpu_cache = std::make_shared<GPUPageCache>(config.gpu_cache_config.value());
        cache_manager->gpu_cache = gpu_cache;
      }
      cache_manager->cpu_background_flush();
      gpu_cache->gpu_background_flush();
    } else {
      SPDLOG_CRITICAL("GPU ONLY MODE, NO PREFIX CACHE");
      gpu_cache = std::make_shared<GPUPageCache>(config.gpu_cache_config.value());
    }
  }
};

std::shared_ptr<KVC2Interface> create_kvc2(KVC2Config config) {
  NumTokenPerBlock = config.num_token_per_page;
  EvictCount = config.evict_count;
  // SPDLOG_WARN("Sizeof KVC2Config {} here", sizeof(KVC2Config));
  return std::make_shared<KVC2>(config);
}

DoubleCacheHandle::~DoubleCacheHandle() {
  if (kvc2_top->config.gpu_only) {
    kvc2_top->gpu_cache->gpu_only_free_cols(gpu_only_block_idx);
  } else {
    for_all_cache_block_entry([](std::shared_ptr<CacheBlockEntry>& block_entry) {
      block_entry->lock_guard();
      if (block_entry->with_key == false && block_entry->data != nullptr) {
        block_entry->free_on_cpu();
      }
      return true;
    });
  }
};

void DoubleCacheHandle::get_handles() {
  size_t new_count = 0, total_count = 0;
  auto get_info_handles = [this, &new_count, &total_count](
                              CacheInfo info, std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& layers) {
    auto total_block_count = div_up(estimated_length, NumTokenPerBlock);
    for (size_t l = 0; l < info.hidden_layer_count(); l++) {
      auto hashes = match.matched_hashes(info, l);
      layers[l].resize(total_block_count, nullptr);
      for (size_t i = 0; i < total_block_count; i++) {
        std::optional<CacheEntryManager::Key> key = std::nullopt;
        if (i < hashes.size())
          key = hashes[i];
        bool is_new;
        total_count += 1;
        layers[l][i] = this->kvc2_top->cache_manager->get(is_new, info.element_size(NumTokenPerBlock), key);
        if (is_new)
          new_count += 1;
        layers[l][i]->cache_info = info;
        layers[l][i]->layer = l;
      }
    }
  };

  if (kvc2_top->config.k_cache_on)
    get_info_handles(k_info(), k_cache_handles);
  if (kvc2_top->config.v_cache_on)
    get_info_handles(v_info(), v_cache_handles);
  SPDLOG_INFO("New Handles: {}/{}", new_count, total_count);
}

bool DoubleCacheHandle::to_gpu() {
  std::promise<bool> p;
  to_gpu_async([&p](bool ok) { p.set_value(ok); });
  return p.get_future().get();
}

void DoubleCacheHandle::to_gpu_async(std::function<void(bool)> call_back) {
  if (enable_alt) {
    assert(false);
    // size_t page_size = kvc2_top->config.num_token_per_page;
    // BlockLength count =
    //     div_up(TokenLength(std::ceil(match_by_blocks.partial_count() * page_size *
    //     kvc2_top->config.recompute_ratio)),
    //            page_size);
    // if (alloc_attached_blocks(count) == false) {
    //   SPDLOG_WARN("Cannot allocate attached GPU block");
    //   call_back(false);
    //   return;
    // } else {
    //   SPDLOG_INFO("Allocated {} attached GPU blocks", count);
    // }
  }

  // don't wait here
  if (any_match() == false) {
    SPDLOG_INFO("No match, No need to load to gpu");
    call_back(true);
    return;
  }

  auto gpu_io_helper = gpu_io(kvc2_top->gpu_cache.get(), 0, match_range_length(), IO_Read);
  gpu_io_helper->call_back = [call_back]() { call_back(true); };

  // Ok this is very stupid, but I have to do this for now
  std::thread([gpu_io_helper]() { gpu_io_helper->wait(); }).detach();
}

bool DoubleCacheHandle::alloc_attached_blocks(BlockLength count) {
  // attached_vertical_handles.resize(count);
  // for (size_t i = 0; i < count; i++) {
  //   attached_vertical_handles[i] = std::shared_ptr<DoubleVerticalBlocksHandle>(new DoubleVerticalBlocksHandle);
  //   attached_vertical_handles[i]->gpu_only = true;
  // }
  // return kvc2_top->gpu_cache->alloc_pages(attached_vertical_handles);
  return true;
}

std::vector<size_t> DoubleCacheHandle::get_gpu_attached_block_idx() {
  std::vector<size_t> re;
  // for (auto& h : attached_vertical_handles) {
  //   re.push_back(h->gpu_block_idx.value());
  // }
  return re;
}

void CacheBlockEntry::set_key(TokensHash key, std::shared_ptr<CacheBlockEntry> me) {
  assert(with_key == false);
  with_key = true;
  hash = key;
  // SPDLOG_DEBUG("Insert New Gen KVCache, key {}", key);
  std::lock_guard<std::mutex> manager_lg(manager->lock);
  if (manager->key_entry_map.contains(me->hash)) {
    SPDLOG_WARN("Duplicate key {}", me->hash);
  } else {
    manager->insert(me);
  }
}

std::vector<size_t> DoubleCacheHandle::get_gpu_block_idx() {
  if (kvc2_top->config.gpu_only) {
    return gpu_only_block_idx;
  } else {
    std::vector<size_t> re;
    for (auto& handle : k_cache_handles[0]) {
      re.push_back(handle->gpu_block_idx.value());
    }
    return re;
  }
}

/*
length : total length of tokens (including matched tokens)
  1. update key, insert CacheBlock hash to lru
  2. set dirty flag
  3. update prefix tree, allocate new disk location
*/
void DoubleCacheHandle::append_tokens(Token* all_tokens, TokenLength length) {
  if (kvc2_top->config.gpu_only) {
    return;
  }
  TimeObserver time_observer(kvc2_top->met->append_tokens_time_ms);
  if (enable_alt) {
    SPDLOG_WARN("Append Tokens Not Implemented for Alternative Path");
    return;
  }
  if (length > estimated_length) {
    SPDLOG_ERROR("Length {} exceed estimated length {}", length, estimated_length);
    assert(false);
  }
  size_t match_length = matched_length();

  if (length < match_length) {
    SPDLOG_WARN("Length {} less than match length {}", length, match_length);
    assert(false);
  }

  if (length > ids.size()) {
    ids.insert(ids.end(), all_tokens + ids.size(), all_tokens + length);
  }

  static const auto num_token_per_page = kvc2_top->config.num_token_per_page;

  if (match_length % num_token_per_page != 0) {
    SPDLOG_ERROR("Match length {} is not multiple of num_token_per_page {}", match_length, num_token_per_page);
    assert(false);
  }

  if (match_length + num_token_per_page > length) {
    // SPDLOG_DEBUG("append_tokens No need to update");
    return;
  }
  SPDLOG_DEBUG("Append Tokens to {}", length);
  auto pre_match_length = match_length;
  // set gpu dirty flag
  size_t new_added_block_count = 0;
  while (match_length + num_token_per_page <= length) {
    match_length += num_token_per_page;
    new_added_block_count += 1;
  }

  // update prefix tree
  match.prefix = kvc2_top->tree->new_prefix_node(match.prefix, pre_match_length, ids.data(), match_length).get();
  match.match_length = match_length;

  // alloc disk location for new added prefix
  auto disk_cache = kvc2_top->disk_cache.get();
  Location k_loc{0, 0}, v_loc{0, 0};
  if (is_k_cache_on) {
    k_loc = disk_cache->allocate(k_info(), new_added_block_count);
    k_seg_locs.add_location(match.prefix->start_length / NumTokenPerBlock, k_loc);
    match.prefix->update_location(k_info(), k_loc);
  }
  if (is_v_cache_on) {
    v_loc = disk_cache->allocate(v_info(), new_added_block_count);
    v_seg_locs.add_location(match.prefix->start_length / NumTokenPerBlock, v_loc);
    match.prefix->update_location(v_info(), v_loc);
  }

  // update cache handles
  auto update_cache_handles = [this, pre_match_length, length](
                                  CacheInfo info, std::vector<std::vector<std::shared_ptr<CacheBlockEntry>>>& layers,
                                  Location loc) {
    TokensHasher hasher;
    for (Layer l = 0; l < info.hidden_layer_count(); l++) {
      hasher.reset(info.hash_value());
      hasher.update_raw(&l, sizeof(l));
      hasher.update(ids.data(), pre_match_length);
      auto page_count_start = pre_match_length / num_token_per_page;
      for (size_t i = pre_match_length; i + num_token_per_page <= length; i += num_token_per_page) {
        auto page_count = i / num_token_per_page;
        hasher.update(ids.data() + i, num_token_per_page);
        auto block = layers[l][page_count];
        {
          auto lg = block->lock_guard();
          block->idx = loc.start_idx + page_count - page_count_start;
          block->set_key(hasher.get(), block);
          if (l == 0 && info.is_key_cache) {
            block->gpu_cc.tc.set_has_data();
          }
          block->gpu_cc.dirty.store(true);
        }
      }
    }
  };

  if (is_k_cache_on) {
    update_cache_handles(k_info(), k_cache_handles, k_loc);
  }
  if (is_v_cache_on) {
    update_cache_handles(v_info(), v_cache_handles, v_loc);
  }

  // kvc2_top->block_cache->debug();
}

void CacheBlockEntry::flush_back_async(IO_Helper<CacheBlockEntry>& helper,
                                       std::vector<std::atomic_bool*>& dirty_flags) {
  auto kvc2_top = manager->config.kvc2_top;
  auto allocator = kvc2_top->disk_cache->get_allocator(cache_info);
  // if (layer == 0) {
  //   SPDLOG_DEBUG("Flush {} to {}", fmt::ptr(this), idx);
  // }
  io_with(kvc2_top->io_dealer.get(), helper, allocator->get_store(layer), layer, idx, IOOption::IO_Write);
  dirty_flags.push_back(&cpu_cc.dirty);
}

void CacheEntryManager::cpu_background_flush() {
  if (background_flush_back.get() == nullptr) {
    SPDLOG_INFO("Starting CPU Background flush");
    background_flush_back = std::unique_ptr<periodic::PeriodicTask>(new periodic::PeriodicTask([this]() {
      // Timer t("CPU Flush");
      std::vector<std::atomic_bool*> dirty_cpus;
      std::vector<std::unique_lock<CacheBlockEntry::MutexT>> entry_uls;
      IO_Helper<CacheBlockEntry> io_helper(nullptr, [&dirty_cpus]() {
        for (auto& flag : dirty_cpus) {
          flag->store(false);
        }
        if (dirty_cpus.size() > 0)
          SPDLOG_DEBUG("{} dirty CPU pages flushed.", dirty_cpus.size());
      });
      {
        std::lock_guard<std::mutex> ul(lock);
        for (auto& e : usage_list) {
          auto ul = e->try_lock();
          if (ul.owns_lock()) {
            if (e->cpu_cc.dirty.load()) {
              entry_uls.push_back(std::move(ul));
              e->flush_back_async(io_helper, dirty_cpus);
            }
          }
          // if (dirty_cpus.size() == 100) {
          //   break;
          // }
        }
      }

      io_helper.finish_add_taks();
      io_helper.wait();
    }));
  } else {
    SPDLOG_ERROR("Flush Thread Already Started");
  }
}

void GPUPageCache::gpu_background_flush() {
  if (background_flush_back.get() == nullptr) {
    SPDLOG_INFO("Starting GPU Background flush");
    background_flush_back = std::unique_ptr<periodic::PeriodicTask>(new periodic::PeriodicTask([this]() {
      // Timer t("GPU Flush");

      std::vector<size_t> dirty_cols;
      std::vector<CacheBlockEntry*> entries;
      std::vector<std::unique_lock<CacheBlockEntry::MutexT>> uls;
      BatchPromise promise(config.gpu_devices_id.size());
      auto reqs = basic_request(cudaMemcpyDeviceToHost, [&promise]() { promise.set(); });

      for (size_t i = 0; i < config.total_kvcache_pages; i++) {
        std::lock_guard<std::mutex> lg(this->lock);
        auto col_uls = try_lock_col(i);
        if (col_uls.empty())
          continue;
        for (size_t l = 0; l < config.layer_count; l++) {
          if (config.k_cache_on &&
              (occupations[l][i]->gpu_cc.dirty.load() == false || occupations[l][i]->cpu_cc.dirty.load()))
            goto next_gpu_page;
          if (config.v_cache_on &&
              (v_occupations[l][i]->gpu_cc.dirty.load() == false || v_occupations[l][i]->cpu_cc.dirty.load()))
            goto next_gpu_page;
        }

        dirty_cols.push_back(i);
        for (size_t l = 0; l < config.layer_count; l++) {
          // occupations[l][i]->alloc_on_cpu_no_lock();
          if (config.k_cache_on)
            entries.push_back(occupations[l][i].get());
          if (config.v_cache_on)
            entries.push_back(v_occupations[l][i].get());
        }
        append_col_to_request(reqs, occupations, v_occupations, i);
        for (auto& ul : col_uls) {
          uls.push_back(std::move(ul));
        }
      next_gpu_page:
        continue;
      }

      submit_requests(reqs);
      promise.get_shared_fut().wait();
      if (dirty_cols.empty() == false)
        SPDLOG_INFO("GPU Flushed Back {} cols", dirty_cols.size());

      for (auto& entry : entries) {
        entry->cpu_cc.tc.set_has_data();
        // we have locks here
        entry->cpu_cc.dirty.store(true);
      }
      for (auto& col : dirty_cols) {
        for (size_t l = 0; l < config.layer_count; l++) {
          if (config.k_cache_on)
            occupations[l][col]->gpu_cc.dirty.store(false);
          if (config.v_cache_on)
            v_occupations[l][col]->gpu_cc.dirty.store(false);
        }
      }
      if (dirty_cols.empty() == false) {
        debug();
      }
    }));
  } else {
    SPDLOG_ERROR("Flush Thread Already Started");
  }
}

}  // namespace kvc2
