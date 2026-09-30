#ifndef CPUINFER_OPERATOR_KT_WEIGHT_ARENA_HPP
#define CPUINFER_OPERATOR_KT_WEIGHT_ARENA_HPP

// File-backed arena for per-expert BufferB weight blocks.
//
// The CPU MoE kernels allocate one ~4-5 MiB std::aligned_alloc(64, ...) block
// per CPU-resident expert matrix (gate / up / down) to hold the repacked,
// quantized weights. For a large MoE that is tens of GiB of *anonymous* memory
// per model, which the OOM killer targets and which can only be reclaimed via
// swap. Those blocks are written once during weight load and are read-only for
// the rest of the process; nothing benefits from them being anonymous.
//
// When `KtWeightArena` is opened with a directory, each such block is instead a
// MAP_SHARED slice of one backing file per (layer, NUMA part). File-backed pages
// are clean once written, so the kernel can reclaim a cold expert's pages
// without swap and refault them from the file (NVMe) on the rare cold route.
// A machine whose CPU expert working set is close to total RAM can then serve
// the model, trading a little cold-route latency for not running out of memory.
//
// Enabled via GeneralMOEConfig::mmap_weights_dir (server flag
// --kt-mmap-experts-dir). Empty string / unopened arena => unchanged
// std::aligned_alloc path, zero overhead.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__linux__)
#include <numa.h>
#include <sched.h>
#endif

class KtWeightArena {
 public:
  KtWeightArena() = default;
  KtWeightArena(const KtWeightArena&) = delete;
  KtWeightArena& operator=(const KtWeightArena&) = delete;

  ~KtWeightArena() {
    for (auto& m : maps_) ::munmap(m.first, m.second);
    if (fd_ >= 0) ::close(fd_);
  }

  bool enabled() const { return fd_ >= 0; }

  // Open the backing file <dir>/bufferb_L<layer>_n<numa>.bin, reserved for
  // `blocks` blocks of at most `max_block_bytes` each. Idempotent; a failure to
  // open or size the file logs and leaves the arena disabled (callers then fall
  // back to std::aligned_alloc automatically).
  void open(const std::string& dir, int layer_idx, int numa_part, size_t blocks, size_t max_block_bytes) {
    if (dir.empty() || fd_ >= 0) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string path = dir + "/bufferb_L" + std::to_string(layer_idx) + "_n" + std::to_string(numa_part) + ".bin";
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);  // no O_TRUNC: a restart re-maps and re-writes over it
    if (fd_ < 0) {
      fprintf(stderr, "[kt-mmap] open(%s): %s -- falling back to anonymous heap for this layer\n", path.c_str(),
              strerror(errno));
      return;
    }
    numa_node_ = current_numa_node();
    size_t blk = round_up(max_block_bytes);
    cap_ = blocks * blk + kSlack;
    if (::ftruncate(fd_, static_cast<off_t>(cap_)) != 0)
      fprintf(stderr, "[kt-mmap] ftruncate(%s, %zu): %s\n", path.c_str(), cap_, strerror(errno));
    fprintf(stderr, "[kt-mmap] layer %d numa %d -> %s (reserve %.1f GiB, node %d)\n", layer_idx, numa_part,
            path.c_str(), cap_ / (1024.0 * 1024 * 1024), numa_node_);
  }

  // Allocate one block. Returns a pointer usable exactly like the old
  // std::aligned_alloc(64, n) result. Sets *file_backed so the caller knows
  // whether it must std::free() the pointer itself (false) or leave it to this
  // arena's destructor (true). On any failure, falls back to std::aligned_alloc
  // and *file_backed = false.
  void* alloc(size_t n, bool* file_backed) {
    *file_backed = false;
    if (fd_ < 0) return std::aligned_alloc(64, n);

    size_t rounded = round_up(n);
    if (off_ + rounded > cap_) {  // grow if a layer needs more than the estimate
      cap_ = off_ + rounded + kSlack;
      if (::ftruncate(fd_, static_cast<off_t>(cap_)) != 0)
        fprintf(stderr, "[kt-mmap] ftruncate grow %zu: %s\n", cap_, strerror(errno));
    }
    off_t off = static_cast<off_t>(off_);
    off_ += rounded;

    void* a = ::mmap(nullptr, rounded, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, off);
    if (a == MAP_FAILED) {
      fprintf(stderr, "[kt-mmap] mmap(off=%lld, sz=%zu): %s -- anonymous fallback\n", static_cast<long long>(off),
              rounded, strerror(errno));
      return std::aligned_alloc(64, n);
    }
    ::madvise(a, rounded, MADV_RANDOM);  // cold-route refaults are scattered, not sequential
#if defined(__linux__)
    // Keep first-touch local on multi-socket. The load memcpy runs on this
    // NUMA part's pinned subpool thread, so plain first-touch is already local
    // on a single-socket box; this makes it explicit for dual-socket.
    if (numa_node_ >= 0 && numa_available() >= 0) numa_tonode_memory(a, rounded, numa_node_);
#endif
    maps_.emplace_back(a, rounded);
    *file_backed = true;
    return a;
  }

 private:
  static constexpr size_t kGranularity = 2u * 1024 * 1024;   // 2 MiB mmap granularity
  static constexpr size_t kSlack = 64ull * 1024 * 1024;      // headroom over the block estimate

  static size_t round_up(size_t n) { return (n + (kGranularity - 1)) & ~(kGranularity - 1); }

  static int current_numa_node() {
#if defined(__linux__)
    if (numa_available() >= 0) {
      int cpu = sched_getcpu();
      if (cpu >= 0) return numa_node_of_cpu(cpu);
    }
#endif
    return -1;
  }

  int fd_ = -1;
  int numa_node_ = -1;
  size_t off_ = 0;
  size_t cap_ = 0;
  std::vector<std::pair<void*, size_t>> maps_;
};

#endif  // CPUINFER_OPERATOR_KT_WEIGHT_ARENA_HPP
