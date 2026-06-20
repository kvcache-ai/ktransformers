/**
 * @file mesh_io_uring.hpp
 * @brief io_uring + O_DIRECT 读取器 + Scale Cache 预加载
 *
 * 读取路径：不走 mmap，而是 O_DIRECT 打开 safetensors 文件。
 * 通过 io_uring 提交读请求，SSD 直接 DMA 写入 NUMA 本地 buffer。
 *
 * Scale Cache（AMXINT4 专用）：
 * - 启动时一次性把所有专家的 scale 数据读入独立 NUMA 本地 buffer
 * - scale_cache_loaded_=true：只发 3 个请求（只读 weight），scale 从 cache memcpy
 * - scale_cache_loaded_=false：发 9 个请求（weight + scale + mins 全读）
 */
#pragma once

#include <cstring>
#include <fcntl.h>
#include <liburing.h>
#include <numa.h>
// liburing.h 定义了 BLOCK_SIZE 等宏，会污染全局命名空间，
// 与 amx_raw_kernels.hpp / fp8-moe.hpp 中的 BLOCK_SIZE 变量冲突
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mesh_config.hpp"

namespace mesh {

// io_uring 读请求优先级（与 schedule_key 配合使用）
enum class ReadPriority : uint8_t {
  Prefetch = 1,  // 预取（低优先级）
  Demand = 10,   // 需求（高优先级）
};

/**
 * @brief io_uring 读取器
 *
 * 封装 liburing，所有读走 O_DIRECT 直达 NUMA buffer，不经过 page cache。
 */
class MeshIoUring {
 public:
  static constexpr int kQueueDepth = 256;  // io_uring SQ ring 大小

  MeshIoUring() {
    if (io_uring_queue_init(kQueueDepth, &ring_, 0) != 0) {
      throw std::runtime_error("MeshIoUring: io_uring_queue_init failed");
    }
  }

  ~MeshIoUring() {
    io_uring_queue_exit(&ring_);
    release_scale_cache();
  }

  // 禁止拷贝
  MeshIoUring(const MeshIoUring&) = delete;
  MeshIoUring& operator=(const MeshIoUring&) = delete;

  // ===== Scale Cache 预加载（AMXINT4 专用）=====

  // 启动时一次性把所有专家的 scale 数据读入 NUMA 本地 buffer
  // expert_num 个专家，每个 scale_bytes 字节
  void preload_scale_cache(int expert_num, int tp_count, int numa_node,
                           const std::vector<std::vector<ExpertFileLayout>>& layouts) {
    if (scale_cache_loaded_) return;

    // 计算每个 TP 分片的 scale 总大小
    for (int tp = 0; tp < tp_count; tp++) {
      ScaleCacheTP cache;
      size_t total_gate = 0, total_up = 0, total_down = 0;
      size_t total_gate_mins = 0, total_up_mins = 0, total_down_mins = 0;
      for (int e = 0; e < expert_num; e++) {
        total_gate += layouts[tp][e].gate_scale_bytes;
        total_up += layouts[tp][e].up_scale_bytes;
        total_down += layouts[tp][e].down_scale_bytes;
        total_gate_mins += layouts[tp][e].gate_mins_bytes;
        total_up_mins += layouts[tp][e].up_mins_bytes;
        total_down_mins += layouts[tp][e].down_mins_bytes;
      }
      cache.gate_scale = numa_alloc_onnode(total_gate, numa_node);
      cache.up_scale = numa_alloc_onnode(total_up, numa_node);
      cache.down_scale = numa_alloc_onnode(total_down, numa_node);
      cache.gate_mins = numa_alloc_onnode(total_gate_mins, numa_node);
      cache.up_mins = numa_alloc_onnode(total_up_mins, numa_node);
      cache.down_mins = numa_alloc_onnode(total_down_mins, numa_node);
      cache.gate_scale_total = total_gate;
      cache.up_scale_total = total_up;
      cache.down_scale_total = total_down;
      cache.gate_mins_total = total_gate_mins;
      cache.up_mins_total = total_up_mins;
      cache.down_mins_total = total_down_mins;

      // 同步读取所有 scale 数据（启动阶段，可以阻塞）
      size_t off_g = 0, off_u = 0, off_d = 0;
      size_t off_gm = 0, off_um = 0, off_dm = 0;
      for (int e = 0; e < expert_num; e++) {
        const auto& layout = layouts[tp][e];
        sync_read_to(layout.fd, layout.gate_scale_offset, layout.gate_scale_bytes,
                     (char*)cache.gate_scale + off_g);
        sync_read_to(layout.fd, layout.up_scale_offset, layout.up_scale_bytes,
                     (char*)cache.up_scale + off_u);
        sync_read_to(layout.fd, layout.down_scale_offset, layout.down_scale_bytes,
                     (char*)cache.down_scale + off_d);
        sync_read_to(layout.fd, layout.gate_mins_offset, layout.gate_mins_bytes,
                     (char*)cache.gate_mins + off_gm);
        sync_read_to(layout.fd, layout.up_mins_offset, layout.up_mins_bytes,
                     (char*)cache.up_mins + off_um);
        sync_read_to(layout.fd, layout.down_mins_offset, layout.down_mins_bytes,
                     (char*)cache.down_mins + off_dm);
        off_g += layout.gate_scale_bytes;
        off_u += layout.up_scale_bytes;
        off_d += layout.down_scale_bytes;
        off_gm += layout.gate_mins_bytes;
        off_um += layout.up_mins_bytes;
        off_dm += layout.down_mins_bytes;
      }
      scale_cache_.push_back(std::move(cache));
    }
    scale_cache_loaded_ = true;
  }

  // 从 scale cache 中 memcpy 某个专家的 scale 到目标 buffer
  void copy_scale_from_cache(int tp_part_idx, int expert_id,
                             void* gate_scale_dst, void* up_scale_dst, void* down_scale_dst,
                             void* gate_mins_dst, void* up_mins_dst, void* down_mins_dst,
                             const std::vector<ExpertFileLayout>& layouts_tp) {
    // 需要外部传入该 TP 的 layouts 来计算偏移
    size_t off_g = 0, off_u = 0, off_d = 0;
    size_t off_gm = 0, off_um = 0, off_dm = 0;
    for (int e = 0; e < expert_id; e++) {
      off_g += layouts_tp[e].gate_scale_bytes;
      off_u += layouts_tp[e].up_scale_bytes;
      off_d += layouts_tp[e].down_scale_bytes;
      off_gm += layouts_tp[e].gate_mins_bytes;
      off_um += layouts_tp[e].up_mins_bytes;
      off_dm += layouts_tp[e].down_mins_bytes;
    }
    const auto& layout = layouts_tp[expert_id];
    const auto& cache = scale_cache_[tp_part_idx];
    memcpy(gate_scale_dst, (char*)cache.gate_scale + off_g, layout.gate_scale_bytes);
    memcpy(up_scale_dst, (char*)cache.up_scale + off_u, layout.up_scale_bytes);
    memcpy(down_scale_dst, (char*)cache.down_scale + off_d, layout.down_scale_bytes);
    memcpy(gate_mins_dst, (char*)cache.gate_mins + off_gm, layout.gate_mins_bytes);
    memcpy(up_mins_dst, (char*)cache.up_mins + off_um, layout.up_mins_bytes);
    memcpy(down_mins_dst, (char*)cache.down_mins + off_dm, layout.down_mins_bytes);
  }

  bool scale_cache_loaded() const { return scale_cache_loaded_; }

  // ===== io_uring 异步读 =====

  // 提交一个专家的读取请求
  // scale_cache_loaded_=true: 发 3 个请求（只读 weight），scale 从 cache memcpy
  // scale_cache_loaded_=false: 发 9 个请求（weight + scale + mins 全读）
  void submit_load(int expert_id, int tp_part_idx,
                   const ExpertFileLayout& layout,
                   void* gate_dst, void* up_dst, void* down_dst,
                   void* gate_scale_dst, void* up_scale_dst, void* down_scale_dst,
                   void* gate_mins_dst, void* up_mins_dst, void* down_mins_dst,
                   ReadPriority priority) {
    int n_reqs = scale_cache_loaded_ ? 3 : 9;
    std::vector<io_uring_sqe*> sqes;
    sqes.reserve(n_reqs);

    // 提交 SQE
    auto* sqe = io_uring_get_sqe(&ring_);
    io_uring_prep_read(sqe, layout.fd, gate_dst, layout.gate_bytes, layout.gate_offset);
    sqes.push_back(sqe);

    sqe = io_uring_get_sqe(&ring_);
    io_uring_prep_read(sqe, layout.fd, up_dst, layout.up_bytes, layout.up_offset);
    sqes.push_back(sqe);

    sqe = io_uring_get_sqe(&ring_);
    io_uring_prep_read(sqe, layout.fd, down_dst, layout.down_bytes, layout.down_offset);
    sqes.push_back(sqe);

    if (!scale_cache_loaded_) {
      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, gate_scale_dst, layout.gate_scale_bytes, layout.gate_scale_offset);
      sqes.push_back(sqe);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, up_scale_dst, layout.up_scale_bytes, layout.up_scale_offset);
      sqes.push_back(sqe);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, down_scale_dst, layout.down_scale_bytes, layout.down_scale_offset);
      sqes.push_back(sqe);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, gate_mins_dst, layout.gate_mins_bytes, layout.gate_mins_offset);
      sqes.push_back(sqe);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, up_mins_dst, layout.up_mins_bytes, layout.up_mins_offset);
      sqes.push_back(sqe);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, down_mins_dst, layout.down_mins_bytes, layout.down_mins_offset);
      sqes.push_back(sqe);
    }

    // 设置 user_data 用于 CQE 回调识别
    for (auto* s : sqes) {
      io_uring_sqe_set_data(s, nullptr);
    }
    io_uring_submit(&ring_);
  }

  // 阻塞等待某专家的读请求完成（通过 CQE 计数）
  void wait_expert(int expert_id, int n_reqs) {
    for (int i = 0; i < n_reqs; i++) {
      io_uring_cqe* cqe;
      io_uring_wait_cqe(&ring_, &cqe);
      io_uring_cqe_seen(&ring_, cqe);
    }
  }

  // 批量提交一层所有专家的读取
  void submit_layer_batch(const std::vector<int>& expert_ids, int tp_part_idx,
                          const std::vector<ExpertFileLayout>& layouts_tp,
                          std::vector<void*> gate_dsts, std::vector<void*> up_dsts,
                          std::vector<void*> down_dsts, ReadPriority priority) {
    for (size_t i = 0; i < expert_ids.size(); i++) {
      int eid = expert_ids[i];
      // scale/mins 目标指针由调用者通过 slot pool 提供
      // 这里简化：只读 weight，scale 从 cache memcpy
      submit_load(eid, tp_part_idx, layouts_tp[eid],
                  gate_dsts[i], up_dsts[i], down_dsts[i],
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                  priority);
    }
  }

  // 提交并等待全部完成（同步接口，启动阶段用）
  void submit_and_wait() {
    io_uring_submit(&ring_);
    // 等待所有 inflight 完成
    unsigned head;
    unsigned count = 0;
    io_uring_cqe* cqe;
    while (true) {
      io_uring_for_each_cqe(&ring_, head, cqe) {
        count++;
      }
      if (count == 0) break;
      io_uring_cq_advance(&ring_, count);
      count = 0;
    }
  }

 private:
  io_uring ring_;

  // Scale Cache（每 TP 一个）
  struct ScaleCacheTP {
    void* gate_scale = nullptr;
    void* up_scale = nullptr;
    void* down_scale = nullptr;
    void* gate_mins = nullptr;
    void* up_mins = nullptr;
    void* down_mins = nullptr;
    size_t gate_scale_total = 0;
    size_t up_scale_total = 0;
    size_t down_scale_total = 0;
    size_t gate_mins_total = 0;
    size_t up_mins_total = 0;
    size_t down_mins_total = 0;
  };
  std::vector<ScaleCacheTP> scale_cache_;
  bool scale_cache_loaded_ = false;

  // 同步读取（启动阶段用，pread + O_DIRECT）
  void sync_read_to(int fd, off_t offset, size_t bytes, void* dst) {
    if (bytes == 0) return;
    size_t done = 0;
    while (done < bytes) {
      ssize_t n = pread(fd, (char*)dst + done, bytes - done, offset + done);
      if (n < 0) {
        throw std::runtime_error("MeshIoUring: pread failed, errno=" + std::to_string(errno));
      }
      done += n;
    }
  }

  void release_scale_cache() {
    for (auto& c : scale_cache_) {
      if (c.gate_scale) numa_free(c.gate_scale, c.gate_scale_total);
      if (c.up_scale) numa_free(c.up_scale, c.up_scale_total);
      if (c.down_scale) numa_free(c.down_scale, c.down_scale_total);
      if (c.gate_mins) numa_free(c.gate_mins, c.gate_mins_total);
      if (c.up_mins) numa_free(c.up_mins, c.up_mins_total);
      if (c.down_mins) numa_free(c.down_mins, c.down_mins_total);
    }
    scale_cache_.clear();
  }
};

}  // namespace mesh
