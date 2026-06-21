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

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
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

  // 启动时一次性把某层所有专家的 scale 数据读入 NUMA 本地 buffer
  // A4 fix: 改为按层预加载，因为每层的 scale 数据不同
  // 辅因2 fix: 按 TP 分片传入对应 NUMA 节点，避免跨 NUMA 访问
  // expert_num 个专家，每个 scale_bytes 字节
  void preload_scale_cache(int expert_num, int tp_count,
                           const std::vector<int>& numa_nodes,
                           const std::vector<std::vector<ExpertFileLayout>>& layouts_tp,
                           int layer_idx) {
    if (scale_cache_loaded_per_layer_.size() > (size_t)layer_idx &&
        scale_cache_loaded_per_layer_[layer_idx]) return;

    // 确保 scale_cache_ 有足够层
    if ((int)scale_cache_.size() <= layer_idx) {
      scale_cache_.resize(layer_idx + 1);
    }

    auto& layer_cache = scale_cache_[layer_idx];
    // 计算每个 TP 分片的 scale 总大小，按 TP 对应 NUMA 节点分配
    for (int tp = 0; tp < tp_count; tp++) {
      int numa_node = numa_nodes[tp % numa_nodes.size()];  // 辅因2: 按 TP 选 NUMA
      ScaleCacheTP cache;
      size_t total_gate = 0, total_up = 0, total_down = 0;
      size_t total_gate_mins = 0, total_up_mins = 0, total_down_mins = 0;
      for (int e = 0; e < expert_num; e++) {
        total_gate += layouts_tp[tp][e].gate_scale_bytes;
        total_up += layouts_tp[tp][e].up_scale_bytes;
        total_down += layouts_tp[tp][e].down_scale_bytes;
        total_gate_mins += layouts_tp[tp][e].gate_mins_bytes;
        total_up_mins += layouts_tp[tp][e].up_mins_bytes;
        total_down_mins += layouts_tp[tp][e].down_mins_bytes;
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
        const auto& layout = layouts_tp[tp][e];
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
      layer_cache.push_back(std::move(cache));
    }
    // 第二轮根因1 fix: 按层设置标志，而非全局标志
    if ((int)scale_cache_loaded_per_layer_.size() <= layer_idx) {
      scale_cache_loaded_per_layer_.resize(layer_idx + 1, false);
    }
    scale_cache_loaded_per_layer_[layer_idx] = true;
  }

  // 从 scale cache 中 memcpy 某个专家的 scale 到目标 buffer
  // A4 fix: 加 layer_idx 参数
  // 辅因1 fix: 支持 nullptr 参数（slot 中无 mins 空间时跳过 mins memcpy）
  void copy_scale_from_cache(int layer_idx, int tp_part_idx, int expert_id,
                             void* gate_scale_dst, void* up_scale_dst, void* down_scale_dst,
                             void* gate_mins_dst, void* up_mins_dst, void* down_mins_dst,
                             const std::vector<ExpertFileLayout>& layouts_tp) {
    if ((int)scale_cache_.size() <= layer_idx) return;
    auto& layer_cache = scale_cache_[layer_idx];
    if ((int)layer_cache.size() <= tp_part_idx) return;

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
    const auto& cache = layer_cache[tp_part_idx];
    // scale 数据必须拷贝（BufferB 的 d 指针依赖）
    if (gate_scale_dst) memcpy(gate_scale_dst, (char*)cache.gate_scale + off_g, layout.gate_scale_bytes);
    if (up_scale_dst) memcpy(up_scale_dst, (char*)cache.up_scale + off_u, layout.up_scale_bytes);
    if (down_scale_dst) memcpy(down_scale_dst, (char*)cache.down_scale + off_d, layout.down_scale_bytes);
    // mins 数据可选拷贝（slot 中无 mins 空间时传 nullptr 跳过）
    if (gate_mins_dst) memcpy(gate_mins_dst, (char*)cache.gate_mins + off_gm, layout.gate_mins_bytes);
    if (up_mins_dst) memcpy(up_mins_dst, (char*)cache.up_mins + off_um, layout.up_mins_bytes);
    if (down_mins_dst) memcpy(down_mins_dst, (char*)cache.down_mins + off_dm, layout.down_mins_bytes);
  }

  // 第二轮根因1 fix: 按层查询 scale cache 是否已预加载
  bool scale_cache_loaded(int layer_idx = -1) const {
    if (layer_idx < 0) {
      // 无参：返回是否任意层已加载（用于粗略判断）
      return !scale_cache_loaded_per_layer_.empty();
    }
    return (size_t)layer_idx < scale_cache_loaded_per_layer_.size() &&
           scale_cache_loaded_per_layer_[layer_idx];
  }

  // ===== io_uring 异步读 =====

  // A6 fix: PendingRequest 跟踪每个专家读取的完成状态
  // 一个专家的读取可能涉及 3-9 个 SQE，所有 SQE 完成后触发 on_complete
  struct PendingRequest {
    int layer_idx = -1;
    int tp_part_idx = -1;
    int expert_id = -1;
    int total_reqs = 0;
    std::atomic<int> completed_reqs{0};
    std::function<void(int, int, int)> on_complete;  // (layer, tp, expert)
  };

  // 提交一个专家的读取请求
  // scale_cache_loaded_=true: 发 3 个请求（只读 weight），scale 从 cache memcpy
  // scale_cache_loaded_=false: 发 9 个请求（weight + scale + mins 全读）
  // A6 fix: 加 layer_idx 和 on_complete 回调，CQE 完成后触发 mark_cached
  void submit_load(int expert_id, int tp_part_idx,
                   const ExpertFileLayout& layout,
                   void* gate_dst, void* up_dst, void* down_dst,
                   void* gate_scale_dst, void* up_scale_dst, void* down_scale_dst,
                   void* gate_mins_dst, void* up_mins_dst, void* down_mins_dst,
                   ReadPriority priority,
                   int layer_idx = -1,
                   std::function<void(int, int, int)> on_complete = nullptr) {
    // 第二轮根因1 fix: 按层判断 scale cache 是否已预加载
    // Bug 2 fix: BF16 路径 gate_scale_bytes == 0，无 scale/mins，只发 3 个 SQE
    bool has_scale = (layout.gate_scale_bytes > 0);
    bool layer_scale_loaded = has_scale && scale_cache_loaded(layer_idx);
    // Bug 1 fix: down_stride > 0 表示 BF16 down_proj TP 切片不连续，用 pread 逐行同步读取
    // 此时 down 不走 io_uring，n_reqs 减 1
    bool down_scattered = (layout.down_stride > 0 && layout.down_rows > 0);
    int n_reqs = layer_scale_loaded ? 3 : (has_scale ? 9 : 3);
    if (down_scattered) n_reqs -= 1;  // down 用 pread，不走 io_uring

    // A6: 创建 PendingRequest 跟踪完成状态
    auto* pending = new PendingRequest{
      layer_idx, tp_part_idx, expert_id, n_reqs, 0, std::move(on_complete)
    };

    // 提交 SQE，user_data 指向 pending
    auto* sqe = io_uring_get_sqe(&ring_);
    io_uring_prep_read(sqe, layout.fd, gate_dst, layout.gate_bytes, layout.gate_offset);
    io_uring_sqe_set_data(sqe, pending);

    sqe = io_uring_get_sqe(&ring_);
    io_uring_prep_read(sqe, layout.fd, up_dst, layout.up_bytes, layout.up_offset);
    io_uring_sqe_set_data(sqe, pending);

    // Bug 5 fix: 先提交 gate/up SQE，再 pread down，让 io_uring 与 pread 并行
    // 原代码 pread 在 io_uring_submit 之前，阻塞期间 io_uring 空闲
    if (!down_scattered) {
      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, down_dst, layout.down_bytes, layout.down_offset);
      io_uring_sqe_set_data(sqe, pending);
    }

    if (!layer_scale_loaded && has_scale) {
      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, gate_scale_dst, layout.gate_scale_bytes, layout.gate_scale_offset);
      io_uring_sqe_set_data(sqe, pending);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, up_scale_dst, layout.up_scale_bytes, layout.up_scale_offset);
      io_uring_sqe_set_data(sqe, pending);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, down_scale_dst, layout.down_scale_bytes, layout.down_scale_offset);
      io_uring_sqe_set_data(sqe, pending);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, gate_mins_dst, layout.gate_mins_bytes, layout.gate_mins_offset);
      io_uring_sqe_set_data(sqe, pending);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, up_mins_dst, layout.up_mins_bytes, layout.up_mins_offset);
      io_uring_sqe_set_data(sqe, pending);

      sqe = io_uring_get_sqe(&ring_);
      io_uring_prep_read(sqe, layout.fd, down_mins_dst, layout.down_mins_bytes, layout.down_mins_offset);
      io_uring_sqe_set_data(sqe, pending);
    }

    // Bug 5 fix: 先 submit 所有 SQE，让 io_uring 开始 DMA
    io_uring_submit(&ring_);

    // Bug 5 fix: down_scattered 的 pread 在 submit 之后执行，与 io_uring 并行
    if (down_scattered) {
      // Bug 1 fix: BF16 down_proj [E,H,I] 行主序，TP 沿 I 切不连续
      // 用 pread 逐行同步读取到 slot buffer 的连续区域
      size_t row_bytes = layout.down_bytes / layout.down_rows;
      char* dst = static_cast<char*>(down_dst);
      off_t src_off = layout.down_offset;
      for (int r = 0; r < layout.down_rows; r++) {
        // Bug 6 fix: pread 加错误处理，失败 abort 避免后续 use-after-free
        // 注意：不能 throw 或 delete pending，因为 gate/up SQE 已提交，
        // CQE 处理时会访问 pending，throw/delete 会导致 use-after-free
        ssize_t ret = pread(layout.fd, dst + r * row_bytes, row_bytes, src_off);
        if (ret < 0) {
          fprintf(stderr,
                  "[MESH] pread down scattered failed at row %d/%d, expert=%d, "
                  "errno=%d (%s), aborting\n",
                  r, layout.down_rows, expert_id, errno, strerror(errno));
          std::abort();
        }
        if (ret < (ssize_t)row_bytes) {
          // 部分读，补齐剩余（理论上 pread 对普通文件应一次读完）
          ssize_t done = ret;
          while (done < (ssize_t)row_bytes) {
            ret = pread(layout.fd, dst + r * row_bytes + done,
                        row_bytes - done, src_off + done);
            if (ret < 0) {
              fprintf(stderr,
                      "[MESH] pread down scattered partial fail, errno=%d (%s), "
                      "aborting\n", errno, strerror(errno));
              std::abort();
            }
            done += ret;
          }
        }
        src_off += layout.down_stride;
      }
    }
  }

  // A6: 处理已完成的 CQE，触发 on_complete 回调
  // 非阻塞：只处理当前已有的 CQE，不等待
  void process_cqes() {
    unsigned head;
    unsigned count = 0;
    io_uring_cqe* cqe;

    io_uring_for_each_cqe(&ring_, head, cqe) {
      auto* pending = static_cast<PendingRequest*>(io_uring_cqe_get_data(cqe));
      if (pending) {
        // 检查读取错误
        if (cqe->res < 0) {
          fprintf(stderr, "[MESH] io_uring read error: expert=%d tp=%d res=%d (%s)\n",
                  pending->expert_id, pending->tp_part_idx, cqe->res, strerror(-cqe->res));
        }
        int completed = pending->completed_reqs.fetch_add(1) + 1;
        if (completed == pending->total_reqs) {
          // 所有 SQE 完成，触发回调
          if (pending->on_complete) {
            pending->on_complete(pending->layer_idx, pending->tp_part_idx, pending->expert_id);
          }
          delete pending;
        }
      }
      count++;
    }
    if (count > 0) {
      io_uring_cq_advance(&ring_, count);
    }
  }

  // 阻塞等待某专家的读请求完成（通过 CQE 计数）
  // A6: 改为处理 CQE 并触发回调
  void wait_expert(int expert_id, int n_reqs) {
    for (int i = 0; i < n_reqs; i++) {
      io_uring_cqe* cqe;
      io_uring_wait_cqe(&ring_, &cqe);
      auto* pending = static_cast<PendingRequest*>(io_uring_cqe_get_data(cqe));
      if (pending) {
        if (cqe->res < 0) {
          fprintf(stderr, "[MESH] io_uring read error in wait_expert: expert=%d res=%d (%s)\n",
                  pending->expert_id, cqe->res, strerror(-cqe->res));
        }
        int completed = pending->completed_reqs.fetch_add(1) + 1;
        if (completed == pending->total_reqs) {
          if (pending->on_complete) {
            pending->on_complete(pending->layer_idx, pending->tp_part_idx, pending->expert_id);
          }
          delete pending;
        }
      }
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
  // A6: 处理 CQE 并触发 on_complete 回调
  void submit_and_wait() {
    io_uring_submit(&ring_);
    // 等待所有 inflight 完成，处理回调
    while (true) {
      unsigned head;
      unsigned count = 0;
      io_uring_cqe* cqe;
      io_uring_for_each_cqe(&ring_, head, cqe) {
        auto* pending = static_cast<PendingRequest*>(io_uring_cqe_get_data(cqe));
        if (pending) {
          if (cqe->res < 0) {
            fprintf(stderr, "[MESH] io_uring read error in submit_and_wait: expert=%d res=%d (%s)\n",
                    pending->expert_id, cqe->res, strerror(-cqe->res));
          }
          int completed = pending->completed_reqs.fetch_add(1) + 1;
          if (completed == pending->total_reqs) {
            if (pending->on_complete) {
              pending->on_complete(pending->layer_idx, pending->tp_part_idx, pending->expert_id);
            }
            delete pending;
          }
        }
        count++;
      }
      if (count == 0) {
        // 没有更多 CQE，检查是否还有 inflight SQE
        unsigned inflight = io_uring_sq_ready(&ring_);
        if (inflight == 0) break;
        // 还有 inflight，等待一个 CQE
        io_uring_cqe* wait_cqe;
        io_uring_wait_cqe(&ring_, &wait_cqe);
        continue;
      }
      io_uring_cq_advance(&ring_, count);
    }
  }

 private:
  io_uring ring_;

  // Scale Cache（A4 fix: [layer][tp] 每层独立）
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
  // A4 fix: [layer][tp] 的 scale cache
  std::vector<std::vector<ScaleCacheTP>> scale_cache_;
  // 第二轮根因1 fix: 改为按层标志，避免全局标志导致 l>=1 层跳过预加载
  std::vector<bool> scale_cache_loaded_per_layer_;

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
    for (auto& layer_cache : scale_cache_) {
      for (auto& c : layer_cache) {
        if (c.gate_scale) numa_free(c.gate_scale, c.gate_scale_total);
        if (c.up_scale) numa_free(c.up_scale, c.up_scale_total);
        if (c.down_scale) numa_free(c.down_scale, c.down_scale_total);
        if (c.gate_mins) numa_free(c.gate_mins, c.gate_mins_total);
        if (c.up_mins) numa_free(c.up_mins, c.up_mins_total);
        if (c.down_mins) numa_free(c.down_mins, c.down_mins_total);
      }
    }
    scale_cache_.clear();
  }
};

}  // namespace mesh
