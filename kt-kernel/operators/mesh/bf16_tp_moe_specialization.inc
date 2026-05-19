// This file is included at top level after AMX_BF16_MOE_TP is defined.
// It owns the concrete TP_MOE specialization needed by BF16 MESH paths.

template <typename K>
class TP_MOE<AMX_BF16_MOE_TP<K>> : public TP_MOE_Common<AMX_BF16_MOE_TP<K>> {
 public:
  using Base = TP_MOE_Common<AMX_BF16_MOE_TP<K>>;
  using Base::Base;

#ifndef _WIN32
  struct Bf16ExpertCacheBlobLayout {
    int fd = -1;
    off_t offset = 0;
    size_t bytes = 0;
    size_t gate_full_bytes = 0;
    size_t up_full_bytes = 0;
    size_t down_full_bytes = 0;
    std::vector<size_t> gate_offsets;
    std::vector<size_t> up_offsets;
  };

  bool build_bf16_expert_cache_blob_layout(int expert_id, Bf16ExpertCacheBlobLayout* layout) const {
    if (layout == nullptr || this->tp_count <= 0) return false;
    layout->gate_offsets.assign(this->tp_count, 0);
    layout->up_offsets.assign(this->tp_count, 0);

    const auto& gate0 =
        mesh::file_slot_at(this->config, 0, this->config.gate_file_slots, "gate.weight", expert_id, true);
    if (gate0.fd < 0 || gate0.size == 0) return false;
    layout->fd = gate0.fd;
    layout->offset = gate0.offset;

    size_t gate_prefix = 0;
    for (int tp = 0; tp < this->tp_count; ++tp) {
      const auto& gate =
          mesh::file_slot_at(this->config, tp, this->config.gate_file_slots, "gate.weight", expert_id, true);
      if (gate.fd != layout->fd || gate.offset != layout->offset + static_cast<off_t>(gate_prefix)) {
        return false;
      }
      layout->gate_offsets[tp] = gate_prefix;
      gate_prefix += gate.size;
    }
    layout->gate_full_bytes = gate_prefix;

    const off_t up_base = layout->offset + static_cast<off_t>(layout->gate_full_bytes);
    size_t up_prefix = 0;
    for (int tp = 0; tp < this->tp_count; ++tp) {
      const auto& up = mesh::file_slot_at(this->config, tp, this->config.up_file_slots, "up.weight", expert_id, true);
      if (up.fd != layout->fd || up.offset != up_base + static_cast<off_t>(up_prefix)) {
        return false;
      }
      layout->up_offsets[tp] = up_prefix;
      up_prefix += up.size;
    }
    layout->up_full_bytes = up_prefix;

    const off_t down_base =
        layout->offset + static_cast<off_t>(layout->gate_full_bytes + layout->up_full_bytes);
    for (int tp = 0; tp < this->tp_count; ++tp) {
      const auto& down =
          mesh::file_slot_at(this->config, tp, this->config.down_file_slots, "down.weight", expert_id, true);
      if (down.fd != layout->fd || down.offset != down_base || down.size == 0) {
        return false;
      }
      if (tp == 0) {
        layout->down_full_bytes = down.size;
      } else if (layout->down_full_bytes != down.size) {
        return false;
      }
    }

    layout->bytes = layout->gate_full_bytes + layout->up_full_bytes + layout->down_full_bytes;
    return layout->bytes > 0;
  }

  struct Bf16PendingCacheBlobPromotion {
    int expert_id = -1;
    Bf16ExpertCacheBlobLayout layout;
    void* blob_raw = nullptr;
    std::vector<uint8_t> pack_needed;
  };

  struct Bf16CacheBlobReaderBatch {
    ktransformers::AsyncExpertReader* reader = nullptr;
    std::vector<ktransformers::AsyncExpertReader::ReadRequest> read_batch;
    std::vector<uint64_t> request_ids;
  };

  size_t bf16_cache_blob_read_chunk_bytes() const {
    constexpr size_t kDefaultChunkBytes = 1u << 20;
    const char* raw = std::getenv("KT_MESH_BF16_BLOB_READ_CHUNK_KB");
    if (raw == nullptr || raw[0] == '\0') {
      return kDefaultChunkBytes;
    }
    char* end = nullptr;
    const long parsed_kib = std::strtol(raw, &end, 10);
    if (end == raw || parsed_kib <= 0) {
      return kDefaultChunkBytes;
    }
    size_t bytes = static_cast<size_t>(parsed_kib) * 1024u;
    bytes = (bytes / 512u) * 512u;
    return bytes == 0 ? kDefaultChunkBytes : bytes;
  }

  size_t bf16_cache_blob_experts_per_wave() const {
    const char* raw = std::getenv("KT_MESH_BF16_BLOB_EXPERTS_PER_WAVE");
    if (raw == nullptr || raw[0] == '\0') {
      return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || parsed <= 0) {
      return 0;
    }
    return static_cast<size_t>(parsed);
  }

  void append_bf16_cache_blob_read_requests(int expert_id,
                                            const Bf16ExpertCacheBlobLayout& layout,
                                            void* blob_raw,
                                            size_t chunk_bytes,
                                            std::vector<ktransformers::AsyncExpertReader::ReadRequest>* read_batch) const {
    if (blob_raw == nullptr || read_batch == nullptr) return;
    chunk_bytes = chunk_bytes == 0 ? (1u << 20) : chunk_bytes;
    size_t cursor = 0;
    while (cursor < layout.bytes) {
      const size_t read_bytes = std::min(chunk_bytes, layout.bytes - cursor);
      read_batch->push_back(ktransformers::AsyncExpertReader::ReadRequest{
          expert_id,
          layout.fd,
          static_cast<char*>(blob_raw) + cursor,
          read_bytes,
          layout.offset + static_cast<off_t>(cursor),
          0});
      cursor += read_bytes;
    }
  }

  bool preload_bf16_cache_blob_experts(const int64_t* expert_ids, int count, int qlen) {
    if (this->config.io_backend != IOBackend::IOURING || this->config.async_reader == nullptr || count <= 0) {
      return false;
    }

    std::vector<int> active_experts;
    std::vector<int> expert_counts;
    mesh::collect_unique_cpu_experts(this->config, expert_ids, count, active_experts, &expert_counts);
    if (active_experts.empty()) {
      return true;
    }

    const bool enable_wave_mode = mesh::bf16_wave_resident_enabled();
    const int cache_capacity = !this->tps.empty() ? this->tps[0]->cache_capacity_for_debug() : 0;
    const int wave_capacity = cache_capacity > 0 ? cache_capacity : (int)active_experts.size();
    if (enable_wave_mode && qlen > 1 && wave_capacity > 0 && (int)active_experts.size() > wave_capacity) {
      return false;
    }

    std::stable_sort(active_experts.begin(), active_experts.end(), [&](int lhs, int rhs) {
      if (expert_counts[lhs] != expert_counts[rhs]) {
        return expert_counts[lhs] > expert_counts[rhs];
      }
      return lhs < rhs;
    });

    const size_t blob_read_chunk_bytes = bf16_cache_blob_read_chunk_bytes();
    const size_t blob_experts_per_wave = bf16_cache_blob_experts_per_wave();
    const size_t reserve_requests_per_expert =
        std::max<size_t>(1, ((6u << 20) + blob_read_chunk_bytes - 1) / blob_read_chunk_bytes);
    std::vector<ktransformers::AsyncExpertReader*> readers;
    readers.reserve(this->config.async_readers.size() + 1);
    for (auto* reader : this->config.async_readers) {
      if (reader != nullptr) {
        readers.push_back(reader);
      }
    }
    if (readers.empty() && this->config.async_reader != nullptr) {
      readers.push_back(this->config.async_reader);
    }
    if (readers.empty()) {
      return false;
    }
    if (mesh::bf16_cache_blob_trace_enabled()) {
      std::fprintf(stderr,
                   "[BF16_CACHE_BLOB] layer=%d qlen=%d active=%zu chunk_kib=%zu experts_per_wave=%zu readers=%zu\n",
                   this->config.layer_idx,
                   qlen,
                   active_experts.size(),
                   blob_read_chunk_bytes / 1024u,
                   blob_experts_per_wave,
                   readers.size());
    }

    std::vector<Bf16PendingCacheBlobPromotion> pending;
    pending.reserve(active_experts.size());
    std::vector<Bf16CacheBlobReaderBatch> reader_batches(readers.size());
    for (size_t i = 0; i < readers.size(); ++i) {
      reader_batches[i].reader = readers[i];
      reader_batches[i].read_batch.reserve(
          ((active_experts.size() + readers.size() - 1) / readers.size()) * reserve_requests_per_expert);
    }

    auto blob_numa_node_for_reader = [&](size_t reader_idx) {
      if (this->config.pool != nullptr && !this->config.pool->config.subpool_numa_map.empty()) {
        const size_t numa_idx =
            std::min(reader_idx, this->config.pool->config.subpool_numa_map.size() - 1);
        return this->config.pool->config.subpool_numa_map[numa_idx];
      }
      return 0;
    };

    auto cancel_pending = [&]() {
      for (auto& item : pending) {
        for (int tp = 0; tp < this->tp_count; ++tp) {
          if (item.pack_needed[tp]) this->tps[tp]->cancel_external_iouring_pack(item.expert_id);
        }
      }
    };
    auto free_pending_blobs = [&]() {
      for (auto& item : pending) {
        if (item.blob_raw != nullptr) {
          numa_free(item.blob_raw, item.layout.bytes);
          item.blob_raw = nullptr;
        }
      }
    };
    auto reset_reader_batches = [&]() {
      for (auto& batch : reader_batches) {
        batch.read_batch.clear();
        batch.request_ids.clear();
      }
    };
    auto flush_pending = [&]() {
      bool has_reads = false;
      for (const auto& batch : reader_batches) {
        has_reads = has_reads || !batch.read_batch.empty();
      }
      if (!has_reads) {
        pending.clear();
        reset_reader_batches();
        return;
      }

      for (auto& batch : reader_batches) {
        if (batch.read_batch.empty()) continue;
        mesh::record_iouring_read_stats(this->config.cache_stats, this->config.enable_cache_stats, batch.read_batch);
        batch.request_ids = batch.reader->submit_reads(batch.read_batch);
        if (batch.request_ids.size() != batch.read_batch.size()) {
          cancel_pending();
          free_pending_blobs();
          reset_reader_batches();
          pending.clear();
          throw std::runtime_error("BF16 MESH cache-blob batched submit returned wrong request count");
        }
      }

      for (auto& batch : reader_batches) {
        if (batch.request_ids.empty()) continue;
        const bool read_ok = batch.reader->wait_for_requests(batch.request_ids, 30000);
        if (!read_ok) {
          cancel_pending();
          free_pending_blobs();
          reset_reader_batches();
          pending.clear();
          throw std::runtime_error("BF16 MESH cache-blob batched io_uring read failed");
        }
      }

      bool pack_failed = false;
      for (auto& item : pending) {
        const auto* gate_base = static_cast<const ggml_bf16_t*>(item.blob_raw);
        const auto* up_base = reinterpret_cast<const ggml_bf16_t*>(
            reinterpret_cast<const char*>(item.blob_raw) + item.layout.gate_full_bytes);
        const auto* down_base = reinterpret_cast<const ggml_bf16_t*>(
            reinterpret_cast<const char*>(item.blob_raw) + item.layout.gate_full_bytes + item.layout.up_full_bytes);

        std::vector<uint8_t> pack_failed_tp(this->tp_count, 0);
        this->config.pool->dispense_backend()->do_numa_job([&, this](int tp) {
          if (!item.pack_needed[tp]) return;
          const auto* gate_raw = reinterpret_cast<const ggml_bf16_t*>(
              reinterpret_cast<const char*>(gate_base) + item.layout.gate_offsets[tp]);
          const auto* up_raw = reinterpret_cast<const ggml_bf16_t*>(
              reinterpret_cast<const char*>(up_base) + item.layout.up_offsets[tp]);
          if (!this->tps[tp]->pack_iouring_expert_from_raw_slices(
                  item.expert_id, gate_raw, up_raw, down_base, false)) {
            pack_failed_tp[tp] = 1;
          }
        });
        for (uint8_t failed : pack_failed_tp) {
          pack_failed = pack_failed || failed != 0;
        }
      }

      free_pending_blobs();
      reset_reader_batches();
      pending.clear();
      if (pack_failed) {
        throw std::runtime_error("BF16 MESH cache-blob pack failed");
      }
    };

    for (int expert_id : active_experts) {
      Bf16PendingCacheBlobPromotion item;
      item.expert_id = expert_id;
      if (!build_bf16_expert_cache_blob_layout(expert_id, &item.layout)) {
        cancel_pending();
        free_pending_blobs();
        return false;
      }
      item.pack_needed.assign(this->tp_count, 0);
      bool any_pack_needed = false;
      for (int tp = 0; tp < this->tp_count; ++tp) {
        if (this->tps[tp]->begin_external_iouring_pack(expert_id, false)) {
          item.pack_needed[tp] = 1;
          any_pack_needed = true;
        }
      }
      if (!any_pack_needed) {
        continue;
      }
      const size_t reader_idx = pending.size() % reader_batches.size();
      item.blob_raw = numa_alloc_onnode(item.layout.bytes, blob_numa_node_for_reader(reader_idx));
      if (item.blob_raw == nullptr) {
        for (int tp = 0; tp < this->tp_count; ++tp) {
          if (item.pack_needed[tp]) this->tps[tp]->cancel_external_iouring_pack(expert_id);
        }
        cancel_pending();
        free_pending_blobs();
        throw std::runtime_error("BF16 MESH cache-blob batch failed to allocate raw buffer");
      }
      append_bf16_cache_blob_read_requests(
          expert_id, item.layout, item.blob_raw, blob_read_chunk_bytes, &reader_batches[reader_idx].read_batch);
      pending.push_back(std::move(item));
      if (blob_experts_per_wave > 0 && pending.size() >= blob_experts_per_wave) {
        flush_pending();
      }
    }

    flush_pending();
    return true;
  }
#endif

  void load_weights() override {
    auto& config = this->config;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const bool use_iouring = config.io_backend == IOBackend::IOURING;
    if (!use_iouring && config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }

    const bool use_per_expert_ptrs = !config.gate_projs.empty();
    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;

    if (use_iouring) {
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        this->tps[i]->set_weight_buffers(tpc.gate_proj, tpc.up_proj, tpc.down_proj);
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });
    } else {
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;

        tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];
        tpc.up_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];
        tpc.down_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];

        const size_t gate_up_weight_src_offset = (size_t)i * tp_weight_elems;
        const size_t down_weight_src_col_offset = (size_t)i * (size_t)tpc.intermediate_size;

        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&](int expert_id) {
              const size_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);

              ggml_bf16_t* gate_dst = (ggml_bf16_t*)tpc.gate_proj + expert_id * tp_weight_elems;
              ggml_bf16_t* up_dst = (ggml_bf16_t*)tpc.up_proj + expert_id * tp_weight_elems;
              ggml_bf16_t* down_dst = (ggml_bf16_t*)tpc.down_proj + expert_id * tp_weight_elems;

              const ggml_bf16_t* gate_src;
              const ggml_bf16_t* up_src;
              const ggml_bf16_t* down_src;

              if (use_per_expert_ptrs) {
                gate_src = (const ggml_bf16_t*)config.gate_projs[0][logical_expert_id] + gate_up_weight_src_offset;
                up_src = (const ggml_bf16_t*)config.up_projs[0][logical_expert_id] + gate_up_weight_src_offset;
                down_src = (const ggml_bf16_t*)config.down_projs[0][logical_expert_id];
              } else {
                gate_src = (const ggml_bf16_t*)config.gate_proj + logical_expert_id * full_weight_elems +
                           gate_up_weight_src_offset;
                up_src =
                    (const ggml_bf16_t*)config.up_proj + logical_expert_id * full_weight_elems + gate_up_weight_src_offset;
                down_src = (const ggml_bf16_t*)config.down_proj + logical_expert_id * full_weight_elems;
              }

              std::memcpy(gate_dst, gate_src, tp_weight_elems * sizeof(ggml_bf16_t));
              std::memcpy(up_dst, up_src, tp_weight_elems * sizeof(ggml_bf16_t));

              for (int row = 0; row < config.hidden_size; row++) {
                const size_t src_row_offset =
                    (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
                const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
                std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset,
                            (size_t)tpc.intermediate_size * sizeof(ggml_bf16_t));
              }
            },
            nullptr);
      });

      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        this->tps[i]->set_weight_buffers(tpc.gate_proj, tpc.up_proj, tpc.down_proj);
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });

      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        delete[] (ggml_bf16_t*)tpc.gate_proj;
        delete[] (ggml_bf16_t*)tpc.up_proj;
        delete[] (ggml_bf16_t*)tpc.down_proj;
      });
    }

    this->weights_loaded = true;
  }

  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental) {
    if (this->weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Not Loaded");
    }
#ifndef _WIN32
    const int qlen = *qlen_ptr;
    (void)preload_bf16_cache_blob_experts(expert_ids, qlen * k, qlen);
#endif
    Base::forward(qlen_ptr, k, expert_ids, weights, input, output, incremental);
  }

  void forward_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output,
                       bool incremental) {
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output,
            incremental);
  }

  void forward_binding_with_scores(intptr_t qlen_ptr,
                                   int k,
                                   intptr_t expert_ids,
                                   intptr_t weights,
                                   intptr_t input,
                                   intptr_t output,
                                   bool incremental,
                                   intptr_t router_scores,
                                   int score_rows,
                                   int score_cols,
                                   int score_transform) {
    this->observe_router_scores_binding(router_scores, score_rows, score_cols, score_transform);
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output,
            incremental);
  }

  /**
   * @brief Write weights to GPU buffer for all TP parts
   *
   * BF16 version - no scales needed, scale_ptrs parameters are kept for interface compatibility.
   */
  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->weights_loaded == false) {
      throw std::runtime_error("Not Loaded");
    }
    if (this->tps.empty()) {
      throw std::runtime_error("No TP parts initialized");
    }
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w2_weight_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Weight pointer arrays size must match gpu_tp_count");
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }

#include "../mesh/bf16_tp_moe_runtime.inc"
};
