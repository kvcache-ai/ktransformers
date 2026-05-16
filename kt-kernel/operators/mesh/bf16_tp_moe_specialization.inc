// This file is included at top level after AMX_BF16_MOE_TP is defined.
// It owns the concrete TP_MOE specialization needed by BF16 MESH paths.

template <typename K>
class TP_MOE<AMX_BF16_MOE_TP<K>> : public TP_MOE_Common<AMX_BF16_MOE_TP<K>> {
 public:
  using Base = TP_MOE_Common<AMX_BF16_MOE_TP<K>>;
  using Base::Base;

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

    if (config.use_mmap || use_iouring) {
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
