// This file is included at top level after AMX_MOE_TP is defined.
// It owns the concrete TP_MOE specialization needed by MESH resident-cache paths.

template <typename K>
class TP_MOE<AMX_MOE_TP<K>> : public TP_MOE_Common<AMX_MOE_TP<K>> {
 public:
  using Base = TP_MOE_Common<AMX_MOE_TP<K>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    if (config.io_backend == IOBackend::IOURING) {
      printf("TP Load from io_uring file slots\n");
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });
      this->weights_loaded = true;
    } else if (config.gate_projs.empty() == false) {
      printf("TP Load from loader\n");
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        this->tps[i]->set_weight_buffers(tpc.gate_proj, tpc.up_proj, tpc.down_proj);
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });
      this->weights_loaded = true;
    } else if (config.gate_proj != nullptr) {
      printf("From BF16\n");
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = this->tp_configs[i];
        size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;
        tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.up_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.down_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        if (tpc.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
                memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                memcpy((ggml_bf16_t*)tpc.up_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.up_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((ggml_bf16_t*)tpc.down_proj + expert_id * tpc.hidden_size * tpc.intermediate_size +
                             col * tpc.intermediate_size,
                         (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                             col * config.intermediate_size + i * tpc.intermediate_size,
                         sizeof(ggml_bf16_t) * tpc.intermediate_size);
                }
              },
              nullptr);
        }
      }

      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        this->tps[i]->set_weight_buffers(tpc.gate_proj, tpc.up_proj, tpc.down_proj);
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });

      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = this->tp_configs[i];
        delete[] (ggml_bf16_t*)(tpc.gate_proj);
        delete[] (ggml_bf16_t*)(tpc.up_proj);
        delete[] (ggml_bf16_t*)(tpc.down_proj);
      }

      this->weights_loaded = true;
    } else if (config.path != "") {
      printf("TP Load from file %s\n", config.path.c_str());
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });
      this->weights_loaded = true;
    } else {
      throw std::runtime_error("no weight source");
    }
  }

#include "../mesh/amx_tp_moe_mesh_methods.inc"

#include "../mesh/amx_tp_moe_merge_results.inc"
};
