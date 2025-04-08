#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#define FMT_HEADER_ONLY
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"

#include "scheduler.h"
#include <optional>

#include "arithmetic.hpp"
#include "atomic_ptr_with_flags.hpp"
#include "easy_format.hpp"
#include "metrics.h"
#include "mpsc.hpp"
#include "timer.hpp"
#include <atomic>
#include <cassert>
#include <future>
#include <memory>
#include <queue>

#include "kvc2.h"

using json = nlohmann::json;

namespace scheduler {

void Settings::auto_derive() {
  gpu_device_count = gpu_device_id.size();
  if (torch::cuda::is_available()) {
    size_t gpu_count = torch::cuda::device_count();
    SPDLOG_INFO("Number of available GPUs: {}, want {}", gpu_count,
                gpu_device_count);
    if (gpu_count < gpu_device_count) {
      SPDLOG_ERROR("Not enough GPUs available.");
      exit(0);
    }
    for (size_t i = 0; i < gpu_device_count; i++) {
      devices.push_back(torch::Device(torch::kCUDA, gpu_device_id[i]));
    }
  } else {
    SPDLOG_ERROR("CUDA is not available on this system.");
    exit(0);
  }

  if (model_settings.num_k_heads % gpu_device_count != 0) {
    SPDLOG_ERROR("num_k_heads {} is not divisible by gpu_device_count {}",
                 model_settings.num_k_heads, gpu_device_count);
    assert(false);
  }

  size_t gpu_memory_available = gpu_memory_size * memory_utilization_percentage;
  if (gpu_memory_available * gpu_device_count <
      model_settings.params_nbytes()) {
    SPDLOG_ERROR("GPU memory size {}G is smaller than {}G",
                 gpu_memory_available * gpu_device_count / 1e9,
                 model_settings.params_nbytes() / 1e9);
    assert(false);
  }

  assert(model_settings.k_head_dim % model_settings.num_k_heads == 0);
  size_t head_per_gpu = model_settings.num_k_heads / gpu_device_count;
  size_t gpu_memory_for_kv_cache =
      gpu_memory_available /*- model_settings.params_nbytes() /
                              gpu_device_count*/
      ;
  SPDLOG_INFO(
      "Each GPU Total: {}MiB, Model Params: {}MiB, KVCache: {}MiB, Left: {}MiB",
      gpu_memory_size / (1 << 20),
      model_settings.params_nbytes() / gpu_device_count / (1 << 20),
      gpu_memory_for_kv_cache / (1 << 20),
      (gpu_memory_size - gpu_memory_available) / (1 << 20));
  size_t kv_cache_on_cnt = (size_t)(k_cache_on) + (size_t)(v_cache_on);
  size_t max_total_kvcache_pages =
      gpu_memory_for_kv_cache /
      (kv_cache_on_cnt * head_per_gpu * model_settings.k_head_dim *
       model_settings.bytes_per_kv_cache_element * page_size *
       model_settings.layer_count);
  if (total_kvcache_pages.has_value()) {
    if (total_kvcache_pages.value() > max_total_kvcache_pages) {
      SPDLOG_ERROR(
          "total_kvcache_pages {} is larger than max_total_kvcache_pages {}",
          total_kvcache_pages.value(), max_total_kvcache_pages);
      assert(false);
    }
  } else {
    total_kvcache_pages = max_total_kvcache_pages;
    SPDLOG_INFO("total_kvcache_pages is auto derived as {}",
                max_total_kvcache_pages);
  }

  if (page_size % 256 != 0) {
    SPDLOG_ERROR("page_size {} is not divisible by 256", page_size);
    assert(false);
  }
  if (page_size < 256) {
    SPDLOG_ERROR("page_size {} is smaller than 256", page_size);
    assert(false);
  }
}

std::string BatchQueryTodo::debug() {
  std::string re = "BatchQueryTodo: ";
  re += "QueryIDs: ";
  for (auto &id : query_ids) {
    re += std::to_string(id) + " ";
  }
  return re;
}

bool BatchQueryTodo::empty() {
  return prefill_mini_batches.empty() && decode_mini_batches.empty();
}

struct QueryMaintainer;

struct Query {
  QueryID id;
  torch::Tensor query_token;
  TokenLength prompt_length;
  TokenLength no_kvcache_from;
  TokenLength estimated_length;

  SampleOptions sample_options;

  UserID user_id;
  std::optional<int> SLO_TTFT_ms;
  std::optional<int> SLO_TBT_ms;

  std::vector<std::vector<int>> stop_criteria;

  // status
  // Query status changed by this order
  enum Status { Received, Preparing, Ready, Prefill, Decode, Done };
  Status plan_status = Received;
  TokenLength active_position; // the position where no kvcache now
  TokenLength plan_position;   // the position where no kvcache now, in plan
  size_t prepare_try_count = 0;
  std::shared_ptr<kvc2::DoubleCacheHandleInterface> kvc2_handle = nullptr;

  // derived from kvc2_handle
  torch::Tensor block_index; // block indexes

  struct QueryContext {
    ModelName model_name;
    QuantType quant_type;
    kvc2::KVC2Interface *kvc2_interface;
    QueryMaintainer *query_maintainer;
    Metrics *met;
  } ctx;

  void after_load(bool ok);

  void to_status(Status to);

  void export_metrics() {
    ctx.met->query_count(status_to_string(plan_status))->Increment(1);
  }

  Query(QueryID id, QueryAdd query_add, QueryContext context)
      : id(id), prompt_length(query_add.query_length), no_kvcache_from(0),
        estimated_length(query_add.estimated_length),
        sample_options(query_add.sample_options), user_id(query_add.user_id),
        SLO_TTFT_ms(query_add.SLO_TTFT_ms), SLO_TBT_ms(query_add.SLO_TBT_ms),
        stop_criteria(query_add.stop_criteria), ctx(context) {
    std::vector<int64_t> shape = {int64_t(query_add.estimated_length)};
    query_token =
        torch::zeros(shape, torch::TensorOptions().dtype(torch::kInt32));
    assert(query_token.is_contiguous());
    if (query_token.is_contiguous() == false) {
      SPDLOG_ERROR("Query Token must be contiguous!");
      exit(1);
    }

    memcpy(query_token.data_ptr(), query_add.query_token.data(),
           query_add.query_length * sizeof(Token));

    no_kvcache_from = 0; // maybe match prefix later
    export_metrics();
  }

  Token &token_at(size_t idx) {
    return reinterpret_cast<Token *>(query_token.data_ptr())[idx];
  }

  void absorb_update(const QueryUpdate &update) {
    SPDLOG_DEBUG("{}", update.debug());
    active_position = update.active_position;
    kvc2_handle->append_tokens(&token_at(0),
                               active_position); // active_position is length -1
    if (update.is_prefill) {
      if (active_position == prompt_length) {
        token_at(active_position) = update.generated_token;
        ctx.met->generated_tokens->Increment(1);
      }
    } else {
      token_at(active_position) = update.generated_token;
      ctx.met->generated_tokens->Increment(1);
    }

    if (update.decode_done || active_position == estimated_length - 1) {
      to_status(Done);
    }
  }

  void absorb_prefill_task(const PrefillTask &task) {
    auto &[id, start, length] = task;
    this->plan_position = start + length;
    if (this->plan_position == prompt_length) {
      to_status(Decode);
    }
  }

  void absorb_decode_task([[maybe_unused]] const QueryID &task) {
    this->plan_position += 1;
  }

  PrefillTask get_prefill_task(size_t prefill_length) {
    if (prefill_length + plan_position > prompt_length) {
      prefill_length = prompt_length - plan_position;
    }
    return {id, plan_position, prefill_length};
  }

  static std::string status_to_string(Status status) {
    switch (status) {
    case Received:
      return "Received";
    case Preparing:
      return "Preparing";
    case Ready:
      return "Ready";
    case Prefill:
      return "Prefill";
    case Decode:
      return "Decode";
    case Done:
      return "Done";
    }
    assert(false);
  }

  void debug() {
    std::string status_string = status_to_string(plan_status);

    SPDLOG_DEBUG("Query {}, prompt_length {}, estimated_length {}, plan status "
                 "{}, plan position {} "
                 "active position {}",
                 id, prompt_length, estimated_length, status_string,
                 plan_position, active_position);
  }
};

std::string QueryUpdate::debug() const {
  return fmt::format("Query {}, ok {}, is_prefill {}, done {}, active_position "
                     "{}, gen token {}",
                     id, ok, is_prefill, decode_done, active_position,
                     generated_token);
}

using Q = std::shared_ptr<Query>;

struct KVC2_Maintainer {
  Settings settings;

  std::vector<torch::Tensor> k_cache;
  std::vector<torch::Tensor> v_cache;
  std::shared_ptr<kvc2::KVC2Interface> kvc2_interface;

  KVC2_Maintainer(Settings settings) : settings(settings) {
    // SPDLOG_WARN("Creating KVC2 Instance {}", settings.kvc2_root_path);
    assert(settings.kvc2_root_path.size() > 0);

    // SPDLOG_WARN("Sizeof KVC2Config {} upper", sizeof(kvc2::KVC2Config));
    kvc2::GPUPageCacheConfig gpu_cache_config{
        .gpu_only = settings.gpu_only,
        .gpu_devices_id = settings.gpu_device_id,
        .layer_count = settings.model_settings.layer_count,
        .total_kvcache_pages = settings.total_kvcache_pages.value(),
        .num_token_per_page = settings.page_size,
        .num_k_heads = settings.model_settings.num_k_heads,
        .k_head_dim = settings.use_self_defined_head_dim
                          ? settings.self_defined_head_dim
                          : settings.model_settings.k_head_dim,
        .full_kv_cache_on_each_gpu = settings.full_kv_cache_on_each_gpu,
        .k_cache_on = settings.k_cache_on,
        .v_cache_on = settings.v_cache_on,
        .tensor_type = torch::kBFloat16,
    };

    auto model_configs_path =
        std::filesystem::path(settings.kvc2_config_path) / "model_configs.json";
    load_model_configs(model_configs_path);
    auto my_model_config = ModelConfig();
    my_model_config.load_from(
        std::filesystem::path(settings.model_settings.model_path) /
        "config.json");
    model_configs[settings.model_name] = my_model_config;
    dump_model_configs(model_configs_path);

    kvc2::KVC2Config kvc2_config = {
        .k_cache_on = settings.k_cache_on,
        .v_cache_on = settings.v_cache_on,
        .gpu_only = settings.gpu_only,
        .load_from_disk = settings.load_from_disk,
        .save_to_disk = settings.save_to_disk,
        .path = settings.kvc2_root_path,
        .config_path = settings.kvc2_config_path,
        .num_token_per_page = settings.page_size,
        .memory_pool_size = size_t(settings.memory_pool_size_GB * 1e9),
        .evict_count = settings.evict_count,
        .gpu_cache_config = gpu_cache_config,
        .metrics_port = settings.kvc2_metrics_port,
    };
    kvc2_interface = kvc2::create_kvc2(kvc2_config);
    if (settings.load_from_disk)
      kvc2_interface->load();

    SPDLOG_DEBUG("KVC2 created ok");

    auto [k_cache, v_cache] = kvc2_interface->get_kvcache();
    this->k_cache = k_cache;
    this->v_cache = v_cache;
  }
};

using EventAddQuery = std::pair<QueryAdd, std::promise<QueryID> *>;
using EventUpdateQuery = BatchQueryUpdate;
using EventTakenBatch = std::shared_ptr<BatchQueryTodo>;
struct EventPrepare {
  QueryID query_id;
  bool first_try;
};
struct EventPrepared {
  QueryID query_id;
  bool ok;
};

struct EventQueryStatus {
  QueryID query_id;
  Query::Status now_status;
};
struct EventSchedule {};

using Event =
    std::variant<EventAddQuery, EventUpdateQuery, EventTakenBatch, EventPrepare,
                 EventPrepared, EventQueryStatus, EventSchedule>;

template <typename T> std::string event_name(const T &event);

template <> std::string event_name(const EventAddQuery &) {
  return "EventAddQuery";
}

template <> std::string event_name(const EventUpdateQuery &) {
  return "EventUpdateQuery";
}

template <> std::string event_name(const EventTakenBatch &) {
  return "EventTakenBatch";
}
template <> std::string event_name(const EventPrepare &) {
  return "EventPrepare";
}

template <> std::string event_name(const EventPrepared &) {
  return "EventPrepared";
}

template <> std::string event_name(const EventQueryStatus &) {
  return "EventQueryStatus";
}

template <> std::string event_name(const EventSchedule &) {
  return "EventSchedule";
}

// 用 std::visit 实现对 variant 的 event_name
std::string event_name(const Event &event) {
  return std::visit([](const auto &e) { return event_name(e); }, event);
}

static_assert(std::is_copy_constructible<Event>::value);
static_assert(std::is_move_constructible<Event>::value);

struct QueryMaintainer : public Scheduler {
  // only get access by event loop
  Settings settings;
  QueryID query_id_counter = NoQueryID + 1;
  std::map<QueryID, Q> query_map;
  std::shared_ptr<KVC2_Maintainer> kvc2_maintainer;

  std::shared_ptr<Metrics> met;
  // multi-thread visit
  std::atomic_bool stop_flag = false;
  // TODO consider correctness of event loop
  MPSCQueueConsumerLock<Event> event_loop_queue;

  // std::binary_semaphore batch_ready{0};
  AtomicPtrWithFlag<BatchQueryTodo> next_batch;

  QueryMaintainer() = default;

  void gen_batch_query_todo(BatchQueryTodo *re, const std::set<Q> &queries) {
    std::vector<std::vector<QueryID>> d_batch(2);
    size_t last_decode_batch = 0;
    size_t prefill_num = 0;
    size_t decode_num = 0;
    size_t preill_length = 0;
    for (auto &q : queries) {
      if (q->plan_status == Query::Prefill) {
        prefill_num += 1;
      }
      if (q->plan_status == Query::Decode) {
        decode_num += 1;
      }
    }
    if (prefill_num >= 2 ||
        (prefill_num == 1 && settings.max_batch_size - 2 < decode_num)) {
      preill_length = settings.recommended_chunk_prefill_token_count;
    } else {
      preill_length = settings.recommended_chunk_prefill_token_count * 2;
    }
    for (auto &q : queries) {
      re->query_ids.push_back(q->id);
      re->query_tokens.push_back(q->query_token);
      re->query_lengths.push_back(q->prompt_length);
      if (q->plan_status == Query::Prefill) {
        re->prefill_mini_batches.push_back(q->get_prefill_task(preill_length));
        assert(re->prefill_mini_batches.size() <= 2);
      }
      if (q->plan_status == Query::Decode) {
        d_batch[last_decode_batch].push_back(q->id);
        // last_decode_batch = 1 - last_decode_batch;
        if (d_batch[last_decode_batch].size() == settings.max_batch_size - 1) {
          last_decode_batch += 1;
          assert(last_decode_batch < 2);
        }
      }
      re->block_indexes.push_back(q->block_index);
      re->sample_options.push_back(q->sample_options);
      re->stop_criteria.push_back(q->stop_criteria);
    }

    re->attn_masks = std::nullopt;
    re->rope_ranges = std::nullopt;

    for (auto &b : d_batch) {
      if (b.empty())
        continue;
      re->decode_mini_batches.push_back(b);
    }

    met->batch_count("Generated")->Increment(1);
  }

  // Interface

  void init(Settings settings) override {
    SPDLOG_INFO("\nScheduler Settings:\n"
                "  model_name: {}\n"
                "  quant_type: {}\n"
                "    model_path: {}\n"
                "    params_count: {}\n"
                "    layer_count: {}\n"
                "    num_k_heads: {}\n"
                "    k_head_dim: {}\n"
                "    bytes_per_params: {}\n"
                "    bytes_per_kv_cache_element: {}\n"
                "  page_size: {}\n"
                "  gpu_device_id: {}\n"
                "  gpu_memory_size: {}\n"
                "  memory_utilization_percentage: {}\n"
                "  max_batch_size: {}\n"
                "  recommended_chunk_prefill_token_count: {}\n"
                "  sched_metrics_port: {}\n"
                "  kvc2_config_path: {}\n"
                "  kvc2_root_path: {}\n"
                "  memory_pool_size_GB: {}\n"
                "  evict_count: {}\n"
                "  kvc2_metrics_port: {}\n"
                "  load_from_disk: {}\n"
                "  save_to_disk: {}\n"
                "  strategy_name: {}\n"
                "  gpu_device_count: {}\n",
                settings.model_name, settings.quant_type,
                settings.model_settings.model_path,
                settings.model_settings.params_count,
                settings.model_settings.layer_count,
                settings.model_settings.num_k_heads,
                settings.model_settings.k_head_dim,
                settings.model_settings.bytes_per_params,
                settings.model_settings.bytes_per_kv_cache_element,

                settings.page_size, format_vector(settings.gpu_device_id),
                readable_number(settings.gpu_memory_size),
                settings.memory_utilization_percentage, settings.max_batch_size,
                settings.recommended_chunk_prefill_token_count,
                settings.sched_metrics_port, settings.kvc2_config_path,
                settings.kvc2_root_path, settings.memory_pool_size_GB,
                settings.evict_count, settings.kvc2_metrics_port,
                settings.load_from_disk, settings.save_to_disk,
                settings.strategy_name, settings.gpu_device_count);

    this->settings = settings;
    kvc2_maintainer =
        std::shared_ptr<KVC2_Maintainer>(new KVC2_Maintainer(settings));
    MetricsConfig met_conf = {
        .endpoint = "0.0.0.0:" + std::to_string(settings.sched_metrics_port),
        .model_name = settings.model_name,
        .gpu_count = settings.gpu_device_count,
    };

    SPDLOG_INFO("Creating scheduler metrics exporter on {}", met_conf.endpoint);
    met = std::make_shared<Metrics>(met_conf);
    met->fn_every_sec = [](Metrics *met) {
      auto generated_tokens = met->generated_tokens->Collect().counter.value;
      SPDLOG_INFO("Last Sec Generated Tokens {}", generated_tokens);
    };
  }
  Query::QueryContext get_query_context() {
    return Query::QueryContext{
        .model_name = settings.model_name,
        .quant_type = settings.quant_type,
        .kvc2_interface = kvc2_maintainer->kvc2_interface.get(),
        .query_maintainer = this,
        .met = met.get(),
    };
  }

  QueryID add_query(QueryAdd query_add) override {
    std::promise<QueryID> p;
    event_loop_queue.enqueue(EventAddQuery(query_add, &p));
    return p.get_future().get();
  }

  void cancel_query(QueryID id) override {
    SPDLOG_INFO("Cancel Query");
    SPDLOG_INFO("sched:{} Cancel Query", fmt::ptr(this));
    auto it = query_map.find(id);
    if (it == query_map.end()) {
      SPDLOG_ERROR("Query {} is not found", id);
      return;
    }
    query_map.erase(it);
  }

  // Here this function update last batch results and get the next batch
  // in most cases, the batch is ready,
  // if not, busy wait to get it
  std::shared_ptr<BatchQueryTodo>
  update_last_batch(BatchQueryUpdate updates) override {
    event_loop_queue.enqueue(updates);

    // Busy Wait
    while (true) {
      auto [ptr, is_new] = next_batch.touch_load();
      // SPDLOG_INFO("ptr {} is_new {}", fmt::ptr(ptr), is_new);
      if (is_new) {
        // SPDLOG_DEBUG("New Batch {}", fmt::ptr(ptr));
        auto re = std::shared_ptr<BatchQueryTodo>(ptr);
        event_loop_queue.enqueue(re);
        return re;
      } else {
        // // here to busy wait
        // SPDLOG_INFO("Not New");
        // using namespace std::chrono_literals;
        // std::this_thread::sleep_for(1s);
      }
    }
  }

  InferenceContext get_inference_context() override {
    InferenceContext re;
    re.k_cache = kvc2_maintainer->k_cache;
    re.v_cache = kvc2_maintainer->v_cache;
    // kvc2_maintainer->k_cache[0][0][0][0][0][0] = 42; // test whether we pass
    // this to inference loop
    return re;
  }

  virtual void strategy_add_query(Q new_query) = 0;
  virtual void strategy_update_query(const EventUpdateQuery &update) = 0;
  virtual void strategy_taken_batch(const EventTakenBatch &batch) = 0;
  virtual void strategy_prepare(const EventPrepare &prepare) = 0;
  virtual void strategy_prepared(const EventPrepared &prepared) = 0;
  virtual void strategy_query_status(const EventQueryStatus &query_status) = 0;
  virtual void strategy_schedule(const EventSchedule &event,
                                 BatchQueryTodo *new_batch) = 0;

  void tackle_event(EventAddQuery &event) {
    auto &query_add = event.first;
    QueryID id = query_id_counter;
    event.second->set_value(id);
    query_id_counter += 1;
    Q new_query(new Query(id, query_add, get_query_context()));
    query_map[id] = new_query;
    SPDLOG_INFO("New Query {} is added", id);
    strategy_add_query(new_query);
  }

  void tackle_event(const EventUpdateQuery &update) {
    // SPDLOG_INFO("Tackle Update Query");
    for (auto &u : update) {
      if (u.ok == false) {
        SPDLOG_ERROR("Query {} is not exectued OK", u.id);
        exit(1);
      }
      auto q = query_map[u.id];
      if (q->plan_status == Query::Status::Prefill ||
          q->plan_status == Query::Status::Decode) {
        q->absorb_update(u);
      } else {
        SPDLOG_DEBUG(
            "Query {} is not in Prefill or Decode status, do not update it",
            u.id);
      }
    }
    strategy_update_query(update);
  }

  void tackle_event(const EventTakenBatch &batch) {
    met->batch_count("Taken")->Increment(1);
    for (auto &task : batch->prefill_mini_batches) {
      auto [id, s, l] = task;
      if (l == 0)
        continue;
      query_map.at(id)->absorb_prefill_task(task);
    }
    for (auto &mini_batch : batch->decode_mini_batches) {
      for (auto &id : mini_batch) {
        query_map.at(id)->absorb_decode_task(id);
      }
    }

    strategy_taken_batch(batch);
  }

  void tackle_event(const EventPrepare &event) { strategy_prepare(event); }
  void tackle_event(const EventPrepared &event) { strategy_prepared(event); }
  void tackle_event(const EventQueryStatus &event) {
    strategy_query_status(event);
  }

  void tackle_event(const EventSchedule &event) {
    // SPDLOG_INFO("Tackle Schedule Event");

    HistogramTimerWrapper t(met->schedule_time);

    BatchQueryTodo *new_batch = new BatchQueryTodo;
    strategy_schedule(event, new_batch);
    // if (new_batch->query_ids.empty()) {
    //   SPDLOG_INFO("Nothing todo");
    //   delete new_batch;
    //   return;
    // }
    auto [old_batch, flag] = next_batch.exchange(new_batch, true);
    if (new_batch->empty() == false) {
      SPDLOG_DEBUG("set new batch {}", fmt::ptr(new_batch));
    }
    if (flag) {
      SPDLOG_INFO("Batch {} is not consumed", fmt::ptr(old_batch));
      delete old_batch;
    }
  }

  void run() override {
    std::thread([this]() {
      SPDLOG_WARN("Starting Scheduler Event Loop");
      while (stop_flag.load() == false) {
        auto event = event_loop_queue.dequeue();
        met->event_count(event_name(event))->Increment(1);
        std::visit(
            [this](auto event) {
              using T = std::decay_t<decltype(event)>;
              // SPDLOG_INFO("Event Loop: {}", typeid(T).name());
              if constexpr (std::is_same_v<T, EventAddQuery>) {
                tackle_event(event);
              } else if constexpr (std::is_same_v<T, EventUpdateQuery>) {
                tackle_event(event);
              } else if constexpr (std::is_same_v<T, EventTakenBatch>) {
                tackle_event(event);
              } else if constexpr (std::is_same_v<T, EventPrepare>) {
                tackle_event(event);
              } else if constexpr (std::is_same_v<T, EventPrepared>) {
                tackle_event(event);
              } else if constexpr (std::is_same_v<T, EventQueryStatus>) {
                tackle_event(event);
              } else if constexpr (std::is_same_v<T, EventSchedule>) {
                tackle_event(event);
              } else {
                SPDLOG_ERROR("Should not be here");
                assert(false);
              }
            },
            event);
        if (event_loop_queue.size() == 0 &&
            std::holds_alternative<EventSchedule>(event) == false) {
          // if this is not a schedule event, we need to schedule one
          event_loop_queue.enqueue(EventSchedule());
        }
      }
    }).detach();
  }

  void stop() override { stop_flag.store(true); }

  ~QueryMaintainer() {
    kvc2_maintainer->kvc2_interface->save();
    stop();
  }
};

void Query::to_status(Status to) {
  SPDLOG_DEBUG("Calling to status query {}, to {}", id, status_to_string(to));
  switch (to) {
  case Received:
    assert(false);
    break;
  case Preparing:
    SPDLOG_INFO("Preparing Query {} {}", id,
                prepare_try_count > 0
                    ? (std::to_string(prepare_try_count) + " Try")
                    : "");
    prepare_try_count += 1;

    ctx.kvc2_interface->lookup_to_gpu_async(
        ctx.model_name, ctx.quant_type,
        static_cast<kvc2::Token *>(query_token.data_ptr()), prompt_length,
        estimated_length,
        [this](std::shared_ptr<kvc2::DoubleCacheHandleInterface> handle) {
          if (handle == nullptr) {
            SPDLOG_INFO("Get handle from kvc2 Failed.");
            this->after_load(false);
          } else {
            SPDLOG_INFO("Get handle from kvc2 Success.");
            this->kvc2_handle = handle;
            this->to_status(Ready);
            this->after_load(true);
          }
        });
    break;
  case Ready:
    SPDLOG_INFO("Ready Query {}", id);
    break;
  case Prefill:
    SPDLOG_INFO("Prefilling Query {}", id);
    // assert(plan_status == Received);
    plan_position = kvc2_handle->matched_length();

    if (prompt_length - plan_position == 0) {
      assert(prompt_length > 0);
      plan_position -= 1;
    }
    break;
  case Decode:
    SPDLOG_INFO("Decoding Query {}", id);
    // assert(plan_status == Prefill);
    break;
  case Done:
    SPDLOG_INFO("Finish Query {}", id);
    kvc2_handle = nullptr;
    ctx.query_maintainer->event_loop_queue.enqueue(EventQueryStatus{
        .query_id = id,
        .now_status = to,
    });
    // assert(plan_status == Decode);
    break;
  }
  plan_status = to;
  export_metrics();
}

void Query::after_load(bool ok) {
  if (ok) {
    size_t page_count =
        div_up(estimated_length, ctx.query_maintainer->settings.page_size);
    std::vector<int64_t> shape;
    shape.push_back(page_count);
    block_index =
        torch::zeros(shape, torch::TensorOptions().dtype(torch::kInt32))
            .contiguous();
    auto ptr = reinterpret_cast<int32_t *>(block_index.data_ptr());
    auto vec_idx = kvc2_handle->get_gpu_block_idx();
    for (size_t i = 0; i < vec_idx.size(); i++) {
      ptr[i] = vec_idx[i];
    }
    no_kvcache_from = kvc2_handle->matched_length();
  }
  if (ok) {
    ctx.query_maintainer->event_loop_queue.enqueue(EventPrepared{
        .query_id = id,
        .ok = ok,
    });
  } else {
    ctx.query_maintainer->event_loop_queue.enqueue(EventPrepare{
        .query_id = id,
        .first_try = false,
    });
  }
}

struct FCFS_single_prefill : public QueryMaintainer {
  std::queue<Q> queue;
  std::queue<Q> ready_queue;

  bool has_query_preparing = false;
  std::optional<EventPrepare> wait_done_prepare = std::nullopt;

  std::set<Q> active_query; // on going queries for LLMs

  // interface all these are executed in a single thread
  void strategy_add_query(Q new_query) override {
    queue.push(new_query);
    if (has_query_preparing == false) {
      has_query_preparing = true;
      auto next_q = queue.front();
      queue.pop();
      event_loop_queue.enqueue(EventPrepare{next_q->id, true});
    }
  }

  void strategy_update_query(const EventUpdateQuery &update) override {
    for (auto u : update) {
      auto &q = query_map[u.id];
      if (q->plan_status == Query::Done) {
        active_query.erase(q);
      }
    }
  }

  void strategy_taken_batch(const EventTakenBatch &batch) override {
    for (auto &q : batch->query_ids) {
      if (query_map[q]->plan_status != Query::Done) {
        active_query.insert(query_map[q]);
      }
    }
  }

  void strategy_prepare(const EventPrepare &prepare) override {
    if (prepare.first_try) {
      auto &q = query_map[prepare.query_id];
      q->to_status(Query::Preparing);
    } else {
      assert(wait_done_prepare.has_value() == false);
      wait_done_prepare = prepare;
      wait_done_prepare->first_try = true;
    }
  }

  void strategy_prepared(const EventPrepared &prepared) override {
    assert(prepared.ok);
    ready_queue.push(query_map[prepared.query_id]);
    if (queue.empty() == false) {
      auto next_q_prepare = queue.front();
      queue.pop();
      event_loop_queue.enqueue(EventPrepare{next_q_prepare->id, true});

    } else {
      has_query_preparing = false;
    }
  }

  void strategy_query_status(const EventQueryStatus &query_status) override {
    if (query_status.now_status == Query::Done) {
      if (wait_done_prepare.has_value()) {
        event_loop_queue.enqueue(wait_done_prepare.value());
        wait_done_prepare = std::nullopt;
      }
    }
  }

  void strategy_schedule([[maybe_unused]] const EventSchedule &event,
                         BatchQueryTodo *new_batch) override {
    bool have_prefill = false;
    for (auto &q : active_query) {
      if (q->plan_status == Query::Prefill) {
        have_prefill = true;
      }
    }

    if (have_prefill == false && ready_queue.empty() == false &&
        active_query.size() < settings.max_batch_size) {
      auto &next_q = ready_queue.front();
      ready_queue.pop();

      SPDLOG_INFO("Active query {}", next_q->id);
      active_query.insert(next_q);
      next_q->to_status(Query::Prefill);
    }
    if (active_query.empty() == false)
      SPDLOG_INFO("Active Query Size {}", active_query.size());
    for (auto &q : active_query) {
      q->debug();
    }
    gen_batch_query_todo(new_batch, active_query);
  }
};

struct FCFS : public FCFS_single_prefill {
  void strategy_schedule([[maybe_unused]] const EventSchedule &event,
                         BatchQueryTodo *new_batch) override {
    int prefill_count = 0;
    const int max_prefill_count = 2;
    for (auto &q : active_query) {
      if (q->plan_status == Query::Prefill) {
        prefill_count += 1;
      }
    }

    while (prefill_count < max_prefill_count && ready_queue.empty() == false &&
           active_query.size() < settings.max_batch_size) {
      auto next_q = ready_queue.front();
      ready_queue.pop();

      SPDLOG_INFO("Active query {}", next_q->id);
      active_query.insert(next_q);
      next_q->to_status(Query::Prefill);
      prefill_count += 1;
    }
    if (active_query.empty() == false) {
      SPDLOG_DEBUG("Active Query Size {}", active_query.size());
    }
    for (auto &q : active_query) {
      q->debug();
    }
    gen_batch_query_todo(new_batch, active_query);
  }
};

std::shared_ptr<Scheduler> create_scheduler(Settings settings) {
  spdlog::set_level(spdlog::level::debug);
  std::shared_ptr<Scheduler> re;
  SPDLOG_INFO("Using Strategy {}", settings.strategy_name);
  if (settings.strategy_name == "FCFS-single-prefill") {
    re = std::shared_ptr<Scheduler>(new FCFS_single_prefill());
  } else if (settings.strategy_name == "FCFS") {
    re = std::shared_ptr<Scheduler>(new FCFS());
  } else {
    SPDLOG_ERROR("Unknown strategy {}", settings.strategy_name);
  }
  re->init(settings);
  return re;
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SampleOptions, temperature, top_p);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(QueryAdd, query_token, query_length,
                                   estimated_length, sample_options, user_id,
                                   SLO_TTFT_ms, SLO_TBT_ms);

std::string QueryAdd::serialize() {
  json j = *this;
  return j.dump();
}

QueryAdd QueryAdd::deserialize(const std::string &input) {
  json j = json::parse(input);
  return j.get<QueryAdd>();
}

}; // namespace scheduler
