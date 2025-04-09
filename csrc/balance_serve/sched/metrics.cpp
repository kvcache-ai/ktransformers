#include "metrics.h"
#include <iostream>

// 构造函数
Metrics::Metrics(const MetricsConfig &config)
    : registry_(std::make_shared<prometheus::Registry>()),
      exposer_(config.endpoint), stop_uptime_thread_(false),
      start_time_(std::chrono::steady_clock::now()) {
  // 定义统一的桶大小，最大为 10000 ms (10 s)
  std::vector<double> common_buckets = {
      0.001, 0.005, 0.01,  0.05,  0.1,    0.5,    1.0,    5.0,
      10.0,  50.0,  100.0, 500.0, 1000.0, 5000.0, 10000.0}; // 毫秒

  // 注册 TTFT_ms Histogram
  auto &TTFT_family = prometheus::BuildHistogram()
                          .Name(std::string(METRIC_PREFIX) + "_TTFT_ms")
                          .Help("Time to first token in milliseconds")
                          .Register(*registry_);
  TTFT_ms = &TTFT_family.Add({{"model", config.model_name}}, common_buckets);

  // 注册 TBT_ms Histogram
  auto &TBT_family = prometheus::BuildHistogram()
                         .Name(std::string(METRIC_PREFIX) + "_TBT_ms")
                         .Help("Time between tokens in milliseconds")
                         .Register(*registry_);
  TBT_ms = &TBT_family.Add({{"model", config.model_name}}, common_buckets);

  // 注册 schedule_time Histogram
  auto &schedule_time_family =
      prometheus::BuildHistogram()
          .Name(std::string(METRIC_PREFIX) + "_schedule_time_ms")
          .Help("Time to generate schedule in milliseconds")
          .Register(*registry_);
  schedule_time =
      &schedule_time_family.Add({{"model", config.model_name}}, common_buckets);

  // 注册 generated_tokens Counter
  auto &generated_tokens_family =
      prometheus::BuildCounter()
          .Name(std::string(METRIC_PREFIX) + "_generated_tokens_total")
          .Help("Total generated tokens")
          .Register(*registry_);
  generated_tokens =
      &generated_tokens_family.Add({{"model", config.model_name}});

  // 注册 throughput_query Gauge
  auto &throughput_query_family =
      prometheus::BuildGauge()
          .Name(std::string(METRIC_PREFIX) + "_throughput_query")
          .Help("Throughput per second based on queries")
          .Register(*registry_);
  throughput_query =
      &throughput_query_family.Add({{"model", config.model_name}});

  // 注册 throughput_generated_tokens Gauge
  auto &throughput_generated_tokens_family =
      prometheus::BuildGauge()
          .Name(std::string(METRIC_PREFIX) + "_throughput_generated_tokens")
          .Help("Throughput per second based on generated tokens")
          .Register(*registry_);
  throughput_generated_tokens =
      &throughput_generated_tokens_family.Add({{"model", config.model_name}});

  // 注册 event_count Counter family
  event_count_family_ =
      &prometheus::BuildCounter()
           .Name(std::string(METRIC_PREFIX) + "_event_count_total")
           .Help("Count of various events")
           .Register(*registry_);

  batch_count_family_ =
      &prometheus::BuildCounter()
           .Name(std::string(METRIC_PREFIX) + "_batch_count_total")
           .Help("Count of various batch by status")
           .Register(*registry_);

  // 注册 query_count Counter family
  query_count_family_ =
      &prometheus::BuildCounter()
           .Name(std::string(METRIC_PREFIX) + "_query_count_total")
           .Help("Count of queries by status")
           .Register(*registry_);

  // 注册 uptime_ms Gauge
  auto &uptime_family = prometheus::BuildGauge()
                            .Name(std::string(METRIC_PREFIX) + "_uptime_ms")
                            .Help("Uptime of the scheduler in milliseconds")
                            .Register(*registry_);
  uptime_ms = &uptime_family.Add({{"model", config.model_name}});

  // 注册 GPU 利用率 Gauges
  auto &gpu_util_family =
      prometheus::BuildGauge()
          .Name(std::string(METRIC_PREFIX) + "_gpu_utilization_ratio")
          .Help("Current GPU utilization ratio (0 to 1)")
          .Register(*registry_);
  for (size_t i = 0; i < config.gpu_count; ++i) {
    gpu_utilization_gauges.push_back(&gpu_util_family.Add(
        {{"gpu_id", std::to_string(i)}, {"model", config.model_name}}));
  }

  // 将 Registry 注册到 Exposer 中
  exposer_.RegisterCollectable(registry_);

  // 启动 uptime 更新线程
  StartUptimeUpdater();
}

// 析构函数
Metrics::~Metrics() { StopUptimeUpdater(); }

// 启动 uptime 更新线程
void Metrics::StartUptimeUpdater() {
  uptime_thread_ = std::thread([this]() {
    while (!stop_uptime_thread_) {
      auto now = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> uptime_duration =
          now - start_time_;
      uptime_ms->Set(uptime_duration.count());
      // fn_every_sec(this);
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  });
}

// 停止 uptime 更新线程
void Metrics::StopUptimeUpdater() {
  stop_uptime_thread_ = true;
  if (uptime_thread_.joinable()) {
    uptime_thread_.join();
  }
}

// 获取 event_count 指标
prometheus::Counter *Metrics::event_count(const std::string &type) {
  return &event_count_family_->Add({{"type", type}}); // 可根据需要添加更多标签
}

// 获取 query_count 指标
prometheus::Counter *Metrics::query_count(const std::string &status) {
  return &query_count_family_->Add(
      {{"status", status}}); // 可根据需要添加更多标签
}

prometheus::Counter *Metrics::batch_count(const std::string &type) {
  return &batch_count_family_->Add({{"type", type}});
}
