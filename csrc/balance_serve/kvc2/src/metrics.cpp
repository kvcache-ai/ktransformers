#include "metrics.h"

namespace kvc2 {

Metrics::Metrics(const MetricsConfig& config)
    : registry_(std::make_shared<prometheus::Registry>()), exposer_(config.endpoint) {
  // 注册 prefix_nodes Counter
  auto& prefix_nodes_family = prometheus::BuildCounter()
                                  .Name(std::string(METRIC_PREFIX) + "_prefix_nodes")
                                  .Help("Number of prefix nodes")
                                  .Register(*registry_);
  prefix_nodes = &prefix_nodes_family.Add({});

  // 注册 prefix_block_count Counter
  auto& prefix_block_count_family = prometheus::BuildCounter()
                                        .Name(std::string(METRIC_PREFIX) + "_prefix_block_count")
                                        .Help("Number of prefix blocks")
                                        .Register(*registry_);
  prefix_block_count = &prefix_block_count_family.Add({});

  // 定义统一的桶大小，最大为 10000 ms (10 s)
  std::vector<double> common_buckets = {1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0};

  // 注册 raw_insert_time_ms Histogram
  auto& raw_insert_time_ms_family = prometheus::BuildHistogram()
                                        .Name(std::string(METRIC_PREFIX) + "_raw_insert_time_ms")
                                        .Help("function raw insert's time in milliseconds")
                                        .Register(*registry_);
  raw_insert_time_ms = &raw_insert_time_ms_family.Add({}, common_buckets);

  // 注册 lookup_time_ms Histogram
  auto& lookup_time_ms_family = prometheus::BuildHistogram()
                                    .Name(std::string(METRIC_PREFIX) + "_lookup_time_ms")
                                    .Help("function lookup's time in milliseconds")
                                    .Register(*registry_);
  lookup_time_ms = &lookup_time_ms_family.Add({}, common_buckets);

  // 注册 lookup_prefixmatch_length Histogram
  auto& lookup_prefixmatch_length_family = prometheus::BuildHistogram()
                                               .Name(std::string(METRIC_PREFIX) + "_lookup_prefixmatch_length")
                                               .Help("function lookup's prefix match length")
                                               .Register(*registry_);
  lookup_prefixmatch_length = &lookup_prefixmatch_length_family.Add({}, common_buckets);

  // 注册 matched_length_percentage Histogram
  auto& matched_length_percentage_family = prometheus::BuildHistogram()
                                               .Name(std::string(METRIC_PREFIX) + "_matched_length_percentage")
                                               .Help("function matched length percentage")
                                               .Register(*registry_);
  matched_length_percentage = &matched_length_percentage_family.Add({}, common_buckets);

  // 注册 disk_usage Gauge
  auto& disk_usage_family =
      prometheus::BuildGauge().Name(std::string(METRIC_PREFIX) + "_disk_usage").Help("disk usage").Register(*registry_);
  disk_usage = &disk_usage_family.Add({});

  // 注册 memory_pool_size Gauge
  memory_pool_size_family_ = &prometheus::BuildGauge()
                                  .Name(std::string(METRIC_PREFIX) + "_memory_pool_size")
                                  .Help("memory pool size")
                                  .Register(*registry_);

  // 注册 memory_pool_node_count Gauge
  memory_pool_node_count_family_ = &prometheus::BuildGauge()
                                        .Name(std::string(METRIC_PREFIX) + "_memory_pool_node_count")
                                        .Help("memory pool node count")
                                        .Register(*registry_);

  // 注册 lru_entry_count Gauge
  lru_entry_count_family_ = &prometheus::BuildGauge()
                                 .Name(std::string(METRIC_PREFIX) + "_lru_entry_count")
                                 .Help("lru entry count")
                                 .Register(*registry_);

  // 注册 gpu_page_count Gauge
  gpu_page_count_family_ = &prometheus::BuildGauge()
                                .Name(std::string(METRIC_PREFIX) + "_gpu_page_count")
                                .Help("gpu page count")
                                .Register(*registry_);

  // 注册 append_tokens_time_ms Histogram
  auto& append_tokens_time_ms_family = prometheus::BuildHistogram()
                                           .Name(std::string(METRIC_PREFIX) + "_append_tokens_time_ms")
                                           .Help("append tokens time in milliseconds")
                                           .Register(*registry_);
  append_tokens_time_ms = &append_tokens_time_ms_family.Add({}, common_buckets);

  // 注册 gpu_flush_back_time_ms Histogram
  auto& gpu_flush_back_time_ms_family = prometheus::BuildHistogram()
                                            .Name(std::string(METRIC_PREFIX) + "_gpu_flush_back_time_ms")
                                            .Help("gpu flush back time in milliseconds")
                                            .Register(*registry_);
  gpu_flush_back_time_ms = &gpu_flush_back_time_ms_family.Add({}, common_buckets);

  // 注册 cpu_flush_back_time_ms Histogram
  auto& cpu_flush_back_time_ms_family = prometheus::BuildHistogram()
                                            .Name(std::string(METRIC_PREFIX) + "_cpu_flush_back_time_ms")
                                            .Help("cpu flush back time in milliseconds")
                                            .Register(*registry_);
  cpu_flush_back_time_ms = &cpu_flush_back_time_ms_family.Add({}, common_buckets);

  exposer_.RegisterCollectable(registry_);
}

// 析构函数
Metrics::~Metrics() {
  // 停止指标暴露
  // exposer_.Stop();
}

// 获取 memory_pool_size 指标
prometheus::Gauge* Metrics::memory_pool_size(const std::string& type) {
  return &memory_pool_size_family_->Add({{"type", type}});
}

// 获取 memory_pool_node_count 指标
prometheus::Gauge* Metrics::memory_pool_node_count(const std::string& type) {
  return &memory_pool_node_count_family_->Add({{"type", type}});
}

// 获取 lru_entry_count 指标
prometheus::Gauge* Metrics::lru_entry_count(const std::string& type) {
  return &lru_entry_count_family_->Add({{"type", type}});
}

// 获取 gpu_page_count 指标
prometheus::Gauge* Metrics::gpu_page_count(std::string type) {
  return &gpu_page_count_family_->Add({{"type", type}});
}

TimeObserver::TimeObserver(prometheus::Histogram* h) {
  histogram_ = h;
  timer_.start();
}

TimeObserver::~TimeObserver() {
  timer_.stop();
  histogram_->Observe(timer_.elapsedNs() / 1e6);  // ns -> ms
}

}  // namespace kvc2