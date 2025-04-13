#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "prometheus/counter.h"
#include "prometheus/exposer.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"

#include "utils/timer.hpp"

namespace kvc2 {

// 指标前缀宏定义
#define METRIC_PREFIX "kvc2"

struct MetricsConfig {
  std::string endpoint;  // 监听端点，如 "0.0.0.0:8080"
};

class Metrics {
 public:
  // 构造函数传入 MetricsConfig
  Metrics(const MetricsConfig& config);
  ~Metrics();

  // 禁止拷贝和赋值
  Metrics(const Metrics&) = delete;
  Metrics& operator=(const Metrics&) = delete;

  // 指标指针
  prometheus::Counter* prefix_nodes;
  prometheus::Counter* prefix_block_count;

  prometheus::Histogram* raw_insert_time_ms;
  prometheus::Histogram* lookup_time_ms;
  prometheus::Histogram* lookup_prefixmatch_length;
  prometheus::Histogram* matched_length_percentage;

  prometheus::Gauge* disk_usage;

  prometheus::Gauge* memory_pool_size(const std::string& type);
  prometheus::Gauge* memory_pool_node_count(const std::string& type);

  prometheus::Gauge* lru_entry_count(const std::string& type);
  prometheus::Gauge* gpu_page_count(std::string type);

  prometheus::Histogram* append_tokens_time_ms;
  prometheus::Histogram* gpu_flush_back_time_ms;
  prometheus::Histogram* cpu_flush_back_time_ms;

 private:
  std::shared_ptr<prometheus::Registry> registry_;
  prometheus::Exposer exposer_;

  prometheus::Family<prometheus::Gauge>* memory_pool_size_family_;
  prometheus::Family<prometheus::Gauge>* memory_pool_node_count_family_;
  prometheus::Family<prometheus::Gauge>* lru_entry_count_family_;
  prometheus::Family<prometheus::Gauge>* gpu_page_count_family_;
};

class TimeObserver {
 public:
  TimeObserver(prometheus::Histogram* h);
  ~TimeObserver();

 private:
  Timer timer_;
  prometheus::Histogram* histogram_;
};

}  // namespace kvc2