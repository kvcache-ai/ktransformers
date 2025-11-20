#ifndef Metrics_H
#define Metrics_H

#include <atomic>
#include <chrono>
#include <memory>
#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>
#include <string>
#include <thread>
#include <vector>

#include "timer.hpp"
// 指标前缀宏定义
#define METRIC_PREFIX "scheduler"
class Metrics;

// 配置结构体
struct MetricsConfig {
  std::string endpoint;
  std::string model_name; // 模型名称，如 "gpt-4"
  size_t gpu_count;       // GPU数量
};

// Metrics 类，根据配置初始化 Prometheus 指标
class Metrics {
public:
  // 构造函数传入 MetricsConfig
  Metrics(const MetricsConfig &config);
  ~Metrics();

  // 禁止拷贝和赋值
  Metrics(const Metrics &) = delete;
  Metrics &operator=(const Metrics &) = delete;

  std::function<void(Metrics *)> fn_every_sec;

  // 指标指针
  prometheus::Gauge *uptime_ms;
  prometheus::Histogram *TTFT_ms;
  prometheus::Histogram *TBT_ms;
  prometheus::Histogram *schedule_time;
  prometheus::Gauge *throughput_query;
  prometheus::Gauge *throughput_generated_tokens;
  prometheus::Counter *generated_tokens;
  std::vector<prometheus::Gauge *> gpu_utilization_gauges;

  // 计数器家族
  prometheus::Counter *event_count(const std::string &type);
  prometheus::Counter *query_count(const std::string &status);
  prometheus::Counter *batch_count(const std::string &type);

private:
  std::shared_ptr<prometheus::Registry> registry_;
  prometheus::Exposer exposer_;

  // 计数器家族
  prometheus::Family<prometheus::Counter> *event_count_family_;
  prometheus::Family<prometheus::Counter> *batch_count_family_;
  prometheus::Family<prometheus::Counter> *query_count_family_;

  // 线程和控制变量用于更新 uptime_ms
  std::thread uptime_thread_;
  std::atomic<bool> stop_uptime_thread_;

  // 启动 uptime 更新线程
  void StartUptimeUpdater();
  // 停止 uptime 更新线程
  void StopUptimeUpdater();

  // 记录程序启动时间
  std::chrono::steady_clock::time_point start_time_;
};

struct HistogramTimerWrapper {
  prometheus::Histogram *histogram;
  Timer timer;
  inline HistogramTimerWrapper(prometheus::Histogram *histogram)
      : histogram(histogram), timer() {
    timer.start();
  }
  inline ~HistogramTimerWrapper() { histogram->Observe(timer.elapsedMs()); }
};

#endif // Metrics_H
