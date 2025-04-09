#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

class Statistics {
public:
  // Increment the counter for a given key by a specified value (default is 1)
  void increment_counter(const std::string &key, int64_t value = 1) {
    counters_[key] += value;
  }

  int64_t &get_counter(const std::string &key) { return counters_[key]; }

  // Start the timer for a given key
  void start_timer(const std::string &key) {
    active_timers_[key] = std::chrono::high_resolution_clock::now();
  }

  // Stop the timer for a given key and update the total time and count
  void stop_timer(const std::string &key) {
    auto start_it = active_timers_.find(key);
    if (start_it != active_timers_.end()) {
      auto duration =
          std::chrono::high_resolution_clock::now() - start_it->second;
      timings_[key].total_time += duration;
      timings_[key].count += 1;
      active_timers_.erase(start_it);
    } else {
      // Handle error: stop_timer called without a matching start_timer
      std::cerr << "Warning: stop_timer called for key '" << key
                << "' without a matching start_timer.\n";
    }
  }

  // Print out the collected statistical information
  void report() const {
    std::cout << "Counters:\n";
    for (const auto &kv : counters_) {
      std::cout << "  " << kv.first << ": " << kv.second << "\n";
    }
    std::cout << "\nTimers:\n";
    for (const auto &kv : timings_) {
      std::cout << "  " << kv.first << ": count = " << kv.second.count
                << ", total_time = " << kv.second.total_time.count() << "s"
                << ", average_time = "
                << (kv.second.count > 0
                        ? kv.second.total_time.count() / kv.second.count
                        : 0)
                << "s\n";
    }
  }

private:
  // Mapping from key to counter
  std::unordered_map<std::string, int64_t> counters_;

  // Struct to hold timing information for a key
  struct TimingInfo {
    int64_t count = 0;
    std::chrono::duration<double> total_time =
        std::chrono::duration<double>::zero();
  };

  // Mapping from key to timing information
  std::unordered_map<std::string, TimingInfo> timings_;

  // Mapping from key to the start time of active timers
  std::unordered_map<std::string,
                     std::chrono::high_resolution_clock::time_point>
      active_timers_;
};

#endif // STATISTICS_HPP
