#pragma once
#include "readable_number.hpp"
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

inline std::string doubleToStringR2(double value) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << value;
  return stream.str();
}

class Timer {
public:
  std::string name;
  bool tmp_timer = false;

  Timer() {}
  Timer(std::string name) : name(name), tmp_timer(true) { start(); }
  ~Timer() {
    if (tmp_timer) {
      std::cout << name << " " << elapsedMs() << " ms" << std::endl;
    }
  }

  void start() {
    m_startTime = std::chrono::high_resolution_clock::now();
    assert(m_isRunning == false);
    m_isRunning = true;
  }

  void stop() {
    m_endTime = std::chrono::high_resolution_clock::now();
    assert(m_isRunning == true);
    m_isRunning = false;
    m_runningNs += elapsedNs();
  }

  double elapsedNs() {
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime;

    if (m_isRunning) {
      endTime = std::chrono::high_resolution_clock::now();
    } else {
      endTime = m_endTime;
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime -
                                                                m_startTime)
        .count();
  }

  void printElapsedMilliseconds() {
    std::cout << elapsedNs() / 1e6 << " ms" << std::endl;
  }

  static std::string ns_to_string(double duration) {
    auto nano_sec = duration;
    if (nano_sec >= 1000) {
      auto mirco_sec = nano_sec / 1000.0;
      if (mirco_sec >= 1000) {
        auto milli_sec = mirco_sec / 1000.0;
        if (milli_sec >= 1000) {
          auto seconds = milli_sec / 1000.0;

          if (seconds >= 60.0) {
            auto minutes = seconds / 60.0;

            if (minutes >= 60.0) {
              auto hours = minutes / 60.0;
              return doubleToStringR2(hours) + " h";
            } else {
              return doubleToStringR2(minutes) + " min";
            }
          } else {
            return doubleToStringR2(seconds) + " sec";
          }
        } else {
          return doubleToStringR2(milli_sec) + " ms";
        }
      } else {
        return doubleToStringR2(mirco_sec) + " us";
      }
    } else {
      return doubleToStringR2(nano_sec) + " ns";
    }
  }

  double runningTimeNs() { return m_runningNs; }

  std::string runningTime() {
    auto duration = m_runningNs;
    return ns_to_string(duration);
  }

  std::string elapsedTime() { return ns_to_string(elapsedNs()); }
  double elapsedMs() { return elapsedNs() / 1e6; }
  std::string report_throughput(size_t op_cnt) {
    double ops = op_cnt / elapsedMs() * 1000;
    return readable_number(ops) + "op/s";
  }

  void merge(Timer &other) {
    assert(m_isRunning == false);
    assert(other.m_isRunning == false);
    m_runningNs += other.runningTimeNs();
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_endTime;
  bool m_isRunning = false;
  double m_runningNs = 0.0;
};

class Counter {
public:
  Counter() {}

  std::map<std::string, size_t> counters;

  void inc(const char *name, size_t num) { counters[name] += num; };
  void print() {
    for (auto &p : counters) {
      std::cout << p.first << " : " << p.second << std::endl;
    }
  };
};
