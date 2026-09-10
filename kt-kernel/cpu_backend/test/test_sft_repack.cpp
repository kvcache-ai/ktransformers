// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../operators/sft_repack.hpp"

namespace {
using namespace std::chrono_literals;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_versions_and_failed_publication() {
  sft::RepackState slot;
  const auto fp8 = sft::RepackState::new_version();
  const auto bf16 = sft::RepackState::new_version();
  const auto reloaded = sft::RepackState::new_version();
  std::vector<unsigned char> storage;
  int preparations = 0;
  auto prepare = [&](auto version, size_t bytes, unsigned char value) {
    slot.ensure_ready(version, [&] {
      ++preparations;
      storage.assign(bytes, value);
    });
    require(storage.size() == bytes && storage.back() == value, "wrong shared buffer owner");
  };
  prepare(fp8, 128, 8);
  prepare(fp8, 128, 8);
  require(preparations == 1, "prefetched weights were repacked by consumer");
  prepare(bf16, 256, 16);
  prepare(fp8, 128, 8);
  prepare(reloaded, 128, 9);
  require(preparations == 4, "dtype/owner change or reload reused stale weights");

  bool caught = false;
  try {
    slot.ensure_ready(bf16, [&] {
      storage.front() = 99;
      throw std::runtime_error("injected partial repack");
    });
  } catch (const std::runtime_error&) {
    caught = true;
  }
  require(caught, "repack failure was swallowed");
  prepare(reloaded, 128, 9);
  require(preparations == 5 && storage.front() == 9, "failed repack published stale readiness");
}

void test_async_producer_and_cpu_consumer() {
  sft::RepackTask task;
  std::promise<void> producer_started, release_producer, consumer_attempted;
  auto release = release_producer.get_future();
  int weights = 0;
  task.submit([&] {
    producer_started.set_value();
    release.wait();
    weights = 42;
  });
  producer_started.get_future().get();
  // Host/GPU work can run while the preparation task is still in flight.
  require(task.pending(), "submit was synchronous");
  auto consumer = std::async(std::launch::async, [&] {
    consumer_attempted.set_value();
    auto execution = sft::acquire_cpu_execution();
    return weights;
  });
  consumer_attempted.get_future().get();
  const bool consumer_blocked = consumer.wait_for(20ms) == std::future_status::timeout;
  release_producer.set_value();
  task.wait();
  require(consumer_blocked, "CPU consumer entered the non-reentrant executor during repack");
  require(consumer.get() == 42, "CPU consumer observed incomplete weights");
  require(!task.pending(), "wait did not retire the task");
  task.wait();
}

void test_producer_waits_for_previous_consumer() {
  sft::RepackTask task;
  std::promise<void> started;
  auto started_future = started.get_future();
  bool producer_blocked;
  {
    auto execution = sft::acquire_cpu_execution();
    task.submit([&] { started.set_value(); });
    producer_blocked = started_future.wait_for(20ms) == std::future_status::timeout;
  }
  task.wait();
  require(producer_blocked, "prefetch overwrote an active consumer's buffer");
}

void test_exception_and_retry() {
  sft::RepackTask task;
  task.submit([] { throw std::runtime_error("async error"); });
  bool caught = false;
  try {
    task.wait();
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "async error";
  }
  require(caught, "wait did not propagate the producer exception");
  require(!task.pending(), "failed task was not retired");
  int result = 0;
  task.submit([&] { result = 7; });
  task.wait();
  require(result == 7, "task could not be reused after failure");
}

void test_destruction_drains() {
  int completed = 0;
  {
    sft::RepackTask task;
    task.submit([&] { completed = 1; });
  }
  require(completed == 1, "destructor left a task using captured resources");
  {
    sft::RepackTask task;
    task.submit([] { throw std::runtime_error("abandoned error"); });
  }
}
}  // namespace

int main() {
  test_versions_and_failed_publication();
  test_async_producer_and_cpu_consumer();
  test_producer_waits_for_previous_consumer();
  test_exception_and_retry();
  test_destruction_drains();
  std::puts("SFT repack lifecycle: PASS");
}
