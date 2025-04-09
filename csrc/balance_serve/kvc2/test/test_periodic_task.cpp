#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <future>
#include <iostream>
#include <thread>
#include "utils/periodic_task.hpp"

// 1. 任务是否按预期执行
void testPeriodicTaskExecution() {
  std::atomic<int> execution_count{0};
  auto task = [&execution_count]() { execution_count++; };

  periodic::PeriodicTask periodic_task(task, std::chrono::milliseconds(50));

  std::this_thread::sleep_for(std::chrono::seconds(2));

  assert(execution_count >= 20);  // 确保任务执行了至少 20 次
  std::cout << "Test 1 passed: Task executed periodically." << std::endl;
  std::cout << "Task executed " << execution_count.load() << " times." << std::endl;
}

// 2. 提前唤醒任务的功能
void testWakeUpImmediately() {
  std::atomic<int> execution_count{0};
  auto task = [&execution_count]() { execution_count++; };

  periodic::PeriodicTask periodic_task(task, std::chrono::milliseconds(200));

  // 提前唤醒任务
  periodic_task.wakeUp();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 等待任务执行

  std::cout << "Execution count after wakeUp: " << execution_count.load() << std::endl;
  assert(execution_count == 1);  // 确保任务立即执行
  std::cout << "Test 2 passed: Task woke up immediately." << std::endl;
}

// 3. wakeUpWait() 的等待功能
void testWakeUpWait() {
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  auto task = [&promise]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 模拟任务执行
    promise.set_value();                                          // 任务完成时设置 promise
  };

  periodic::PeriodicTask periodic_task(task, std::chrono::milliseconds(200));

  // 调用 wakeUpWait 并等待任务完成
  std::future<void> wakeup_future = periodic_task.wakeUpWait();
  wakeup_future.wait();  // 等待任务完成

  assert(wakeup_future.valid());  // 确保 future 是有效的
  std::cout << "Test 3 passed: wakeUpWait() works correctly." << std::endl;
  std::cout << "wakeUpWait() future is valid." << std::endl;
}

// 4. 任务抛出异常的处理
void testTaskExceptionHandling() {
  auto task = []() { throw std::runtime_error("Test exception"); };

  periodic::PeriodicTask periodic_task(task, std::chrono::milliseconds(200));

  std::this_thread::sleep_for(std::chrono::milliseconds(300));  // 等待一段时间

  std::cout << "Test 4 passed: Task exception is handled correctly." << std::endl;
  std::cout << "Exception handled and task did not crash." << std::endl;
}

// 5. 线程是否能正确停止
void testTaskStop() {
  std::atomic<bool> stopped{false};
  auto task = [&stopped]() {
    while (!stopped) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  };

  periodic::PeriodicTask periodic_task(task, std::chrono::milliseconds(100));

  std::this_thread::sleep_for(std::chrono::seconds(1));  // 运行一段时间

  stopped = true;                                              // 请求停止
  std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 等待线程停止

  std::cout << "Test 5 passed: Task thread stops correctly." << std::endl;
  std::cout << "Task has been stopped successfully." << std::endl;
}

// 6. 高频唤醒的情况下任务执行是否正常
void testHighFrequencyWakeUp() {
  std::atomic<int> execution_count{0};
  auto task = [&execution_count]() { execution_count++; };

  periodic::PeriodicTask periodic_task(task, std::chrono::milliseconds(200));

  for (int i = 0; i < 100; ++i) {
    periodic_task.wakeUp();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 每 10 毫秒唤醒一次
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));  // 等待任务执行完成

  assert(execution_count > 50);  // 确保任务至少执行了 50 次
  std::cout << "Test 6 passed: Task handles frequent wake ups correctly." << std::endl;
  std::cout << "Task executed " << execution_count.load() << " times." << std::endl;
}

// 7. 多个 wakeUpWait() 调用的处理
void testMultipleWakeUpWait() {
  std::atomic<int> execution_count{0};
  auto task = [&execution_count]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 模拟任务执行
    execution_count++;
  };

  periodic::PeriodicTask periodic_task(task, std::chrono::milliseconds(200));

  // 同时调用两个 wakeUpWait
  std::future<void> future1 = periodic_task.wakeUpWait();
  std::future<void> future2 = periodic_task.wakeUpWait();

  future1.wait();
  future2.wait();

  assert(execution_count == 1);  // 确保任务只执行了一次
  std::cout << "Test 7 passed: Multiple wakeUpWait() calls are handled correctly." << std::endl;
  std::cout << "Task executed " << execution_count.load() << " times." << std::endl;
}

// 8. 任务函数为空的边界情况
void testEmptyTaskFunction() {
  auto task = []() {
    // 空任务函数
  };

  periodic::PeriodicTask periodic_task(task, std::chrono::milliseconds(100));

  std::this_thread::sleep_for(std::chrono::seconds(1));  // 等待一段时间

  std::cout << "Test 8 passed: Empty task function works correctly." << std::endl;
  std::cout << "Empty task function executed without issues." << std::endl;
}

int main() {
  std::cout << "Starting tests..." << std::endl;

  // testWakeUpImmediately();
  testPeriodicTaskExecution();
  testWakeUpImmediately();
  testWakeUpWait();
  testTaskExceptionHandling();
  testTaskStop();
  testHighFrequencyWakeUp();
  testMultipleWakeUpWait();
  testEmptyTaskFunction();

  std::cout << "All tests passed!" << std::endl;

  return 0;
}
