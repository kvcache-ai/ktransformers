/*
 * @Author: Xie Weiyu ervinxie@qq.com
 * @Date: 2024-11-21 06:35:47
 * @LastEditors: Xie Weiyu ervinxie@qq.com
 * @LastEditTime: 2024-11-21 06:35:50
 * @FilePath: /kvc2/src/utils/spin_lock.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置:
 * https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#include <atomic>
#include <chrono>
#include <thread>

class SpinLock {
 public:
  SpinLock() { flag.clear(); }

  void lock() {
    const int max_delay = 1024;  // Maximum delay in microseconds
    int delay = 1;               // Initial delay in microseconds

    while (flag.test_and_set(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::microseconds(delay));
      delay *= 2;
      if (delay > max_delay) {
        delay = max_delay;
      }
    }
  }

  void unlock() { flag.clear(std::memory_order_release); }

 private:
  std::atomic_flag flag = ATOMIC_FLAG_INIT;
};
