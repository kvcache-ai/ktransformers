
#include <unistd.h>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "page_aligned_memory_pool.cpp"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#define FMT_HEADER_ONLY
#include "spdlog/spdlog.h"

// 每个线程执行的任务
void thread_task(PageAlignedMemoryPool& pool) {
  std::mt19937 gen(123);
  std::vector<std::pair<void*, size_t>> allocated;
  size_t cnt = 40000;
  for (size_t i = 0; i < cnt; ++i) {
    // 随机分配一个大小
    size_t size = (gen() % 100 + 1) * 4096 * 4;
    void* ptr = pool.alloc(size);
    // SPDLOG_DEBUG(pool.debug());
    if (ptr) {
      pool.free(ptr, size);
      //   allocated.push_back({ptr, size});
    }
    // sleep((int)(gen() % 1000) / 1000.0);
  }
  // free all memory
  for (auto& p : allocated) {
    pool.free(p.first, p.second);
  }
}

int main(int argc, char* argv[]) {
  spdlog::set_level(spdlog::level::debug);

  // 创建一个内存池
  PageAlignedMemoryPool pool(40ll * 1024 * 1024 * 1024);  // 40 G

  // 创建线程
  const int num_threads = 32;
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_task, std::ref(pool));
  }

  // 等待所有线程完成
  for (auto& t : threads) {
    t.join();
  }

  // 输出调试信息
  std::cout << pool.debug() << std::endl;

  return 0;
}