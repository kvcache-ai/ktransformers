#include <mutex>
#include <queue>
#include "utils/lock_free_queue.hpp"

#define STDQ

int main() {
  const int num_producers = 48;
  const int num_items = 1e6;

#ifdef STDQ
  std::mutex lock;
  std::queue<int> queue;
#else
  MPSCQueue<int> queue;
#endif

  auto start_time = std::chrono::high_resolution_clock::now();

  // Launch multiple producer threads
  std::vector<std::thread> producers;
  for (int i = 0; i < num_producers; ++i) {
    producers.emplace_back([&queue, i
#ifdef STDQ
                            ,
                            &lock
#endif
    ]() {
      for (int j = 0; j < num_items; ++j) {
#ifdef STDQ
        std::lock_guard<std::mutex> guard(lock);
        queue.push(i * num_items + j);
#else
        queue.enqueue(std::make_shared<int>(i * num_items + j));
#endif
      }
    });
  }

  // Consumer thread
  std::thread consumer([&queue, num_producers
#ifdef STDQ
                        ,
                        &lock
#endif
  ]() {
    int count = 0;
    while (count < num_producers * num_items) {
#ifdef STDQ
      std::lock_guard<std::mutex> guard(lock);
      if (!queue.empty()) {
        queue.pop();
        count++;
      }
#else
      if (auto item = queue.dequeue()) {
        count++;
      }
#endif
    }
  });

  // Wait for all producers to finish
  for (auto& producer : producers) {
    producer.join();
  }

  // Wait for the consumer to finish
  consumer.join();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

#ifdef STDQ
  std::cout << "std::queue with mutex ";
#else
  std::cout << "lock free queue ";
#endif

  std::cout << "Processed " << num_producers * num_items / 1e6 << "M items in " << duration << " milliseconds "
            << num_producers * num_items / 1e3 / duration << " MOps." << std::endl;

  return 0;
}