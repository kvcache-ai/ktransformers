#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "utils/lock_free_queue.hpp"

struct Item {
  int value;
  std::promise<void> promise;
};

int main() {
  MPSCQueue<Item> queue;

  std::vector<std::thread> producers;
  const int num_producers = 4;
  const int items_per_producer = 5;

  // 启动生产者线程
  for (int i = 0; i < num_producers; ++i) {
    producers.emplace_back([&queue, i]() {
      for (int j = 0; j < items_per_producer; ++j) {
        auto item = std::make_shared<Item>();
        item->value = i * items_per_producer + j;
        std::future<void> future = item->promise.get_future();
        queue.enqueue(item);
        future.wait();  // 等待消费者处理完成
      }
    });
  }

  // 启动消费者线程
  std::thread consumer([&queue, num_producers, items_per_producer]() {
    int total_items = num_producers * items_per_producer;
    int processed = 0;
    while (processed < total_items) {
      std::shared_ptr<Item> item = queue.dequeue();
      if (item) {
        std::cout << "Consumed item with value: " << item->value << std::endl;
        item->promise.set_value();  // 通知生产者
        ++processed;
      } else {
        // 如果队列为空，可以选择休眠或让出线程
        std::this_thread::yield();
      }
    }
  });

  // 等待所有线程完成
  for (auto& producer : producers) {
    producer.join();
  }
  consumer.join();

  return 0;
}