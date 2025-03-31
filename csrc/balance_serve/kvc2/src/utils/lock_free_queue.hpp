#include <atomic>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

template <typename T>
class MPSCQueue {
  struct Node {
    std::shared_ptr<T> data;
    std::atomic<Node*> next;

    Node() : next(nullptr) {}
    Node(std::shared_ptr<T> data_) : data(std::move(data_)), next(nullptr) {}
  };

  std::atomic<Node*> head;
  Node* tail;

 public:
  std::atomic_size_t enqueue_count = 0;
  size_t dequeue_count = 0;
  MPSCQueue() {
    Node* dummy = new Node();
    head.store(dummy, std::memory_order_relaxed);
    tail = dummy;
  }

  ~MPSCQueue() {
    // 清理剩余的节点
    Node* node = tail;
    while (node) {
      Node* next = node->next.load(std::memory_order_relaxed);
      delete node;
      node = next;
    }
  }

  // 生产者调用
  void enqueue(std::shared_ptr<T> data) {
    enqueue_count.fetch_add(1);
    Node* node = new Node(std::move(data));
    Node* prev_head = head.exchange(node, std::memory_order_acq_rel);
    prev_head->next.store(node, std::memory_order_release);
  }

  // 消费者调用
  std::shared_ptr<T> dequeue() {
    Node* next = tail->next.load(std::memory_order_acquire);
    if (next) {
      std::shared_ptr<T> res = std::move(next->data);
      delete tail;
      tail = next;
      dequeue_count += 1;
      return res;
    }
    return nullptr;
  }
};