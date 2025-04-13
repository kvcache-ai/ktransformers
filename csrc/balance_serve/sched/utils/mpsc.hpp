#include <atomic>
#include <cassert>
#include <iostream>
#include <optional>
#include <semaphore>

template <typename T> class MPSCQueue {
  struct Node {
    T data;
    std::atomic<Node *> next;

    Node() : next(nullptr) {}
    Node(T data_) : data(std::move(data_)), next(nullptr) {}
  };

  std::atomic<Node *> head;
  Node *tail;

public:
  std::atomic_size_t enqueue_count = 0;
  size_t dequeue_count = 0;
  MPSCQueue() {
    Node *dummy = new Node();
    head.store(dummy, std::memory_order_seq_cst);
    tail = dummy;
  }

  ~MPSCQueue() {
    Node *node = tail;
    while (node) {
      Node *next = node->next.load(std::memory_order_seq_cst);
      delete node;
      node = next;
    }
  }

  // 生产者调用
  void enqueue(T data) {
    enqueue_count.fetch_add(1);
    Node *node = new Node(std::move(data));
    Node *prev_head = head.exchange(node, std::memory_order_seq_cst);
    prev_head->next.store(node, std::memory_order_seq_cst);
  }

  // 消费者调用
  std::optional<T> dequeue() {
    Node *next = tail->next.load(std::memory_order_seq_cst);
    if (next) {
      T res = std::move(next->data);
      delete tail;
      tail = next;
      dequeue_count += 1;
      return res;
    }
    return std::nullopt;
  }

  size_t size() { return enqueue_count.load() - dequeue_count; }
};

template <typename T> class MPSCQueueConsumerLock {
  MPSCQueue<T> queue;
  std::counting_semaphore<> sema{0};

public:
  void enqueue(T data) {
    queue.enqueue(std::move(data));
    // std::atomic_thread_fence(std::memory_order_seq_cst);// Inserting this
    // because the memory order might be wrong, I am also not that sure about
    // this.
    sema.release();
  }

  T dequeue() {
    auto re = queue.dequeue();
    if (re.has_value()) {
      while (sema.try_acquire() == false) {
        std::cerr
            << __FILE__ << ":" << __FUNCTION__
            << " sema try acquire should be success, retrying, please check"
            << std::endl;
        // assert(false);
      }
      return re.value();
    }
    sema.acquire();
    return queue.dequeue().value();
  }

  template <typename Rep, typename Period>
  std::optional<T> try_dequeue_for(std::chrono::duration<Rep, Period> dur) {
    auto re = queue.dequeue();
    if (re.has_value()) {
      while (sema.try_acquire() == false) {
        std::cerr
            << __FILE__ << ":" << __FUNCTION__
            << " sema try acquire should be success, retrying, please check"
            << std::endl;
        // assert(false);
      }
      return re.value();
    }

    if (sema.try_acquire_for(dur)) {
      return queue.dequeue().value();
    } else {
      return std::nullopt;
    }
  }

  size_t size() { return queue.size(); }
};
