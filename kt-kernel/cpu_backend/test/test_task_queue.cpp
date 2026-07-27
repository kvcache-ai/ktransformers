#include "../task_queue.h"

#include <atomic>
#include <cassert>
#include <stdexcept>
#include <string>

int main() {
  TaskQueue queue;
  std::atomic<int> completed{0};

  queue.enqueue([&] { completed.fetch_add(1); });
  queue.enqueue([] { throw std::runtime_error("first task failure"); });
  queue.enqueue([&] { completed.fetch_add(1); });

  bool caught = false;
  try {
    queue.sync(0);
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "first task failure";
  }
  assert(caught);
  assert(completed.load() == 2);

  queue.sync(0);
  queue.enqueue([&] { completed.fetch_add(1); });
  queue.sync(0);
  assert(completed.load() == 3);

  queue.enqueue([] { throw std::runtime_error("first"); });
  queue.enqueue([] { throw std::runtime_error("second"); });
  try {
    queue.sync(0);
    assert(false);
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "first");
  }

  return 0;
}
