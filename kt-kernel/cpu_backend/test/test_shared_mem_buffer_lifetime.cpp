#include "../shared_mem_buffer.h"

#include <numa.h>
#include <sched.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

void expect_equal(int actual, int expected, const char* message) {
  if (actual != expected) {
    throw std::runtime_error(std::string(message) + ": expected " + std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

MemoryRequest counting_request(size_t bytes, int& update_count) {
  MemoryRequest request;
  request.append_function([&update_count](void*) { ++update_count; }, bytes);
  return request;
}

class ScratchOwner {
 public:
  explicit ScratchOwner(SharedMemBuffer& buffer) : buffer_(buffer) {}
  ~ScratchOwner() { buffer_.dealloc(this); }

  void alloc(size_t bytes, int& update_count) { buffer_.alloc(this, counting_request(bytes, update_count)); }

 private:
  SharedMemBuffer& buffer_;
};

class NumaScratchOwner {
 public:
  NumaScratchOwner(SharedMemBufferNuma& buffer, int numa) : buffer_(buffer), numa_(numa) {}
  ~NumaScratchOwner() { buffer_.dealloc(numa_, this); }

  void alloc(size_t bytes, int& update_count) { buffer_.alloc(numa_, this, counting_request(bytes, update_count)); }

 private:
  SharedMemBufferNuma& buffer_;
  int numa_;
};

void test_destroyed_owner_is_not_replayed() {
  SharedMemBuffer buffer;
  int destroyed_updates = 0;

  {
    ScratchOwner owner(buffer);
    owner.alloc(64, destroyed_updates);
    expect_equal(destroyed_updates, 1, "initial request update");
  }

  int growing_updates = 0;
  ScratchOwner growing_owner(buffer);
  growing_owner.alloc(128, growing_updates);

  expect_equal(destroyed_updates, 1, "destroyed owner replayed during growth");
  expect_equal(growing_updates, 1, "growing request update");
}

void test_live_owner_keeps_multiple_request_groups() {
  SharedMemBuffer buffer;
  int first_updates = 0;
  int second_updates = 0;
  int growing_updates = 0;
  int second_growth_updates = 0;
  ScratchOwner growing_owner(buffer);

  {
    ScratchOwner multi_request_owner(buffer);
    multi_request_owner.alloc(64, first_updates);
    multi_request_owner.alloc(32, second_updates);
    growing_owner.alloc(128, growing_updates);

    expect_equal(first_updates, 2, "first live request was not replayed");
    expect_equal(second_updates, 2, "second live request was not replayed");
    expect_equal(growing_updates, 1, "first growth request update");
  }

  growing_owner.alloc(256, second_growth_updates);

  expect_equal(first_updates, 2, "destroyed multi-request owner replayed");
  expect_equal(second_updates, 2, "destroyed multi-request owner replayed");
  expect_equal(growing_updates, 2, "live owner's earlier request was not replayed");
  expect_equal(second_growth_updates, 1, "second growth request update");
}

void test_numa_owner_deallocation() {
  SharedMemBufferNuma buffer;
  const int cpu = sched_getcpu();
  const int detected_numa = cpu >= 0 ? numa_node_of_cpu(cpu) : 0;
  const int numa = detected_numa >= 0 ? detected_numa : 0;
  int destroyed_updates = 0;

  {
    NumaScratchOwner owner(buffer, numa);
    owner.alloc(64, destroyed_updates);
    expect_equal(destroyed_updates, 1, "initial NUMA request update");
  }

  int growing_updates = 0;
  NumaScratchOwner growing_owner(buffer, numa);
  growing_owner.alloc(128, growing_updates);

  expect_equal(destroyed_updates, 1, "destroyed NUMA owner replayed during growth");
  expect_equal(growing_updates, 1, "NUMA growing request update");
}

}  // namespace

int main() {
  try {
    test_destroyed_owner_is_not_replayed();
    test_live_owner_keeps_multiple_request_groups();
    test_numa_owner_deallocation();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "shared memory buffer lifetime test failed: %s\n", error.what());
    return 1;
  }

  std::puts("shared memory buffer lifetime test passed");
  return 0;
}
