#pragma once

#include <assert.h>
#include <algorithm>  // std::sort
#include <atomic>
#include <bitset>
#include <cstddef>  // size_t
#include <mutex>    // std::mutex
#include <vector>

constexpr size_t PageSize = 4096;

/// PageAlignedMemoryPool 类的声明
struct PageAlignedMemoryPool {
 private:
  constexpr static size_t Blocks = 16;

  void* data = nullptr;

  size_t total_size = 0, total_pages = 0;

  std::atomic_size_t now_block = 0;
  std::atomic_size_t allocated = 0;  // allocated_size
  std::atomic_size_t alloc_count = 0;
  std::atomic_size_t free_count = 0;

  std::mutex lock[Blocks];
  size_t page_per_block = 0;
  void* first_page[Blocks];
  size_t count_page[Blocks];
  std::vector<int8_t> bitmap[Blocks];
  void* alloc_in_block(size_t block_index, size_t alloc_size);

 public:
  /// 构造函数和析构函数
  explicit PageAlignedMemoryPool(size_t size_in_bytes);
  ~PageAlignedMemoryPool();

  /// 禁用拷贝和移动
  PageAlignedMemoryPool(PageAlignedMemoryPool&& other) = delete;
  PageAlignedMemoryPool& operator=(PageAlignedMemoryPool&& other) = delete;
  PageAlignedMemoryPool(const PageAlignedMemoryPool& other) = delete;
  PageAlignedMemoryPool& operator=(const PageAlignedMemoryPool& other) = delete;

  /// 成员函数
  size_t page_count();
  size_t page_padded_size(size_t size);

  void* alloc(size_t size);
  std::vector<void*> alloc_multiple(size_t size, size_t count);
  void free(void* data, size_t size);
  void defragment();
  std::string debug();
};
