#include "page_aligned_memory_pool.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#define FMT_HEADER_ONLY
#include "spdlog/spdlog.h"

#include "utils/arithmetic.hpp"
#include "utils/easy_format.hpp"

/// 构造函数
PageAlignedMemoryPool::PageAlignedMemoryPool(size_t size_in_bytes) {
  total_size = (size_in_bytes / PageSize) * PageSize;
  // 对齐分配。C++17 对齐方式写法，如果编译器不支持可以改用其它方法
  data = ::operator new[](total_size, std::align_val_t(PageSize));
  total_pages = total_size / PageSize;

  assert(total_pages >= Blocks);
  page_per_block = total_pages / Blocks;

  for (size_t block_index = 0; block_index < Blocks; block_index++) {
    first_page[block_index] = reinterpret_cast<void*>(reinterpret_cast<intptr_t>(data) +
                                                      static_cast<intptr_t>(block_index) * page_per_block * PageSize);
    count_page[block_index] =
        block_index == Blocks - 1 ? (total_pages - page_per_block * (Blocks - 1)) : page_per_block;
    SPDLOG_DEBUG("first_page[{}] = {}, count_page[{}] = {}", block_index,
                 reinterpret_cast<intptr_t>(first_page[block_index]) - reinterpret_cast<intptr_t>(data), block_index,
                 count_page[block_index]);
    bitmap[block_index].resize(count_page[block_index], 0);
  }
  SPDLOG_INFO("PageAlignedMemoryPool with size {} Mbytes, {} pages", total_size / (1 << 20), page_count());
}

/// 析构函数
PageAlignedMemoryPool::~PageAlignedMemoryPool() {
  if (data) {
    // 注意：需要与分配时的对齐方式对应
    ::operator delete[](data, std::align_val_t(PageSize));
    data = nullptr;
  }
}

/// 返回总页数
size_t PageAlignedMemoryPool::page_count() {
  return total_size / PageSize;
}

/// 返回按整页对齐后的字节数
size_t PageAlignedMemoryPool::page_padded_size(size_t size) {
  return div_up(size, PageSize) * PageSize;
}

void* PageAlignedMemoryPool::alloc_in_block(size_t block_index, size_t alloc_size) {
  std::lock_guard<std::mutex> guard(lock[block_index]);
  size_t free_pages = 0;
  for (size_t i = 0; i < count_page[block_index]; i++) {
    if (bitmap[block_index][i] == 0) {
      free_pages++;
      if (free_pages == alloc_size) {
        size_t page_index = i + 1 - free_pages;
        for (size_t page = page_index; page < page_index + alloc_size; page++) {
          bitmap[block_index][page] = 1;
          // SPDLOG_DEBUG("alloc page {} in block {}", page, block_index);
        }
        return reinterpret_cast<void*>(reinterpret_cast<intptr_t>(first_page[block_index]) + page_index * PageSize);
      }
    } else {
      free_pages = 0;
    }
  }
  return nullptr;
}

/// 分配函数
void* PageAlignedMemoryPool::alloc(size_t size) {
  size_t alloc_size = div_up(size, PageSize);
  auto cnt = now_block.fetch_add(1, std::memory_order_relaxed);
  for (size_t i = 0; i < Blocks; i++) {
    auto result = alloc_in_block((i + cnt) % Blocks, alloc_size);
    if (result != nullptr) {
      allocated.fetch_add(alloc_size * PageSize, std::memory_order_relaxed);
      alloc_count.fetch_add(1, std::memory_order_relaxed);
      return result;
    }
  }
  return nullptr;
}

/// 释放函数
void PageAlignedMemoryPool::free(void* p, size_t size) {
  auto alloc_size = div_up(size, PageSize);
  size_t block_index = (reinterpret_cast<intptr_t>(p) - reinterpret_cast<intptr_t>(data)) / page_per_block / PageSize;
  size_t page_index = (reinterpret_cast<intptr_t>(p) - reinterpret_cast<intptr_t>(first_page[block_index])) / PageSize;

  std::lock_guard<std::mutex> guard(lock[block_index]);

  for (size_t page = page_index; page < page_index + alloc_size; page++)
    bitmap[block_index][page] = 0;

  allocated.fetch_sub(alloc_size * PageSize, std::memory_order_relaxed);
  free_count.fetch_add(1, std::memory_order_relaxed);
}
// TODO: too slow
std::vector<void*> PageAlignedMemoryPool::alloc_multiple(size_t size, size_t count) {
  std::vector<void*> result;
  for (size_t i = 0; i < count; i++) {
    auto p = alloc(size);
    if (p == nullptr) {
      for (auto ptr : result) {
        free(ptr, size);
      }
      return {};
    }
    result.push_back(p);
  }
  return result;
}

void PageAlignedMemoryPool::defragment() {}

/// 调试打印
std::string PageAlignedMemoryPool::debug() {
  return fmt::format("PageAlignedMemoryPool: total_size: {}MB, allocated: {}, alloc/free count: {}/{}\n",
                     readable_number(total_size), readable_number(size_t(allocated)), size_t(alloc_count),
                     size_t(free_count));
}
