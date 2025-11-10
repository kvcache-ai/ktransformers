#include <atomic>

template <typename T> struct AtomicPtrWithFlag {
  constexpr static uint64_t mask = 1ull << 63;
  std::atomic_uint64_t ptr = 0;

  std::pair<T *, bool>
  load(std::memory_order order = std::memory_order_seq_cst) {
    uint64_t val = ptr.load(order);
    return {reinterpret_cast<T *>(val & (~mask)), val & mask};
  }

  void store(T *p, bool flag,
             std::memory_order order = std::memory_order_seq_cst) {
    ptr.store(reinterpret_cast<uint64_t>(p) | (flag ? mask : 0), order);
  }

  std::pair<T *, bool>
  exchange(T *p, bool flag,
           std::memory_order order = std::memory_order_seq_cst) {
    uint64_t val =
        ptr.exchange(reinterpret_cast<uint64_t>(p) | (flag ? mask : 0), order);
    return {reinterpret_cast<T *>(val & (~mask)), val & mask};
  }

  std::pair<T *, bool>
  touch_load(std::memory_order order = std::memory_order_seq_cst) {
    uint64_t val = ptr.fetch_and(~mask, order);
    return {reinterpret_cast<T *>(val & (~mask)), val & mask};
  }

  bool check_flag(std::memory_order order = std::memory_order_seq_cst) {
    return ptr.load(order) & mask;
  }
};
