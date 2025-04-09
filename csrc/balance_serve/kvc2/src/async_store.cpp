
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <filesystem>
#include <future>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>

#include "utils/lock_free_queue.hpp"

#include "async_store.hh"

namespace async_store {

struct ArrayStore {
  static const size_t DeviceBlockSize = 512;

  const size_t element_size;
  const size_t element_size_aligned;

  size_t size;

  size_t size_in_bytes() { return size * element_size_aligned; }

  std::filesystem::path data_path;

  void extend(size_t to) {
    if (to <= size) {
      return;
    }
    // TODO: extend file
    size = to;
    // LOG_INFO("Extend file to `, size `", to, size_in_bytes());
  }

  ArrayStore(size_t element_size, size_t size, std::filesystem::path data_path)
      : element_size(element_size),
        element_size_aligned((element_size + DeviceBlockSize - 1) / DeviceBlockSize),
        data_path(data_path) {
    // TODO: prefix cache
  }

  void read(size_t index, void* buffer) {
    // TODO: read from file
  }
  void write(size_t index, void* buffer) {
    // TODO: write to file
  }
};

ArrayStore* create_or_open_store(size_t element_size, size_t size, std::filesystem::path data_path) {
  return new ArrayStore(element_size, size, data_path);
}

void close_store(ArrayStore* store) {
  delete store;
}

size_t capacity(ArrayStore* store) {
  return store->size;
}

void extend(ArrayStore* store, size_t to) {
  store->extend(to);
}

template <typename T>
struct ArrayStoreT {
  ArrayStore store;
  ArrayStoreT(size_t element_count, std::filesystem::path data_path) : store(sizeof(T), element_count, data_path) {}

  void read(size_t index, void* output) { store.read(index, output); }

  void write(size_t index, T& value) { store.write(index, &value); }
  void write(size_t index, void* value) { store.write(index, value); }
};

std::string request_to_string(IORequest* req) {
  return fmt::format("IOReqeust {} {} to {}[{}]", req->write ? "Write" : "Read ", req->data,
                     req->store->data_path.c_str(), req->index);
}

struct IODealerImpl {
  MPSCQueue<IORequest> ioQueue;
  uint64_t io_cnt = 0;
  size_t io_amount = 0;
  bool use_io_uring;
  int IO_DEPTH;

  bool stop = false;
  IODealerImpl(bool use_io_uring, int IO_DEPTH) : use_io_uring(use_io_uring), IO_DEPTH(IO_DEPTH) {}

  void queue_consumer() {
    // TODO:
  }

  void io_perf() {
    // TODO:
  }

  void io_dealer() {
    // TODO:
  }
};

IODealer::IODealer(bool use_io_uring, int IO_DEPTH) {
  io_impl = new IODealerImpl(use_io_uring, IO_DEPTH);
}

IODealer::~IODealer() {
  stop();
  delete io_impl;
}

void IODealer::enqueue(std::shared_ptr<IORequest> req) {
  io_impl->ioQueue.enqueue(req);
}

std::thread IODealer::start_io_thread() {
  return std::thread([this]() { io_impl->io_dealer(); });
}
void IODealer::stop() {
  if (io_impl->stop) {
    return;
  }
  // LOG_INFO("Stopping IO Dealer");
  io_impl->stop = true;
}

}  // namespace async_store
