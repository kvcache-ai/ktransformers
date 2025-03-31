#pragma once
#include <cstddef>
#include <filesystem>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#define FMT_HEADER_ONLY
#include "spdlog/spdlog.h"

#include "io_helper.hpp"

namespace async_store {

struct ArrayStore;

ArrayStore* create_or_open_store(size_t element_size, size_t size, std::filesystem::path data_path);
void close_store(ArrayStore* store);
size_t capacity(ArrayStore* store);
void extend(ArrayStore* store, size_t to);



struct IORequest {
  ArrayStore* store;
  bool write;
  void* data;
  size_t index;

  // for sync
  bool need_promise = false;
  BatchPromise* promise;
};

std::string request_to_string(IORequest* req);

struct IODealerImpl;
struct IODealer {
  IODealerImpl* io_impl;

  IODealer(bool use_io_uring = false, int IO_DEPTH = 128);
  ~IODealer();
  IODealer(const IODealer&) = delete;
  IODealer& operator=(const IODealer&) = delete;
  IODealer(IODealer&&) = default;
  IODealer& operator=(IODealer&&) = default;

  void enqueue(std::shared_ptr<IORequest> req);
  std::thread start_io_thread();
  void stop();
};

}  // namespace async_store
