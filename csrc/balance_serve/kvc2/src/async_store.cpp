
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

#include <photon/common/alog.h>
#include <photon/common/io-alloc.h>
#include <photon/fs/localfs.h>
#include <photon/photon.h>
#include <photon/thread/thread11.h>
#include "utils/lock_free_queue.hpp"

#include "async_store.hh"

namespace async_store {

#ifdef USE_IO_URING
static int io_engine_type = photon::fs::ioengine_iouring;
#else
static int io_engine_type = photon::fs::ioengine_libaio;
#endif

struct ArrayStore {
  photon::mutex lock;
  static const size_t DeviceBlockSize = 512;

  const size_t element_size;
  const size_t element_size_aligned;

  size_t size;

  size_t size_in_bytes() { return size * element_size_aligned; }

  std::filesystem::path data_path;
  std::unique_ptr<photon::fs::IFile> file;

  void extend(size_t to) {
    if (to <= size) {
      return;
    }
    file->ftruncate(to * element_size_aligned);
    size = to;
    LOG_INFO("Extend file to `, size `", to, size_in_bytes());
  }

  ArrayStore(size_t element_size, size_t size, std::filesystem::path data_path)
      : element_size(element_size),
        element_size_aligned(align_up(element_size, DeviceBlockSize)),
        data_path(data_path) {
    double write_amplification = element_size_aligned * 1.0 / element_size;
    if (write_amplification > 1.1) {
      LOG_WARN("Warning: write amplification is ` for `", write_amplification, data_path.c_str());
    }

    if (std::filesystem::exists(data_path)) {
      LOG_INFO("Opening `", data_path.c_str());
      this->file = std::unique_ptr<photon::fs::IFile>(
          photon::fs::open_localfile_adaptor(data_path.c_str(), O_RDWR | O_DIRECT, 0664, io_engine_type));
    } else {
      LOG_INFO("Creating `", data_path.c_str());
      this->file = std::unique_ptr<photon::fs::IFile>(
          photon::fs::open_localfile_adaptor(data_path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0664, io_engine_type));
    }
    if (file.get() == nullptr) {
      LOG_ERROR("Error opening file");
    }
    struct stat buf;
    file->fstat(&buf);
    this->size = buf.st_size / element_size_aligned;

    extend(size);
  }

  void read(size_t index, void* buffer) {
    size_t ret = file->pread(buffer, element_size, index * element_size_aligned);
    if (ret != element_size) {
      perror("Error reading from file");
      LOG_ERROR("Error reading to file ` ` `, ret `", buffer, element_size, index * element_size_aligned, ret);
    }
    file->fdatasync();
    file->fsync();
  }
  void write(size_t index, void* buffer) {
    size_t ret = file->pwrite(buffer, element_size, index * element_size_aligned);
    if (ret != element_size) {
      perror("Error writing to file");

      LOG_ERROR("Error writing to file ` ` ` `, ret `", file.get(), buffer, element_size, index * element_size_aligned,
                ret);
    }
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
    while (stop == false) {
      if (auto request = ioQueue.dequeue(); request) {
        if (request->write) {
          request->store->write(request->index, request->data);
        } else {
          request->store->read(request->index, request->data);
        }
        io_cnt += 1;
        io_amount += request->store->element_size_aligned;

        if (request->need_promise) {
          // LOG_INFO("Set Promise `",request->promise);
          request->promise->set();
        }
        // photon::thread_yield();
      } else {
        // 队列为空，避免忙等
        photon::thread_usleep(10);
        // photon::thread_yield();
      }
    }
  }

  void io_perf() {
    LOG_INFO("IO Depth `", IO_DEPTH);
    while (stop == false) {
      photon::thread_sleep(1);
      if (io_cnt == 0) {
        continue;
      }
      LOG_INFO("IO queue remaining: ` , processed ` M.  IO count: ` Kops, ` M/s",
               (ioQueue.enqueue_count - ioQueue.dequeue_count), ioQueue.dequeue_count / 1e6, io_cnt / 1e3,
               io_amount / 1e6);
      io_cnt = 0;
      io_amount = 0;
    }
  }

  void io_dealer() {
    int ev_engine = use_io_uring ? photon::INIT_EVENT_IOURING : photon::INIT_EVENT_EPOLL;
    int io_engine = use_io_uring ? photon::INIT_IO_NONE : photon::INIT_IO_LIBAIO;
    int fs_io_engine = use_io_uring ? photon::fs::ioengine_iouring : photon::fs::ioengine_libaio;
    io_engine_type = fs_io_engine;
    int ret = photon::init(ev_engine, io_engine, photon::PhotonOptions{.libaio_queue_depth = 512});
    if (ret != 0) {
      LOG_ERROR("PHOTON INIT FAILED");
      exit(1);
    }
    DEFER(photon::fini());
    std::vector<photon::join_handle*> handles;

    handles.push_back(photon::thread_enable_join(photon::thread_create11([this]() { io_perf(); })));

    LOG_INFO("Initializing IO Dealer");
    for (int i = 0; i < IO_DEPTH; i++) {
      handles.push_back(photon::thread_enable_join(photon::thread_create11([this]() { queue_consumer(); })));
    }
    for (auto& handle : handles) {
      photon::thread_join(handle);
    }
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
  LOG_INFO("Stopping IO Dealer");
  io_impl->stop = true;
}

}  // namespace async_store
