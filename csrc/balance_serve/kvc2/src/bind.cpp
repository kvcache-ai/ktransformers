// #include <pybind11/functional.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <memory>
// #include <thread>
// #include <vector>
// #include "kvc2.h"
// #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
// #define FMT_HEADER_ONLY
// #include "spdlog/spdlog.h"
// #include "utils/arithmetic.hpp"

// namespace py = pybind11;

// PYBIND11_MODULE(kvc2_ext, m) {
//   // Bind KVC2Config struct
//   py::class_<kvc2::KVC2Config>(m, "KVC2Config")
//       .def(py::init<>())
//       .def_readwrite("path", &kvc2::KVC2Config::path)
//       .def_readwrite("block_length", &kvc2::KVC2Config::num_token_per_page)
//       .def_readwrite("memory_pool_size", &kvc2::KVC2Config::memory_pool_size)
//       .def_readwrite("evict_count", &kvc2::KVC2Config::evict_count);

//   // Bind CacheInfo struct
//   py::class_<kvc2::CacheInfo>(m, "CacheInfo")
//       .def(py::init<>())
//       .def_readwrite("model_name", &kvc2::CacheInfo::model_name)
//       .def_readwrite("is_key_cache", &kvc2::CacheInfo::is_key_cache)
//       .def_readwrite("quant_type", &kvc2::CacheInfo::quant_type)
//       .def("hidden_layer_count", &kvc2::CacheInfo::hidden_layer_count)
//       .def("path", &kvc2::CacheInfo::path, py::arg("which_layer") = std::nullopt)
//       .def("__eq__", &kvc2::CacheInfo::operator==)
//       .def("element_size", &kvc2::CacheInfo::element_size)
//       .def("hash_value", &kvc2::CacheInfo::hash_value);

//   // Bind KVC2HandleInterface class
//   py::class_<kvc2::KVC2HandleInterface, std::shared_ptr<kvc2::KVC2HandleInterface>>(m, "KVC2HandleInterface")
//       .def("matched_length", &kvc2::SingleCacheHandleInterface::matched_length)
//       .def("handle_data", &kvc2::KVC2HandleInterface::handle_data);

//   // Bind KVC2Interface class
//   py::class_<kvc2::KVC2Interface, std::shared_ptr<kvc2::KVC2Interface>>(m, "KVC2Interface")
//       .def("start_io_thread", [](kvc2::KVC2Interface& self) { self.start_io_thread(); })
//       .def("stop_io_thread", &kvc2::KVC2Interface::stop_io_thread)
//       .def("load", &kvc2::KVC2Interface::load)
//       .def("save", &kvc2::KVC2Interface::save)
//       .def("raw_insert", &kvc2::KVC2Interface::raw_insert)
//       .def("raw_read", &kvc2::KVC2Interface::raw_read)
//       .def("lookup", &kvc2::KVC2Interface::lookup);

//   // Bind create_kvc2 function
//   m.def("create_kvc2", &kvc2::create_kvc2, py::arg("config"));
// }