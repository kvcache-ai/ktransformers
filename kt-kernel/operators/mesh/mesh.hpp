/**
 * @file mesh.hpp
 * @brief MESH 公共头文件
 *
 * MESH 是 KTransformers 的专家权重驻留管理插件。
 * 只需 include 本文件即可使用全部 MESH 接口。
 *
 * 设计约束：
 * - 所有 MESH 代码集中在本目录，对原版 KT 仅做最小化 hook 修改
 * - mesh_enabled=false 时所有 hook 编译期优化为空操作
 * - 不修改 AMX 内核、NUMA 张量并行、CUDA Graph 调度或 FlashInfer
 */
#pragma once

#include "mesh_config.hpp"
#include "mesh_slot_pool.hpp"
#include "mesh_io_uring.hpp"
#include "mesh_scheduler.hpp"
#include "mesh_eviction.hpp"
#include "mesh_prefill.hpp"
#include "mesh_decode.hpp"
#include "mesh_handoff.hpp"
#include "mesh_residency.hpp"
#include "mesh_hook.hpp"
