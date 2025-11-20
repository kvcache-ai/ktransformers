/**
 * @Description  :
 * @Author       : chenht2022, Jianwei Dong
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : Jianwei Dong
 * @LastEditTime : 2024-08-26 22:47:06
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
// Python bindings
#include "cpu_backend/cpuinfer.h"
#if !defined(KTRANSFORMERS_USE_ROCM) && !defined(KTRANSFORMERS_USE_XPU)
#include "device_launch_parameters.h"
#endif
#include "llamafile/flags.h"
#include "operators/kvcache/kvcache.h"
#include "operators/llamafile/linear.h"
#include "operators/llamafile/mlp.h"
#include "operators/llamafile/moe.h"
#include "operators/llamafile/sft_moe.h"

#if defined(__x86_64__) && defined(__HAS_AVX512F__) && defined(__HAS_AMX__)
#include "operators/amx/moe.hpp"
#include "operators/amx/sft_moe.hpp"
#endif

#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <cstdint>
#include <iostream>
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals;

// Binding functions for the KVCache class
class KVCacheBindings {
  public:
    class AttnBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            const ggml_fp16_t *q_in;
            ggml_fp16_t *output;
            float *attn_lse;
            int layer_idx;
            int generate_token_idx;
            int q_len;
            int batch_size;
            int max_block_num;
            int *block_table;
            int *cache_seqlens;
            int pick_block_num;
            int init_block_num;
            int local_block_num;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                &KVCache::attn, args_->kv_cache, args_->q_in, args_->output,
                args_->attn_lse, args_->layer_idx, args_->generate_token_idx,
                args_->q_len, args_->batch_size, args_->max_block_num,
                args_->block_table, args_->cache_seqlens, args_->pick_block_num,
                args_->init_block_num, args_->local_block_num);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t q_in, intptr_t output,
                           intptr_t attn_lse, int layer_idx,
                           int generate_token_idx, int q_len, int batch_size,
                           int max_block_num, intptr_t block_table,
                           intptr_t cache_seqlens, int pick_block_num,
                           int init_block_num, int local_block_num) {
            Args *args = new Args{nullptr,
                                  &kv_cache,
                                  (const ggml_fp16_t *)q_in,
                                  (ggml_fp16_t *)output,
                                  (float *)attn_lse,
                                  layer_idx,
                                  generate_token_idx,
                                  q_len,
                                  batch_size,
                                  max_block_num,
                                  (int *)block_table,
                                  (int *)cache_seqlens,
                                  pick_block_num,
                                  init_block_num,
                                  local_block_num};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

    class GetAllKVCacheOneLayerBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            int layer_id;
            ggml_fp16_t *k_in;
            ggml_fp16_t *v_in;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&KVCache::get_all_kvcache_one_layer,
                                     args_->kv_cache, args_->layer_id,
                                     args_->k_in, args_->v_in);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t k_in, intptr_t v_in,
                           int layer_id) {
            Args *args = new Args{nullptr, &kv_cache, layer_id,
                                  (ggml_fp16_t *)k_in, (ggml_fp16_t *)v_in};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

    class GetAndUpdateKVCacheFp16Bindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            ggml_fp16_t *k_in;
            ggml_fp16_t *v_in;
            int layer_id;
            int *block_table;
            int batch_size;
            int max_block_num;
            int *cache_seqlens;
            int q_len;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&KVCache::get_and_update_kvcache_fp16,
                                     args_->kv_cache, args_->k_in, args_->v_in,
                                     args_->layer_id, args_->block_table,
                                     args_->batch_size, args_->max_block_num,
                                     args_->cache_seqlens, args_->q_len);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t k_in, intptr_t v_in,
                           int layer_id, intptr_t block_table, int batch_size,
                           int max_block_num, intptr_t cache_seqlens,
                           int q_len) {
            Args *args = new Args{nullptr,
                                  &kv_cache,
                                  (ggml_fp16_t *)k_in,
                                  (ggml_fp16_t *)v_in,
                                  layer_id,
                                  (int *)block_table,
                                  batch_size,
                                  max_block_num,
                                  (int *)cache_seqlens,
                                  q_len};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class GetKVCacheFp16Bindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            ggml_fp16_t *k_in;
            ggml_fp16_t *v_in;
            int layer_id;
            int *block_table;
            int batch_size;
            int max_block_num;
            int *cache_seqlens;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                &KVCache::get_kvcache_fp16, args_->kv_cache, args_->k_in,
                args_->v_in, args_->layer_id, args_->block_table,
                args_->batch_size, args_->max_block_num, args_->cache_seqlens);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t k_in, intptr_t v_in,
                           int layer_id, intptr_t block_table, int batch_size,
                           int max_block_num, intptr_t cache_seqlens) {
            Args *args = new Args{nullptr,
                                  &kv_cache,
                                  (ggml_fp16_t *)k_in,
                                  (ggml_fp16_t *)v_in,
                                  layer_id,
                                  (int *)block_table,
                                  batch_size,
                                  max_block_num,
                                  (int *)cache_seqlens};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

    class UpdateKVCacheFp16Bindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            ggml_fp16_t *k_in;
            ggml_fp16_t *v_in;
            int layer_id;
            int *block_table;
            int batch_size;
            int max_block_num;
            int *cache_seqlens;
            int q_len;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&KVCache::update_kvcache_fp16,
                                     args_->kv_cache, args_->k_in, args_->v_in,
                                     args_->layer_id, args_->block_table,
                                     args_->batch_size, args_->max_block_num,
                                     args_->cache_seqlens, args_->q_len);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t k_in, intptr_t v_in,
                           int layer_id, intptr_t block_table, int batch_size,
                           int max_block_num, intptr_t cache_seqlens,
                           int q_len) {
            Args *args = new Args{nullptr,
                                  &kv_cache,
                                  (ggml_fp16_t *)k_in,
                                  (ggml_fp16_t *)v_in,
                                  layer_id,
                                  (int *)block_table,
                                  batch_size,
                                  max_block_num,
                                  (int *)cache_seqlens,
                                  q_len};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

    class UpdateImportanceBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            const ggml_fp16_t *importance;
            int layer_id;
            int *block_table;
            int batch_size;
            int max_block_num;
            int *offset;
            int width;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                &KVCache::update_importance, args_->kv_cache, args_->importance,
                args_->layer_id, args_->block_table, args_->batch_size,
                args_->max_block_num, args_->offset, args_->width);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t importance, int layer_id,
                           intptr_t block_table, int batch_size,
                           int max_block_num, intptr_t offset, int width) {
            Args *args = new Args{nullptr,
                                  &kv_cache,
                                  (const ggml_fp16_t *)importance,
                                  layer_id,
                                  (int *)block_table,
                                  batch_size,
                                  max_block_num,
                                  (int *)offset,
                                  width};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

    class AttnWithKVCacheBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            const ggml_fp16_t *q_in;
            const ggml_fp16_t *k_in;
            const ggml_fp16_t *v_in;
            ggml_fp16_t *output;
            float *attn_lse;
            int layer_idx;
            int generate_token_idx;
            int q_len;
            int batch_size;
            int max_block_num;
            int *block_table;
            int *cache_seqlens;
            int topk;
            int local;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                &KVCache::attn_with_kvcache, args_->kv_cache, args_->q_in,
                args_->k_in, args_->v_in, args_->output, args_->attn_lse,
                args_->layer_idx, args_->generate_token_idx, args_->q_len,
                args_->batch_size, args_->max_block_num, args_->block_table,
                args_->cache_seqlens, args_->topk, args_->local);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t q_in, intptr_t k_in,
                           intptr_t v_in, intptr_t output, intptr_t attn_lse,
                           int layer_idx, int generate_token_idx, int q_len,
                           int batch_size, int max_block_num,
                           intptr_t block_table, intptr_t cache_seqlens,
                           int topk, int local) {
            Args *args = new Args{nullptr,
                                  &kv_cache,
                                  (const ggml_fp16_t *)q_in,
                                  (const ggml_fp16_t *)k_in,
                                  (const ggml_fp16_t *)v_in,
                                  (ggml_fp16_t *)output,
                                  (float *)attn_lse,
                                  layer_idx,
                                  generate_token_idx,
                                  q_len,
                                  batch_size,
                                  max_block_num,
                                  (int *)block_table,
                                  (int *)cache_seqlens,
                                  topk,
                                  local};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

    class ClearImportanceAllLayersBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            int *block_table;
            int *cache_seqlens;
            int batch_size;
            int max_block_num;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&KVCache::clear_importance_all_layers,
                                     args_->kv_cache, args_->block_table,
                                     args_->cache_seqlens, args_->batch_size,
                                     args_->max_block_num);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t block_table,
                           intptr_t cache_seqlens, int batch_size,
                           int max_block_num) {
            Args *args = new Args{nullptr,
                                  &kv_cache,
                                  (int *)block_table,
                                  (int *)cache_seqlens,
                                  batch_size,
                                  max_block_num};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

    class CalcAnchorAllLayersBindinds {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            int *block_table;
            int *cache_seqlens;
            int batch_size;
            int max_block_num;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&KVCache::calc_anchor_all_layers,
                                     args_->kv_cache, args_->block_table,
                                     args_->cache_seqlens, args_->batch_size,
                                     args_->max_block_num);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t block_table,
                           intptr_t cache_seqlens, int batch_size,
                           int max_block_num) {
            Args *args = new Args{nullptr,
                                  &kv_cache,
                                  (int *)block_table,
                                  (int *)cache_seqlens,
                                  batch_size,
                                  max_block_num};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

    class LoadKVCacheBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            std::string tensor_file_path;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&KVCache::load_kvcache, args_->kv_cache,
                                     args_->tensor_file_path);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, std::string tensor_file_path) {
            Args *args =
                new Args{nullptr, &kv_cache, (std::string)tensor_file_path};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class DumpKVCacheBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            KVCache *kv_cache;
            int *block_table;
            int cache_total_len;
            std::string tensor_file_path;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&KVCache::dump_kvcache, args_->kv_cache,
                                     args_->block_table, args_->cache_total_len,
                                     args_->tensor_file_path);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(KVCache &kv_cache, intptr_t block_table,
                           int cache_total_len, std::string tensor_file_path) {
            Args *args =
                new Args{nullptr, &kv_cache, (int *)block_table,
                         cache_total_len, (std::string)tensor_file_path};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

class LinearBindings {
  public:
    class WarmUpBindinds {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            Linear *linear;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&Linear::warm_up, args_->linear);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(Linear &linear) {
            Args *args = new Args{nullptr, &linear};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            Linear *linear;
            int qlen;
            const void *input;
            void *output;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&Linear::forward, args_->linear,
                                     args_->qlen, args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(Linear &linear, int qlen, intptr_t input,
                           intptr_t output) {
            Args *args = new Args{nullptr, &linear, qlen, (const void *)input,
                                  (void *)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

class MLPBindings {
  public:
    class WarmUpBindinds {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            MLP *mlp;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&MLP::warm_up, args_->mlp);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(MLP &mlp) {
            Args *args = new Args{nullptr, &mlp};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            MLP *mlp;
            int qlen;
            const void *input;
            void *output;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&MLP::forward, args_->mlp, args_->qlen,
                                     args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(MLP &mlp, int qlen, intptr_t input,
                           intptr_t output) {
            Args *args = new Args{nullptr, &mlp, qlen, (const void *)input,
                                  (void *)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

class MOEBindings {
  public:
    class WarmUpBindinds {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            MOE *moe;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&MOE::warm_up, args_->moe);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(MOE &moe) {
            Args *args = new Args{nullptr, &moe};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            MOE *moe;
            int qlen;
            int k;
            const uint64_t *expert_ids;
            const float *weights;
            const void *input;
            void *output;
            int *batch_size_tensor;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                &MOE::forward, args_->moe, args_->qlen, args_->k,
                args_->expert_ids, args_->weights, args_->input, args_->output, args_->batch_size_tensor);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(MOE &moe, int qlen, int k, intptr_t expert_ids,
                           intptr_t weights, intptr_t input, intptr_t output, intptr_t batch_size_tensor) {
            Args *args = new Args{nullptr,
                                  &moe,
                                  qlen,
                                  k,
                                  (const uint64_t *)expert_ids,
                                  (const float *)weights,
                                  (const void *)input,
                                  (void *)output,
                                  (int *)batch_size_tensor};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

namespace {
	inline void sft_moe_forward_wrapper(
			SFT_MOE& self,
			int qlen, int k,
			const uint64_t* expert_ids,
			const float*     weights,
			const void*      input,
			void*            output,
			Backend*         backend)
	{
		self.ensure_fwd_cache(qlen, k);
		self.forward(qlen, k, expert_ids, weights,
					input, output,
					backend,
					self.fwd_cache_ptr());
	}

	inline void sft_moe_backward_wrapper(
			SFT_MOE& self,
			int layer_idx,
			int qlen, int k,
			const uint64_t* expert_ids,
			const float*     weights,
			const void*      input,
			const void*      grad_output,
			void*            grad_input,
			Backend*         backend)
	{
		self.backward(layer_idx, qlen, k, expert_ids, weights,
					input, grad_output, grad_input,
					backend,
					self.fwd_cache_ptr());
	}
}

class SFT_MOEBindings {
  public:
    class WarmUpBindinds {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            SFT_MOE *moe;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&SFT_MOE::warm_up, args_->moe);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(SFT_MOE &moe) {
            Args *args = new Args{nullptr, &moe};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            SFT_MOE *moe;
            int qlen;
            int k;
            const uint64_t *expert_ids;
            const float *weights;
            const void *input;
            void *output;
        };
        // static void inner(void *args) {
        //     Args *args_ = (Args *)args;
        //     args_->cpuinfer->enqueue(
        //         &SFT_MOE::forward, args_->moe, args_->qlen, args_->k,
        //         args_->expert_ids, args_->weights, args_->input, args_->output);
        // }
		static void inner(void *args) {
			Args *args_ = static_cast<Args *>(args);
			args_->cpuinfer->enqueue(
				&sft_moe_forward_wrapper,   // 使用包装函数
				args_->moe, 
				args_->qlen, args_->k,
				args_->expert_ids,
				args_->weights,
				args_->input,
				args_->output);
		}
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(SFT_MOE &moe, int qlen, int k, intptr_t expert_ids,
                           intptr_t weights, intptr_t input, intptr_t output) {
            Args *args = new Args{nullptr,
                                  &moe,
                                  qlen,
                                  k,
                                  (const uint64_t *)expert_ids,
                                  (const float *)weights,
                                  (const void *)input,
                                  (void *)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
	// FIXME: need fit the args setting with the backward of MoE
	class BackwardBindings {
    public:
		struct Args {
			CPUInfer* cpuinfer;
			SFT_MOE* moe;
			int layer_idx;
			int qlen;
			int k;
			const uint64_t* expert_ids;
			const float* weights;
			const void* input;
			const void* grad_output;
			void* grad_input;
		};

        // static void inner(void* args) {
        //     Args* args_ = static_cast<Args*>(args);
        //     args_->cpuinfer->enqueue(&SFT_MOE::backward, args_->moe, 
        //         args_->qlen, args_->k,
        //         args_->expert_ids, args_->weights,
		// 		args_->input,
		// 		args_->grad_output,
		// 		args_->grad_input);
        // }

		static void inner(void *args) {
			Args *args_ = static_cast<Args *>(args);
			args_->cpuinfer->enqueue(
				&sft_moe_backward_wrapper,  // 使用包装函数
				args_->moe,
				args_->layer_idx,
				args_->qlen, args_->k,
				args_->expert_ids,
				args_->weights,
				args_->input,
				args_->grad_output,
				args_->grad_input);
		}

        static std::pair<intptr_t, intptr_t> cpuinfer_interface(
            SFT_MOE& moe, int layer_idx, int qlen, int k, 
            intptr_t expert_ids, intptr_t weights,
    		intptr_t input,
            intptr_t grad_output, intptr_t grad_input) {
            
            Args* args = new Args{
				nullptr, &moe, layer_idx, qlen, k,
				reinterpret_cast<const uint64_t*>(expert_ids),
				reinterpret_cast<const float*>(weights),
				reinterpret_cast<const void*>(input), 
				reinterpret_cast<const void*>(grad_output),
				reinterpret_cast<void*>(grad_input)
			};
            return std::make_pair(
                reinterpret_cast<intptr_t>(&inner),
                reinterpret_cast<intptr_t>(args));
        }
    };
};

#if defined(__x86_64__) && defined(__HAS_AVX512F__) && defined(__HAS_AMX__)
template<class T>
class AMX_MOEBindings {
  public:
    class WarmUpBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            AMX_MOE<T> *moe;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&AMX_MOE<T>::warm_up, args_->moe);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(AMX_MOE<T> &moe) {
            Args *args = new Args{nullptr, &moe};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class LoadWeightsBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            AMX_MOE<T> *moe;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&AMX_MOE<T>::load_weights, args_->moe);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(AMX_MOE<T> &moe) {
            Args *args = new Args{nullptr, &moe};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            AMX_MOE<T> *moe;
            int qlen;
            int k;
            const uint64_t *expert_ids;
            const float *weights;
            const void *input;
            void *output;
            int *batch_size_tensor;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                &AMX_MOE<T>::forward, args_->moe, args_->qlen, args_->k,
                args_->expert_ids, args_->weights, args_->input, args_->output, args_->batch_size_tensor);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(AMX_MOE<T> &moe, int qlen, int k, intptr_t expert_ids,
                        intptr_t weights, intptr_t input, intptr_t output, intptr_t batch_size_tensor) {
            Args *args = new Args{nullptr,
                                &moe,
                                qlen,
                                k,
                                (const uint64_t *)expert_ids,
                                (const float *)weights,
                                (const void *)input,
                                (void *)output,
                                (int *)batch_size_tensor};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};
#endif

#if defined(__x86_64__) && defined(__HAS_AVX512F__) && defined(__HAS_AMX__)
template<class T>
class SFT_AMX_MOEBindings {
  public:
    class WarmUpBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            SFT_AMX_MOE<T> *moe;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&SFT_AMX_MOE<T>::warm_up, args_->moe);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(SFT_AMX_MOE<T> &moe) {
            Args *args = new Args{nullptr, &moe};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class LoadWeightsBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            SFT_AMX_MOE<T> *moe;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&SFT_AMX_MOE<T>::load_weights, args_->moe);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(SFT_AMX_MOE<T> &moe) {
            Args *args = new Args{nullptr, &moe};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            SFT_AMX_MOE<T> *moe;
            int qlen;
            int k;
            const uint64_t *expert_ids;
            const float *weights;
            const void *input;
            void *output;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                &SFT_AMX_MOE<T>::forward, args_->moe, args_->qlen, args_->k,
                args_->expert_ids, args_->weights, args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(SFT_AMX_MOE<T> &moe, int qlen, int k, intptr_t expert_ids,
                        intptr_t weights, intptr_t input, intptr_t output) {
            Args *args = new Args{nullptr,
                                &moe,
                                qlen,
                                k,
                                (const uint64_t *)expert_ids,
                                (const float *)weights,
                                (const void *)input,
                                (void *)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };

	class BackwardBindings {
    public:
		struct Args {
			CPUInfer* cpuinfer;
			SFT_AMX_MOE<T> *moe;
			int qlen;
			int k;
			const uint64_t* expert_ids;
			const float* weights;
            const void* input;
			const void* output_grad;
			void* input_grad;
		};

		static void inner(void *args) {
			Args *args_ = static_cast<Args *>(args);
			args_->cpuinfer->enqueue(
				&SFT_AMX_MOE<T>::backward,
				args_->moe,
				args_->qlen, args_->k,
				args_->expert_ids,
				args_->weights,
                args_->input,
				args_->output_grad,
				args_->input_grad);
		}

        static std::pair<intptr_t, intptr_t> cpuinfer_interface(
            SFT_AMX_MOE<T> &moe, int qlen, int k, 
            intptr_t expert_ids, intptr_t weights,
            intptr_t input,
            intptr_t output_grad, intptr_t input_grad) {
            
            Args* args = new Args{
				nullptr, &moe, qlen, k,
				(const uint64_t*)expert_ids,
				(const float*)weights,
                (const void*)input,
				(const void*)output_grad,
				(void*)input_grad
			};
            return std::make_pair(
                (intptr_t)&inner,
                (intptr_t)args);
        }
    };
};
#endif

PYBIND11_MODULE(cpuinfer_ext, m) {
    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<int>())
        .def("submit", &CPUInfer::submit)
        .def("submit_with_cuda_stream", &CPUInfer::submit_with_cuda_stream)
        .def("sync", &CPUInfer::sync)
        .def("sync_with_cuda_stream", &CPUInfer::sync_with_cuda_stream);

    auto linear_module = m.def_submodule("linear");
    py::class_<LinearConfig>(linear_module, "LinearConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride,
                         int group_max_len, intptr_t proj, int proj_type,
                         int hidden_type) {
            return LinearConfig(hidden_size, intermediate_size, stride,
                                group_max_len, (void *)proj,
                                (ggml_type)proj_type, (ggml_type)hidden_type);
        }));
    py::class_<Linear>(linear_module, "Linear")
        .def(py::init<LinearConfig>())
        .def("warm_up", &LinearBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &LinearBindings::ForwardBindings::cpuinfer_interface);

    auto mlp_module = m.def_submodule("mlp");
    py::class_<MLPConfig>(mlp_module, "MLPConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride,
                         int group_max_len, intptr_t gate_proj,
                         intptr_t up_proj, intptr_t down_proj, int gate_type,
                         int up_type, int down_type, int hidden_type) {
            return MLPConfig(hidden_size, intermediate_size, stride,
                             group_max_len, (void *)gate_proj, (void *)up_proj,
                             (void *)down_proj, (ggml_type)gate_type,
                             (ggml_type)up_type, (ggml_type)down_type,
                             (ggml_type)hidden_type);
        }));
    py::class_<MLP>(mlp_module, "MLP")
        .def(py::init<MLPConfig>())
        .def("warm_up", &MLPBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &MLPBindings::ForwardBindings::cpuinfer_interface);

    auto moe_module = m.def_submodule("moe");
    py::class_<MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size,
                         int intermediate_size, int stride, int group_min_len,
                         int group_max_len, intptr_t gate_proj,
                         intptr_t up_proj, intptr_t down_proj, int gate_type,
                         int up_type, int down_type, int hidden_type) {
            return MOEConfig(expert_num, routed_expert_num, hidden_size,
                             intermediate_size, stride, group_min_len,
                             group_max_len, (void *)gate_proj, (void *)up_proj,
                             (void *)down_proj, (ggml_type)gate_type,
                             (ggml_type)up_type, (ggml_type)down_type,
                             (ggml_type)hidden_type);
        }));
    py::class_<MOE>(moe_module, "MOE")
        .def(py::init<MOEConfig>())
        .def("warm_up", &MOEBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &MOEBindings::ForwardBindings::cpuinfer_interface);

    auto sft_moe_module = m.def_submodule("sft_moe");
    py::class_<SFT_MOEConfig>(sft_moe_module, "SFT_MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size,
                         int intermediate_size, int stride, int group_min_len,
                         int group_max_len, intptr_t gate_proj,
                         intptr_t up_proj, intptr_t down_proj, int gate_type,
                         int up_type, int down_type, int hidden_type) {
            return SFT_MOEConfig(expert_num, routed_expert_num, hidden_size,
                             intermediate_size, stride, group_min_len,
                             group_max_len, (void *)gate_proj, (void *)up_proj,
                             (void *)down_proj, (ggml_type)gate_type,
                             (ggml_type)up_type, (ggml_type)down_type,
                             (ggml_type)hidden_type);
        }));
    py::class_<SFT_MOE>(sft_moe_module, "SFT_MOE")
        .def(py::init<SFT_MOEConfig>())
        .def("warm_up", &SFT_MOEBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &SFT_MOEBindings::ForwardBindings::cpuinfer_interface)
		.def("backward", &SFT_MOEBindings::BackwardBindings::cpuinfer_interface);

    #if defined(__x86_64__) && defined(__HAS_AVX512F__) && defined(__HAS_AMX__)
    py::class_<AMX_MOEConfig>(moe_module, "AMX_MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size,
                         int intermediate_size,
                         int max_len, intptr_t gate_proj,
                         intptr_t up_proj, intptr_t down_proj) {
            return AMX_MOEConfig(expert_num, routed_expert_num, hidden_size,
                                 intermediate_size, 
                                 max_len, (void *)gate_proj,
                                 (void *)up_proj, (void *)down_proj);
        }));

    py::class_<AMX_MOE<amx::GemmKernel224BF>>(moe_module, "AMXBF16_MOE")
        .def(py::init<AMX_MOEConfig>())
        .def("warm_up", &AMX_MOEBindings<amx::GemmKernel224BF>::WarmUpBindings::cpuinfer_interface)
        .def("load_weights", &AMX_MOEBindings<amx::GemmKernel224BF>::LoadWeightsBindings::cpuinfer_interface)
        .def("forward", &AMX_MOEBindings<amx::GemmKernel224BF>::ForwardBindings::cpuinfer_interface);
    py::class_<AMX_MOE<amx::GemmKernel224Int8>>(moe_module, "AMXInt8_MOE")
        .def(py::init<AMX_MOEConfig>())
        .def("warm_up", &AMX_MOEBindings<amx::GemmKernel224Int8>::WarmUpBindings::cpuinfer_interface)
        .def("load_weights", &AMX_MOEBindings<amx::GemmKernel224Int8>::LoadWeightsBindings::cpuinfer_interface)
        .def("forward", &AMX_MOEBindings<amx::GemmKernel224Int8>::ForwardBindings::cpuinfer_interface);

    #endif

	#if defined(__x86_64__) && defined(__HAS_AVX512F__) && defined(__HAS_AMX__)
    py::class_<SFT_AMX_MOEConfig>(sft_moe_module, "SFT_AMX_MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size,
                         int intermediate_size,
                         int max_len, intptr_t gate_proj,
                         intptr_t up_proj, intptr_t down_proj) {
            return SFT_AMX_MOEConfig(expert_num, routed_expert_num, hidden_size,
                                 intermediate_size, 
                                 max_len, (void *)gate_proj,
                                 (void *)up_proj, (void *)down_proj);
        }));

    py::class_<SFT_AMX_MOE<amx::GemmKernel224BF>>(sft_moe_module, "SFT_AMXBF16_MOE")
        .def(py::init<SFT_AMX_MOEConfig>())
        .def("warm_up", &SFT_AMX_MOEBindings<amx::GemmKernel224BF>::WarmUpBindings::cpuinfer_interface)
        .def("load_weights", &SFT_AMX_MOEBindings<amx::GemmKernel224BF>::LoadWeightsBindings::cpuinfer_interface)
        .def("forward", &SFT_AMX_MOEBindings<amx::GemmKernel224BF>::ForwardBindings::cpuinfer_interface)
		.def("backward", &SFT_AMX_MOEBindings<amx::GemmKernel224BF>::BackwardBindings::cpuinfer_interface);

    py::class_<SFT_AMX_MOE<amx::GemmKernel224Int8>>(sft_moe_module, "SFT_AMXInt8_MOE")
        .def(py::init<SFT_AMX_MOEConfig>())
        .def("warm_up", &SFT_AMX_MOEBindings<amx::GemmKernel224Int8>::WarmUpBindings::cpuinfer_interface)
        .def("load_weights", &SFT_AMX_MOEBindings<amx::GemmKernel224Int8>::LoadWeightsBindings::cpuinfer_interface)
        .def("forward", &SFT_AMX_MOEBindings<amx::GemmKernel224Int8>::ForwardBindings::cpuinfer_interface)
		.def("backward", &SFT_AMX_MOEBindings<amx::GemmKernel224Int8>::BackwardBindings::cpuinfer_interface);

    #endif

    auto kvcache_module = m.def_submodule("kvcache");

    py::enum_<AnchorType>(kvcache_module, "AnchorType")
        .value("FIXED", AnchorType::FIXED_ANCHOR)
        .value("DYNAMIC", AnchorType::DYNAMIC)
        .value("QUEST", AnchorType::QUEST)
        .value("BLOCK_MAX", AnchorType::BLOCK_MAX)
        .value("BLOCK_MEAN", AnchorType::BLOCK_MEAN);
    py::enum_<ggml_type>(kvcache_module, "ggml_type")
        .value("FP16", ggml_type::GGML_TYPE_F16)
        .value("FP32", ggml_type::GGML_TYPE_F32)
        .value("Q4_0", ggml_type::GGML_TYPE_Q4_0)
        .value("Q8_0", ggml_type::GGML_TYPE_Q8_0);
    py::enum_<RetrievalType>(kvcache_module, "RetrievalType")
        .value("LAYER", RetrievalType::LAYER)
        .value("KVHEAD", RetrievalType::KVHEAD)
        .value("QHEAD", RetrievalType::QHEAD);

    py::class_<KVCacheConfig>(kvcache_module, "KVCacheConfig")
        .def(py::init<int, int, int, int, int, int, AnchorType, ggml_type,
                      RetrievalType, int, int, int, int, int, int>())
        .def_readwrite("layer_num", &KVCacheConfig::layer_num)
        .def_readwrite("kv_head_num", &KVCacheConfig::kv_head_num)
        .def_readwrite("q_head_num", &KVCacheConfig::q_head_num)
        .def_readwrite("head_dim", &KVCacheConfig::head_dim)
        .def_readwrite("block_len", &KVCacheConfig::block_len)
        .def_readwrite("anchor_num", &KVCacheConfig::anchor_num)
        .def_readwrite("anchor_type", &KVCacheConfig::anchor_type)
        .def_readwrite("kv_type", &KVCacheConfig::kv_type)
        .def_readwrite("retrieval_type", &KVCacheConfig::retrieval_type)
        .def_readwrite("layer_step", &KVCacheConfig::layer_step)
        .def_readwrite("token_step", &KVCacheConfig::token_step)
        .def_readwrite("layer_offset", &KVCacheConfig::layer_offset)
        .def_readwrite("max_block_num", &KVCacheConfig::max_block_num)
        .def_readwrite("max_batch_size", &KVCacheConfig::max_batch_size)
        .def_readwrite("max_thread_num", &KVCacheConfig::max_thread_num);
    py::class_<KVCache>(kvcache_module, "KVCache")
        .def(py::init<KVCacheConfig>())
        .def("get_cache_total_len", &KVCache::get_cache_total_len)
        .def("update_cache_total_len",
             [](KVCache &kvcache, int cache_total_len) {
                 kvcache.update_cache_total_len(cache_total_len);
             })
        .def("attn", &KVCacheBindings::AttnBindings::cpuinfer_interface)
        .def(
            "get_all_kvcache_one_layer",
            &KVCacheBindings::GetAllKVCacheOneLayerBindings::cpuinfer_interface)
        .def("get_and_update_kvcache_fp16",
             &KVCacheBindings::GetAndUpdateKVCacheFp16Bindings::
                 cpuinfer_interface)
        .def("get_kvcache_fp16",
             &KVCacheBindings::GetKVCacheFp16Bindings::cpuinfer_interface)
        .def("update_kvcache_fp16",
             &KVCacheBindings::UpdateKVCacheFp16Bindings::cpuinfer_interface)
        .def("update_importance",
             &KVCacheBindings::UpdateImportanceBindings::cpuinfer_interface)
        .def("attn_with_kvcache",
             &KVCacheBindings::AttnWithKVCacheBindings::cpuinfer_interface)
        .def("clear_importance_all_layers",
             &KVCacheBindings::ClearImportanceAllLayersBindings::
                 cpuinfer_interface)
        .def("calc_anchor_all_layers",
             &KVCacheBindings::CalcAnchorAllLayersBindinds::cpuinfer_interface);
}