import torch
import flashinfer
import gc
try:
    from flash_attn import flash_attn_with_kvcache
    print("found flash_attn")
    
except ImportError:
    print("flash_attn not found, flashinfer unit test needed it. If you are using balance serve, ignore this.")

from typing import Union, Optional

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

setup_seed(998244353)

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.bfloat16)
global_dtype=torch.bfloat16
global_device=torch.device("cuda",0)
torch.cuda.set_device(0)
torch.backends.cudnn.enabled =True
torch.backends.cudnn.benchmark = True

class flashInferAttn():
	
	float_workspace_buffer = None
	def __init__(self,
			max_batch_token,
			max_batch_size,
			max_pages,
			device = "cuda:0",
			kv_layout: str = "NHD",
			use_cuda_graph: bool = False,
			) -> None:
		self.device = device
		self.max_batch_token = max_batch_token
		self.kv_layout = kv_layout
		self.use_cuda_graph = use_cuda_graph
		if flashInferAttn.float_workspace_buffer is None:
			flashInferAttn.float_workspace_buffer = torch.empty(max_batch_token * 1024 * 1024, dtype=torch.uint8, device=device)
		self.qo_indptr_buf = torch.empty((max_batch_size+1,), dtype=torch.int32, device=device)
		self.paged_kv_indptr_buf = torch.empty((max_batch_size+1,), dtype=torch.int32, device=device)
		self.paged_kv_indices_buf = torch.empty((max_pages,), dtype=torch.int32, device=device)
		self.paged_kv_last_page_len_buf = torch.empty((max_batch_size,), dtype=torch.int32, device=device)
		self.batch_size_tensor_buf = torch.empty((1,), dtype=torch.int32, device=device)
		self.num_tokens_tensor_buf = torch.empty((1,), dtype=torch.uint32, device=device)
	
		# TODO: custom mask
		self.custom_mask_buf = None
		self.qk_indptr_buf = None
		self.warpper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
			flashInferAttn.float_workspace_buffer,
			self.kv_layout,
			use_cuda_graph=self.use_cuda_graph,
			qo_indptr_buf=self.qo_indptr_buf,
			paged_kv_indptr_buf=self.paged_kv_indptr_buf,
			paged_kv_indices_buf=self.paged_kv_indices_buf,
			paged_kv_last_page_len_buf=self.paged_kv_last_page_len_buf,
			backend = "fa2",
		)

	def plan(self,
		qo_indptr: torch.Tensor,
		paged_kv_indptr: torch.Tensor,
		paged_kv_indices: torch.Tensor,
		paged_kv_last_page_len: torch.Tensor,
		batch_size_tensor: torch.Tensor,
		num_tokens_tensor: torch.Tensor,
		num_qo_heads: int,
		num_kv_heads: int,
		head_dim: int,
		page_size: int,
		causal: bool = True, 
		pos_encoding_mode: str = "NONE",
		q_data_type: Union[str, torch.dtype] = torch.bfloat16,
		kv_data_type: Optional[Union[str, torch.dtype]] = None):
		
		self.batch_size_tensor_buf.copy_(batch_size_tensor, non_blocking=True)
		self.num_tokens_tensor_buf.copy_(num_tokens_tensor, non_blocking=True)
		self.page_size = page_size
		self.warpper.plan(
			qo_indptr,
			paged_kv_indptr,
			paged_kv_indices,
			paged_kv_last_page_len,
			num_qo_heads,
			num_kv_heads,
			head_dim,
			page_size,
			causal = causal,
			pos_encoding_mode = pos_encoding_mode,
			q_data_type = q_data_type,
			kv_data_type = kv_data_type
			)

	def calc_batch_indices(self, ragged_size = None):
		if self.use_cuda_graph:
			self.batch_indices, self.positions = flashinfer.get_batch_indices_positions(
				self.qo_indptr_buf, flashinfer.get_seq_lens(self.paged_kv_indptr_buf, self.paged_kv_last_page_len_buf, self.page_size), self.batch_size_tensor_buf, self.max_batch_token)
		else:
			self.batch_indices, self.positions = flashinfer.get_batch_indices_positions(
				self.warpper._qo_indptr_buf, flashinfer.get_seq_lens(self.warpper._paged_kv_indptr_buf, self.warpper._paged_kv_last_page_len_buf, self.page_size), self.batch_size_tensor_buf, ragged_size)

	def forward(self, q, k_cache, v_cache, k, v):
		if self.use_cuda_graph:
			flashinfer.page.append_paged_kv_cache(k, v, self.batch_indices, self.positions, (k_cache, v_cache), self.paged_kv_indices_buf, self.paged_kv_indptr_buf, self.paged_kv_last_page_len_buf, self.num_tokens_tensor_buf)
			return self.warpper.run(q, (k_cache, v_cache))
		else:
			flashinfer.page.append_paged_kv_cache(k, v, self.batch_indices, self.positions, (k_cache, v_cache), self.warpper._paged_kv_indices_buf, self.warpper._paged_kv_indptr_buf, self.warpper._paged_kv_last_page_len_buf, self.num_tokens_tensor_buf)
			return self.warpper.run(q, (k_cache, v_cache))


def testCudaGraph():
	
	# use max batch to create buffer
	batch_decode = 8
	prefill_chunk = 48
	past_kv_0 = 4090
	past_kv_1 = 4096
	raged_size = prefill_chunk + batch_decode
	num_key_value_heads = 8
	head_dim = 128
	num_attention_heads = 64
	page_size = 256
	num_pages_per_seq = (past_kv_1 + page_size - 1) // page_size
	total_num_pages = (num_pages_per_seq + 1) * (batch_decode + 1) + prefill_chunk // page_size
	attn = flashInferAttn(raged_size, batch_decode+1, total_num_pages, use_cuda_graph=True)

	batch_size_tensor = torch.tensor([batch_decode + 1], device=global_device, dtype=torch.int32)
	
	k_caches = []	
	v_caches = []
	ks = []
	vs = []
	qs = []
	for layer_idx in range(3):
		k_caches.append(torch.randn(total_num_pages, page_size, num_key_value_heads, head_dim, device=global_device, dtype=torch.bfloat16))
		v_caches.append(torch.randn(total_num_pages, page_size, num_key_value_heads, head_dim, device=global_device, dtype=torch.bfloat16))
		ks.append(torch.randn(raged_size, num_key_value_heads, head_dim, device=global_device, dtype=torch.bfloat16))
		vs.append(torch.randn(raged_size, num_key_value_heads, head_dim, device=global_device, dtype=torch.bfloat16))
		qs.append(torch.randn(raged_size, num_attention_heads, head_dim, device=global_device, dtype=torch.bfloat16))
	
	# warmup and capture small batch
	past_kv_0 = 250
	past_kv_1 = 256
	num_pages_per_seq = (past_kv_1 + page_size - 1) // page_size
	total_num_pages = (num_pages_per_seq + 1) * (batch_decode + 1) + prefill_chunk // page_size
	q_indptr = torch.empty((batch_decode + 2,), dtype=torch.int32, device=global_device)
	q_indptr[0] = 0
	q_indptr[1:] = torch.arange(prefill_chunk, prefill_chunk + batch_decode + 1, device=global_device, dtype=torch.int32)
	kv_indptr = torch.arange(0, batch_decode + 2, device=global_device, dtype=torch.int32) * num_pages_per_seq
	kv_indices = torch.arange(0, total_num_pages, device=global_device, dtype=torch.int32)
	kv_last_page_len = torch.empty((batch_decode + 1,), dtype=torch.int32, device=global_device)
	kv_last_page_len[:1+batch_decode//2] = int((past_kv_0 - 1) % page_size + 1)
	kv_last_page_len[1+batch_decode//2:] = int((past_kv_1 - 1) % page_size + 1)

	print(q_indptr)
	print(kv_indptr)
	print(kv_indices)
	print(kv_last_page_len)
	attn.plan(q_indptr,
			kv_indptr,
			kv_indices,
			kv_last_page_len,
			batch_size_tensor,
			num_attention_heads,
			num_key_value_heads,
			head_dim,
			page_size,
			causal = True,
			pos_encoding_mode="NONE",
			q_data_type=torch.bfloat16)

	attn.calc_batch_indices(raged_size)
	for layer_idx in range(3):
		attn.forward(qs[layer_idx], k_caches[layer_idx], v_caches[layer_idx], ks[layer_idx], vs[layer_idx])
		torch.cuda.synchronize()

	outs = []
	g = torch.cuda.CUDAGraph()
	with torch.cuda.graph(g):
		for layer_idx in range(3):
			outs.append(attn.forward(qs[layer_idx], k_caches[layer_idx], v_caches[layer_idx], ks[layer_idx], vs[layer_idx]))
	g.replay()
	
	kv_last_page_len[:1+batch_decode//2] = int(past_kv_0)
	kv_last_page_len[1+batch_decode//2:] = int(past_kv_1)
	for layer_idx in range(3):
		for i in range(batch_decode + 1):
			
			qi = qs[layer_idx][q_indptr[i] : q_indptr[i + 1]]
			o_ref_i = flash_attn_with_kvcache(
				qi.unsqueeze(0),
				k_caches[layer_idx],
				v_caches[layer_idx],
				causal=True,
				block_table=kv_indices[kv_indptr[i]:kv_indptr[i+1]].unsqueeze(0),
				cache_seqlens=torch.tensor([past_kv_0 if i < 1+batch_decode//2 else past_kv_1], device=global_device, dtype=torch.int32)
			)
			o_i = outs[layer_idx][q_indptr[i] : q_indptr[i + 1]]
			print(layer_idx, i)
			torch.testing.assert_close(o_i.unsqueeze(0), o_ref_i, rtol=5e-3, atol=5e-3)

	# run another batch size use capture cuda graph
	past_kv_0 = 4090
	past_kv_1 = 4096
	prefill_chunk = 24
	batch_decode = 4
	num_pages_per_seq = (past_kv_1 + page_size - 1) // page_size
	total_num_pages = (num_pages_per_seq + 1) * (batch_decode + 1) + prefill_chunk // page_size
	batch_size_tensor = torch.tensor([batch_decode + 1], device=global_device, dtype=torch.int32)
	num_tokens_tensor = torch.tensor([batch_decode + prefill_chunk], device=global_device, dtype=torch.int32)

	q_indptr = torch.empty((batch_decode + 2,), dtype=torch.int32, device=global_device)
	q_indptr[0] = 0
	q_indptr[1:] = torch.arange(prefill_chunk, prefill_chunk + batch_decode + 1, device=global_device, dtype=torch.int32)
	kv_indptr = torch.arange(0, batch_decode + 2, device=global_device, dtype=torch.int32) * num_pages_per_seq
	kv_indices = torch.arange(0, total_num_pages, device=global_device, dtype=torch.int32)
	kv_last_page_len = torch.empty((batch_decode + 1,), dtype=torch.int32, device=global_device)
	kv_last_page_len[:1+batch_decode//2] = int((past_kv_0 - 1) % page_size + 1)
	kv_last_page_len[1+batch_decode//2:] = int((past_kv_1 - 1) % page_size + 1)
	attn.plan(q_indptr,
			kv_indptr,
			kv_indices,
			kv_last_page_len,
			batch_size_tensor,
			num_attention_heads,
			num_key_value_heads,
			head_dim,
			page_size,
			causal = True,
			pos_encoding_mode="NONE",
			q_data_type=torch.bfloat16)
	attn.calc_batch_indices(raged_size)
	g.replay()
	
	kv_last_page_len[:1+batch_decode//2] = int(past_kv_0)
	kv_last_page_len[1+batch_decode//2:] = int(past_kv_1)
	for layer_idx in range(3):
		for i in range(batch_decode + 1):
			
			qi = qs[layer_idx][q_indptr[i] : q_indptr[i + 1]]
			o_ref_i = flash_attn_with_kvcache(
				qi.unsqueeze(0),
				k_caches[layer_idx],
				v_caches[layer_idx],
				causal=True,
				block_table=kv_indices[kv_indptr[i]:kv_indptr[i+1]].unsqueeze(0),
				cache_seqlens=torch.tensor([past_kv_0 if i < 1+batch_decode//2 else past_kv_1], device=global_device, dtype=torch.int32)
			)
			o_i = outs[layer_idx][q_indptr[i] : q_indptr[i + 1]]
			print(layer_idx, i)
			torch.testing.assert_close(o_i.unsqueeze(0), o_ref_i, rtol=5e-3, atol=5e-3)
			


def testAttentionFlashInfer(	
	):
	batch_decode = 32
	prefill_chunk = 64
	past_kv_0 = 510
	past_kv_1 = 512
	raged_size = prefill_chunk + batch_decode
	num_key_value_heads = 8
	head_dim = 128
	num_attention_heads = 64
	cases = 1
	page_size = 32
	num_pages_per_seq = (past_kv_1 + page_size - 1) // page_size
	total_num_pages = (num_pages_per_seq + 1) * (batch_decode + 1) + prefill_chunk // page_size
	workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
	qs = []
	kvs = []
	q_indptrs = []
	kv_indptrs = []
	kv_indicess = []
	kv_last_page_lens = []
	wrappers = []
	for case_id in range(cases):
		kvs.append(torch.randn(total_num_pages, 2, page_size, num_key_value_heads, head_dim, device=global_device, dtype=torch.bfloat16))
		qs.append(torch.randn(raged_size, num_attention_heads, head_dim, device=global_device, dtype=torch.bfloat16))
		q_indptr = torch.empty((batch_decode + 2,), dtype=torch.int32, device=global_device)
		q_indptr[0] = 0
		q_indptr[1:] = torch.arange(prefill_chunk, prefill_chunk + batch_decode + 1, device=global_device, dtype=torch.int32)
		q_indptrs.append(q_indptr)
		kv_indptrs.append(torch.arange(0, batch_decode + 2, device=global_device, dtype=torch.int32) * num_pages_per_seq)
		kv_indicess.append(torch.arange(0, total_num_pages, device=global_device, dtype=torch.int32))
		kv_last_page_len = torch.empty((batch_decode + 1,), dtype=torch.int32, device=global_device)
		kv_last_page_len[:1+batch_decode//2] = int((past_kv_0 - 1) % page_size + 1)
		kv_last_page_len[1+batch_decode//2:] = int((past_kv_1 - 1) % page_size + 1)
		kv_last_page_lens.append(kv_last_page_len)
		wrappers.append(flashinfer.BatchPrefillWithPagedKVCacheWrapper(
			workspace_buffer,
			"NHD",
			use_cuda_graph=True,
			qo_indptr_buf=q_indptrs[case_id],
			paged_kv_indptr_buf=kv_indptrs[case_id],
			paged_kv_indices_buf=kv_indicess[case_id],
			paged_kv_last_page_len_buf=kv_last_page_lens[case_id],
		))
		wrappers[case_id].plan(
			q_indptrs[case_id],
			kv_indptrs[case_id],
			kv_indicess[case_id],
			kv_last_page_lens[case_id],
			num_attention_heads,
			num_key_value_heads,
			head_dim,
			page_size,
			causal = True,
			pos_encoding_mode="ROPE_LLAMA",
			q_data_type=torch.bfloat16
		)
					
	def custom_forward(case_id):
		out = wrappers[case_id].run(qs[case_id], kvs[case_id])
	
	custom_forward(0)

# testCudaGraph()
# pass