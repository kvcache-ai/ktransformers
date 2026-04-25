'''
Date: 2024-11-14 12:23:45
LastEditors: Xie Weiyu ervinxie@qq.com
LastEditTime: 2024-11-25 08:59:23
'''
import logging
import torch
from torch import nn
from transformers import GenerationConfig

from flashinfer.sampling import (
	min_p_sampling_from_probs,
	top_k_renorm_probs,
	top_k_top_p_sampling_from_logits,
	top_p_renorm_probs,
)

logger = logging.getLogger(__name__)

class SamplingOptions():
	# Batched sampling params
	temperatures: torch.Tensor
	top_ps: torch.Tensor
	top_ks: torch.Tensor
	min_ps: torch.Tensor

	# All requests use greedy sampling
	is_all_greedy: bool

	# Dispatch in CUDA graph
	need_min_p_sampling: bool
	
	def __init__(self, bsz = 1, device = torch.device('cuda'), pretrained_config:GenerationConfig = None, temperatures: torch.Tensor = None, top_ps: torch.Tensor = None):
		if pretrained_config is None and temperatures is None:
			self.temperatures = torch.full((bsz, 1), 0, device=device, dtype=torch.float32)
			self.top_ps = torch.ones((bsz, 1), device=device, dtype=torch.float32)
			self.top_ks = torch.ones((bsz, 1), device=device, dtype=torch.float32)
			self.need_min_p_sampling = False
			self.is_all_greedy = True
		else:
			if temperatures is not None:
				self.temperatures = temperatures.unsqueeze(-1)
			else:
				self.temperatures = torch.full((bsz, 1), pretrained_config.temperature, device=device, dtype=torch.float32)
			
			if top_ps is not None:
				self.top_ps = top_ps.unsqueeze(-1)
			else:	
				self.top_ps = torch.full((bsz, 1), pretrained_config.top_p, device=device, dtype=torch.float32)
			self.top_ks = torch.full((bsz, 1), pretrained_config.top_k, device=device, dtype=torch.float32)
			self.need_min_p_sampling = False
			self.is_all_greedy = False

class Sampler(nn.Module):
	def __init__(self):
		super().__init__()
	
	def forward(
		self,
		logits: torch.Tensor,
		sampling_config: SamplingOptions = None,
	):
		if sampling_config == None:
			sampling_config = SamplingOptions()

		logits = logits.contiguous()
		origin_logits = logits.clone()
		if sampling_config.is_all_greedy:
			# Use torch.argmax if all requests use greedy sampling
			probs = logits
			batch_next_token_ids = torch.argmax(logits, -1)
		else:
			# Post process logits
			logits.div_(sampling_config.temperatures)
			max_top_k_round, batch_size = 32, logits.shape[0]
			if sampling_config.need_min_p_sampling:
				probs = torch.softmax(logits, dim=-1)
				logits = None
				del logits
				probs = top_k_renorm_probs(probs, sampling_config.top_ks)
				probs = top_p_renorm_probs(probs, sampling_config.top_ps)
				batch_next_token_ids = min_p_sampling_from_probs(
					probs, sampling_config.min_ps
				)
				temperature_0_idx = torch.where(sampling_config.temperatures == 0)[0]
				batch_next_token_ids[temperature_0_idx] = torch.argmax(origin_logits[temperature_0_idx], -1).to(torch.int32)
			else:
				# TODO: use different kernel when don't need top_k or top_p
				# @TODO get probs
				probs = logits
				batch_next_token_ids = top_k_top_p_sampling_from_logits(
					logits,
					sampling_config.top_ks,
					sampling_config.top_ps,
					filter_apply_order="joint",
				)
				temperature_0_idx = torch.where(sampling_config.temperatures == 0)[0]
				batch_next_token_ids[temperature_0_idx] = torch.argmax(origin_logits[temperature_0_idx], -1).to(torch.int32)
			
		return batch_next_token_ids.to(torch.int32), probs