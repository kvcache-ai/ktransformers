'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import torch
from typing import Dict

class CUDAGraphRunner:

    def __init__(self):
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        model,
        cur_token,
        position_ids,
        cache_position,
        past_key_values,
        **kwargs,
    ) -> None:
        assert self.graph is None
        # Capture the graph.
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        #self.graph.enable_debug_mode()
        self.model = model
        inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to("cuda")
        with torch.cuda.graph(self.graph):
            logits=model(inputs_embeds=inputs_embeds, 
                         position_ids=position_ids,
                         cache_position=cache_position,
                         past_key_values=past_key_values,
                         **kwargs)[0]
        past_key_values.change_seq_length(-1)
        torch.cuda.synchronize()
        #self.graph.debug_dump("cuda_graph_hooked.dot")

        # Save the input and output buffers.
        self.input_buffers = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
        self.output_buffers = {"logits": logits}
        return

    def forward(
        self,
        cur_token,
        position_ids,
        cache_position,
    ) -> torch.Tensor:
        # Copy the input tensors to the input buffers.
        inputs_embeds = self.model.model.embed_tokens(cur_token.to("cpu"))
        self.input_buffers["inputs_embeds"].copy_(inputs_embeds)
        self.input_buffers["position_ids"].copy_(position_ids)
        self.input_buffers["cache_position"].copy_(cache_position)

        # Run the graph.
        #print("begin replay")
        #time.sleep(1)
        self.graph.replay()
        torch.cuda.synchronize()
        # Return the output tensor.
        return self.output_buffers["logits"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
