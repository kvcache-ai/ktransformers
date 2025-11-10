'''
Description :
Author      : Boxin Zhang
Version     : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
'''
from typing import Dict

import threading
import torch
import torch_npu


class NPUGraphRunner:

    def __init__(self, deviceId):
        torch.npu.set_compile_mode(jit_compile=False)
        self.deviceId = deviceId
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}
        self.past_key_value = None

    def init(self, batch_size, seq_length):
        self.graph = torch.npu.NPUGraph()
        self.main_stream = torch_npu.npu.Stream(device=self.deviceId)
        self.share_experts_stream = torch_npu.npu.Stream(device=self.deviceId)
        self.logits = torch.zeros((batch_size, seq_length, 7168), dtype=torch.float16).to(self.deviceId)  # deepseekV3 hidden_size
        self.workspace = None
        self.model_capture = True
        torch_npu.npu._subscribe_report(self.main_stream)

    def destroy(self):
        torch_npu.npu._unsubscribe_report(self.main_stream)
        del self.graph
        destory_runner(self.deviceId)

    def capture(
            self,
            model,
            cur_token,
            position_ids,
            cache_position,
            past_key_values,
            main_device,
            **kwargs,
    ) -> None:
        inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to(main_device)
        with torch.no_grad():
            with torch.npu.graph(self.graph, stream=self.main_stream, auto_dispatch_capture=True):
                logits = model(inputs_embeds=inputs_embeds,
                            position_ids=position_ids,
                            cache_position=cache_position,
                            past_key_values=past_key_values,
                            is_prefill=False,
                            **kwargs)
        self.input_buffers = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
        self.output_buffers = {
            "logits": logits,
        }

    def forward(
            self,
            inputs_embeds,
            position_ids,
            cache_position,
    ) -> torch.Tensor:
        thread = threading.Thread(target=self.graph.update, kwargs={"cpu_update_input": [{"actual_seq_lengths_kv": self.past_key_value.position}]})
        thread.start()

        self.input_buffers["inputs_embeds"].copy_(inputs_embeds)
        self.input_buffers["position_ids"].copy_(position_ids)
        self.input_buffers["cache_position"].copy_(cache_position)
        torch_npu.npu.synchronize()
        with torch_npu.npu.stream(self.main_stream):
            # Run the graph.
            self.graph.replay()
        thread.join()

        # Return the output tensor.
        return self.output_buffers["logits"]

    def launch_callback(self, func, data, block, stream):
        torch_npu.npu._launch_host_func(stream, func, data)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

runner_dict = dict()

def check_runner(deviceId: int):
    runner = runner_dict.get(deviceId)
    if runner is None:
        return True
    else:
        return False

def destory_runner(deviceId: int):
    # print("the new NPUGraphRunner and deviceId is ", deviceId)
    runner = runner_dict.get(deviceId)
    if runner is not None:
        runner_dict[deviceId] = None

def get_or_create_runner(deviceId: int):
    runner = runner_dict.get(deviceId)
    if runner is None:
        runner = NPUGraphRunner(deviceId)
        runner_dict[deviceId] = runner
    return runner