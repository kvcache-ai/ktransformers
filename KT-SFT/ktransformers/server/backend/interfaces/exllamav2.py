import sys, os
from typing import AsyncIterator, Dict, Tuple

import torch

from ..args import ConfigArgs, default_args

from ..base import BackendInterfaceBase, ThreadContext
from ktransformers.server.schemas.assistants.runs import RunObject


from ..args import *

class ExllamaThreadContext(ThreadContext):
    def __init__(self, run: RunObject, args: ConfigArgs = default_args) -> None:
        super().__init__(run,args)
        
    def get_interface(self):
        return 

    def get_local_messages(self):
        raise NotImplementedError




class ExllamaInterface(BackendInterfaceBase):
    
    def __init__(self, args: ConfigArgs = ...):
        raise NotImplementedError
    
    def tokenize_prompt(self, prompt: str) -> torch.Tensor:
        raise NotImplementedError
    
    async def inference(self,local_messages,request_unique_id:Optional[str])->AsyncIterator:
        raise NotImplementedError
    



