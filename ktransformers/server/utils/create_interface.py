#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : qiyuxinlin
Date         : 2024-07-25 11:50:16
Version      : 1.0.0
LastEditors  : qiyuxinlin 
LastEditTime : 2024-07-25 12:54:48
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
from ktransformers.server.config.config import Config
from ktransformers.server.backend.args import ConfigArgs
from ktransformers.server.backend.context_manager import ThreadContextManager
from ktransformers.server.backend.interfaces.exllamav2 import ExllamaInterface
from ktransformers.server.backend.interfaces.transformers import TransformersInterface
from ktransformers.server.backend.interfaces.ktransformers import KTransformersInterface
def create_interface(config: Config, default_args: ConfigArgs):
    if config.backend_type=='transformers':
        from ktransformers.server.backend.interfaces.transformers import  TransformersInterface as BackendInterface
    elif config.backend_type == 'exllamav2':
        from ktransformers.server.backend.interfaces.exllamav2 import  ExllamaInterface as BackendInterface
    elif config.backend_type == 'ktransformers':
        from ktransformers.server.backend.interfaces.ktransformers import  KTransformersInterface as BackendInterface
    else:
        raise NotImplementedError(f'{config.backend_type} not implemented')
    GlobalInterface.interface = BackendInterface(default_args)
    GlobalContextManager.context_manager = ThreadContextManager(GlobalInterface.interface)

class GlobalContextManager:
    context_manager: ThreadContextManager
class GlobalInterface:
    interface:  TransformersInterface | KTransformersInterface | ExllamaInterface
    
def get_thread_context_manager() -> ThreadContextManager:
    return GlobalContextManager.context_manager
def get_interface() -> TransformersInterface | KTransformersInterface | ExllamaInterface:
    return GlobalInterface.interface