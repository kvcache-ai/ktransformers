#!/usr/bin/env python
# coding=utf-8
"""
Description  :
Author       : unicornchan
Date         : 2024-06-11 16:35:42
Version      : 1.0.0
LastEditors  : WuHao
LastEditTime : 2024-08-12 06:31:14
"""
import os
import shutil
import yaml

from ktransformers.server.config.singleton import Singleton
from typing import Optional


class Config(metaclass=Singleton):
    """Singleton pattern Config class, used to get all configurations."""

    CONFIG_FILE_NAME = "config.yaml"

    @staticmethod
    def load() -> dict:
        """load config file

        Returns:
            dict: all configs
        """
        base_path: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_yaml: str = os.path.join(base_path, "configs", Config.CONFIG_FILE_NAME)

        user_path: str = os.path.expanduser("~")
        localstore_path: str = os.path.join(user_path, ".ktransformers")
        config_path: str = os.path.join(localstore_path, Config.CONFIG_FILE_NAME)
        if not os.path.exists(config_yaml):
            print(f"Can't find config file, {config_yaml}")
            exit(-1)
        if not os.path.exists(localstore_path):
            os.mkdir(localstore_path)
        if not os.path.exists(config_path):
            shutil.copyfile(config_yaml, config_path)
        with open(config_path, "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
        return config

    @staticmethod
    def to_path(path: str) -> str:
        """
        process file path
        """
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        real_path = path if os.path.isabs(path) else os.path.join(base_path, path)
        return real_path

    def __init__(self):
        cfg = Config.load()
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.user_path: str = os.path.expanduser("~")
        self.localstore_path: str = os.path.join(self.user_path, ".ktransformers")
        # log configs
        self.log_dir = os.path.join(self.base_path, Config.to_path(cfg["log"]["dir"]))
        self.log_file = cfg["log"]["file"]
        self.log_level = cfg["log"]["level"]
        self.backup_count = cfg["log"]["backup_count"]

        # server configs
        self.server: dict = cfg.get("server", {})
        self.server_ip = self.server.get("ip", "0.0.0.0")
        self.server_port = self.server.get("port", 9016)
        self.api_key = self.server.get("api_key", "")

        # db configs
        self.db_configs: dict = cfg.get("db", {})
        self.db_type = self.db_configs.get("type", "")
        self.db_host = os.path.join(self.base_path, self.db_configs.get("host", ""))
        self.db_port = self.db_configs.get("port", "")
        self.db_name = self.db_configs.get("database", "")
        self.db_pool_size = self.db_configs.get("pool_size")
        self.db_database = self.db_configs.get("database", "")

        # user config
        self.user_config: dict = cfg.get("user", {})
        self.user_secret_key = self.user_config.get("secret_key", "")
        self.user_algorithm = self.user_config.get("algorithm", "")
        self.user_force_think = self.user_config.get("force_think", False)

        # model config
        self.model: dict = cfg.get("model", {})
        self.backend_type: str = self.model.get("type", "transformers")
        self.model_dir: str = self.model.get("path", "")
        # to make sure it consistent with previous version
        self.model_path: str = self.model_dir
        self.model_name: str = self.model.get("name", "")
        self.model_device: str = self.model.get("device", "cuda:0")
        self.gguf_path: Optional[str] = self.model.get("gguf_path", None)
        self.use_cuda_graph = self.model.get("use_cuda_graph", True)
        self.trust_remote_code = self.model.get("trust_remote_code", True)
        # self.model_cache_lens = self.model.get("cache_lens")
        self.optimize_config_path: Optional[str] = self.model.get(
            "optimize_config_path", None
        )
        self.paged = self.model.get("paged", True)

        self.total_context = self.model.get("total_context", 2**18)
        self.max_batch_size = self.model.get("max_batch_size", 20 if self.paged else 1)
        self.chunk_prefill_size = self.model.get("chunk_prefill_size", 8192)
        
        self.max_new_tokens = self.model.get("max_new_tokens", 2000)
        self.json_mode = self.model.get("json_mode", False)
        self.healing = self.model.get("healing", False)
        self.ban_strings: Optional[list] = self.model.get("ban_strings", None)
        self.gpu_split: Optional[str] = self.model.get("gpu_split", None)
        self.length: Optional[int] = self.model.get("length", None)
        self.rope_scale: Optional[float] = self.model.get("rope_scale", None)
        self.rope_alpha: Optional[float] = self.model.get("rope_alpha", None)
        self.no_flash_attn = self.model.get("no_flash_attn", False)
        self.low_mem = self.model.get("low_mem", False)
        self.experts_per_token: Optional[int] = self.model.get("experts_per_token", None)
        self.load_q4 = self.model.get("load_q4", False)
        self.fast_safetensors = self.model.get("fast_safetensors", False)
        self.draft_model_dir: Optional[str] = self.model.get("draft_model_dir", None)
        self.no_draft_scale = self.model.get("no_draft_scale", False)
        self.modes = self.model.get("modes", False)
        self.mode = self.model.get("mode", "llama")
        self.username = self.model.get("username", "User")
        self.botname = self.model.get("botname", "Chatbort")
        self.system_prompt: Optional[str] = self.model.get("system_prompt", None)
        self.temperature = self.model.get("temperature", 0.95)
        self.smoothing_factor = self.model.get("smoothing_factor", 0.0)
        self.dynamic_temperature: Optional[str] = self.model.get("dynamic_temperature", None)
        self.top_k = self.model.get("top_k", 50)
        self.top_p = self.model.get("top_p", 0.8)
        self.top_a = self.model.get("top_a", 0.0)
        self.skew = self.model.get("skew", 0.0)
        self.typical = self.model.get("typical", 0.0)
        self.repetition_penalty = self.model.get("repetition_penalty", 1.01)
        self.frequency_penalty = self.model.get("frequency_penalty", 0.0)
        self.presence_penalty = self.model.get("presence_penalty", 0.0)
        self.max_response_tokens = self.model.get("max_response_tokens", 300)
        self.response_chunk = self.model.get("response_chunk", 250)
        self.no_code_formatting = self.model.get("no_code_formatting", False)
        self.cache_8bit = self.model.get("cache_8bit", False)
        self.cache_q4 = self.model.get("cache_q4", True)
        self.ngram_decoding = self.model.get("ngram_decoding", False)
        self.print_timings = self.model.get("print_timings", False)
        self.amnesia = self.model.get("amnesia", False)
        self.batch_size = self.model.get("batch_size", 1)
        self.cache_lens = self.model.get("cache_lens", 4096)
        self.device = self.model.get("device", "cuda:2")

        # web config
        self.web: dict = cfg.get("web", {})
        self.web_cross_domain: bool = self.web.get("open_cross_domain", True)
        self.mount_web: bool = self.web.get("mount", False)

        self.ext: dict = cfg.get("ext", {})
        self.cpu_infer = self.ext.get("cpu_infer", 10)

        # file config
        self.local_store_configs: dict = cfg.get("local_store", {})
        self.file_upload_dir: str = os.path.join(
            self.localstore_path, self.local_store_configs.get("file_upload_dir", "")
        )
        self.assistant_store_dir: str = os.path.join(
            self.localstore_path, self.local_store_configs.get("assistant_store_dir", "")
        )

        # long context config
        self.long_context_config: dict = cfg.get("long_context", {})
        self.chunk_size = self.long_context_config.get("chunk_size", 4096)
        self.max_seq_len = self.long_context_config.get("max_seq_len", 32000)
        self.block_size = self.long_context_config.get("block_size", 128)
        self.local_windows_len = self.long_context_config.get("local_windows_len", 4096)
        self.second_select_num = self.long_context_config.get("second_select_num", 32)
        self.anchor_type = self.long_context_config.get("anchor_type", "DYNAMIC")
        self.kv_type = self.long_context_config.get("kv_type", "FP16")
        self.dense_layer_num = self.long_context_config.get("dense_layer_num", 2)
        self.anchor_num = self.long_context_config.get("anchor_num", 1)
        self.preselect_block = self.long_context_config.get("preselect_block", True)
        self.head_select_mode = self.long_context_config.get("head_select_mode", "SHARED")
        self.preselect_block_count = self.long_context_config.get("preselect_block_count", 32)
        self.layer_step = self.long_context_config.get("layer_step", 1)
        self.token_step = self.long_context_config.get("token_step", 100)

        # local chat
        self.local_chat_config: dict = cfg.get("local_chat", {})
        self.prompt_file = self.local_chat_config.get("prompt_file", None)
