#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : unicornchan
Date         : 2024-06-11 16:35:42
Version      : 1.0.0
LastEditors  : chenxl 
LastEditTime : 2024-07-27 01:55:42
'''
import os
import yaml

from ktransformers.server.config.singleton import Singleton


class Config(metaclass=Singleton):
    """Singleton pattern Config class, used to get all configurations.
    """
    CONFIG_FILE_NAME = "config.yaml"

    @staticmethod
    def load() -> dict:
        """load config file

        Returns:
            dict: all configs
        """
        base_path: str = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__)))
        config_yaml: str = os.path.join(
            base_path, "configs", Config.CONFIG_FILE_NAME)
        if not os.path.exists(config_yaml):
            print(f"Can't find config file, {config_yaml}")
            exit(-1)
        with open(config_yaml, 'r', encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
        return config

    @staticmethod
    def to_path(path: str) -> str:
        """
        process file path
        """
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        real_path = path if os.path.isabs(
            path) else os.path.join(base_path, path)
        return real_path

    def __init__(self):
        cfg = Config.load()
        self.base_path = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__)))
        # log configs
        self.log_dir = os.path.join(self.base_path, Config.to_path(cfg["log"]["dir"]))
        self.log_file = cfg["log"]["file"]
        self.log_level = cfg["log"]["level"]
        self.backup_count = cfg["log"]["backup_count"]

        # server configs
        self.server: dict = cfg.get("server",{})
        self.server_ip = self.server.get("ip", "0.0.0.0")
        self.server_port = self.server.get("port", 9016)

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
        
        # model config
        self.model:dict = cfg.get("model", {})
        self.backend_type: str = self.model.get("type", "transformers")
        self.model_path: str = self.model.get("path", "")
        self.model_name: str = self.model.get("name", "")
        self.model_device: str = self.model.get("device", "cuda:0")
        self.gguf_path: str = self.model.get("gguf_path", "")
        
        # web config
        self.web: dict = cfg.get("web", {})
        self.web_cross_domain: bool = self.web.get("open_cross_domain", True)
        self.mount_web: bool = self.web.get("mount", False)
        
        self.ext: dict = cfg.get("ext", {})
        self.cpu_infer = self.ext.get("cpu_infer", 10)
