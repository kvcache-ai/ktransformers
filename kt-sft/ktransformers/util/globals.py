import os

class _GlobalConfig:
    def __init__(self):
        self._config = {
            "mod": 'infer', # infer or sft
        }

    def get(self, key, default=None):
        return self._config.get(key, default)

    def set(self, key, value):
        self._config[key] = value

    def update(self, **kwargs):
        self._config.update(kwargs)

    def all(self):
        return self._config

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value):
        self._config[key] = value

GLOBAL_CONFIG = _GlobalConfig()
