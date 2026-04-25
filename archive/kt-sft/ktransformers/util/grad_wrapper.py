from functools import wraps
import torch, yaml, pathlib

import os, sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_dir)

from ktransformers.util.globals import GLOBAL_CONFIG

# print(f"start_sit: {GLOBAL_CONFIG._config['mod']}")

def maybe_no_grad(_func=None):
    # print(f"maybe_sit: {GLOBAL_CONFIG._config['mod']}")
    
    def decorator(func):
        # print(f"decorate_sit: {GLOBAL_CONFIG._config['mod']}")
        def wrapper(*args, **kwargs):
            # print(f"wrap_sit: {GLOBAL_CONFIG._config['mod']}")
            if GLOBAL_CONFIG._config["mod"] == "sft":
                return func(*args, **kwargs)
            elif GLOBAL_CONFIG._config["mod"] == "infer":
                with torch.no_grad():
                    return func(*args, **kwargs)
        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
