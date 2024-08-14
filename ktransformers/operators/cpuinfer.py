import sys, os
from typing import Any
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
import cpuinfer_ext
from ktransformers.server.config.config import Config
class CPUInfer:
    cpu_infer = None
    def __init__(self, cpu_infer:int = Config().cpu_infer):
        if CPUInfer.cpu_infer is None:
            CPUInfer.cpu_infer = cpuinfer_ext.CPUInfer(cpu_infer)
        
    def __getattribute__(self, __name: str) -> Any:
        return CPUInfer.cpu_infer.__getattribute__(__name)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        return CPUInfer.cpu_infer.__setattr__(__name, __value)