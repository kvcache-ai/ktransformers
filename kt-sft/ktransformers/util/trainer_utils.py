
from accelerate import Accelerator
import torch.nn as nn

class KAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("device_placement", False)
        super().__init__(*args, **kwargs)

    def prepare_model(self, model, *args, **kwargs):
        return model

    def prepare(self, *args, **kwargs):
        prepped = []
        for obj in args:
            if isinstance(obj, nn.Module):
                prepped.append(self.prepare_model(obj, **kwargs))
            else:
                prepped.append(super().prepare(obj, **kwargs))
        return tuple(prepped) if len(prepped) > 1 else prepped[0]