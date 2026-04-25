import torch
from collections import OrderedDict
from torch.nn.modules import Module

_ORIG_MODULE_INIT = Module.__init__

def _patched_module_init(self, *args, **kwargs):
    torch._C._log_api_usage_once("python.nn_module")

    if self.call_super_init is False and bool(kwargs):
        raise TypeError(
            f"{type(self).__name__}.__init__() got an unexpected keyword argument '{next(iter(kwargs))}'"
        )
    if self.call_super_init is False and bool(args):
        raise TypeError(
            f"{type(self).__name__}.__init__() takes 1 positional argument but {len(args) + 1} were given"
        )

    object.__setattr__(self, "training", True)
    object.__setattr__(self, "_parameters", {})
    object.__setattr__(self, "_buffers", {})
    object.__setattr__(self, "_non_persistent_buffers_set", set())
    object.__setattr__(self, "_backward_pre_hooks", OrderedDict())
    object.__setattr__(self, "_backward_hooks", OrderedDict())
    object.__setattr__(self, "_is_full_backward_hook", None)
    object.__setattr__(self, "_forward_hooks", OrderedDict())
    object.__setattr__(self, "_forward_hooks_with_kwargs", OrderedDict())
    object.__setattr__(self, "_forward_hooks_always_called", OrderedDict())
    object.__setattr__(self, "_forward_pre_hooks", OrderedDict())
    object.__setattr__(self, "_forward_pre_hooks_with_kwargs", OrderedDict())
    object.__setattr__(self, "_state_dict_hooks", OrderedDict())
    object.__setattr__(self, "_state_dict_pre_hooks", OrderedDict())
    object.__setattr__(self, "_load_state_dict_pre_hooks", OrderedDict())
    object.__setattr__(self, "_load_state_dict_post_hooks", OrderedDict())

    if not (hasattr(self, "orig_module") and isinstance(self.orig_module, torch.nn.modules.linear.Linear)):
        object.__setattr__(self, "_modules", {})

    if self.call_super_init:
        object.__init__(self)

def install_patch():
    Module.__init__ = _patched_module_init

def restore_patch():
    Module.__init__ = _ORIG_MODULE_INIT

install_patch()