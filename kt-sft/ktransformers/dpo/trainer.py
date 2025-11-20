from trl import DPOTrainer
import os
from packaging import version
import inspect
import functools
from typing import Union, Any, Dict, List

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import Dataset as TorchDataset


from transformers.training_args import OptimizerNames

from transformers.trainer_utils import seed_worker
from transformers.utils import (
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
    is_torch_hpu_available,
    is_accelerate_available,
    is_apex_available,
    logging,
)

from ktransformers.util.trainer_utils import KAccelerator


if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration
from accelerate import __version__ as accelerate_version
if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
if is_sagemaker_mp_enabled():
    from transformers.trainer_utils import smp_forward_backward

logger = logging.get_logger(__name__)

class KTDporainer(DPOTrainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # only save LoRA adapter, including adapter_config.json
        self.model.save_pretrained(output_dir)

    def _move_model_to_device(self, model, device):
        print("[KTrainer] Due to the placement feature in KTransformers, skip moving model to", device)
        return model

    def _wrap_model(self, model, training=True, dataloader=None):
        self.model_wrapped = model
        return model

    def create_accelerator_and_postprocess(self):
        # We explicitly don't rely on the `Accelerator` to do gradient accumulation
        grad_acc_kwargs = {}
        if is_accelerate_available("0.28.0") and self.args.accelerator_config.gradient_accumulation_kwargs is not None:
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        # check if num_steps is attempted to be passed in gradient_accumulation_kwargs
        if "num_steps" in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                # raise because we do not know which setting is intended.
                raise ValueError(
                    "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`"
                    "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`."
                )
            else:
                self.args.gradient_accumulation_steps = grad_acc_kwargs["num_steps"]

        accelerator_config = self.args.accelerator_config.to_dict()

        if is_accelerate_available("0.28.0"):
            # Extract dataloader config params from accelerator config
            dataloader_params = ["split_batches", "dispatch_batches", "even_batches", "use_seedable_sampler"]
            dataloader_config_dict = {param: accelerator_config.pop(param) for param in dataloader_params if param in accelerator_config}
            if DataLoaderConfiguration is None:
                raise ImportError("Your accelerate does not provide DataLoaderConfiguration but Trainer expects it.")
            dataloader_config = DataLoaderConfiguration(**dataloader_config_dict)
            if is_accelerate_available("1.1.0"):
                dataloader_config.data_seed = self.args.data_seed
        else:
            dataloader_config = None

        non_blocking = accelerator_config.pop("non_blocking", False)
        if not is_accelerate_available("0.30.0"):
            if non_blocking:
                raise ImportError(
                    "`non_blocking` is only supported in accelerate v0.30.0 and above. Please upgrade accelerate to use this feature."
                )
        else:
            if non_blocking and not self.args.dataloader_pin_memory:
                logger.warning \
                    ("`non_blocking` is enabled but `dataloader_pin_memory` is not. For best performance, enable both.")
            if dataloader_config is not None:
                dataloader_config.non_blocking = non_blocking

        accelerator_config.pop("gradient_accumulation_kwargs", None)

        args = {
            "deepspeed_plugin": self.args.deepspeed_plugin,
            "device_placement": False,
        }

        if is_accelerate_available("0.28.0"):
            args["dataloader_config"] = dataloader_config
        else:
            args.update(accelerator_config)

        if getattr(self.args, "tp_size", 1) > 1:
            self.is_tp_enabled = True
            if version.parse(accelerate_version) > version.parse("1.3.0") and TorchTensorParallelPlugin is not None:
                args["torch_tp_plugin"] = TorchTensorParallelPlugin(tp_size=self.args.tp_size)
            else:
                raise ValueError("Requires accelerate>1.3.0 to use Tensor Parallelism.")

        self.accelerator = KAccelerator(**args)

        try:
            self.accelerator.state.device_ids = [0]
            self.accelerator.state.num_processes = 1
            self.accelerator.state.num_gpus = 1
        except Exception:
            pass

        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        if "use_gather_object" in inspect.signature(self.gather_function).parameters.keys():
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.is_tp_enabled = getattr(self.accelerator.state, "torch_tp_plugin", None) is not None
        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            for param in ["limit_all_gathers", "activation_checkpointing"]:
                setattr(fsdp_plugin, param, self.args.fsdp_config.get(param, getattr(fsdp_plugin, param)))
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                    "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                    "when using FSDP."
                )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

        # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
        if (
                self.args.save_only_model
                and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
                and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

        # `auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3
        if (
                self.is_deepspeed_enabled
                and self.accelerator.state.deepspeed_plugin.zero_stage == 3
                and self.args.auto_find_batch_size
        ):
            raise ValueError(
                "`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3. Please consider using Zero-2, Zero-1, or FSDP"
            )
        if (
                self.args.save_only_model
                and self.is_fsdp_enabled
                and "SHARDED_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)
        ):
            raise ValueError("save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'")

        if dataloader_config is not None:
            dataloader_config.split_batches = False
            dataloader_config.dispatch_batches = False
            dataloader_config.even_batches = False

    def post_training_step(self, loss):
        if loss.device != self.args.device:
            ret = loss.to(self.args.device, non_blocking=True)
        return loss

    def training_step(
            self,
            model: torch.nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            num_items_in_batch=None
    ) -> torch.Tensor:

        ret = super().training_step(model, inputs,  num_items_in_batch=num_items_in_batch)
        ret = self.post_training_step(ret)
        return ret