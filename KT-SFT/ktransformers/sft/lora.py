from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import Trainer
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
from packaging import version
import os
import inspect
import functools
from typing import Union, Any, Dict, List

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import Dataset as TorchDataset

from peft import LoraConfig, TaskType
from datasets import Dataset
from torchviz import make_dot
from tqdm import tqdm
import os, json
from pathlib import Path
from accelerate import Accelerator
if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration
from accelerate import __version__ as accelerate_version
if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
if is_sagemaker_mp_enabled():
    from transformers.trainer_utils import smp_forward_backward

from ktransformers.sft.peft_utils.mapping import get_peft_model

logger = logging.get_logger(__name__)

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

class KTrainer(Trainer):
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
                logger.warning("`non_blocking` is enabled but `dataloader_pin_memory` is not. For best performance, enable both.")
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
            
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader with per_device_train_batch_size
        (no implicit multipliers by number of visible GPUs).
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if is_datasets_available():
            try:
                import datasets
                if isinstance(train_dataset, datasets.Dataset):
                    train_dataset = self._remove_unused_columns(train_dataset, description="training")
                else:
                    data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
            except Exception:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            if self.args.dataloader_num_workers > 0 and self.args.dataloader_prefetch_factor is not None:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dl = DataLoader(train_dataset, **dataloader_params)

        try:
            prepared = self.accelerator.prepare(dl, device_placement=[False])
        except TypeError:
            prepared = self.accelerator.prepare(dl)

        return prepared
    
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:  # type: ignore
                scaled_loss.backward()
        else:
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            if getattr(self.accelerator, "distributed_type", None) and \
               str(self.accelerator.distributed_type) == "DistributedType.DEEPSPEED":
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

        ret = loss.detach()
        if ret.device != self.args.device:
            ret = ret.to(self.args.device, non_blocking=True)

        if os.environ.get("KT_DBG_STEP", "0") == "1" and not hasattr(self, "_kt_dbg_once"):
            try:
                print(f"[KT-DBG] args.device={self.args.device}  loss(before)={loss.device}  loss(return)={ret.device}")
            except Exception:
                pass
            self._kt_dbg_once = True

        return ret

class SFTJsonListDataset(TorchDataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_len: int = 512):
        super().__init__()
        with open(path, "r", encoding="utf-8") as f:
            self.samples: List[Dict] = json.load(f)
        self.tok = tokenizer
        self.max_len = max_len

    @staticmethod
    def build_example(ins: str, inp: str, out: str) -> Dict[str, str]:
        ins = (ins or "").strip()
        inp = (inp or "").strip()
        out = (out or "").strip()
        prompt = (ins + inp) if ins else inp
        return {"prompt": prompt, "response": out}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        eg = self.build_example(rec.get("instruction", ""), rec.get("input", ""), rec.get("output", ""))

        prompt_ids = self.tok(
            eg["prompt"],
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        response_ids = self.tok(
            eg["response"],
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        eos_id = self.tok.eos_token_id
        input_ids = prompt_ids + response_ids + ([eos_id] if eos_id is not None else [])
        input_ids = input_ids[: self.max_len]

        labels = [-100] * min(len(prompt_ids), self.max_len)
        tail = input_ids[len(labels):]
        labels = labels + tail
        labels = labels[: self.max_len]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

def lora_and_load_adapter(model, tokenizer, sft_data_path, save_adapter_path):
    
    Path(save_adapter_path).mkdir(parents=True, exist_ok=True)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", # FOR DeepSeek-V2-Lite
            "q_a_proj", # FOR DeepSeek-V3&R1
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "shared_experts.gate_proj",
            "shared_experts.up_proj",
            "shared_experts.down_proj",
        ],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    train_dataset = SFTJsonListDataset(sft_data_path, tokenizer, max_len=512)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=save_adapter_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        # max_steps=30, # TODO: FOR TEST, will override any value given in num_train_epochs
        learning_rate=1e-4,
        fp16=False,
        logging_steps=10,
        save_steps=200,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )
    
    debug_path = os.path.join(save_adapter_path, "model_infra_debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump({"model": str(model)}, f, ensure_ascii=False, indent=2)
    
    # output = model(input_ids=torch.tensor([[1,2,3]], dtype=torch.int32, device="cuda:0"))
    # loss = output.logits.mean()
        
    # dot = make_dot(loss, params=dict(model.named_parameters()))
    # dot.render("KT_compute_cpuinfer_moe_model_graph", format="svg")
    
    trainer = KTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    # model.gradient_checkpointing_enable()
    # if hasattr(model, "enable_input_require_grads"):
    #     model.enable_input_require_grads()
    
    trainer.train()

def inject_lora_layer(model, use_adapter_path):

    cfg_path = os.path.join(use_adapter_path, "adapter_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    task_type_str = (data.get("task_type") or "CAUSAL_LM").upper()
    bias = data.get("bias", "none")
    if bias in (None, False):
        bias = "none"
    if data.get("lora_bias") is True and bias == "none":
        bias = "lora_only"

    tmods = data.get("target_modules")
    if isinstance(tmods, str):
        tmods = [m.strip() for m in tmods.split(",") if m.strip()]

    mts = data.get("modules_to_save", None)
    if isinstance(mts, str):
        mts = [m.strip() for m in mts.split(",") if m.strip()]

    rank_pattern = data.get("rank_pattern") or None
    alpha_pattern = data.get("alpha_pattern") or None

    lora_config = LoraConfig(
        r=data.get("r", 8),
        lora_alpha=data.get("lora_alpha", 32),
        lora_dropout=float(data.get("lora_dropout", 0.0)),
        bias=bias,
        task_type=TaskType[task_type_str],
        target_modules=tmods,
        modules_to_save=mts,
        init_lora_weights=bool(data.get("init_lora_weights", True)),
        inference_mode=bool(data.get("inference_mode", True)),
        use_rslora=bool(data.get("use_rslora", False)),
        use_dora=bool(data.get("use_dora", False)),
    )
    print(f"lora_config:{lora_config.__dict__}")
    
    # model = inject_adapter_in_model(lora_config, model)
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.eval()