
from collections.abc import Mapping
from typing import Union, Optional
from accelerate import Accelerator
import torch.nn as nn
import torch

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


def nested_detach(
    tensors: Union["torch.Tensor", list["torch.Tensor"], tuple["torch.Tensor"], dict[str, "torch.Tensor"]],
    clone: bool = False,
):
    r"""Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t, clone=clone) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t, clone=clone) for k, t in tensors.items()})

    if isinstance(tensors, torch.Tensor):
        if clone:
            return tensors.detach().clone()
        else:
            return tensors.detach()
    else:
        return tensors

def get_batch_logps(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    label_pad_token_id: int = -100,
    ld_alpha: Optional[float] = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    r"""Compute the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.

    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    valid_length = loss_mask.sum(-1)
    if ld_alpha is not None:
        num_examples = labels.shape[0] // 2
        chosen_lengths = valid_length[:num_examples]
        rejected_lengths = valid_length[num_examples:]
        min_lengths = torch.min(chosen_lengths, rejected_lengths)
        start_positions = torch.argmax(loss_mask.int(), dim=1)
        public_lengths = start_positions + torch.cat([min_lengths, min_lengths], dim=0)

        seq_len = labels.shape[-1]
        position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

        ld_mask = position_ids < public_lengths.unsqueeze(1)
        front_mask = (ld_mask * loss_mask).float()
        rear_mask = (~ld_mask * loss_mask).float()

        front_logps = (per_token_logps * front_mask).sum(-1)
        rear_logps = (per_token_logps * rear_mask).sum(-1)
        logps = front_logps + ld_alpha * rear_logps
    else:
        logps = (per_token_logps * loss_mask).sum(-1)

    return logps, valid_length