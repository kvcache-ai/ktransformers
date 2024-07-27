#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import argparse
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterable, Iterator, Sequence, TypeVar, cast

import math
import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

logger = logging.getLogger("hf-to-gguf")


###### MODEL DEFINITIONS ######

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


AnyModel = TypeVar("AnyModel", bound="type[Model]")


class Model:
    _model_classes: dict[str, type[Model]] = {}

    dir_model: Path
    ftype: gguf.LlamaFileType
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    model_name: str | None
    part_names: list[str]
    is_safetensors: bool
    hparams: dict[str, Any]
    block_count: int
    tensor_map: gguf.TensorNameMap
    tensor_names: set[str] | None
    fname_out: Path
    gguf_writer: gguf.GGUFWriter

    # subclasses should define this!
    model_arch: gguf.MODEL_ARCH

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, is_big_endian: bool, use_temp_file: bool, eager: bool, model_name: str | None):
        if type(self) is Model:
            raise TypeError(f"{type(self).__name__!r} should not be directly instantiated")
        self.dir_model = dir_model
        self.ftype = ftype
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.lazy = not eager
        self.model_name = model_name
        self.part_names = Model.get_model_part_names(self.dir_model, "model", ".safetensors")
        self.is_safetensors = len(self.part_names) > 0
        if not self.is_safetensors:
            self.part_names = Model.get_model_part_names(self.dir_model, "pytorch_model", ".bin")
        self.hparams = Model.load_hparams(self.dir_model)
        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer"])
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        self.tensor_names = None
        if self.ftype == gguf.LlamaFileType.GUESSED:
            # NOTE: can't use field "torch_dtype" in config.json, because some finetunes lie.
            _, first_tensor = next(self.get_tensors())
            if first_tensor.dtype == torch.float16:
                logger.info(f"choosing --outtype f16 from first tensor type ({first_tensor.dtype})")
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
            else:
                logger.info(f"choosing --outtype bf16 from first tensor type ({first_tensor.dtype})")
                self.ftype = gguf.LlamaFileType.MOSTLY_BF16
        ftype_up: str = self.ftype.name.partition("_")[2].upper()
        ftype_lw: str = ftype_up.lower()
        # allow templating the file name with the output ftype, useful with the "auto" ftype
        self.fname_out = fname_out.parent / fname_out.name.format(ftype_lw, outtype=ftype_lw, ftype=ftype_lw, OUTTYPE=ftype_up, FTYPE=ftype_up)
        self.gguf_writer = gguf.GGUFWriter(path=None, arch=gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file)

    @classmethod
    def __init_subclass__(cls):
        # can't use an abstract property, because overriding it without type errors
        # would require using decorated functions instead of simply defining the property
        if "model_arch" not in cls.__dict__:
            raise TypeError(f"Missing property 'model_arch' for {cls.__name__!r}")

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        tensor_names_from_parts: set[str] = set()

        if len(self.part_names) > 1:
            self.tensor_names = set()
            index_name = "model.safetensors" if self.is_safetensors else "pytorch_model.bin"
            index_name += ".index.json"
            logger.info(f"gguf: loading model weight map from '{index_name}'")
            with open(self.dir_model / index_name, "r", encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
                weight_map = index.get("weight_map")
                if weight_map is None or not isinstance(weight_map, dict):
                    raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                self.tensor_names.update(weight_map.keys())
        else:
            self.tensor_names = tensor_names_from_parts

        for part_name in self.part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                tensor_names_from_parts.update(model_part.keys())

                for name in model_part.keys():
                    data = model_part.get_tensor(name) if self.is_safetensors else model_part[name]
                    if self.lazy:
                        data = LazyTorchTensor.from_eager(data)
                    yield name, data

        # only verify tensor name presence; it doesn't matter if they are not in the right files
        if len(sym_diff := tensor_names_from_parts.symmetric_difference(self.tensor_names)) > 0:
            raise ValueError(f"Mismatch between weight map and model parts for tensor names: {sym_diff}")

    def format_tensor_name(self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight") -> str:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            raise ValueError(f"Missing {key!r} for MODEL_TENSORS of {self.model_arch!r}")
        name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in name:
            assert bid is not None
            name = name.format(bid=bid)
        return name + suffix

    def match_model_tensor_name(self, name: str, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        self.gguf_writer.add_embedding_length(n_embd)
        logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["intermediate_size", "n_inner"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_head_count(n_head)
        logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (f_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        return [(self.map_tensor_name(name), data_torch)]

    def extra_f32_tensors(self, name: str, new_name: str, bid: int | None, n_dims: int) -> bool:
        del name, new_name, bid, n_dims  # unused

        return False

    def extra_f16_tensors(self, name: str, new_name: str, bid: int | None, n_dims: int) -> bool:
        del name, new_name, bid, n_dims  # unused

        return False

    def write_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data in ((n, d.squeeze().numpy()) for n, d in self.modify_tensors(data_torch, name, bid)):
                data: np.ndarray = data  # type hint
                n_dims = len(data.shape)
                data_dtype = data.dtype
                data_qtype: gguf.GGMLQuantizationType | None = None

                # when both are True, f32 should win
                extra_f32 = self.extra_f32_tensors(name, new_name, bid, n_dims)
                extra_f16 = self.extra_f16_tensors(name, new_name, bid, n_dims)

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
                extra_f32 = any(cond for cond in (
                    extra_f32,
                    n_dims == 1,
                    new_name.endswith("_norm.weight"),
                ))

                # Some tensor types are always in float32
                extra_f32 = extra_f32 or any(self.match_model_tensor_name(new_name, key, bid) for key in (
                    gguf.MODEL_TENSOR.FFN_GATE_INP,
                    gguf.MODEL_TENSOR.POS_EMBD,
                    gguf.MODEL_TENSOR.TOKEN_TYPES,
                ))

                # if f16 desired, convert any float32 2-dim weight tensors to float16
                extra_f16 = any(cond for cond in (
                    extra_f16,
                    (name.endswith(".weight") and n_dims >= 2),
                ))

                if self.ftype != gguf.LlamaFileType.ALL_F32 and extra_f16 and not extra_f32:
                    if self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                        data = gguf.quantize_bf16(data)
                        assert data.dtype == np.int16
                        data_qtype = gguf.GGMLQuantizationType.BF16

                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0 and gguf.can_quantize_to_q8_0(data):
                        data = gguf.quantize_q8_0(data)
                        assert data.dtype == np.uint8
                        data_qtype = gguf.GGMLQuantizationType.Q8_0

                    else:  # default to float16 for quantized tensors
                        if data_dtype != np.float16:
                            data = data.astype(np.float16)
                        data_qtype = gguf.GGMLQuantizationType.F16

                if data_qtype is None:  # by default, convert to float32
                    if data_dtype != np.float32:
                        data = data.astype(np.float32)
                    data_qtype = gguf.GGMLQuantizationType.F32

                shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype) if data.dtype == np.uint8 else data.shape

                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

                self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    def write(self):
        self.write_tensors()
        self.gguf_writer.write_header_to_file(self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    def write_vocab(self):
        self.gguf_writer.write_header_to_file(self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)

        part_names.sort()

        return part_names

    @staticmethod
    def load_hparams(dir_model: Path):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: AnyModel) -> AnyModel:
            for name in names:
                cls._model_classes[name] = modelcls
            return modelcls
        return func

    @classmethod
    def from_model_architecture(cls, arch: str) -> type[Model]:
        try:
            return cls._model_classes[arch]
        except KeyError:
            raise NotImplementedError(f'Architecture {arch!r} not supported!') from None

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert-hf-to-gguf-update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggerganov/llama.cpp/pull/6920
    # Marker: Start get_vocab_base_pre
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {chktok}")
        logger.debug(f"chkhsh: {chkhsh}")

        res = None

        # NOTE: if you get an error here, you need to update the convert-hf-to-gguf-update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/BAAI/bge-small-en-v1.5
            res = "bert-bge"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/mosaicml/mpt-7b
            res = "mpt"
        if chkhsh == "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34":
            # ref: https://huggingface.co/bigcode/starcoder2-3b
            res = "starcoder"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3":
            # ref: https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b
            res = "stablelm2"
        if chkhsh == "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff":
            # ref: https://huggingface.co/smallcloudai/Refact-1_6-base
            res = "refact"
        if chkhsh == "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8":
            # ref: https://huggingface.co/CohereForAI/c4ai-command-r-v01
            res = "command-r"
        if chkhsh == "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea":
            # ref: https://huggingface.co/Qwen/Qwen1.5-7B
            res = "qwen2"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/allenai/OLMo-1.7-7B-hf
            res = "olmo"
        if chkhsh == "a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e":
            # ref: https://huggingface.co/databricks/dbrx-base
            res = "dbrx"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-en
            res = "jina-v2-en"
        if chkhsh == "171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-es
            res = "jina-v2-es"
        if chkhsh == "27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-de
            res = "jina-v2-de"
        if chkhsh == "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d":
            # ref: https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct
            res = "smaug-bpe"
        if chkhsh == "c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360":
            # ref: https://huggingface.co/LumiOpen/Poro-34B-chat
            res = "poro-chat"
        if chkhsh == "7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-code
            res = "jina-v2-code"

        if res is None:
            logger.warning("\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert-hf-to-gguf-update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert-hf-to-gguf-update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggerganov/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning("**************************************************************************************")
            logger.warning("\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
        logger.debug(f"chkhsh: {chkhsh}")

        return res
        # Marker: End get_vocab_base_pre

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_qwen(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
            merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

        # for this kind of tokenizer, added_vocab is not a subset of vocab, so they need to be combined
        added_vocab = tokenizer.special_tokens
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **added_vocab}.items()}

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
        special_vocab.merges = merges
        # only add special tokens when they were not already loaded from config.json
        if len(special_vocab.special_token_ids) == 0:
            special_vocab._set_special_token("bos", tokenizer.special_tokens["<|endoftext|>"])
            special_vocab._set_special_token("eos", tokenizer.special_tokens["<|endoftext|>"])
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.special_tokens["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNKNOWN] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if (token_id >= vocab_size):
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_llama_hf(self):
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


@Model.register("GPTNeoXForCausalLM")
class GPTNeoXModel(Model):
    model_arch = gguf.MODEL_ARCH.GPTNEOX

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]

        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(
            int(self.hparams["rotary_pct"] * (self.hparams["hidden_size"] // self.hparams["num_attention_heads"])),
        )
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_parallel_residual(self.hparams.get("use_parallel_residual", True))
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_eps"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))

        tensors: list[tuple[str, Tensor]] = []

        if re.match(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", name):
            # Map bloom-style qkv_linear to gpt-style qkv_linear
            # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
            # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
            qkv_weights = data_torch.reshape((n_head, 3, n_embed // n_head, n_embed))
            data_torch = torch.cat(
                (
                    qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 2, :, :].reshape((-1, n_embed)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.weight")
        elif re.match(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.bias", name):
            qkv_bias = data_torch.reshape((n_head, 3, n_embed // n_head))
            data_torch = torch.cat(
                (
                    qkv_bias[:, 0, :].reshape((n_embed,)),
                    qkv_bias[:, 1, :].reshape((n_embed,)),
                    qkv_bias[:, 2, :].reshape((n_embed,)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.bias")

        tensors.append((self.map_tensor_name(name), data_torch))

        return tensors


@Model.register("BloomForCausalLM")
class BloomModel(Model):
    model_arch = gguf.MODEL_ARCH.BLOOM

    def set_gguf_parameters(self):
        self.gguf_writer.add_name("Bloom")
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        self.gguf_writer.add_feed_forward_length(4 * n_embed)
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))

        name = re.sub(r'transformer\.', '', name)

        tensors: list[tuple[str, Tensor]] = []

        if re.match(r"h\.\d+\.self_attention\.query_key_value\.weight", name):
            # Map bloom-style qkv_linear to gpt-style qkv_linear
            # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
            # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
            qkv_weights = data_torch.reshape((n_head, 3, n_embed // n_head, n_embed))
            data_torch = torch.cat(
                (
                    qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 2, :, :].reshape((-1, n_embed)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.weight")
        elif re.match(r"h\.\d+\.self_attention\.query_key_value\.bias", name):
            qkv_bias = data_torch.reshape((n_head, 3, n_embed // n_head))
            data_torch = torch.cat(
                (
                    qkv_bias[:, 0, :].reshape((n_embed,)),
                    qkv_bias[:, 1, :].reshape((n_embed,)),
                    qkv_bias[:, 2, :].reshape((n_embed,)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.bias")

        tensors.append((self.map_tensor_name(name), data_torch))

        if name == "word_embeddings.weight":
            assert self.tensor_names is not None

            # TODO: tie them at runtime, don't duplicate in the model file
            if all(s not in self.tensor_names for s in ("lm_head.weight", "output.weight")):
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch))

        return tensors


@Model.register("MPTForCausalLM")
class MPTModel(Model):
    model_arch = gguf.MODEL_ARCH.MPT

    def set_vocab(self):
        try:
            self._set_vocab_gpt2()
        except Exception:
            # Fallback for SEA-LION model
            self._set_vocab_sentencepiece()
            self.gguf_writer.add_add_bos_token(False)
            self.gguf_writer.add_pad_token_id(3)
            self.gguf_writer.add_eos_token_id(1)
            self.gguf_writer.add_unk_token_id(0)

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layers"]
        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["d_model"])
        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        if kv_n_heads := self.hparams["attn_config"].get("kv_n_heads"):
            self.gguf_writer.add_head_count_kv(kv_n_heads)
        self.gguf_writer.add_layer_norm_eps(1e-5)
        if self.hparams["attn_config"]["clip_qkv"] is not None:
            self.gguf_writer.add_clamp_kqv(self.hparams["attn_config"]["clip_qkv"])
        if self.hparams["attn_config"]["alibi"]:
            self.gguf_writer.add_max_alibi_bias(self.hparams["attn_config"]["alibi_bias_max"])
        else:
            self.gguf_writer.add_max_alibi_bias(0.0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        if "scales" in name:
            new_name = self.map_tensor_name(name, try_suffixes=(".weight", ".bias", ".scales"))
            new_name = new_name.replace("scales", "act.scales")
        else:
            new_name = self.map_tensor_name(name, try_suffixes=(".weight", ".bias"))

        return [(new_name, data_torch)]


@Model.register("OrionForCausalLM")
class OrionModel(Model):
    model_arch = gguf.MODEL_ARCH.ORION

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)
        hf_repo = self.hparams.get("_name_or_path", "")

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            raise ValueError("gguf: can not find ctx length parameter.")

        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_source_hf_repo(hf_repo)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        # note: config provides rms norm but it is actually layer norm
        # ref:  https://huggingface.co/OrionStarAI/Orion-14B-Chat/blob/276a17221ce42beb45f66fac657a41540e71f4f5/modeling_orion.py#L570-L571
        self.gguf_writer.add_layer_norm_eps(self.hparams["rms_norm_eps"])


@Model.register("BaichuanForCausalLM", "BaiChuanForCausalLM")
class BaichuanModel(Model):
    model_arch = gguf.MODEL_ARCH.BAICHUAN

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)
        hf_repo = self.hparams.get("_name_or_path", "")

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            raise ValueError("gguf: can not find ctx length parameter.")

        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_source_hf_repo(hf_repo)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        tensors: list[tuple[str, Tensor]] = []

        if bid is not None and name == f"model.layers.{bid}.self_attn.W_pack.weight":
            logger.info(f"Unpacking and permuting layer {bid}")
            tensors = [
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid),
                    self._reverse_hf_permute_part(data_torch, 0, head_count, head_count)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid),
                    self._reverse_hf_permute_part(data_torch, 1, head_count, head_count_kv)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid),
                    self._reverse_hf_part(data_torch, 2)),
            ]
        else:
            tensors = [(self.map_tensor_name(name), data_torch)]

        return tensors

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def _reverse_hf_permute_part(
        self, weights: Tensor, n_part: int, n_head: int, n_head_kv: int | None = None,
    ) -> Tensor:
        r = weights.shape[0] // 3
        return self._reverse_hf_permute(weights[r * n_part:r * n_part + r, ...], n_head, n_head_kv)

    def _reverse_hf_part(self, weights: Tensor, n_part: int) -> Tensor:
        r = weights.shape[0] // 3
        return weights[r * n_part:r * n_part + r, ...]


@Model.register("XverseForCausalLM")
class XverseModel(Model):
    model_arch = gguf.MODEL_ARCH.XVERSE

    def set_vocab(self):
        assert (self.dir_model / "tokenizer.json").is_file()
        dir_model = self.dir_model
        hparams = self.hparams

        tokens: list[bytes] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model)
        vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        reverse_vocab: dict[int, str] = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for token_id in range(vocab_size):
            token_text = reverse_vocab[token_id].encode('utf-8')
            # replace "\x00" to string with length > 0
            if token_text == b"\x00":
                toktype = gguf.TokenType.BYTE  # special
                token_text = f"<{token_text}>".encode('utf-8')
            elif re.fullmatch(br"<0x[0-9A-Fa-f]{2}>", token_text):
                toktype = gguf.TokenType.BYTE  # special
            elif reverse_vocab[token_id] in added_vocab:
                if tokenizer.added_tokens_decoder[token_id].special:
                    toktype = gguf.TokenType.CONTROL
                else:
                    toktype = gguf.TokenType.USER_DEFINED
            else:
                toktype = gguf.TokenType.NORMAL

            tokens.append(token_text)
            toktypes.append(toktype)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)
        hf_repo = self.hparams.get("_name_or_path", "")

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            raise ValueError("gguf: can not find ctx length parameter.")

        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_source_hf_repo(hf_repo)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        # HF models permute some of the tensors, so we need to undo that
        if name.endswith("q_proj.weight"):
            data_torch = self._reverse_hf_permute(data_torch, head_count, head_count)
        if name.endswith("k_proj.weight"):
            data_torch = self._reverse_hf_permute(data_torch, head_count, head_count_kv)

        return [(self.map_tensor_name(name), data_torch)]

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )


@Model.register("FalconForCausalLM", "RWForCausalLM")
class FalconModel(Model):
    model_arch = gguf.MODEL_ARCH.FALCON

    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        self.gguf_writer.add_name("Falcon")
        self.gguf_writer.add_context_length(2048)  # not in config.json
        self.gguf_writer.add_tensor_data_layout("jploski")  # qkv tensor transform
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # QKV tensor transform
        # The original query_key_value tensor contains n_head_kv "kv groups",
        # each consisting of n_head/n_head_kv query weights followed by one key
        # and one value weight (shared by all query heads in the kv group).
        # This layout makes it a big pain to work with in GGML.
        # So we rearrange them here,, so that we have n_head query weights
        # followed by n_head_kv key weights followed by n_head_kv value weights,
        # in contiguous fashion.
        # ref: https://github.com/jploski/ggml/blob/falcon40b/examples/falcon/convert-hf-to-ggml.py

        if "query_key_value" in name:
            n_head = self.find_hparam(["num_attention_heads", "n_head"])
            n_head_kv = self.find_hparam(["num_kv_heads", "n_head_kv"], optional=True) or 1
            head_dim = self.hparams["hidden_size"] // n_head

            qkv = data_torch.view(n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
            q = qkv[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
            k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
            v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
            data_torch = torch.cat((q, k, v)).reshape_as(data_torch)

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("GPTBigCodeForCausalLM")
class StarCoderModel(Model):
    model_arch = gguf.MODEL_ARCH.STARCODER

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("StarCoder")
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@Model.register("GPTRefactForCausalLM")
class RefactModel(Model):
    model_arch = gguf.MODEL_ARCH.REFACT

    def set_vocab(self):
        super().set_vocab()

        # TODO: how to determine special FIM tokens automatically?
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False,
                                          special_token_types = ['prefix', 'suffix', 'middle', 'fsep', 'eot'])
        special_vocab._set_special_token("prefix", 1)
        special_vocab._set_special_token("suffix", 3)
        special_vocab._set_special_token("middle", 2)
        special_vocab._set_special_token("fsep",   4) # is this correct?
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("Refact")
        # refact uses Alibi. So this is from config.json which might be used by training.
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])

        self.gguf_writer.add_feed_forward_length(ff_dim)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        n_head = self.hparams["n_head"]
        n_head_kv = 1
        head_dim = self.hparams["n_embd"] // n_head

        tensors: list[tuple[str, Tensor]] = []

        if bid is not None:
            if name == f"transformer.h.{bid}.attn.kv.weight":
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), data_torch[:n_head_kv * head_dim]))
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), data_torch[n_head_kv * head_dim:]))
            elif name == f"transformer.h.{bid}.attn.q.weight":
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), data_torch))
            elif name == f"transformer.h.{bid}.mlp.gate_up_proj.weight":
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), data_torch[:ff_dim]))
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), data_torch[ff_dim:]))

        if len(tensors) == 0:
            tensors.append((self.map_tensor_name(name), data_torch))

        return tensors


@Model.register("StableLmForCausalLM", "StableLMEpochForCausalLM", "LlavaStableLMEpochForCausalLM")
class StableLMModel(Model):
    model_arch = gguf.MODEL_ARCH.STABLELM

    def set_vocab(self):
        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            # StableLM 2 1.6B uses a vocab in a similar format to Qwen's vocab
            self._set_vocab_qwen()

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        rotary_factor = self.find_hparam(["partial_rotary_factor", "rope_pct"])
        self.gguf_writer.add_rope_dimension_count(int(rotary_factor * (hparams["hidden_size"] // hparams["num_attention_heads"])))
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])
        self.gguf_writer.add_parallel_residual(hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True)
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(["layer_norm_eps", "norm_eps"]))
        self.gguf_writer.add_file_type(self.ftype)

    _q_norms: list[dict[str, Tensor]] | None = None
    _k_norms: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams["num_key_value_heads"]

        if name.find("q_layernorm.norms") != -1:
            assert bid is not None

            if self._q_norms is None:
                self._q_norms = [{} for _ in range(self.block_count)]

            self._q_norms[bid][name] = data_torch

            if len(self._q_norms[bid]) >= n_head:
                return self._stack_qk_norm(bid, n_head, self._q_norms[bid], "q_layernorm")
            else:
                return []

        if name.find("k_layernorm.norms") != -1:
            assert bid is not None

            if self._k_norms is None:
                self._k_norms = [{} for _ in range(self.block_count)]

            self._k_norms[bid][name] = data_torch

            if len(self._k_norms[bid]) >= n_kv_head:
                return self._stack_qk_norm(bid, n_kv_head, self._k_norms[bid], "k_layernorm")
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def _stack_qk_norm(self, bid: int, n_head: int, norms: dict[str, Tensor], layer_name: str = "q_layernorm"):
        datas: list[Tensor] = []
        # extract the norms in order
        for xid in range(n_head):
            ename = f"model.layers.{bid}.self_attn.{layer_name}.norms.{xid}.weight"
            datas.append(norms[ename])
            del norms[ename]
        data_torch = torch.stack(datas, dim=0)

        merged_name = f"model.layers.{bid}.self_attn.{layer_name}.weight"
        new_name = self.map_tensor_name(merged_name)

        return [(new_name, data_torch)]

    def write_tensors(self):
        super().write_tensors()

        if self._q_norms is not None or self._k_norms is not None:
            # flatten two `list[dict[str, Tensor]]` into a single `list[str]`
            norms = (
                [k for d in self._q_norms for k in d.keys()] if self._q_norms is not None else []
            ) + (
                [k for d in self._k_norms for k in d.keys()] if self._k_norms is not None else []
            )
            if len(norms) > 0:
                raise ValueError(f"Unprocessed norms: {norms}")


@Model.register("LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM")
class LlamaModel(Model):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_vocab(self):
        try:
            self. _set_vocab_sentencepiece()
        except FileNotFoundError:
            try:
                self._set_vocab_llama_hf()
            except (FileNotFoundError, TypeError):
                # Llama 3
                self._set_vocab_gpt2()

        # Apply to CodeLlama only (and ignore for Llama 3 with a vocab size of 128256)
        if self.hparams.get("vocab_size", 32000) == 32016:
            special_vocab = gguf.SpecialVocab(
                self.dir_model, load_merges=False,
                special_token_types = ['prefix', 'suffix', 'middle', 'eot']
            )
            special_vocab._set_special_token("prefix", 32007)
            special_vocab._set_special_token("suffix", 32008)
            special_vocab._set_special_token("middle", 32009)
            special_vocab._set_special_token("eot",    32010)
            special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["hidden_size"] // hparams["num_attention_heads"])

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if "add_prefix_space" in tokenizer_config_json:
                    self.gguf_writer.add_add_space_prefix(tokenizer_config_json["add_prefix_space"])

        # Apply to granite small models only
        if self.hparams.get("vocab_size", 32000) == 49152:
            self.gguf_writer.add_add_bos_token(False)

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def write_tensors(self):
        super().write_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("GrokForCausalLM")
class GrokModel(Model):
    model_arch = gguf.MODEL_ARCH.GROK

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_name("Grok")

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find(".moe.") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for wid in ["linear", "linear_1", "linear_v"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"transformer.decoder_layer.{bid}.moe.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"transformer.decoder_layer.{bid}.moe.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("DbrxForCausalLM")
class DbrxModel(Model):
    model_arch = gguf.MODEL_ARCH.DBRX

    def set_gguf_parameters(self):
        ffn_config = self.hparams["ffn_config"]
        attn_config = self.hparams["attn_config"]
        self.gguf_writer.add_name(self.hparams["model_type"])
        self.gguf_writer.add_block_count(self.hparams["n_layers"])

        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_feed_forward_length(ffn_config["ffn_hidden_size"])

        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        self.gguf_writer.add_head_count_kv(attn_config["kv_n_heads"])

        self.gguf_writer.add_rope_freq_base(attn_config["rope_theta"])

        self.gguf_writer.add_clamp_kqv(attn_config["clip_qkv"])
        self.gguf_writer.add_file_type(self.ftype)

        self.gguf_writer.add_expert_count(ffn_config["moe_num_experts"])
        self.gguf_writer.add_expert_used_count(ffn_config["moe_top_k"])

        self.gguf_writer.add_layer_norm_eps(1e-5)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_expert = self.hparams["ffn_config"]["moe_num_experts"]
        n_ff = self.hparams["ffn_config"]["ffn_hidden_size"]
        n_embd = self.hparams["d_model"]

        # Specific behavior for experts tensors: suffix .weight, view as 3D and transpose
        # original implementation expects (n_expert, n_ff, n_embd) for all experts weights
        # But llama.cpp moe graph works differently
        # AND the dimensions in ggml are typically in the reverse order of the pytorch dimensions
        # so (n_expert, n_ff, n_embd) in pytorch is {n_embd, n_ff, n_expert} in ggml_tensor
        exp_tensor_names = {"ffn.experts.mlp.w1": None,       # LLM_TENSOR_FFN_GATE_EXPS ggml_tensor->ne{n_embd, n_ff,   n_expert}
                            "ffn.experts.mlp.w2": (0, 2, 1),  # LLM_TENSOR_FFN_DOWN_EXPS ggml_tensor->ne{n_ff,   n_embd, n_expert}
                            "ffn.experts.mlp.v1": None}       # LLM_TENSOR_FFN_UP_EXPS   ggml_tensor->ne{n_embd, n_ff,   n_expert}
        experts = False

        for exp_tensor_name in exp_tensor_names.keys():
            if name.find(exp_tensor_name) != -1 and name.find(".weight") == -1:
                experts = True
                data_torch = data_torch.view(n_expert, n_ff, n_embd)
                if (permute_tensor := exp_tensor_names[exp_tensor_name]) is not None:
                    data_torch = data_torch.permute(*permute_tensor)
                break

        # map tensor names
        # In MoE models the ffn tensors are typically most of the model weights,
        # and need to be quantizable. Quantize expects tensor names to be suffixed by .weight.
        # Every other model has the weight names ending in .weight,
        # let's assume that is the convention which is not the case for dbrx:
        # https://huggingface.co/databricks/dbrx-instruct/blob/main/model.safetensors.index.json#L15
        new_name = self.map_tensor_name(name if not experts else name + ".weight", try_suffixes=(".weight",))

        return [(new_name, data_torch)]

    def extra_f16_tensors(self, name: str, new_name: str, bid: int | None, n_dims: int) -> bool:
        del name, new_name, bid  # unused

        return n_dims > 1


@Model.register("MiniCPMForCausalLM")
class MiniCPMModel(Model):
    model_arch = gguf.MODEL_ARCH.MINICPM

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        self.gguf_writer.add_name("MiniCPM")
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_llama_hf()

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        # HF models permute some of the tensors, so we need to undo that
        if name.endswith(("q_proj.weight")):
            data_torch = self._reverse_hf_permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight")):
            data_torch = self._reverse_hf_permute(data_torch, n_head, n_kv_head)

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("QWenLMHeadModel")
class QwenModel(Model):
    model_arch = gguf.MODEL_ARCH.QWEN

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def set_vocab(self):
        self._set_vocab_qwen()

    def set_gguf_parameters(self):
        self.gguf_writer.add_name("Qwen")
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@Model.register("Qwen2ForCausalLM")
class Qwen2Model(Model):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()


@Model.register("Qwen2MoeForCausalLM")
class Qwen2MoeModel(Model):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (n_experts := self.hparams.get("num_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (shared_expert_intermediate_size := self.hparams.get('shared_expert_intermediate_size')) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_expert_intermediate_size)
            logger.info(f"gguf: expert shared feed forward length = {shared_expert_intermediate_size}")

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def write_tensors(self):
        super().write_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("GPT2LMHeadModel")
class GPT2Model(Model):
    model_arch = gguf.MODEL_ARCH.GPT2

    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_context_length(self.hparams["n_ctx"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        tensors: list[tuple[str, Tensor]] = []

        # we don't need these
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            return tensors

        if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_proj.weight")):
            data_torch = data_torch.transpose(1, 0)

        new_name = self.map_tensor_name(name)

        tensors.append((new_name, data_torch))

        # note: GPT2 output is tied to (same as) wte in original model
        if new_name == self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD):
            tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch))

        return tensors


@Model.register("PhiForCausalLM")
class Phi2Model(Model):
    model_arch = gguf.MODEL_ARCH.PHI2

    def set_gguf_parameters(self):
        block_count = self.find_hparam(["num_hidden_layers", "n_layer"])

        rot_pct = self.find_hparam(["partial_rotary_factor"])
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])

        self.gguf_writer.add_name("Phi2")
        self.gguf_writer.add_context_length(self.find_hparam(["n_positions", "max_position_embeddings"]))

        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(4 * n_embd)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(["layer_norm_epsilon", "layer_norm_eps"]))
        self.gguf_writer.add_rope_dimension_count(int(rot_pct * n_embd) // n_head)
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_add_bos_token(False)


@Model.register("Phi3ForCausalLM")
class Phi3MiniModel(Model):
    model_arch = gguf.MODEL_ARCH.PHI3

    def set_vocab(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        if not tokenizer_path.is_file():
            raise ValueError(f'Error: Missing {tokenizer_path}')

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNKNOWN] * vocab_size

        for token_id in range(tokenizer.vocab_size()):

            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if (token_id >= vocab_size):
                        logger.debug(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                added_tokens_decoder = tokenizer_config_json.get("added_tokens_decoder", {})
                for token_id, foken_data in added_tokens_decoder.items():
                    token_id = int(token_id)
                    token = foken_data["content"].encode("utf-8")
                    if toktypes[token_id] != SentencePieceTokenTypes.UNKNOWN:
                        assert tokens[token_id] == token
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        tokenizer_file = self.dir_model / 'tokenizer.json'
        if tokenizer_file.is_file():
            with open(tokenizer_file, "r", encoding="utf-8") as f:
                tokenizer_json = json.load(f)
                added_tokens = tokenizer_json.get("added_tokens", [])
                for foken_data in added_tokens:
                    token_id = int(foken_data["id"])
                    token = foken_data["content"].encode("utf-8")
                    if toktypes[token_id] != SentencePieceTokenTypes.UNKNOWN:
                        assert tokens[token_id] == token
                    tokens[token_id] = token
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED
                    if foken_data.get("special"):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        block_count = self.find_hparam(["num_hidden_layers", "n_layer"])

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        n_head_kv = self.find_hparam(["num_key_value_heads", "n_head_kv"])
        rms_eps = self.find_hparam(["rms_norm_eps"])
        max_pos_embds = self.find_hparam(["n_positions", "max_position_embeddings"])
        orig_max_pos_embds = self.find_hparam(["original_max_position_embeddings"])
        rope_dims = n_embd // n_head

        self.gguf_writer.add_name("Phi3")
        self.gguf_writer.add_context_length(max_pos_embds)
        self.gguf_writer.add_rope_scaling_orig_ctx_len(orig_max_pos_embds)
        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(self.find_hparam(["intermediate_size"]))
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_rms_eps(rms_eps)
        self.gguf_writer.add_rope_dimension_count(rope_dims)
        self.gguf_writer.add_rope_freq_base(self.find_hparam(["rope_theta"]))
        self.gguf_writer.add_file_type(self.ftype)

        # write rope scaling for long context (128k) model
        rope_scaling = self.find_hparam(['rope_scaling'], True)
        if (rope_scaling is None):
            return

        scale = max_pos_embds / orig_max_pos_embds

        rope_scaling_type = rope_scaling.get('type', '').lower()
        if len(rope_scaling_type) == 0:
            raise KeyError('Missing the required key rope_scaling.type')

        if rope_scaling_type == 'su':
            attn_factor = math.sqrt(1 + math.log(scale) / math.log(orig_max_pos_embds)) if scale > 1.0 else 1.0
        elif rope_scaling_type == 'yarn':
            attn_factor = 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0
        else:
            raise NotImplementedError(f'The rope scaling type {rope_scaling_type} is not supported yet')

        self.gguf_writer.add_rope_scaling_attn_factors(attn_factor)

        long_factors = rope_scaling.get('long_factor', None)
        short_factors = rope_scaling.get('short_factor', None)

        if long_factors is None or short_factors is None:
            raise KeyError('Missing the required key rope_scaling.long_factor or rope_scaling_short_factor')

        if len(long_factors) != len(short_factors) or len(long_factors) != rope_dims / 2:
            raise ValueError(f'The length of rope long and short factors must be {rope_dims / 2}')

        self.gguf_writer.add_tensor(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ROPE_FACTORS_LONG]  + ".weight", np.array(long_factors, dtype=np.float32))
        self.gguf_writer.add_tensor(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ROPE_FACTORS_SHORT] + ".weight", np.array(short_factors, dtype=np.float32))


@Model.register("PlamoForCausalLM")
class PlamoModel(Model):
    model_arch = gguf.MODEL_ARCH.PLAMO

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_name("PLaMo")
        self.gguf_writer.add_context_length(4096)  # not in config.json
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(5)  # hparams["num_key_value_heads"]) is wrong
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

    def shuffle_attn_q_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(8, 5, 128, 5120)
        data_torch = torch.permute(data_torch, (1, 0, 2, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def shuffle_attn_output_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(5120, 8, 5, 128)
        data_torch = torch.permute(data_torch, (0, 2, 1, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        new_name = self.map_tensor_name(name)

        # shuffle for broadcasting of gqa in ggml_mul_mat
        if new_name.endswith("attn_q.weight"):
            data_torch = self.shuffle_attn_q_weight(data_torch)
        elif new_name.endswith("attn_output.weight"):
            data_torch = self.shuffle_attn_output_weight(data_torch)

        return [(new_name, data_torch)]


@Model.register("CodeShellForCausalLM")
class CodeShellModel(Model):
    model_arch = gguf.MODEL_ARCH.CODESHELL

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("CodeShell")
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_query_groups"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_rope_freq_base(10000.0)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        new_name = self.map_tensor_name(name)

        tensors: list[tuple[str, Tensor]] = [(new_name, data_torch)]

        if new_name == self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD):
            assert self.tensor_names is not None

            if all(s not in self.tensor_names for s in ("lm_head.weight", "output.weight")):
                # copy tok_embd.weight to output.weight
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch))

        return tensors


@Model.register("InternLM2ForCausalLM")
class InternLM2Model(Model):
    model_arch = gguf.MODEL_ARCH.INTERNLM2

    def set_vocab(self):
        # (TODO): Is there a better way?
        # Copy from _set_vocab_sentencepiece, The only difference is that we will treat the character
        # \x00 specially and convert it into an emoji character to prevent it from being mistakenly
        # recognized as an empty string in C++.
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'tokenizer.model'

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            logger.error(f'Error: Missing {tokenizer_path}')
            sys.exit(1)

        sentencepiece_model = model.ModelProto()
        sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
        add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        for token_id in range(vocab_size):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)
            if text == b"\x00":
                # (TODO): fixme
                # Hack here and replace the \x00 characters.
                logger.warning(f"InternLM2 convert token '{text}' to '🐉'!")
                text = "🐉".encode("utf-8")

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    tokens.append(key.encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        old_eos = special_vocab.special_token_ids["eos"]
        if "chat" in os.path.basename(self.dir_model.absolute()):
            # For the chat model, we replace the eos with '<|im_end|>'.
            # TODO: this is a hack, should be fixed
            #       https://github.com/ggerganov/llama.cpp/pull/6745#issuecomment-2067687048
            special_vocab.special_token_ids["eos"] = self._try_get_sft_eos(tokenizer)
            logger.warning(f"Replace eos:{old_eos} with a special token:{special_vocab.special_token_ids['eos']} \
in chat mode so that the conversation can end normally.")

        special_vocab.add_to_gguf(self.gguf_writer)

    def _try_get_sft_eos(self, tokenizer):
        unused_145_list = tokenizer.Encode('[UNUSED_TOKEN_145]')
        im_end_list = tokenizer.Encode('<|im_end|>')
        eos_token = None
        assert (len(unused_145_list) == 1) ^ (len(im_end_list) == 1)
        if len(unused_145_list) == 1:
            eos_token = unused_145_list[0]
        if len(im_end_list) == 1:
            eos_token = im_end_list[0]
        assert eos_token
        return eos_token

    def _hf_permute_qk(self, weights, n_head: int, n_head_kv: int):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def set_gguf_parameters(self):
        self.gguf_writer.add_name("InternLM2")
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_theta"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        num_heads = self.hparams["num_attention_heads"]
        num_kv_heads = self.hparams["num_key_value_heads"]
        hidden_size = self.hparams["hidden_size"]
        q_per_kv = num_heads // num_kv_heads
        head_dim = hidden_size // num_heads
        num_groups = num_heads // q_per_kv

        qkv_pattern = r"model\.layers\.(\d+)\.attention\.wqkv"

        if re.match(qkv_pattern, name):
            bid = re.findall(qkv_pattern, name)[0]
            qkv = data_torch
            # qkv = rearrange(qkv.T, " o (g n i) ->o g n i", g=num_groups, n=q_per_kv + 2, i=head_dim)
            qkv = qkv.T.reshape((-1, num_groups, q_per_kv + 2, head_dim))
            q, k, v = qkv[..., : q_per_kv, :], qkv[..., q_per_kv: q_per_kv + 1, :], qkv[..., q_per_kv + 1: q_per_kv + 2, :]
            # The model weights of q and k equire additional reshape.
            # q = self._hf_permute_qk(rearrange(q, " o g n i ->  o (g n i)").T, num_heads, num_heads)
            q = self._hf_permute_qk(q.reshape((q.shape[0], -1)).T, num_heads, num_heads)
            # k = self._hf_permute_qk(rearrange(k, " o g n i ->  o (g n i)").T, num_heads, num_kv_heads)
            k = self._hf_permute_qk(k.reshape((k.shape[0], -1)).T, num_heads, num_kv_heads)
            # v = rearrange(v, " o g n i ->  o (g n i)").T
            v = v.reshape((v.shape[0], -1)).T
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), k),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), v),
            ]
        else:
            return [(self.map_tensor_name(name), data_torch)]


@Model.register("BertModel", "CamembertModel")
class BertModel(Model):
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)

        # get pooling path
        pooling_path = None
        module_path = self.dir_model / "modules.json"
        if module_path.is_file():
            with open(module_path, encoding="utf-8") as f:
                modules = json.load(f)
            for mod in modules:
                if mod["type"] == "sentence_transformers.models.Pooling":
                    pooling_path = mod["path"]
                    break

        # get pooling type
        if pooling_path is not None:
            with open(self.dir_model / pooling_path / "config.json", encoding="utf-8") as f:
                pooling = json.load(f)
            if pooling["pooling_mode_mean_tokens"]:
                pooling_type = gguf.PoolingType.MEAN
            elif pooling["pooling_mode_cls_token"]:
                pooling_type = gguf.PoolingType.CLS
            else:
                raise NotImplementedError("Only MEAN and CLS pooling types supported")
            self.gguf_writer.add_pooling_type(pooling_type)

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.vocab_size = len(tokens)

        # we need this to validate the size of the token_type embeddings
        # though currently we are passing all zeros to the token_type embeddings
        self.gguf_writer.add_token_type_count(2)  # "Sequence A" or "Sequence B"

        # convert to phantom space vocab
        def phantom(tok):
            if tok.startswith("[") and tok.endswith("]"):
                return tok
            if tok.startswith("##"):
                return tok[2:]
            return "\u2581" + tok
        tokens = list(map(phantom, tokens))

        # add vocab to gguf
        self.gguf_writer.add_tokenizer_model("bert")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # handle special tokens
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # we are only using BERT for embeddings so we don't need the pooling layer
        if name in ("embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"):
            return [] # we don't need these

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("NomicBertModel")
class NomicBertModel(BertModel):
    model_arch = gguf.MODEL_ARCH.NOMIC_BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # the HF config claims n_ctx=8192, but it uses RoPE scaling
        self.hparams["n_ctx"] = 2048

        # SwigLU activation
        assert self.hparams["activation_function"] == "swiglu"
        # this doesn't do anything in the HF version
        assert self.hparams["causal"] is False
        # no bias tensors
        assert self.hparams["qkv_proj_bias"] is False
        assert self.hparams["mlp_fc1_bias"] is False
        assert self.hparams["mlp_fc2_bias"] is False
        # norm at end of layer
        assert self.hparams["prenorm"] is False
        # standard RoPE
        assert self.hparams["rotary_emb_fraction"] == 1.0
        assert self.hparams["rotary_emb_interleaved"] is False
        assert self.hparams["rotary_emb_scale_base"] is None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])


@Model.register("GemmaForCausalLM")
class GemmaModel(Model):
    model_arch = gguf.MODEL_ARCH.GEMMA

    def set_vocab(self):
        self._set_vocab_sentencepiece()

        # TODO: these special tokens should be exported only for the CodeGemma family
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False,
                                          special_token_types = ['prefix', 'suffix', 'middle', 'fsep', 'eot'])
        special_vocab._set_special_token("prefix", 67)
        special_vocab._set_special_token("suffix", 69)
        special_vocab._set_special_token("middle", 68)
        special_vocab._set_special_token("fsep",   70)
        special_vocab._set_special_token("eot",    107)
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"] if "num_key_value_heads" in hparams else hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_key_length(hparams["head_dim"])
        self.gguf_writer.add_value_length(hparams["head_dim"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        # lm_head is not used in llama.cpp, while autoawq will include this tensor in model
        # To prevent errors, skip loading lm_head.weight.
        if name == "lm_head.weight":
            logger.debug(f"Skipping get tensor {name!r} in safetensors so that convert can end normally.")
            return []

        # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("Starcoder2ForCausalLM")
class StarCoder2Model(Model):
    model_arch = gguf.MODEL_ARCH.STARCODER2


@Model.register("MambaForCausalLM", "MambaLMHeadModel")
class MambaModel(Model):
    model_arch = gguf.MODEL_ARCH.MAMBA

    def set_vocab(self):
        vocab_size = self.hparams["vocab_size"]
        # Round vocab size to next multiple of 8
        pad_vocab = self.hparams.get("pad_vocab_size_multiple", 8)
        # pad using ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        vocab_size = -(vocab_size // -pad_vocab) * pad_vocab
        self.hparams["vocab_size"] = vocab_size

        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        elif (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        else:
            # Use the GPT-NeoX tokenizer when no tokenizer files are present
            tokenizer_path = Path(sys.path[0]) / "models" / "ggml-vocab-gpt-neox.gguf"
            logger.warning(f"Using tokenizer from '{os.path.relpath(tokenizer_path, os.getcwd())}'")
            neox_reader = gguf.GGUFReader(tokenizer_path, "r")

            field = neox_reader.get_field(gguf.Keys.Tokenizer.MODEL)
            self.gguf_writer.add_tokenizer_model(bytes(field.parts[-1]).decode("utf-8") if field else "gpt2")

            field = neox_reader.get_field(gguf.Keys.Tokenizer.PRE)
            self.gguf_writer.add_tokenizer_pre(bytes(field.parts[-1]).decode("utf-8") if field else "mpt")

            field = neox_reader.get_field(gguf.Keys.Tokenizer.LIST)
            assert field
            self.gguf_writer.add_token_list([bytes(field.parts[i]) for i in field.data][:vocab_size])

            field = neox_reader.get_field(gguf.Keys.Tokenizer.TOKEN_TYPE)
            assert field
            self.gguf_writer.add_token_types([field.parts[i].tolist()[0] for i in field.data][:vocab_size])

            field = neox_reader.get_field(gguf.Keys.Tokenizer.MERGES)
            assert field
            self.gguf_writer.add_token_merges([bytes(field.parts[i]) for i in field.data])

            field = neox_reader.get_field(gguf.Keys.Tokenizer.BOS_ID)
            self.gguf_writer.add_bos_token_id(field.parts[-1].tolist()[0] if field else 1)

            field = neox_reader.get_field(gguf.Keys.Tokenizer.EOS_ID)
            self.gguf_writer.add_eos_token_id(field.parts[-1].tolist()[0] if field else 0)

            field = neox_reader.get_field(gguf.Keys.Tokenizer.UNK_ID)
            self.gguf_writer.add_unk_token_id(field.parts[-1].tolist()[0] if field else 0)

            field = neox_reader.get_field(gguf.Keys.Tokenizer.PAD_ID)
            self.gguf_writer.add_pad_token_id(field.parts[-1].tolist()[0] if field else 0)

    def set_gguf_parameters(self):
        d_model = self.find_hparam(["hidden_size",       "d_model"])
        d_conv  = self.find_hparam(["conv_kernel",       "d_conv"],  optional=True) or 4
        d_inner = self.find_hparam(["intermediate_size", "d_inner"], optional=True) or 2 * d_model
        d_state = self.find_hparam(["state_size",        "d_state"], optional=True) or 16
        # ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        # ref: https://github.com/state-spaces/mamba/blob/ce59daea3a090d011d6476c6e5b97f6d58ddad8b/mamba_ssm/modules/mamba_simple.py#L58
        dt_rank      = self.find_hparam(["time_step_rank",     "dt_rank"],      optional=True) or -(d_model // -16)
        rms_norm_eps = self.find_hparam(["layer_norm_epsilon", "rms_norm_eps"], optional=True) or 1e-5

        # Fail early for models which don't have a block expansion factor of 2
        assert d_inner == 2 * d_model

        self.gguf_writer.add_name(self.dir_model.name if self.model_name is None else self.model_name)
        self.gguf_writer.add_context_length(2**20) # arbitrary value; for those who use the default
        self.gguf_writer.add_embedding_length(d_model)
        self.gguf_writer.add_feed_forward_length(0) # unused, but seemingly required when loading
        self.gguf_writer.add_head_count(0) # unused, but seemingly required when loading
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(dt_rank)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_file_type(self.ftype)

    _tok_embd = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        output_name = self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT)
        tok_embd_name = self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD)

        new_name = self.map_tensor_name(name)

        if name.endswith(".A_log"):
            logger.debug("A_log --> A ==> " + new_name)
            data_torch = -torch.exp(data_torch)

        # assuming token_embd.weight is seen before output.weight
        if self._tok_embd is not None and new_name == output_name:
            if torch.equal(self._tok_embd, data_torch):
                logger.debug(f"{output_name} is equivalent to {tok_embd_name}, omitting")
                return []
        elif new_name == tok_embd_name:
            self._tok_embd = data_torch

        return [(new_name, data_torch)]

    def extra_f32_tensors(self, name: str, new_name: str, bid: int | None, n_dims: int) -> bool:
        del n_dims  # unused

        return bid is not None and new_name in (
            self.format_tensor_name(n, bid, ".weight" if name.endswith(".weight") else "") for n in [
                gguf.MODEL_TENSOR.SSM_CONV1D,
                gguf.MODEL_TENSOR.SSM_X,
                gguf.MODEL_TENSOR.SSM_DT,
                gguf.MODEL_TENSOR.SSM_A,
                gguf.MODEL_TENSOR.SSM_D,
            ]
        )


@Model.register("CohereForCausalLM")
class CommandR2Model(Model):
    model_arch = gguf.MODEL_ARCH.COMMAND_R

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # max_position_embeddings = 8192 in config.json but model was actually
        # trained on 128k context length
        # aya-23 models don't have model_max_length specified
        self.hparams["max_position_embeddings"] = self.find_hparam(["model_max_length", "max_position_embeddings"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)


@Model.register("OlmoForCausalLM")
@Model.register("OLMoForCausalLM")
class OlmoModel(Model):
    model_arch = gguf.MODEL_ARCH.OLMO

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_layer_norm_eps(1e-5)
        clip_qkv = self.hparams.get("clip_qkv")
        if clip_qkv is not None:
            self.gguf_writer.add_clamp_kqv(clip_qkv)

    # Same as super class, but permuting q_proj, k_proj
    # Copied from: LlamaModel
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith("q_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith("k_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        return [(self.map_tensor_name(name), data_torch)]


@Model.register("JinaBertModel", "JinaBertForMaskedLM")
class JinaBertV2Model(BertModel):
    model_arch = gguf.MODEL_ARCH.JINA_BERT_V2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intermediate_size = self.hparams["intermediate_size"]

    def get_tensors(self):
        for name, data in super().get_tensors():
            if 'gated_layer' in name:
                d1 = data[:self.intermediate_size, :]
                name1 = name.replace('gated_layers', 'gated_layers_w')
                name1 = name1.replace('up_gated_layer', 'gated_layers_v')
                d2 = data[self.intermediate_size:, :]
                name2 = name.replace('gated_layers', 'gated_layers_v')
                name2 = name2.replace('up_gated_layer', 'gated_layers_w')
                yield name1, d1
                yield name2, d2
                continue

            yield name, data

    def set_vocab(self, *args, **kwargs):
        tokenizer_class = 'BertTokenizer'
        with open(self.dir_model / "tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_class = json.load(f)['tokenizer_class']

        if tokenizer_class == 'BertTokenizer':
            super().set_vocab()
        elif tokenizer_class == 'RobertaTokenizer':
            self._set_vocab_gpt2()
            self.gguf_writer.add_token_type_count(2)
        else:
            raise NotImplementedError(f'Tokenizer {tokenizer_class} is not supported for JinaBertModel')
        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(True)


@Model.register("ArcticForCausalLM")
class ArcticModel(Model):
    model_arch = gguf.MODEL_ARCH.ARCTIC

    def set_vocab(self):
        # The reason for using a custom implementation here is that the
        # snowflake-arctic-instruct model redefined tokens 31998 and 31999 from
        # tokenizer.model and used them as BOS and EOS instead of adding new tokens.
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        if not tokenizer_path.is_file():
            logger.error(f'Error: Missing {tokenizer_path}')
            sys.exit(1)

        # Read the whole vocabulary from the tokenizer.model file
        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get('vocab_size', tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNKNOWN] * vocab_size

        for token_id in range(tokenizer.vocab_size()):

            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        # Use the added_tokens_decoder field from tokeniser_config.json as the source
        # of information about added/redefined tokens and modify them accordingly.
        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)

                if "added_tokens_decoder" in tokenizer_config_json:
                    added_tokens_decoder = tokenizer_config_json["added_tokens_decoder"]
                    for token_id, token_json in added_tokens_decoder.items():
                        token_id = int(token_id)
                        if (token_id >= vocab_size):
                            logger.debug(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                            continue

                        token_content = token_json["content"]
                        token_type = SentencePieceTokenTypes.USER_DEFINED
                        token_score = -10000.0

                        # Map unk_token to UNKNOWN, other special tokens to CONTROL
                        # Set the score to 0.0 as in the original tokenizer.model
                        if ("special" in token_json) and token_json["special"]:
                            if token_content == tokenizer_config_json["unk_token"]:
                                token_type = SentencePieceTokenTypes.UNKNOWN
                            else:
                                token_type = SentencePieceTokenTypes.CONTROL
                            token_score = 0.0

                        logger.info(f"Setting added token {token_id} to '{token_content}' (type: {token_type}, score: {token_score:.2f})")
                        tokens[token_id] = token_content.encode("utf-8")
                        toktypes[token_id] = token_type
                        scores[token_id] = token_score

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["hidden_size"] // hparams["num_attention_heads"])

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith("q_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith("k_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def write_tensors(self):
        super().write_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@Model.register("DeepseekV2ForCausalLM")
class DeepseekV2Model(Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if "q_lora_rank" in hparams and hparams["q_lora_rank"] is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hparams["v_head_dim"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "yarn":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])
                self.gguf_writer.add_rope_scaling_orig_ctx_len(self.hparams["rope_scaling"]["original_max_position_embeddings"])
                self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * hparams["rope_scaling"]["mscale_all_dim"])

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def write_tensors(self):
        super().write_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


###### CONVERSION LOGIC ######


# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            lazy=self._lazy,
            args=(self,),
            func=(lambda s: s[0].numpy())
        )

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: torch.dtype, shape: torch.Size) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return LazyTorchTensor._wrap_fn(func)(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--awq-path", type=Path, default=None,
        help="Path to scale awq cache file",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "auto"], default="f16",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file",
    )
    parser.add_argument(
        "--use-temp-file", action="store_true",
        help="use the tempfile library while processing (helpful when running out of memory, process killed)",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dir_model = args.model

    if args.awq_path:
        sys.path.insert(1, str(Path(__file__).parent / 'awq-py'))
        from awq.apply_awq import add_scale_weights  # type: ignore[import-not-found]
        tmp_model_path = args.model / "weighted_model"
        dir_model = tmp_model_path
        if tmp_model_path.is_dir():
            logger.info(f"{tmp_model_path} exists as a weighted model.")
        else:
            tmp_model_path.mkdir(parents=True, exist_ok=True)
            logger.info("Saving new weighted model ...")
            add_scale_weights(str(args.model), str(args.awq_path), str(tmp_model_path))
            logger.info(f"Saved weighted model at {tmp_model_path}.")

    if not dir_model.is_dir():
        logger.error(f'Error: {args.model} is not a directory')
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_model / 'ggml-model-{ftype}.gguf'

    logger.info(f"Loading model: {dir_model.name}")

    hparams = Model.load_hparams(dir_model)

    with torch.inference_mode():
        try:
            model_class = Model.from_model_architecture(hparams["architectures"][0])
        except NotImplementedError:
            logger.error(f"Model {hparams['architectures'][0]} is not supported")
            sys.exit(1)

        model_instance = model_class(dir_model, ftype_map[args.outtype], fname_out, args.bigendian, args.use_temp_file, args.no_lazy, args.model_name)

        logger.info("Set model parameters")
        model_instance.set_gguf_parameters()

        logger.info("Set model tokenizer")
        model_instance.set_vocab()

        model_instance.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        if args.vocab_only:
            logger.info(f"Exporting model vocab to '{model_instance.fname_out}'")
            model_instance.write_vocab()
        else:
            logger.info(f"Exporting model to '{model_instance.fname_out}'")
            model_instance.write()

        logger.info(f"Model successfully exported to '{model_instance.fname_out}'")


if __name__ == '__main__':
    main()
