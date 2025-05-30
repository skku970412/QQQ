# bool-mask 패치
# 1. LlamaModel._update_causal_mask 메서드 교체
# 2. bool-mask 생성
# 3. bool-mask을 CPU에 저장
# 4. bool-mask을 model.model.causal_mask에 저장
# 5. model.model.causal_mask을 CPU에 저장
# 6. model.model.causal_mask을 model.model.causal_mask에 저장

import torch
import torch.nn as nn
import functools
from typing import Optional
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    PretrainedConfig,
)
from .utils import str2torch_dtype, str2torch_device
from accelerate.big_modeling import (
    dispatch_model,
    infer_auto_device_map,
    get_balanced_memory,
)

_MODEL_TYPE = {
    "LlamaForCausalLM": "llama",
    "LLaMAForCausalLM": "llama",
    "Qwen2ForCausalLM": "qwen2",
}
# ----------------- ❶ GPTQ/OOM 대응: bool-mask 패치 -----------------
def _llama_bool_causal_mask(self, attention_mask, inputs_embeds):
    """
    drop-in replacement for LlamaModel._update_causal_mask
    • causal_mask 를 always **bool** 로 유지
    • 필요 크기만 그때그때 생성
    • dtype 변환(x), device 만 맞춰 줌
    => VRAM < 100 MB  /  no OOM
    """
    bsz, seq_len, _ = inputs_embeds.size()

    if (getattr(self, "causal_mask", None) is None) or (
        seq_len > self.causal_mask.shape[-1]
    ):
        # 새(mask) 생성 - still bool
        self.causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=inputs_embeds.device),
            diagonal=1,
        )

    # ★ bool 그대로!  ( 이후 attention 안에서 float16 곱셈 XOR 로 처리됨 )
    causal = self.causal_mask[None, None, :seq_len, :seq_len]

    if attention_mask is not None:
        causal = causal | attention_mask[:, None, None, :]

    return causal
# 실제 메서드 교체
from transformers.models.llama.modeling_llama import LlamaModel
LlamaModel._update_causal_mask = _llama_bool_causal_mask
# ---------------------------------------------------------------


def build_model_and_tokenizer(
    model_path, tokenizer_path, dtype: str, trust_remote_code: bool = True
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    tokenizer.__call__ = functools.partial(tokenizer.__call__, add_special_tokens=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {
        # "load_in_8bit": True,              # 8-bit로 바로 로드 (bitsandbytes 필요)
        # "load_in_4bit": True,            # 4-bit로 하고 싶으면 8bit 대신 사용
        "torch_dtype": torch.float16,      # 커널 dtype
        "device_map": {"": "cpu"},         # 처음엔 전부 CPU
        "max_memory": {"cpu": "40GiB"},    # CPU RAM 한도
        "low_cpu_mem_usage": True,         # 로드 단계 메모리 절약
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, **kwargs
    )
    # model.model.causal_mask = None  
    # ▶▶ 3080 VRAM /OOM 방지—최대 2048 토큰용 마스크만 미리 생성
    max_len = 2048                         # 필요하면 1024 · 4096 등으로 조정
    mask = torch.triu(
        torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1
    )
    # model.model.causal_mask = mask         # CPU 텐서라 VRAM 차지 X
    # ---- ① max_length 만큼만 마스크를 만들고
    max_len = 2048 if hasattr(model.config, "max_length") else 4096
    model.config.max_position_embeddings = max_len

    causal = torch.full(
        (max_len, max_len), True, dtype=torch.bool
    ).triu(diagonal=1)

    # ---- ② CPU bool 텐서 그대로 유지 (VRAM X)
    model.model.causal_mask = causal     # <= bool, device='cpu'
    return model, tokenizer


def get_model_architecture(config):
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_TYPE:
            return _MODEL_TYPE[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_TYPE.keys())}"
    )


def prepare_for_inference(model, device, dtype):
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1
    model.to(str2torch_dtype(dtype))
    if device == "cuda" and torch.cuda.device_count() > 1:
        max_memory = get_balanced_memory(
            model,
            no_split_module_classes=model._no_split_modules,
            dtype=str2torch_dtype(dtype),
        )
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=model._no_split_modules,
            max_memory=max_memory,
            dtype=str2torch_dtype(dtype),
        )
        print(device_map)
        dispatch_model(model, device_map=device_map)
    else:
        model.to(str2torch_device(device))
    model.eval()
    return model


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


import functools


def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)


def get_model_config(
    model_path: str, trust_remote_code: bool = True, revision: Optional[str] = None
) -> PretrainedConfig:
    try:
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, revision=revision
        )
    except ValueError as e:
        if (
            not trust_remote_code
            and "requires you to execute the configuration file" in str(e)
        ):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e
    return config


def get_transformer_layers(model, model_type):
    if model_type in ["llama", "qwen2"]:
        return [layer for layer in model.model.layers]
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_lm_head(model, model_type):
    if model_type in ["llama", "qwen2"]:
        return model.lm_head
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_pre_head_layernorm(model, model_type):
    # NOTE(HandH1998): only support RMSnorm
    if model_type in ["llama", "qwen2"]:
        pre_head_layernorm = model.model.norm
        return pre_head_layernorm
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type in ["llama", "qwen2"]:
        return [model.model.embed_tokens]
    else:
        raise ValueError(f"Unknown model type {model_type}")

def remove_empty_parameters(model):
    state_dict = {}
    for k, v in model.state_dict().items():
        if v.numel() > 0:
            state_dict[k] = v
    return state_dict