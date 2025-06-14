#머델유틸원본근접
# llama 2 모델에서는 준수하게 돌아감
#llama3 는 메모리터짐

# PPL 4.4468
#에다가
# 몽키패치한거
# -> 
# {'results': {'wikitext2': {'ppl': 4042.778564453125}}, 'versions': {'wikitext2': 0}, 'n-shot': {'wikitext2': 0}}
# |  Tasks  |Version|Filter|n-shot|Metric|  Value  |   |Stderr|
# |---------|------:|------|-----:|------|--------:|---|------|
# |wikitext2|      0|      |     0|ppl   |4042.7786|   |      |


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


def build_model_and_tokenizer(
    model_path, tokenizer_path, dtype: str, trust_remote_code: bool = True
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    tokenizer.model_max_length = 2048
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {
        "torch_dtype": str2torch_dtype(dtype),
        "device_map": "auto",
    }
    # 이건 원래코드드
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, trust_remote_code=trust_remote_code, **kwargs
    # )
    # spda써보기
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        device_map="auto",
        attn_implementation="flash_attention_2",
        max_memory={0: "18GB",
                    1: "18GB",
                    2: "18GB",},             # 전체를 일단 CPU에 로드

    )

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