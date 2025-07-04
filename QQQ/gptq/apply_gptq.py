import os
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
CPU = torch.device("cpu")
from QQQ.utils import (
    get_loaders,
    get_model_architecture,
    str2torch_device,
    find_layers,
    recurse_setattr,
    free_memory,
)
from .models import get_gptq_model_func
from .qlinear import QuantLinear


@torch.no_grad()
def apply_gptq(model, gptq_config, args):
    gptq_config.nsamples = args.nsamples
    dataloader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        tokenizer_path=args.tokenizer_path,
        seqlen=args.max_length,
        custom_data_path=args.custom_dataset,
    )
    model_type = get_model_architecture(model.config)
    gptq_func = get_gptq_model_func(model_type)
    device = str2torch_device(args.device)

    quantizers = gptq_func(model, dataloader, device, gptq_config)
    torch.save(quantizers, os.path.join(args.save_path, "quantizers.pth"))

    pack_model(
        model,
        quantizers,
        bits=gptq_config.wbits,
        group_size=gptq_config.groupsize,
    )
    free_memory()
    return model


@torch.no_grad()
def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    force_layer_back_to_cpu: bool = False,
):
    CPU = torch.device("cpu")
    if force_layer_back_to_cpu:
        model.to(CPU)

    # logger.info("Packing model...")
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(
        model,
        quantizers,
        bits,
        group_size,
    )
    qlayers = find_layers(model, [QuantLinear])

    pbar = tqdm(qlayers.keys(), leave=True)
    for name in pbar:
        pbar.set_description(f"Packing {name}...", refresh=True)

        # scale, zero, g_idx, scale_extra = quantizers[name]
        # # so far can only pack layer on CPU
        # layer_device = qlayers[name].device
        # qlayers[name].to(CPU)
        # layers[name], scale, zero, g_idx, scale_extra = (
        #     layers[name].to(CPU),
        #     scale.to(CPU),
        #     zero.to(CPU),
        #     g_idx.to(CPU),
        #     scale_extra.to(CPU) if scale_extra is not None else None,
        # )
        # qlayers[name].pack(layers[name], scale, scale_extra)
        # qlayers[name].to(layer_device)
        # del layers[name]
        # free_memory()
        scale, zero, g_idx, scale_extra = quantizers[name]

        # 1) CPU에서만 pack 수행
        qlayers[name].pack(
            layers[name].to(CPU),
            scale.to(CPU),
            scale_extra.to(CPU) if scale_extra is not None else None
        )

        # 2) CPU-side 객체 즉시 해제
        del layers[name], scale, zero, g_idx, scale_extra
        free_memory()

        # 3) 패킹된 모듈만 원래 장치로 이동
        qlayers[name].to(device=qlayers[name].device)        
        
    print("Model packed.")


def make_quant(
    module,
    names,
    bits,
    group_size,
    trainable: bool = False,
):
    if isinstance(module, QuantLinear):
        return

    for name, submodule in module.named_modules():
        if name in names:
            ori_layer_device = next(submodule.parameters()).device

            if isinstance(submodule, nn.Linear):
                in_features = submodule.in_features
                out_features = submodule.out_features
            elif isinstance(submodule, nn.Conv2d):
                in_features = submodule.in_channels
                out_features = submodule.out_channels
            elif isinstance(submodule, transformers.pytorch_utils.Conv1D):
                in_features = submodule.weight.shape[0]
                out_features = submodule.weight.shape[1]
            bias = submodule.bias is not None
            new_layer = QuantLinear(
                bits,
                group_size,
                in_features,
                out_features,
                bias,
                trainable=trainable,
                weight_dtype=submodule.weight.dtype,
            )
            # new_layer.device = ori_layer_device
            # recurse_setattr(module, name, new_layer.to(ori_layer_device))
            new_layer.device = CPU
            recurse_setattr(module, name, new_layer.to(CPU))