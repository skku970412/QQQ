import sys
sys.path = ['.'] + [p for p in sys.path if 'QQQ_jaeuk' not in p]

import argparse
import logging
import os
import torch
from QQQ.rotation import fuse_layer_norms, rotate_model
from QQQ.smooth import smooth, export_smoothed_model, quantize_model
from QQQ.gptq import apply_gptq
from QQQ.utils import (
    setup_seed,
    build_model_and_tokenizer,
    prepare_for_inference,
    free_memory,
    str2bool,
    remove_empty_parameters,
)

logger = logging.getLogger("QQQ")


# NOTE(HandH1998): If enable smooth, it is recommended to use the default configuration, no need to change
def parse_a_qconfig(args):
    parser = argparse.ArgumentParser(
        description="Activation Quantization Configuration Parser", add_help=False
    )

    parser.add_argument(
        "--a_quantizer",
        dest="quantizer",
        type=str,
        default="TokenFixedFakeQuantize",
        help="Quantizer for activation",
    )
    parser.add_argument(
        "--a_observer",
        dest="observer",
        type=str,
        default="MinMaxObserver",
        help="Observer for activation",
    )
    parser.add_argument(
        "--a_bit",
        dest="bit",
        type=int,
        default=8,
        help="Bit width for activation quantization",
    )
    parser.add_argument(
        "--a_symmetric",
        dest="symmetric",
        type=str2bool,
        default=True,
        help="Symmetric quantization for activation",
    )
    parser.add_argument(
        "--a_ch_axis",
        dest="ch_axis",
        type=int,
        default=0,
        help="Channel axis for activation quantization",
    )

    args, remaining_args = parser.parse_known_args(args)
    return args, remaining_args


# NOTE(HandH1998): If enable smooth, `w_quantizer=FixedQuantize, w_group_size=-1` is for weight per-channel quantizaiton,
# `w_quantizer=GroupFixedQuantize, w_group_size=128` is for weight per-group quantization.
# The other parameters can use the default configuration.
def parse_w_qconfig(args):
    parser = argparse.ArgumentParser(
        description="Weight Quantization Configuration Parser", add_help=False
    )

    parser.add_argument(
        "--w_quantizer",
        dest="quantizer",
        type=str,
        default="FixedQuantize",
        choices=["FixedQuantize", "GroupFixedQuantize"],
        help="Quantizer for weights, (`FixedQuantize` for per-channel, `GroupFixedQuantize` for per-group)",
    )
    parser.add_argument(
        "--w_observer",
        dest="observer",
        type=str,
        default="MinMaxObserver",
        help="Observer for weights",
    )
    parser.add_argument(
        "--w_bit",
        dest="bit",
        type=int,
        default=4,
        help="Bit width for weight quantization",
    )
    parser.add_argument(
        "--w_symmetric",
        dest="symmetric",
        type=str2bool,
        default=True,
        help="Symmetric quantization for weights",
    )
    parser.add_argument(
        "--w_ch_axis",
        dest="ch_axis",
        type=int,
        default=0,
        help="Channel axis for weight quantization (0 for per-channel, -1 for per-layer)",
    )
    parser.add_argument(
        "--w_group_size",
        dest="group_size",
        type=int,
        default=-1,
        choices=[-1, 128],
        help="Group size for weight quantization (-1 for per-channel, 128 for per-group)",
    )

    args, remaining_args = parser.parse_known_args(args)
    return args, remaining_args


# NOTE(HandH1998): If enable smooth, the `calibrate_path` should be changed to your own data path. The other parameters can use the default configuration.
def parse_smooth_args(args):
    parser = argparse.ArgumentParser(
        description="Smooth Configuration Parser", add_help=False
    )
    # Calibration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size of smooth calibration inference",
    )

    # Padding removal
    parser.add_argument(
        "--is_remove_padding",
        dest="is_remove_padding",
        type=str2bool,
        default=True,
        help="Remove padding during quantization",
    )

    # Smooth method
    parser.add_argument(
        "--smooth_method", dest="smooth_method", type=str, default="os+"
    )
    smooth_args, remaining_args = parser.parse_known_args(args)
    smooth_args.a_qconfig, remaining_args = parse_a_qconfig(remaining_args)
    smooth_args.w_qconfig, remaining_args = parse_w_qconfig(remaining_args)
    return smooth_args, remaining_args


# NOTE(HandH1998): `gptq_mse=False` is for `Smooth + GPTQ`, `gptq_mse=True` is for `Rotation + GPTQ`.
# `gptq_groupsize=-1` is for per-channel weight quantization, `gptq_groupsize=128` is for per-group weight quantization
def parse_gptq_args(args):
    parser = argparse.ArgumentParser(
        description="GPTQ Configuration Parser", add_help=False
    )
    parser.add_argument(
        "--gptq_sym",
        dest="sym",
        type=str2bool,
        default=True,
        help="Symmetric quantization for GPTQ, only support sym for now",
    )
    parser.add_argument(
        "--gptq_groupsize",
        dest="groupsize",
        type=int,
        default=-1,
        choices=[-1, 128],
        help="Group size for GPTQ (-1 for per-channel, 128 for per-group), it should be same with w_group_size when enable smooth",
    )
    parser.add_argument(
        "--gptq_mse", dest="mse", type=str2bool, default=True, help="Use MSE for GPTQ"
    )
    parser.add_argument(
        "--gptq_act_order",
        dest="act_order",
        type=str2bool,
        default=True,
        help="Activation order for GPTQ",
    )
    parser.add_argument(
        "--gptq_percdamp",
        dest="percdamp",
        type=float,
        default=0.01,
        help="Percentage damping for GPTQ",
    )

    parser.add_argument(
        "--gptq_wbits",
        dest="wbits",
        type=int,
        default=4,
        help="Bit width for weights in GPTQ",
    )
    parser.add_argument(
        "--gptq_static_groups",
        dest="static_groups",
        type=str2bool,
        default=True,
        help="Use static groups for GPTQ",
    )

    args, remaining_args = parser.parse_known_args(args)
    return args, remaining_args


def parse_rotation_args(args):
    parser = argparse.ArgumentParser(
        description="Rotation Configuration Parser", add_help=False
    )

    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )

    args, remaining_args = parser.parse_known_args(args)
    return args, remaining_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--smooth", type=str2bool, default=False)
    parser.add_argument("--rotation", type=str2bool, default=True)
    parser.add_argument("--max_length", dest="max_length", type=int, default=2048)
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # Calibration
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        choices=["wikitext2", "pile", "ptb", "new_ptb", "c4", "mix"],
        help="Calibration Dataset. If you want to use your own dataset, this should be the default value",
    )
    parser.add_argument(
        "--custom_dataset",
        type=str,
        default="",
        help="Custom Calibration Dataset. It should be your own dataset path. If you want to use the public dataset, this should be the default value",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples",
    )
    args, remaining_args = parser.parse_known_args()
    smooth_args, remaining_args = parse_smooth_args(remaining_args)
    gptq_args, remaining_args = parse_gptq_args(remaining_args)
    rotation_args, remaining_args = parse_rotation_args(remaining_args)
    return args, smooth_args, gptq_args, rotation_args


@torch.no_grad()
def main():
    args, smooth_args, gptq_args, rotation_args = parse_args()
    # set seed
    setup_seed(args.seed)

    # process save_path
    if args.save_path:
        sub_dir_name = args.model_path.split("/")[-1]
        args.save_path = os.path.join(args.save_path, sub_dir_name)
        os.makedirs(args.save_path, exist_ok=True)

    # tokenizer path
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # load model
    model, tokenizer = build_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.dtype
    )
    tokenizer.model_max_length = args.max_length 
    # rotate model
    if args.rotation:
        model = fuse_layer_norms(model)
        model, Q = rotate_model(model, rotation_args, args)
        free_memory()

    # NOTE(HandH1998): No smoothing would give better results for now
    if args.smooth:
        # smooth model
        assert smooth_args.w_qconfig.group_size == gptq_args.groupsize
        model = quantize_model(model, smooth_args, args)
        scale_list = smooth(model, tokenizer, smooth_args, args)
        del model
        del tokenizer
        free_memory()

        # load model and apply smooth scales
        model, tokenizer = build_model_and_tokenizer(
            args.model_path, args.tokenizer_path, args.dtype
        )
        tokenizer.model_max_length = args.max_length 
        if args.rotation:
            # NOTE(HandH1998): smooth scale should work on the rotated model
            model = fuse_layer_norms(model)
            model, _ = rotate_model(model, rotation_args, args, Q)
            free_memory()

        model = export_smoothed_model(model, scale_list)

    # apply gptq
    model = prepare_for_inference(model, args.device, args.dtype)
    model = apply_gptq(model, gptq_args, args)

    # quant_config
    model.config.quantization_config = {
        "group_size": gptq_args.groupsize,
        "quant_method": "qqq",
        "wbits": gptq_args.wbits,
    }

    # save quantized model
    # state_dict = remove_empty_parameters(model)
    # model.save_pretrained(args.save_path, state_dict=state_dict)
    
    
    
    # # Quantized model을 PyTorch pickle(.bin)으로 저장
    state_dict = remove_empty_parameters(model)
    model.save_pretrained(
        args.save_path,
        state_dict=state_dict,
        safe_serialization=False,   # ← 이 플래그를 꺼야 .bin으로 저장됩니다
    )
    # # (선택) .bin을 .pt로 바꾸고 싶다면:
    # bin = os.path.join(args.save_path, "pytorch_model.bin")
    # pt  = os.path.join(args.save_path, "pytorch_model.pt")
    # os.replace(bin, pt)
    # print(f"Saved model_state_dict.pt at {pt}")
    
    tokenizer.save_pretrained(args.save_path)
    logger.info(
        "Quant Finished! The quantized model is saved at {}.".format(args.save_path)
    )


if __name__ == "__main__":
    main()
