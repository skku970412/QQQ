from tqdm import tqdm
import argparse
import collections
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import lm_eval
from lm_eval import tasks, simple_evaluate
from lm_eval.models.huggingface import HFLM
from QQQ.utils import (
    get_model_architecture,
    get_model_config,
    get_loaders,
    pattern_match,
    update_results,
    setup_seed,
)
from QQQ.gptq.models import get_quantized_model_class

import os
import psutil
import subprocess

def get_driver_gpu_used(gpu_index: int = 0) -> float:
    """nvidia-smi를 이용한 GPU 사용량 측정 (MiB)"""
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-i", str(gpu_index)
    ])
    return max(float(x) for x in out.decode().splitlines() if x.strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="path contains model weight and quant config",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="path contains tokenizer",
    )
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


import time

@torch.no_grad()
def eval_model(model, tokenizer, args):
    max_length = args.max_length
    results = {}
    # ---- 시간 및 peak 메모리 측정 준비 ----
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # eval ppl
    if args.eval_ppl:
        for task in ["wikitext2"]:
            _, testloader = get_loaders(
                task,
                seed=0,
                tokenizer_path=args.tokenizer_path,
                seqlen=max_length,
            )
            if "c4" in task:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // max_length

            nlls = []
            infer_times = []  # 각 배치별 시간 저장 (옵션)
            for i in tqdm(range(nsamples)):
                batched_inps = testenc[:, (i * max_length) : ((i + 1) * max_length)].to(
                    model.device
                )
                t0 = time.time()  # 배치별 시간 측정 시작
                outputs = model.model(batched_inps)
                t1 = time.time()  # 배치별 시간 측정 종료
                infer_times.append(t1 - t0)  # 리스트에 추가

                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * max_length) : ((i + 1) * max_length)][
                    :, 1:
                ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * max_length
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * max_length))

            # --- 시간 및 peak 메모리 측정 종료 ---
        # 2. 아래 블록을 바로 이어서 추가하세요
        # --- 시간 및 peak 메모리 측정 종료 ---
        total_time = time.time() - start_time
        peak_mem_pt = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        peak_mem_driver = get_driver_gpu_used(0)  # GPU index 0
        cpu_rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # MB

        result = collections.defaultdict(dict)
        versions = collections.defaultdict(dict)
        n_shot = collections.defaultdict(dict)
        result[task]["ppl"] = ppl.item()
        result[task]["peak_mem_PT_MB"] = peak_mem_pt
        result[task]["peak_mem_nvidia_smi_MB"] = peak_mem_driver
        result[task]["cpu_mem_MB"] = cpu_rss
        result[task]["total_infer_time_sec"] = total_time
        result[task]["avg_infer_time_per_batch_sec"] = sum(infer_times) / len(infer_times)
        versions[task] = 0
        n_shot[task] = 0
        t_results = {
            "results": dict(result),
            "versions": dict(versions),
            "n-shot": dict(n_shot),
        }
        print(t_results)
        update_results(results, t_results)
    print(lm_eval.utils.make_table(results))





if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    config = get_model_config(args.model_path)
    quant_config = config.quantization_config
    # NOTE(HandH1998): delete quantization_config to avoid getting into transformers' quantization method validation,
    # as transformers doesn't support qqq for now
    del config.quantization_config
    model_type = get_model_architecture(config)
    quant_model_class = get_quantized_model_class(model_type)
    model = quant_model_class.from_pretrained(
        args.model_path,
        config=config,
        quant_config=quant_config,
        device_map="sequential",
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    eval_model(model, tokenizer, args)
