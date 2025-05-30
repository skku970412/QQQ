import torch
from torch.nn.modules.module import Module

# ── 여기에 몽키패치 추가 ────────────────────────────────────────────
# _orig_apply = Module._apply
# def _debug_apply(self, fn):
#     for name, buf in self._buffers.items():
#         if buf is not None and buf.device.type != 'meta':
#             print(f"[PATCH] {self.__class__.__name__}.{name}: shape={tuple(buf.shape)} -> fn={fn}")
#     return _orig_apply(self, fn)

# Module._apply = _debug_apply



from tqdm import tqdm
import argparse
import collections
import torch.nn as nn
from transformers import AutoTokenizer
import lm_eval
# from lm_eval import tasks, simple_evaluate
# from lm_eval.models.huggingface import HFLM
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
os.makedirs("offload", exist_ok=True)

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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def eval_model(model, tokenizer, args):
    max_length = args.max_length
    results = {}
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
            for i in tqdm(range(nsamples)):
                batched_inps = testenc[:, (i * max_length) : ((i + 1) * max_length)].to(
                    model.device
                )
                outputs = model.model(batched_inps)
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

            result = collections.defaultdict(dict)
            versions = collections.defaultdict(dict)
            n_shot = collections.defaultdict(dict)
            result[task]["ppl"] = ppl.item()
            versions[task] = 0
            n_shot[task] = 0
            t_results = {
                "results": dict(result),
                "versions": dict(versions),
                "n-shot": dict(n_shot),
            }
            print(t_results)
            update_results(results, t_results)
    # # eval other datasets
    # if args.tasks != "":
    #     task_names = pattern_match(args.tasks.split(","), tasks.TaskManager().all_tasks)
    #     lm = HFLM(
    #         pretrained=model,
    #         backend="causal",
    #         device="cuda",
    #         batch_size=args.batch_size,
    #         tokenizer=tokenizer,
    #         max_lengt=max_length,
    #     )
    #     t_results = simple_evaluate(
    #         lm,
    #         tasks=task_names,
    #         num_fewshot=args.num_fewshot,
    #         batch_size=args.batch_size,
    #     )
    #     update_results(results, t_results)

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
    torch.cuda.empty_cache()

    model = quant_model_class.from_pretrained(
        args.model_path,
        config=config,
        # attn_implementation="flash_attention_2",
        quant_config=quant_config,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    
    
###############################################################

    
#     def move_module_in_chunks(module, device):
#         for child in module.children():
#             move_module_in_chunks(child, device)
#         for name, buf in list(module._buffers.items()):
#             if buf is not None:
#                 size_mb = buf.numel() * buf.element_size() // (1024**2)
#                 if size_mb > 1000:  # 1GB 넘으면 경고
#                     print(f"[WARN] Buffer {name} in {module.__class__.__name__} is {size_mb} MB")
#                 module._buffers[name] = buf.to(device, non_blocking=True)
#         for name, param in list(module._parameters.items()):
#             if param is not None:
#                 size_mb = param.numel() * param.element_size() // (1024**2)
#                 if size_mb > 1000:
#                     print(f"[WARN] Parameter {name} in {module.__class__.__name__} is {size_mb} MB")
#                 module._parameters[name] = param.to(device, non_blocking=True)


#     def remove_large_causal_mask(module, verbose=True):
#         # 모든 하위 모듈 재귀적으로 순회
#         for child in module.children():
#             remove_large_causal_mask(child, verbose)
#         # buffer 중 이름에 'causal_mask' 포함된 것 제거
#         for name in list(module._buffers.keys()):
#             if "causal_mask" in name:
#                 if verbose:
#                     buf = module._buffers[name]
#                     size_mb = buf.numel() * buf.element_size() // (1024**2)
#                     print(f"[REMOVE] Removing buffer {name} ({size_mb} MB) from {module.__class__.__name__}")
#                 del module._buffers[name]


#     remove_large_causal_mask(model)

#     from types import MethodType

#     def safe_update_causal_mask(self, attention_mask, inputs_embeds):
#         seq_length = inputs_embeds.shape[1]
#         device = inputs_embeds.device
#         dtype = torch.float16
#         causal_mask = torch.triu(
#             torch.ones((seq_length, seq_length), device=device, dtype=dtype), diagonal=1
#         )
#         causal_mask = causal_mask[None, None, :, :]
#         return causal_mask
#     model.model._update_causal_mask = MethodType(safe_update_causal_mask, model.model)
#     # 여기에서 patch!
#     for idx, layer in enumerate(model.model.layers):
#         layer.self_attn._update_causal_mask = MethodType(safe_update_causal_mask, layer.self_attn)
#         print(f"Patched self_attn of layer {idx}: {layer.self_attn._update_causal_mask}")

#     move_module_in_chunks(model, torch.device("cuda:0"))
# ###############################################################
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    torch.cuda.empty_cache()

    eval_model(model, tokenizer, args)
