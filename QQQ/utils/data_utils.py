from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import random
import json
from transformers import logging
logging.set_verbosity_error()

def get_wikitext2(nsamples, seed, seqlen, tokenizer_path):
    print("get_wikitext2")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt",add_special_tokens=False )
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt",add_special_tokens=False )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_loaders(
    name="",
    nsamples=128,
    seed=0,
    seqlen=2048,
    tokenizer_path="",
    custom_data_path="",
):

    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer_path)
