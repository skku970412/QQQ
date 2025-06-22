# QQQ: Quality Quattuor‑Bit Quantization for Large Language Models

> **Effortlessly compress — and *accelerate* — LLMs with 4‑bit weights (W4) & 8‑bit activations (A8).**

QQQ is a research‑driven, hardware‑friendly W4A8 post‑training quantization toolkit.
It allows **3.18 B‑parameter and larger models** to run up to **≈ 2.2× faster** than their FP16 counterparts while retaining near‑original perplexity and zero‑shot accuracy.

<div align="center">
  <img src="assets/figures/throughput.png" alt="Throughput comparison" width="600"/>
</div>

---

## 🔑 Key Features

* **W4A8 end‑to‑end pipeline** – adaptive activation smoothing + Hessian‑guided weight compensation.
* **Custom CUDA GEMM kernels** – per‑channel & per‑group W4A8 GEMMs deliver up to **3.7×** speed‑up over cuBLAS FP16.
* **Rotation & GPTQ hooks** – optional weight rotation and MSE‑optimised GPTQ blocks for extra accuracy.
* **Narrow model support** –
  *This fork adds turn‑key scripts for a 3.18 B custom model.*
* **vLLM integration** – one‑line deployment on the high‑throughput vLLM runtime.

---

## 🗂️ Repository Layout

| Path                              | What’s inside                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `assets/figures/`                 | Experiment plots & diagrams                                                   |
| `csrc/`                           | C++ / CUDA kernels and Triton utilities                                       |
| `examples/`                       | Python reference scripts (`quant_model.py`, `eval_model.py`, `test_model.py`) |
| `scripts/`                        | Handy shell wrappers for batch jobs (quantize / eval / infer)                 |
| `third-party/`                    | Vendored code — fast-hadamard-transform …                          |
| Top‑level `*.ipynb`               | Reproducible notebooks for ablation & LLaMA‑3 evaluation                      |
| `environment.yml`, `env_vars.txt` | Conda manifest & env variable template                                        |
| `setup.py` / `requirements.txt`   | PEP‑517 build & minimal pip deps                                              |

📄 *The full directory listing is visible on GitHub* ([github.com](https://github.com/skku970412/QQQ/tree/main))

---

## 🚀 Quick Installation

```bash
# clone & enter
git clone https://github.com/skku970412/QQQ.git
cd QQQ

# create the full Conda env (Python 3.9 / CUDA 12.4)
conda env create -f environment.yml
conda activate qqq-py39

# build C++/CUDA extensions
pip install -v -e .
```

For alternative setups or lighter images, see **README\_conda.md**.

---

## ⚡️ Quick Start

### 1. Quantise a model

```bash
python examples/quant_model.py \
  --model_path  /path/to/fp16-model \
  --tokenizer_path /path/to/tokenizer \
  --dtype float16 \
  --smooth false \   # enable SmoothQuant style smoothing if needed
  --rotation true \   # optional weight rotation
  --dataset wikitext2 --nsamples 128 \
  --w_quantizer FixedQuantize --w_group_size -1 \
  --gptq_mse true --gptq_groupsize -1 \
  --save_path  /path/to/w4a8-model
```

### 2. Evaluate perplexity & zero‑shot accuracy

```bash
python examples/eval_model.py \
  --model_path      /path/to/w4a8-model \
  --tokenizer_path  /path/to/tokenizer \
  --tasks "piqa,winogrande,hellaswag,arc_challenge,arc_easy" \
  --eval_ppl --batch_size 8 --max_length 2048
```

### 3. Inference with vLLM

```
pip install vllm
python vllm_serv.py
```

---

## 🗓️ Changelog (highlights)

* **2025‑03‑12**  Paper accepted at **ICLR 2025 SCI‑FM workshop**.([github.com](https://github.com/HandH1998/QQQ?utm_source=chatgpt.com))
* **2024‑09‑26**  Smooth calibration code refactored; custom datasets supported.
* **2024‑09‑12**  Added Qwen‑2 models (0.5 B → 72 B).
* **2024‑08‑26**  Integrated weight rotation (accuracy ↑, no latency cost).
* **2024‑07‑31**  Merged into **vLLM** master; see linked PR for details.
* **2024‑07‑17**  `quant_config.json` now auto‑embedded in `config.json`.
* **2024‑06‑17**  Pre‑print released on arXiv.
* **2024‑06‑03**  Initial code release.

---

## 🤝 Contributing

Pull requests are welcome — especially for **new model recipes, bug fixes, and kernel optimisations**.
Please create an issue first if you plan a large change.

---

## 📜 License & Citation

QQQ is released under the **Apache 2.0** license.
If you use this codebase or its kernels in your research, please cite:

```bibtex
@article{zhang2024qqq,
  title   = {QQQ: Quality Quattuor-Bit Quantization for Large Language Models},
  author  = {Ying Zhang and Peng Zhang and Mincong Huang and Jingyang Xiang and Yujie Wang and Chao Wang and Yineng Zhang and Lei Yu and Chuan Liu and Wei Lin},
  journal = {arXiv preprint arXiv:2406.09904},
  year    = 2024
}
```

---

*Happy quantising!* 🎉
