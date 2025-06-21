# QQQ — Conda Environment Setup Guide

**Purpose :** spin up a fully reproducible Python + CUDA environment so you can **quantize, evaluate and run W4A8 QQQ models** right after cloning the repo.

---

## 1‑Step Install *(recommended)*

The repo already ships with an `environment.yml` describing every required package (Python 3.9, PyTorch 2.6 + CUDA 12.4, Transformers 4.38.2, etc.). Use it to create the entire environment in one shot.

```bash
# 1 Clone QQQ
git clone https://github.com/skku970412/QQQ.git
cd QQQ

# 2 (optional) delete the hard‑coded prefix line
#    because it contains the path of the author’s machine
sed -i '/^prefix:/d' environment.yml          # Linux / macOS
#  ► Windows PowerShell
#  (Get-Content environment.yml) -notmatch '^prefix:' | Set-Content environment.yml

# 3 Create & activate the conda env (≈ 5‑10 min)
conda env create -f environment.yml
conda activate qqq-py39          # env name comes from environment.yml
```

### Quick sanity‑check

```bash
python - <<'PY'
import torch, transformers, accelerate, platform
print("CUDA ", torch.version.cuda, "| GPU =", torch.cuda.get_device_name(0))
print("PyTorch     ", torch.__version__)
print("Transformers", transformers.__version__)
print("Accelerate  ", accelerate.__version__)
print("Python     ", platform.python_version())
PY
```

If all versions print without errors, you are good to go.

---

## Minimal install — build your own env

Need a lighter test environment or different CUDA version? Start from the **minimal** template below and tweak versions as needed.

<details>
<summary>🔧 `minimal_environment.yml` template</summary>

```yaml
name: qqq-min
channels:
  - conda-forge
  - nvidia
  - defaults
dependencies:
  # Core
  - python=3.9
  - pip
  # GPU stack (CUDA 12.4 — change to cu118 / cu121 … if required)
  - cudatoolkit=12.4
  - pytorch=2.6.0
  - torchvision=0.21.0
  - torchaudio=2.6.0
  # Essential libs
  - accelerate>=1.7
  - zstandard
  - pip:
      - transformers==4.38.2
      - datasets==2.16.1
      - easydict
      - lm_eval==0.4.2
      - fast-hadamard-transform==1.0.4.post1
      - sympy==1.13.1
      - triton==3.2.0
```

</details>

```bash
conda env create -f minimal_environment.yml
conda activate qqq-min
```

> **Different CUDA build?** Pair `pytorch` and `cudatoolkit` with the same CUDA tag (e.g. 11.8) or follow the official `pip install torch==…+cu118` instructions from [https://pytorch.org](https://pytorch.org).

---

## Extra setup & handy tips

| Item                      | Notes                                                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Environment variables** | An example `env_vars.txt` is provided. Typically you only need to adjust `CUDA_HOME` and `LD_LIBRARY_PATH` to your local GPU/CUDA paths.         |
| **Compile C++/CUDA ops**  | Run `pip install -v -e .` once inside the activated env to build QQQ’s custom kernels.                                                           |
| **Jupyter notebooks**     | Add `conda install jupyterlab` if you prefer interactive prototyping.                                                                            |
| **Keeping the env fresh** | Sync with the latest commit via `conda env update -f environment.yml --prune` (or `mamba env update …`).                                         |
| **Memory optimisation**   | Huge models (≥ 30 B) may not fit a single GPU — use vLLM, Hugging Face Accelerate, or 4‑bit KV‑cache tricks (`bitsandbytes`) to shard / offload. |

---

## Troubleshooting FAQ

| Symptom                          | Fix                                                                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `libcublas.so not found`         | Your conda CUDA build and installed NVIDIA driver are mismatched. Check `nvidia-smi` and reinstall `pytorch + cudatoolkit` matching that driver. |
| Conda dependency solving is slow | Install **mamba** for a 2‑3× speed‑up: `conda install -n base -c conda-forge mamba`.                                                             |
| `prefix already exists` error    | A previous env lives at the same path. Remove it with `conda env remove -p <path>` or create a new env name using `-n <new_name>`.               |

---

### 🚀 All set!

You can now run `examples/quant_model.py`, `examples/eval_model.py`, or `examples/test_model.py` to **quantize, benchmark, and generate text with your 3.18 B W4A8 QQQ model**.
