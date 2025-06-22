````markdown
## 📦 QQQ: LLaMA 3.1–8B 전용 W4A8 양자화 툴킷

**QQQ**는 **LLaMA 3.1 8B 모델 전용**으로 설계된 연구 기반의 W4A8(4비트 가중치, 8비트 활성화) 포스트 트레이닝 양자화 툴킷입니다.  
다른 모델은 지원하지 않으며, **오직 LLaMA 3.1–8B**만을 위해 최적화되어 있습니다.

---

### 🔑 주요 기능 (LLaMA 3.1–8B 전용)

- **W4A8 전체 양자화 파이프라인**  
  - SmoothQuant 스타일 활성화 평활화 (선택적)  
  - 헤시안 기반 가중치 보정  
- **커스텀 CUDA GEMM 커널**  
  - per-channel 및 per-group W4A8 연산  
  - cuBLAS FP16 대비 최대 **3.7×** 속도 향상  
- **정확도 향상 옵션**  
  - 선택적 Weight Rotation  
  - MSE 최적화 GPTQ 블록  
- **vLLM 통합 지원**  
  - `--quantization qqq` 옵션으로 Marlin 커널 기반 고속 추론

---

### ✅ 지원 모델

| 모델                | 지원 여부 |
|---------------------|-----------|
| LLaMA 3.1 8B        | ✅ 지원   |
| 기타 모든 모델      | ❌ 미지원 |

> 📌 본 저장소는 **LLaMA 3.1 8B 전용**입니다. 다른 모델에는 적용할 수 없습니다.

---

### 🚀 빠른 설치

```bash
git clone https://github.com/skku970412/QQQ.git
cd QQQ
conda env create -f environment.yml
conda activate qqq-py39
pip install -v -e .
````

---

### ⚡️ 빠른 시작

1. **양자화**

   ```bash
   python examples/quant_model.py \
     --model_path /path/to/llama3.1-8b-fp16 \
     --tokenizer_path /path/to/tokenizer \
     --save_path /path/to/llama3.1-8b-w4a8 \
     --dataset wikitext2 --nsamples 128 \
     --w_quantizer FixedQuantize \
     --gptq_mse true --rotation true
   ```
2. **성능 평가** (Perplexity 및 Zero-Shot Accuracy)

   ```bash
   python examples/eval_model.py \
     --model_path /path/to/llama3.1-8b-w4a8 \
     --tokenizer_path /path/to/tokenizer \
     --tasks "piqa,winogrande,hellaswag,arc_challenge,arc_easy" \
     --eval_ppl --batch_size 8 --max_length 2048
   ```
3. **vLLM 추론**

   ```bash
   pip install vllm
   python vllm_serv.py
   ```

---

### 📝 변경 이력 (하이라이트)(내가한거아님._.)

* **2025-03-12** ICLR 2025 SCI-FM 워크숍 채택
* **2024-09-26** SmoothQuant 리팩토링 및 커스텀 데이터셋 지원
* **2024-09-12** Qwen-2 모델(0.5 B → 72 B) 추가
* **2024-08-26** Weight Rotation 통합 (지연 시간 증가 없음)
* **2024-07-31** vLLM 공식 머지 완료 (PR 참조)
* **2024-07-17** `quant_config.json` 자동 내장
* **2024-06-17** arXiv 사전 인쇄 릴리스
* **2024-06-03** 초기 코드 공개
* 와... ㄷㄷ 레전드

---


📜 **라이선스 및 인용**
Apache 2.0 — 연구에 사용하실 경우 아래 논문을 인용해 주세요:

```bibtex
@article{zhang2024qqq,
  title   = {QQQ: Quality Quattuor-Bit Quantization for Large Language Models},
  author  = {Ying Zhang and Peng Zhang and Mincong Huang and Jingyang Xiang and Yujie Wang and Chao Wang and Yineng Zhang and Lei Yu and Chuan Liu and Wei Lin},
  journal = {arXiv preprint arXiv:2406.09904},
  year    = 2024
}
```

```
```
