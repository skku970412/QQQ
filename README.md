## 📦 QQQ: LLaMA 3.1–8B W4A8 양자화 툴킷

**QQQ**는 **LLaMA 3.1–8B** 모델 전용으로 설계된 연구 기반 **포스트 트레이닝 양자화 툴킷**입니다. 4비트 가중치(W4)와 8비트 활성화(A8)를 적용하여 원본 성능을 거의 유지하면서 추론 속도를 대폭 향상시킵니다.

---

### 🔑 주요 기능 (LLaMA 3.1–8B 전용)

* **완전한 W4A8 양자화 파이프라인**

  * 선택적 SmoothQuant 스타일 활성화 평활화
  * 헤시안 기반 가중치 보정
* **커스텀 CUDA GEMM 커널**

  * 채널별 및 그룹별 W4A8 연산 지원
  * cuBLAS FP16 대비 최대 **3.7×** 속도 향상
* **정확도 향상 옵션**

  * 선택적 가중치 회전(Weight Rotation)
  * MSE 최적화 GPTQ 블록
* **vLLM 통합**

  * `--quantization qqq` 옵션으로 Marlin 커널 기반 고속 추론

---

### ✅ 지원 모델

| 모델           | 지원 여부 |
| ------------ | ----- |
| LLaMA 3.1–8B | ✅ 지원  |
| 기타 모든 모델     | ❌ 미지원 |

> **주의:** QQQ는 오직 **LLaMA 3.1–8B** 전용으로 최적화되어 있습니다.
> 또한 wikitext2 용 ppl만측정함

---

### 🚀 설치 방법

```bash
git clone https://github.com/skku970412/QQQ.git
cd QQQ
conda env create -f environment.yml
conda activate qqq-py39
pip install -v -e .
```

---

### ⚡ 빠른 시작

1. **모델 양자화**

   ```bash
   python examples/quant_model.py \
     --model_path /path/to/llama3.1-8b-fp16 \
     --tokenizer_path /path/to/tokenizer \
     --save_path /path/to/llama3.1-8b-w4a8 \
     --dataset wikitext2 --nsamples 128 \
     --w_quantizer FixedQuantize \
     --gptq_mse true --rotation true
   ```

2. **성능 평가**

   ```bash
   python examples/eval_model.py \
     --model_path /path/to/llama3.1-8b-w4a8 \
     --tokenizer_path /path/to/tokenizer \
     --tasks "ppl" \
     --eval_ppl --batch_size 8 --max_length 2048
   ```

3. **vLLM 추론 서버 실행**

   ```bash
   pip install vllm
   python vllm_serv.py
   ```

---



### 📜 라이선스 및 인용

본 프로젝트는 **Apache 2.0** 라이선스를 따릅니다. 연구에 사용하실 경우, 아래 논문을 인용해주세요:

```bibtex
@article{zhang2024qqq,
  title   = {QQQ: Quality Quattuor-Bit Quantization for Large Language Models},
  author  = {Ying Zhang and Peng Zhang and Mincong Huang and Jingyang Xiang and Yujie Wang and Chao Wang and Yineng Zhang and Lei Yu and Chuan Liu and Wei Lin},
  journal = {arXiv preprint arXiv:2406.09904},
  year    = 2024
}
```
