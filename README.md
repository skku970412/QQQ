## 📦 QQQ: LLaMA 3.1–8B 전용 W4A8 양자화 툴킷

**QQQ**는 **LLaMA 3.1 8B 모델 전용**으로 설계된 연구 기반의 W4A8 (4-bit weight, 8-bit activation) 포스트 트레이닝 양자화(PTQ) 툴킷입니다.  
다른 모델은 지원하지 않으며, 모든 기능은 **LLaMA 3.1–8B에만 최적화**되어 있습니다.

---

### 🔑 주요 기능

- **W4A8 전체 양자화 파이프라인**
  - SmoothQuant 스타일의 활성화 평활화 지원 (선택적)
  - 헤시안 기반 가중치 보정
- **커스텀 CUDA GEMM 커널**
  - per-channel 및 per-group 방식 W4A8 연산
  - cuBLAS FP16 대비 최대 **3.7×** 속도 향상
- **정확도 향상 기술**
  - 선택적 Weight Rotation
  - MSE 최적화 GPTQ 블록
- **vLLM 통합**
  - `--quantization qqq` 옵션으로 Marlin 커널 기반 고속 추론

---

### ✅ 지원 모델

| 모델 이름              | 지원 여부 |
|------------------------|-----------|
| ✅ LLaMA 3.1 8B         | 지원됨     |
| ❌ LLaMA 3.0 / 7B / 13B / 65B | 미지원 |
| ❌ Qwen, Mistral, OPT, Bloom 등 | 미지원 |

> 📌 본 저장소는 **LLaMA 3.1–8B 전용**입니다. 다른 모델에 적용하면 예상치 못한 오류가 발생할 수 있습니다.

---

### ⚙️ 설치

```bash
git clone https://github.com/skku970412/QQQ.git
cd QQQ
conda env create -f environment.yml
conda activate qqq-py39
pip install -v -e .
📜 License & Citation
QQQ is released under the Apache 2.0 license. If you use this codebase or its kernels in your research, please cite:

@article{zhang2024qqq,
  title   = {QQQ: Quality Quattuor-Bit Quantization for Large Language Models},
  author  = {Ying Zhang and Peng Zhang and Mincong Huang and Jingyang Xiang and Yujie Wang and Chao Wang and Yineng Zhang and Lei Yu and Chuan Liu and Wei Lin},
  journal = {arXiv preprint arXiv:2406.09904},
  year    = 2024
}
Happy quantising! 🎉
