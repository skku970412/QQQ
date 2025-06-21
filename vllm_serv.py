import os
# GPU 인덱스 4만 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from vllm import LLM, SamplingParams
# 이하 로직 동일…
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 대화형 REPL:
입력한 프롬프트에 대해 max_new_tokens만큼 생성된 답변을 출력
"""

import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ────────────────────────── 설정값 ──────────────────────────────────────────
MODEL_PATH    = "/home/eiclab/eiclab04/urp2025/QQQ_git/QQQ/qqq-llama3-8b_base/Llama-3.1-8B"
TOKENIZER_PATH= "/home/eiclab/eiclab04/urp2025/QQQ_git/QQQ/qqq-llama3-8b_base/Llama-3.1-8B"
NEW_TOKENS    = 256         # 생성할 토큰 수
TEMPERATURE   = 0.2
TOP_P         = 0.3
# ────────────────────────────────────────────────────────────────────────────

def main():
    # (선택) 특정 GPU만 쓰고 싶다면
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 토크나이저 & LLM 로드
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    llm = LLM(model=MODEL_PATH, tokenizer=TOKENIZER_PATH)

    # 샘플링 파라미터
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.3,          # 적당히 다양성 주기
        top_p=0.9,                # 누적 확률 90%까지
        top_k=50,                 # 상위 50개만
        repetition_penalty=1.2,   # 반복 페널티
        # no_repeat_ngram_size=2,   # 2‑gram 반복 금지
        max_tokens=128,
    )


    print("vLLM REPL. 빈 줄 입력 시 종료합니다.\n")
    while True:
        prompt = input("Prompt> ")
        if not prompt.strip():
            print("Exiting.")
            break

        # 생성
        outputs = llm.generate([prompt], sampling_params)

        # 첫번째(유일한) 요청의 생성 결과 출력
        generated = outputs[0].outputs[0].text
        print(f"\n=== Generated ({len(generated.split()):,} tokens) ===")
        print(generated)
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
