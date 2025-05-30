CUDA_VISIBLE_DEVICES=3,4,5 python3 examples/quant_model.py   --model_path     ./models/Llama-3.1-8B   --tokenizer_path ./models/Llama-3.1-8B   --dtype float16   --smooth false   --rotation true   --dataset wikitext2   --nsamples 128   --w_quantizer FixedQuantize   --w_group_size -1   --gptq_mse true   --gptq_groupsize -1   --save_path ./qqq-llama3-8b_test_more_samples   --batch_size 1


grep -R --include="*.py" "tokenizer(" ./QQQ -n
len(tokenizer)
grep -R --include="*.py" "_update_causal_mask" ./QQQ -n


# activation per-channel quant + weight per-group quant + groupsize 128

CUDA_VISIBLE_DEVICES=1,2,5 python3 examples/quant_model.py   --model_path     ./models/Llama-3.1-8B   --tokenizer_path ./models/Llama-3.1-8B   --dtype float16   --smooth false   --rotation true   --dataset wikitext2   --nsamples 128   --w_quantizer GroupFixedQuantize   --w_group_size 128   --gptq_mse true   --gptq_roupsize 128   --save_path ./qqq-llama3-8b_g128   --batch_size 8g

# llama3.1 8b 퀀트트
CUDA_VISIBLE_DEVICES=2,3,5 python3 examples/quant_model.py   --model_path     ./models/Llama-3.1-8B   --tokenizer_path ./models/Llama-3.1-8B   --dtype float16   --smooth false   --rotation false   --dataset wikitext2   --nsamples 128   --w_quantizer FixedQuantize   --w_group_size -1   --gptq_mse true   --gptq_groupsize -1   --save_path ./qqq-llama3-8b_sdpa --batch_size 1


# llama3.1 8b base perchannel
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3,5 python3 examples/quant_model.py   --model_path     ./models/Llama-3.1-8B     --tokenizer_path ./models/Llama-3.1-8B     --dtype float16   --smooth false   --rotation true   --dataset wikitext2   --nsamples 128   --w_quantizer FixedQuantize   --w_group_size -1   --gptq_mse false   --gptq_groupsize -1   --save_path ./qqq-llama3-8b_base   --batch_size 8
# llama3.1 8b base per



# llama2 7b 퀀트트 

# --gptq_groupsize=-1(per-channel) vs 128(per-group) 조합에 따라 
# 𝐻
# H 블록 크기와 구조가 달라지고, Llama-2 특정 그룹 형태가 PD 조건을 만족하지 못하게 될 수 있습니다.

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3,5 python3 examples/quant_model.py \
  --model_path     /home/eiclab/eiclab04/urp2025/_QQQ_origin/QQQ_jaeuk/Llama-2-7b \
  --tokenizer_path /home/eiclab/eiclab04/urp2025/_QQQ_origin/QQQ_jaeuk/Llama-2-7b \
  --dtype float16 \
  --smooth false \
  --rotation true \
  --dataset wikitext2 \
  --nsamples 128 \
  --w_quantizer FixedQuantize \
  --w_group_size -1 \
  --gptq_mse true \
  --gptq_groupsize -1 \
  --save_path ./qqq-llama2-7b_bolleanmask \
  --batch_size 8

# llama2 7b 퀀트트 mse False -> 돌리기위해
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3,5 python3 examples/quant_model.py \
  --model_path     /home/eiclab/eiclab04/urp2025/_QQQ_origin/QQQ_jaeuk/Llama-2-7b \
  --tokenizer_path /home/eiclab/eiclab04/urp2025/_QQQ_origin/QQQ_jaeuk/Llama-2-7b \
  --dtype float16 \
  --smooth false \
  --rotation true \
  --dataset wikitext2 \
  --nsamples 128 \
  --w_quantizer FixedQuantize \
  --w_group_size -1 \
  --gptq_mse false \
  --gptq_groupsize -1 \
  --save_path ./qqq-llama2-7b_bolleanmask \
  --batch_size 8


# llama eval
# llama2 7b eval
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 examples/eval_model.py   --model_path ./qqq-llama3-8b_test_more_samples/Llama-3.1-8B    --tokenizer_path ./qqq-llama3-8b_test_more_samples/Llama-3.1-8B   --eval_ppl   --batch_size 8   --max_length 2048

# llama2 7b eval
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 examples/eval_model.py   --model_path ./qqq-llama2-7b_nopatch/Llama-2-7b    --tokenizer_path ./qqq-llama2-7b_nopatch/Llama-2-7b   --eval_ppl   --batch_size 8   --max_length 2048




# llama3.1 8b eval
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. python3 examples/eval_model.py   --model_path ./qqq-llama3-8b_flash/Llama-3.1-8B    --tokenizer_path ./qqq-llama3-8b_flash/Llama-3.1-8B   --eval_ppl   --batch_size 8   --max_length 2048


CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 examples/eval_model.py   --model_path ./qqq-llama3-8b_flash/Llama-3.1-8B    --tokenizer_path ./qqq-llama3-8b_flash/Llama-3.1-8B   --eval_ppl   --batch_size 1   --max_length 2048


qqq-llama3-8b_test_more_samples/

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 examples/eval_model.py   --model_path ./qqq-llama3-8b_base/Llama-3.1-8B    --tokenizer_path ./qqq-llama3-8b_base/Llama-3.1-8B   --eval_ppl   --batch_size 1   --max_length 2048

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 examples/eval_model.py   --model_path ./qqq-llama3-8b_g128/Llama-3.1-8B    --tokenizer_path ./qqq-llama3-8b_g128/Llama-3.1-8B   --eval_ppl   --batch_size 1   --max_length 2048