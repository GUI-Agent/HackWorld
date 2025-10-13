# Remember to modify the downloaded model configs following: https://github.com/vllm-project/vllm/issues/15614 if you are using transformers>=4.49
CUDA_VISIBLE_DEVICES=6,7 python3 -m vllm.entrypoints.openai.api_server \
    --model "ByteDance-Seed/UI-TARS-7B-DPO" \
    --max-model-len 32768 \
    --trust-remote-code \
    --port 18688 \
    --host 0.0.0.0 \
    --tensor-parallel-size 2 \
    --dtype=bfloat16

CUDA_VISIBLE_DEVICES=6,7 python3 -m vllm.entrypoints.openai.api_server \
    --model "ByteDance-Seed/UI-TARS-1.5-7B" \
    --max-model-len 32768 \
    --trust-remote-code \
    --port 18688 \
    --host 0.0.0.0 \
    --tensor-parallel-size 2 \
    --dtype=bfloat16