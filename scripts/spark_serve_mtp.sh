#!/bin/bash
# MTP spec-decode server for Qwen3.8-Flash-Next (llama.cpp PR #27836 + grafted MTP head,
# UD-Q4_K_XL-MTP). A/B baseline: swap "--spec-type draft-mtp ..." for "--spec-type none".
set -e
cd ~/qwen3.8-flash-next

GGUF=models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-00001-of-00005.gguf

export LLAMA_ATTN_ROT_DISABLE=1

exec ./llama.cpp/build/bin/llama-server \
    -m "$GGUF" \
    -ngl 999 \
    -ot "per_layer_token_embd.weight=CPU" \
    -fa on \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
    --ctx-size 16384 --parallel 1 \
    --host 0.0.0.0 --port 8080 \
    --no-warmup
