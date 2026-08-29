#!/bin/bash
# 700K ctx for Qwen3.8-Flash-Next (GGUF). YaRN factor 2.7 (262144 * 2.7 = 707,789 >= 700,000).
# q8_0 (FP8) KV is borderline (~119 G, ~1-2 G headroom); the iq4_nl variant is comfortable (~115 G).
# Run detached: setsid nohup bash spark_serve_700k.sh > serve_700k.log 2>&1 < /dev/null &
set -e
cd ~/qwen3.8-flash-next

GGUF=models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

exec ./llama.cpp/build/bin/llama-server \
    -m "$GGUF" \
    -ngl 999 \
    -ot "per_layer_token_embd.weight=CPU" \
    --ctx-size 700000 \
    --parallel 1 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --rope-scaling yarn \
    --rope-scale 2.7 \
    --yarn-orig-ctx 262144 \
    --override-kv qwen4exp.context_length=int:700000 \
    --host 0.0.0.0 --port 8080 \
    --no-warmup
