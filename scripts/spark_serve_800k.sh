#!/bin/bash
# 800K ctx with FP8 (q8_0) KV — the user's question. Expect ~119-121 G used (knife-edge).
# If it swap-thrashes, flip --cache-type-k/v to iq4_nl (~116 G, ~4-5 G headroom).
# YaRN factor ~3.1 for 800K (262144 * 3.1 = 812,646 >= 800,000).
# Run detached: setsid nohup bash spark_serve_800k.sh > serve_800k.log 2>&1 < /dev/null &
set -e
cd /home/maci/qwen3.8-flash-next

GGUF=models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

exec ./llama.cpp/build/bin/llama-server \
    -m "$GGUF" \
    -ngl 999 \
    -ot "per_layer_token_embd.weight=CPU" \
    --ctx-size 800000 \
    --parallel 1 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --rope-scaling yarn \
    --rope-scale 3.1 \
    --yarn-orig-ctx 262144 \
    --override-kv qwen4exp.context_length=int:800000 \
    --host 0.0.0.0 --port 8080 \
    --no-warmup
