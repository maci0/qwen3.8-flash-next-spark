#!/bin/bash
# Safe long-context variant: 512K ctx with FP8 (q8_0) KV.
# Memory math at 512K: QSA KV ~6.4 GB (q8_0) + indexer f16 ~1.5 GB -> ~113-114 G used,
# leaving ~7 G headroom (vs ~1 G at 1M ctx). Recommended for reliable daily long-context use.
# Run detached: setsid nohup bash spark_serve_512k.sh > serve.log 2>&1 < /dev/null &
set -e
cd /home/maci/qwen3.8-flash-next

GGUF=models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

exec ./llama.cpp/build/bin/llama-server \
    -m "$GGUF" \
    -ngl 999 \
    -ot "per_layer_token_embd.weight=CPU" \
    --ctx-size 524288 \
    --parallel 1 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --rope-scaling yarn \
    --rope-scale 2.0 \
    --yarn-orig-ctx 262144 \
    --override-kv qwen4exp.context_length=int:524288 \
    --host 0.0.0.0 --port 8080 \
    --no-warmup
