#!/bin/bash
# Serve UD-Q4_K_XL at 1M context with FP8 (q8_0) KV cache on the Spark.
#   - experts/attention/MTP in GPU unified memory, PLE n-gram table on NVMe (-ot CPU)
#   - FP8 KV: --cache-type-k/v q8_0 (1M ctx -> ~1.2 GB of KV)
#   - 1M ctx: --override-kv bumps model max; static YaRN factor 4 (Qwen's 1M recipe)
#   - single slot (--parallel 1) to keep memory headroom (~9 GB)
# Fallback: spark_serve.sh (16K ctx, 4 slots). Run detached:
#   setsid nohup bash spark_serve_1m.sh > serve.log 2>&1 < /dev/null &
set -e
cd /home/maci/qwen3.8-flash-next

GGUF=models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

exec ./llama.cpp/build/bin/llama-server \
    -m "$GGUF" \
    -ngl 999 \
    -ot "per_layer_token_embd.weight=CPU" \
    --ctx-size 1000000 \
    --parallel 1 \
    --cache-type-k iq4_nl \
    --cache-type-v iq4_nl \
    --rope-scaling yarn \
    --rope-scale 4.0 \
    --yarn-orig-ctx 262144 \
    --override-kv qwen4exp.context_length=int:1000000 \
    --host 0.0.0.0 --port 8080 \
    --no-warmup
