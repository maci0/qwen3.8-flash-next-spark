#!/bin/bash
# Serve UD-Q4_K_XL via llama-server (OpenAI-compatible) on the Spark.
# Strategy (user request): ALL experts/attention/MTP in GPU (unified) memory,
# PLE n-gram table (per_layer_token_embd.weight, 51.2B params) pinned to CPU
# and streamed from the NVMe via mmap (touched only 16 rows/token).
#   -ngl 999                -> everything except overridden tensors on GPU
#   -ot "...=CPU"           -> PLE table stays file-backed on disk
# Cold load takes ~13 min; prints a listening line when ready.
# Run detached: setsid nohup bash spark_serve.sh > serve.log 2>&1 < /dev/null &
set -e
cd /home/maci/qwen3.8-flash-next

GGUF=models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

exec ./llama.cpp/build/bin/llama-server \
    -m "$GGUF" \
    -ngl 999 \
    -ot "per_layer_token_embd.weight=CPU" \
    --ctx-size 16384 \
    --host 0.0.0.0 --port 8080 \
    --no-warmup
