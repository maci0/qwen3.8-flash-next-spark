#!/bin/bash
# 16K config + n-gram speculative decoding (no draft model needed; the MTP head is
# not present in Unsloth's GGUF — conversion/qwen4exp.py drops it, supports_mtp_export=False).
# ngram-mod drafts tokens from context statistics; expect a modest decode speedup.
# Run detached: setsid nohup bash spark_serve_ngram.sh > serve.log 2>&1 < /dev/null &
set -e
cd /home/maci/qwen3.8-flash-next

GGUF=models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

exec ./llama.cpp/build/bin/llama-server \
    -m "$GGUF" \
    -ngl 999 \
    -ot "per_layer_token_embd.weight=CPU" \
    --ctx-size 16384 \
    --parallel 4 \
    --spec-type ngram-mod \
    --spec-draft-n-max 16 \
    --host 0.0.0.0 --port 8080 \
    --no-warmup
