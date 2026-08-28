#!/bin/bash
# Smoke test v2: direct (unpiped) logging so we can see what llama-cli is doing.
# -ngl 0 + mmap = lazy paging from NVMe ("ngrams streamed from disk").
set -e
cd /home/maci/qwen3.8-flash-next

GGUF=models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
BIN=llama.cpp/build/bin/llama-cli
LOG=smoke_raw.log
rm -f "$LOG"

echo "launching llama-cli -> $LOG"
"$BIN" -m "$GGUF" \
    -ngl 0 --ctx-size 8192 \
    --temp 1.0 --top-p 0.95 --top-k 20 \
    --no-warmup \
    -n 64 --no-display-prompt \
    -p "What is 2+2? Answer with just the number." > "$LOG" 2>&1
rc=$?
echo "=== llama-cli exit=$rc ==="
echo "--- log size ---"
ls -la "$LOG"
echo "--- first 40 lines ---"
head -40 "$LOG"
