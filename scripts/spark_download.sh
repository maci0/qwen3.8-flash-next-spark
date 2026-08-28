#!/bin/bash
# Download the Unsloth UD-Q4_K_XL (4-bit) GGUF of Qwen3.8-Flash-Next onto the Spark.
# Run detached on the Spark: setsid nohup bash spark_download.sh > download.log 2>&1 < /dev/null &
set -e
cd ~/qwen3.8-flash-next

python3 -m venv .venv
.venv/bin/pip -q install -U "huggingface_hub[cli]" 2>&1 | tail -1

mkdir -p models/unsloth
.venv/bin/hf download unsloth/Qwen3.8-Flash-Next-GGUF \
    --local-dir models/unsloth/Qwen3.8-Flash-Next-GGUF \
    --include "UD-Q4_K_XL/*" 2>&1 | tail -5

echo "DOWNLOAD DONE"
ls -la models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/
