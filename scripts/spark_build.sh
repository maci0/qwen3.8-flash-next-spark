#!/bin/bash
# Build llama.cpp with qwen4_exp support (PR #27742) on the Spark (ARM64 + CUDA 13, sm_121).
# Run detached on the Spark: setsid nohup bash spark_build.sh > build.log 2>&1 < /dev/null &
set -e
cd ~/qwen3.8-flash-next

git clone https://github.com/ggml-org/llama.cpp.git 2>&1 | tail -1
git -C llama.cpp fetch origin pull/27742/head:pr-27742 2>&1 | tail -1
git -C llama.cpp checkout pr-27742 2>&1 | tail -1

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=121 2>&1 | tail -3

cmake --build llama.cpp/build --config Release -j20 \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split 2>&1 | tail -5

echo "BUILD DONE"
llama.cpp/build/bin/llama-cli --version 2>&1 | head -3
