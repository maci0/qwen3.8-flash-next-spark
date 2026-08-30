#!/bin/bash
# Single-Spark SGLang TP1 with PLE streamed from NVMe (sglang#36567).
# Requires the patched image (qwen38flashnext-dspark:local, built with
# apply_ple_nvme_patches.py). Run on spark1.
set -e
SNAP_HOST=/home/maci/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594
SNAP_CT=/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594

docker rm -f qwen38-tp1 2>/dev/null || true
docker run -d --name qwen38-tp1 --gpus all --ipc host \
  -v /home/maci/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e SGLANG_QWEN4_PLE_NVME_PATH="$SNAP_CT" \
  -e SGLANG_QWEN4_PLE_NVME_BACKEND=mmap \
  -p 8888:8888 \
  qwen38flashnext-dspark:local \
  python3 -m sglang.launch_server \
    --model-path "$SNAP_CT" \
    --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 \
    --chunked-prefill-size 512 --max-running-requests 1 --context-length 32768 \
    --mem-fraction-static 0.84 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
    --host 0.0.0.0 --port 8888
