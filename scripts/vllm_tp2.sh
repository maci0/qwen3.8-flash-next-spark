#!/bin/bash
# vLLM TP2 on 2x DGX Spark for Qwen3.8-Flash-Next (blazux patched image + ray layer, PLE from NVMe).
# Image 'qwen38-flash-dgx-ray' must exist on BOTH nodes (spark1: docker commit after
#   `pip install --no-deps ray` in the base image; then `docker save | gzip | ssh spark2 'gunzip | docker load'`).
# Run on EACH node: RANK=0 bash vllm_tp2.sh (head)  |  RANK=1 bash vllm_tp2.sh (worker)
#
# Topology (stable-cluster design):
#   spark1 qwen38-rayhead     : ray start --head --num-gpus=0  (control plane ONLY; never exits -> cluster ID stable)
#   spark1 qwen38-vllm-tp2-r0 : ray start --address (joins head, owns spark1 GPU) && vllm serve (driver + shard 0)
#   spark2 qwen38-vllm-tp2-r1 : ray start --address (joins head, owns spark2 GPU) && sleep (shard 1 actor)
# vLLM engine failures only restart the vllm container; the Ray cluster survives.
# API on head:8000 (host net). YaRN factor auto (2.0 @ 512K, 4.0 @ 1M).
set -euo pipefail

RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-10.0.1.1}"   # head's CX rail IP
MASTER_PORT="${MASTER_PORT:-26401}"
CTX="${CTX:-512000}"                       # 512K for aggregate; 1000000 for long-context
SEQS="${SEQS:-16}"
GPU_MEM="${GPU_MEM:-0.85}"                 # 0.875 OOM-killed on 300k prefill + MTP; 0.80 long-running
MTP="${MTP:-3}"                            # MTP speculative tokens (0 = off)
IMG="qwen38-flash-dgx-ray"
SNAP=/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594
# NCCL IB plugin needs the RDMA devices (docker does not inject them with --gpus all)
# and unlimited memlock (8MB default kills ibv_create_cq with ENOMEM)
RDMA_DEV="--device /dev/infiniband/rdma_cm --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 --device /dev/infiniband/uverbs2"
ULIMITS="--ulimit memlock=-1 --ulimit stack=67108864"

case "$RANK" in
  0) NODEIP=10.0.1.1 ;;
  1) NODEIP=10.0.1.2 ;;
  *) echo "RANK must be 0 or 1" >&2; exit 1 ;;
esac

if (( CTX > 512000 )); then FACTOR=4.0; else FACTOR=2.0; fi
OVR="{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":${FACTOR},\"original_max_position_embeddings\":262144}}}"

# Shared env: PLE mmap from NVMe, NCCL over CX rail IB, ray memory monitor off (it OOM-kills vLLM workers)
COMMON=(-v "$HOME/.cache/huggingface:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=64 -e VLLM_PLE_MMAP_PREWARM=0
  -e VLLM_QSA_EXACT_TOPK=1 -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_TC=106
  -e NCCL_IB_PCI_RELAXED_ORDERING=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1
  -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=INFO
  -e RAY_memory_monitor_refresh_ms=0
  -e RAY_ADDRESS="$MASTER_ADDR:6379" -e VLLM_HOST_IP="$NODEIP"
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUTE_DSL_ARCH=sm_121a
  -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT="$MASTER_PORT")

if [[ $RANK == 0 ]]; then
  docker rm -f qwen38-vllm-tp2-r0 2>/dev/null || true

  # 1) Stable ray head: control plane only, no GPU advertised. Reuse if already running
  #    (recreating it regenerates the cluster ID and kills every connected raylet).
  if ! docker ps -a --format '{{.Names}}' | grep -qx qwen38-rayhead; then
    docker run -d --name qwen38-rayhead --restart unless-stopped --network host --gpus all \
      -e RAY_memory_monitor_refresh_ms=0 \
      --entrypoint /bin/bash "$IMG" -c \
      "ray start --head --port=6379 --node-ip-address=$NODEIP --num-gpus=0 --include-dashboard=false --disable-usage-stats && sleep infinity"
  fi

  # 2) vLLM container joins the cluster as a ray node owning spark1's GPU (driver + shard 0).
  #    ray 2.58 needs a local raylet so the driver can resolve its node (find_node_ids).
  docker run -d --name qwen38-vllm-tp2-r0 --restart unless-stopped --network host --gpus all \
    --ipc=host --shm-size 16g --cap-add SYS_PTRACE --cap-add IPC_LOCK $RDMA_DEV $ULIMITS \
    "${COMMON[@]}" \
    --entrypoint /bin/bash "$IMG" -c "
      ray start --address=$MASTER_ADDR:6379 --node-ip-address=$NODEIP --num-gpus=1 --disable-usage-stats &&
      exec vllm serve \"$SNAP\" --served-model-name qwen3.8-flash-next \
        --host 0.0.0.0 --port 8000 --load-format safetensors \
        --tensor-parallel-size 2 --distributed-executor-backend ray \
        --max-model-len \"$CTX\" --max-num-seqs \"$SEQS\" --gpu-memory-utilization \"$GPU_MEM\" \
        --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
        --long-prefill-token-threshold 4096 \
        --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\",\"splitting_ops\":[\"vllm::unified_attention_with_output\",\"vllm::unified_mla_attention_with_output\",\"vllm::mamba_mixer2\",\"vllm::mamba_mixer\",\"vllm::short_conv\",\"vllm::qwen3_8_flash_next_ple_short_conv\",\"vllm::qwen3_8_flash_next_qsa_with_output\",\"vllm::linear_attention\",\"vllm::qwen_gdn_attention_core\",\"vllm::qwen_gdn_attention_core_fused_norm_packed\",\"vllm::sparse_attn_indexer\",\"vllm::ple_mmap_lookup\"],\"cudagraph_capture_sizes\":[1,2]}' \
        --no-enable-flashinfer-autotune --kv-cache-dtype auto \
        --hf-overrides '$OVR' \
        --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"max_model_len\":$CTX}' \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
    "
else
  docker rm -f qwen38-vllm-tp2-r1 2>/dev/null || true

  # Worker: join the ray cluster, keep the raylet alive (vLLM shard-1 actor runs here)
  docker run -d --name qwen38-vllm-tp2-r1 --restart unless-stopped --network host --gpus all \
    --ipc=host --shm-size 16g --cap-add SYS_PTRACE --cap-add IPC_LOCK $RDMA_DEV $ULIMITS \
    "${COMMON[@]}" \
    --entrypoint /bin/bash "$IMG" -c \
    "ray start --address=$MASTER_ADDR:6379 --node-ip-address=$NODEIP --num-gpus=1 --disable-usage-stats && sleep infinity"
fi
echo "LAUNCHED RANK=$RANK ($NODEIP)"
