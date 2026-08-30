# vLLM TP2 on 2× DGX Spark — Qwen3.8-Flash-Next NVFP4

Live stack (2026-08-30): `scripts/vllm_tp2.sh` on spark1 (`RANK=0`) + spark2 (`RANK=1`).
Image `qwen38-flash-dgx-ray` (blazux/qwen3.8-Flash-DGX + `pip install --no-deps ray==2.58.0`).
API: `http://192.168.0.211:8000/v1` · served name `qwen3.8-flash-next` · `max_model_len` 512000.

## What this stack is

vLLM tensor-parallel-2 over the CX7 RoCE rail (10.0.1.1 ↔ 10.0.1.2), PLE n-gram table mmapped from NVMe (`VLLM_PLE_MMAP=1`), MTP spec 3, YaRN factor 2.0 @ 512K. The only vLLM build that runs `qwen4_exp` on GB10 today — stock `vllm/vllm-openai:qwen38-flash-next-arm64-cu130` does not; upstream support is open PR [vllm#53896](https://github.com/vllm-project/vllm/pull/53896).

**KV is bf16-only** for this architecture in vLLM (QSA refuses fp8/nvfp4). Pool observed: **3,980,387 tokens → 7.77× concurrency @ 512K**. Compare SGLang TP2, which can do NVFP4 KV and a 1–2M token pool at 1M ctx.

## Measured (this session)

| Run | Tokens | Latency | tok/s |
|---|---|---|---|
| Single-stream chat, code prompt, temp 0.2 | 374 completion / 75 prompt | 12.01 s | **31.14** |
| 8 concurrent, 128-token completions | 1024 completion / 13.79 s wall | 13.79 s | **74.28 agg** (~9.6 t/s each) |

NCCL: `Using network IB`, `NET/IB : Using [0]rocep1s0f1:1/RoCE`, channels via `NET/IB/0`. Init ~1.7 s (connections 1.50 s). Standalone 2-node `all_reduce` over the same env: `RANK0/1 NCCL OK [2.0 × 8]`.

Memory after boot: spark1 117/121 used / 4 avail; spark2 111/121 used / 9 avail. PLE mmap confirmed on both workers (`vllm_ple_mmap.py` stats; `PLE mmap patch applied to ...Qwen3_8FlashNextNGramEmbedding`).

## vs the other stacks (same model, same 2 Sparks)

| Stack | Ctx | KV | PLE | Single t/s | Agg t/s | Notes |
|---|---|---|---|---|---|---|
| **SGLang TP2** | 1M | NVFP4 | host-offload | ~40–44 | 183–234 @ 24 conc | flagship quality + KV density |
| **vLLM TP2** (this) | 512K | bf16 | NVMe mmap | **31.1** | **74 @ 8 conc** | MTP 3, 7.77× @ 512K |
| SGLang TP1 NVMe-PLE | 1M | NVFP4 | NVMe mmap | ~31 | 43–51 @ 4 conc | one Spark |
| llama.cpp GGUF MTP | 16K | f16 | CPU mmap | 27–32 | n/a | 4-bit weights |

vLLM TP2 is in the same single-stream ballpark as SGLang TP1 and llama.cpp, well below SGLang TP2. The gap is expected: bf16 KV (no NVFP4 KV path), PLE from disk, and PIECEWISE graphs with the PLE lookup as a splitting op. 8-way concurrent already saturates decode (~9.6 t/s each, 74 agg); 16-way `bench_serving` is the next measurement.

## Topology (do not collapse)

```
spark1  qwen38-rayhead        ray start --head --num-gpus=0     # control plane, never exits
spark1  qwen38-vllm-tp2-r0    ray start --address && vllm serve # driver + shard 0
spark2  qwen38-vllm-tp2-r1    ray start --address && sleep      # shard 1 actor
```

Ray 2.58 resolves the *driver's* node via local `/tmp/ray` sockets — the driver container must run its own `ray start`. Putting the GCS in the vLLM container is fatal: every engine failure regenerates the cluster ID and kills spark2's raylet (`GCS returned an authentication error` / `ActorHandleNotFoundError`).

## Boot blockers (each one was a full restart)

Documented with repros in `docs/session-log.md` Phase 24:

1. No Ray in the blazux image → `qwen38-flash-dgx-ray` layer. Commit inherits the builder's `bash` entrypoint → `--entrypoint vllm ... serve`.
2. Driver without a local raylet → `No node info found matching attributes: ''`. `address="auto"` starts a bogus single-node cluster; set `RAY_ADDRESS=10.0.1.1:6379`.
3. Cluster-ID churn → dedicated `qwen38-rayhead` (`--num-gpus=0`); script reuses it if running.
4. `--gpus all` does not inject `/dev/infiniband`; default memlock 8 MB kills `ibv_create_cq` → `--device /dev/infiniband/{rdma_cm,uverbs0,uverbs1,uverbs2} --ulimit memlock=-1`.
5. PLE gather captured into the CUDA graph → pinned-copy, then stride (`8192==8193`) asserts. Fix: `--compilation-config` JSON with `cudagraph_mode=PIECEWISE` + 12-op `splitting_ops` (incl. `vllm::ple_mmap_lookup`). Nested `-cc.splitting_ops=...` is a string in this build and pydantic rejects it.

Also: never pin `NCCL_IB_GID_INDEX` (drifts after reboots). `GPU_MEM=0.85` (0.875 OOM'd a 300k prefill + MTP). `--cap-add SYS_PTRACE --cap-add IPC_LOCK` (Yama ptrace + RDMA).

## Knobs

| Env | Default | Meaning |
|---|---|---|
| `CTX` | 512000 | 1000000 auto-sets YaRN factor 4.0 |
| `MTP` | 3 | 0 = off |
| `SEQS` | 16 | max concurrent sequences |
| `GPU_MEM` | 0.85 | 0.80 for long-running |
| `MASTER_ADDR` | 10.0.1.1 | CX rail IP of spark1 |

## Reproduce

```bash
# both nodes already have qwen38-flash-dgx-ray
scp scripts/vllm_tp2.sh spark1:~/vllm_tp2.sh
scp scripts/vllm_tp2.sh spark2:~/vllm_tp2.sh
ssh spark1 'RANK=0 bash ~/vllm_tp2.sh'
ssh spark2 'RANK=1 bash ~/vllm_tp2.sh'
# wait ~8–12 min
curl -s http://192.168.0.211:8000/v1/models
```
