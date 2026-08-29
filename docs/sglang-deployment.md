# SGLang NVFP4 TP2 Deployment — Qwen3.8-Flash-Next on 2× DGX Spark

Status: **LIVE and validated 2026-08-30**. OpenAI API at `http://<head>:8888`, 1M context, NVFP4 checkpoint, fully Docker-contained.

## What this is

The full-quality path for Qwen3.8-Flash-Next: the **NVFP4 checkpoint** (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, 135 GB, experts-only W4A4 quant, in-band with BF16 on GSM8K/AIME) served by **SGLang TP2 across 2× DGX Spark** over a direct CX7 RoCE rail. This is the stack the community proved at 47–70 t/s; our measured numbers below.

## Why a patched image (the SM121 story)

- SGLang qwen4_exp support merged Aug 26 (PR #36497) but is **not in any tagged release**; plain/nightly images are still **broken on GB10 (sm_121)**: TRT-LLM sparse decode silently corrupts long context on SM121 (120k–210k prompts → token-0 loop, sglang#36537), and the FA4 CuTe fallback doesn't compile on GB10. The SM121 Triton packed-QSA kernel (PR #36845) is **still open**.
- Therefore the **MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks** recipe is required: it builds a derivative image (`qwen38flashnext-dspark:local`) from `lmsysorg/sglang:qwen38flashnext` with (1) TRT-LLM excluded on SM121, (2) the Triton packed-varlen QSA fallback, (3) a QSA NVFP4-KV cache patch (not upstream).

## Prerequisites (our verified config)

- 2× DGX Spark (GB10, sm_121, 121 GiB usable each), **direct CX7 link** with RoCE: our pair uses `enp1s0f1np1`/`rocep1s0f1` both sides, IPs `10.0.1.1` (head) / `10.0.1.2` (worker), MTU 9000. (The recipe's default port names/IPs differ — set yours in `.env`.)
- SSH head→worker (key-based; the NVIDIA Sync `spark2` alias re-pointed to the CX IP works).
- Docker on both nodes. NVFP4 repo is **public** (no HF token).

## .env (ours, exact)

See [`scripts/sglang/.env.example`](../scripts/sglang/.env.example). Key values:

```bash
HEAD_CX7_IP=10.0.1.1   WORKER_CX7_IP=10.0.1.2
HEAD_CX7_IF=enp1s0f1np1   WORKER_CX7_IF=enp1s0f1np1
HEAD_CX7_IB=rocep1s0f1   WORKER_CX7_IB=rocep1s0f1
WORKER_HOST=spark2
MEM_FRACTION_STATIC=0.82
CONTEXT_LENGTH=1048576          # 1M ctx (YaRN factor 4 injected)
CHUNKED_PREFILL_SIZE=1024       # must stay 1024 at 1M
SPEC_STEPS=3  SPEC_TOPK=1  SPEC_DRAFT=4   # NEXTN; steps 4 impossible
NVFP4_KV_CACHE=0                # 0 = fp8_e4m3 (faster); 1 = NVFP4 KV (bigger pool)
EXTRA_ARGS=--speculative-attention-mode decode --enable-linear-replayssm-spec
NCCL_MAX_NCHANNELS=8  NCCL_MIN_NCHANNELS=8  NCCL_CUMEM_ENABLE=0
NCCL_IGNORE_CPU_AFFINITY=1  TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

## Two local fixes we had to make

1. **rsync tree-exclude**: a root-owned HF `trees/*.json` cache file (leftover from a container run) made the weight rsync fail with code 23. Patched `start.sh`'s rsync to `--exclude 'trees/'` (`scripts/patch_miaai_rsync.py`). The trees dir is optional HF metadata, not weights.
2. **Page-cache drop without sudo** (required before every boot — the 135 GB rsync page cache otherwise trips SGLang's per-rank memory-balance check / can softlock the box):
   ```bash
   docker run --rm --privileged --pid=host lmsysorg/sglang:qwen38flashnext \
     sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
   ```
   Run on both nodes.

## Boot / resume (one command, idempotent)

```bash
ssh spark1 'cd ~/Qwen3.8-Flash-Next-Dual-DGX-Sparks && ./start.sh serve'
```
First boot: pulls base image (~30 GB), builds patched image on both nodes, downloads 135 GB NVFP4 to the head, rsyncs to the worker over the CX rail (~350 MB/s), boots TP2, waits for readiness (~9–10 min total on warm cache), runs smoke. Weights + image are cached — subsequent boots skip to launch. `./start.sh stop|status|logs|smoke|kv-eval|doctor` also available.

## Validation checklist (all passed)

- `max_total_num_tokens` in the boot log ≥ 1,048,576 — ours: **1,254,528** (fp8+spec-attn-decode) / 1,332,352 (fp8) / **2,056,576 (NVFP4 KV)**.
- `GET /v1/models` → `max_model_len: 1048576`.
- Chat completion returns thinking trace + correct code answer.
- `ple_offload_embedding=True` in server args (PLE table host-pinned, ~11 GB/rank, out of the CUDA static pool).

## Measured results (2026-08-30, single-stream / aggregate)

| Config | Single-stream | Aggregate output (24 conc) | E2E mean |
|---|---|---|---|
| NVFP4 KV | ~36 t/s | — | — |
| fp8_e4m3 KV | ~38–42 t/s | 138.6 tok/s | 15.8 s |
| fp8 + spec-attn-decode | ~40–44 t/s | 147.6–155.2 tok/s | 12.7–14.3 s |
| **+ replayssm-spec (FINAL)** | **~81–103 t/s** | **183–234 tok/s (noisy band)** | 9.4–11.9 s |

More in [`docs/sglang-perf.md`](sglang-perf.md).

## Gotchas / notes

- **UMA softlocks**: booting under a full page cache wedges the box (kernel pings, sshd starved, ~10 min auto-reboot or manual power cycle). Always drop caches before boot. Two incidents logged.
- **mem-fraction 0.85+** risks the DGX OS earlyoom SIGTERM — 0.82 is the validated 1M point.
- **Dual-rail NCCL**: the recipe's preflight accepts a single RoCE device; dual-rail was tried and reverted (not worth patching for unverified ~2%).
- **`--enable-torch-compile`**: breaks CUDA-graph capture on SM121 — keep off.
- **NEXTN steps 4 is impossible** (draft=5 rejected by the compress-ratio cap); 3/1/4 is max.
- **lmcache**: evaluated, deferred — hybrid-layout bugs open for qwen4_exp (LMCache #4771 etc.), and 1M fits without KV offload.
- The user's vLLM DeepSeek-V4 container on the worker was paused to free the node (`docker start vllm-ds4-0731` to restore).

## Reproduce

1. Clone `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` on the head.
2. Copy `.env.example` → `.env`, set your CX IPs/IF/IB + the values above.
3. Apply the rsync patch (or the trees dir fix).
4. `./start.sh serve` (drop caches first on both nodes).

Sources: [MiaAI recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks), [sglang#36497](https://github.com/sgl-project/sglang/pull/36497), [#36845](https://github.com/sgl-project/sglang/pull/36845), [#36537](https://github.com/sgl-project/sglang/issues/36537), [#36796](https://github.com/sgl-project/sglang/issues/36796), [#36797](https://github.com/sgl-project/sglang/issues/36797), [tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark), [pocharlies](https://github.com/pocharlies/qwen38-flash-next-dgx-spark-sglang).
