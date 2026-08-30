# SGLang TP2 Performance Tuning — Qwen3.8-Flash-Next NVFP4 on 2× DGX Spark

Research date: 2026-08-30. Base: [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) recipe (patched SM121 image, 1M YaRN, NVFP4 KV, NEXTN 3/1/4, mem 0.82, chunk 1024) — measured 64.4 tok/s single-stream, 117 ×2, ~153–160 ×8. This doc = the deltas for maximum performance.

## Image: stay on the patched derivative (as of 2026-08-30)

- **sglang PR #36845 (SM121 packed-QSA Triton kernel) is NOT merged.** Plain `dev-cu13`/nightly is still NOT SM121-usable (TRT-LLM sparse decode silently corrupts long context on SM121; FA4 CuTe fallback doesn't compile on GB10). Keep `qwen38flashnext-dspark:local`.
- When #36845 merges: same image, swap the generic Triton fallback for the KDA kernel → expected **~+4%** (measured 1.41–10.66× kernel-level, but only +4.0–4.45% E2E at c=1/4).
- `--enable-torch-compile`: **off** (breaks CUDA-graph capture on SM121 — #36796). No fix seen.

## Near-free wins to add to `.env` (verified sources)

```bash
# ---- NCCL / RoCE (NCCL is ~7% of decode; these are cheap real gains) ----
NCCL_MAX_NCHANNELS=4
NCCL_MIN_NCHANNELS=4
NCCL_CUMEM_ENABLE=0
NCCL_IGNORE_CPU_AFFINITY=1
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# dual rail — our pair HAS the second CX7 rail active (10.0.2.x, roceP2p1s0f1):
HEAD_CX7_IB=rocep1s0f1,roceP2p1s0f1
WORKER_CX7_IB=rocep1s0f1,roceP2p1s0f1
NCCL_CROSS_NIC=1
# verify in boot log: "network IB" (TCP fallback silently costs ~half)

# ---- CUDA graph hygiene ----
EXTRA_ARGS=--disable-prefill-cuda-graph --cuda-graph-max-bs 28 --disable-cuda-graph-padding
```

## Measured on our box (2026-08-30, TP2 @1M, patched image)

| Config | Aggregate output (24 conc) | E2E mean | Single-stream |
|---|---|---|---|
| fp8_e4m3 KV (baseline) | 138.6 tok/s | 15.8 s | ~38–42 t/s |
| fp8 + `--speculative-attention-mode decode` | 147.6–155.2 tok/s | 12.7–14.3 s | ~40–44 t/s |
| **+ `--enable-linear-replayssm-spec` (FINAL)** | **183–234 tok/s (noisy band)** | 9.4–11.9 s | **~81–103 t/s** |

**Single-Spark SGLang TP1 with PLE streamed from NVMe** (sglang#36567, mmap backend, NEXTN 3/1/4, CUDA graphs off, 262K ctx):

| max_running_requests | Aggregate output | Single-stream |
|---|---|---|
| 1 | 18.9 tok/s (serialized) | ~21 t/s (first-boot cold) |
| 4 | **43.0 tok/s** | ~31 t/s (warm) |
| 8 | **50.7 tok/s** | — |

**Final TP1 config (1M ctx, NVFP4 KV, PLE from NVMe)**: `MEM_FRACTION=0.88`, `MAX_RUNNING=4` (auto-clamped — the GDN/mamba state pool is the concurrency limit at full 1M ctx, not the KV pool), pool **1,900,672 tokens** (~2M-class), memory 109/121 used / 12 avail.

**KV-dtype pool ladder at 1M ctx** (mem 0.84 unless noted): bf16 454K → fp8_e4m3 900K → **nvfp4 1.52M** → **nvfp4 1.9M @ mem 0.88**. `max_running_requests`: 2 @ mem 0.84 → 4 @ 0.88 (mamba/GDN state pool scales with the fraction).

Sweet spot ~4–8 concurrent (1→4 = +128%, 4→8 = +18%; bottleneck shifts to the NVMe PLE gather / single-GPU compute). Memory 105/121 used / 16 avail — the 47.7 GiB PLE table is not resident. vs llama.cpp single-Spark GGUF MTP (~27–32 t/s): SGLang TP1 trades ~speed for NVFP4 quality + true NVMe streaming. Tuning room: io_uring backend, `SGLANG_QWEN4_PLE_NVME_CACHE_PAGES` LRU (see below), CUDA graphs, 1M ctx via YaRN.

## Lever A/B series (2026-08-30, each = base + one lever, 24 concurrent)

| Lever | Aggregate (24c) | E2E | Verdict |
|---|---|---|---|
| baseline (spec-attn-decode) | 147.6–155.2 | 12.7–14.3 | — |
| `NCCL_MAX/MIN_NCHANNELS` 4→8 | 150.9 | 12.8 | neutral (kept, harmless) |
| `--schedule-conservativeness 0.5` | 162.1 | 13.9 | ✅ helps alone |
| `--sampling-backend pytorch` | 163.4 | 12.4 | ✅ helps alone |
| **`--enable-linear-replayssm-spec`** | **183–234** | 9.4–11.9 | ✅✅ **winner** (spec-verify memory 11.5→1.8 GB pays off on decode) |
| all three combined | 193.2 | 10.6 | ❌ **negative interaction** — sched05/pytorch help *without* replayssm but hurt *with* it (233.6 → 193.2) |
| `--enable-overlap-schedule` | — | — | ❌ **fails to boot** (recipe's PLE two-batch constraint confirmed) |
| dual-rail NCCL | — | — | ❌ recipe preflight is single-device (reverted) |

Run-to-run variance on this box is ±25% (same config re-measured 183–234); treat aggregate numbers as a band. Single-stream is prompt-dependent (the 600-tok short-prompt series is internally consistent).

**FINAL config**: fp8_e4m3 KV + `--speculative-attention-mode decode --enable-linear-replayssm-spec` + NCCL channels 8, single CX7 rail, NEXTN 3/1/4.

Dual-rail NCCL (both CX7 rails) was attempted — the recipe's preflight only accepts a single RoCE device, so it was reverted (not worth patching for an unverified ~2%). `--enable-torch-compile` stays off; NEXTN 3/1/4 is the max chain (steps 4 rejected).

Pool at 1M: NVFP4 KV 2,056,576 · fp8_e4m3 1,332,352 · fp8+spec-attn-decode 1,254,528 · **final (replayssm) 1,232,832** (all >1M ✓). NVFP4 KV gives max headroom; fp8 is faster — flip = one `.env` line.

## A/B matrix (run on our box once booted; all unmeasured on the patched Triton-fallback stack)

| Knob | Default (MiaAI) | A/B | Why |
|---|---|---|---|
| KV dtype | NVFP4 (`NVFP4_KV_CACHE=1`, pool 2.85M) | `NVFP4_KV_CACHE=0` → fp8_e4m3 (pool 1.75M) | Issue #36797: stock NVFP4-KV = −29% decode vs fp8 on SM121; MiaAI's patched version claims 64.4 t/s — **no clean public A/B of the final patch exists; measure it.** fp8 still fits 1M (1.67×). |
| `--enable-linear-replayssm-spec` | off | on | ReplaySSM spec-verify: spec-scratch 11.5→1.8 GB, `D=0` in mamba sizing; validated on tonyd2wild's TRT-LLM path only. |
| `--speculative-attention-mode decode` | off | on | tonyd2wild uses it; no data on patched path. |
| `--sampling-backend pytorch` | default | pytorch | part of tonyd2wild's token-0-fix stack; may be unneeded with the corrected kernel. |
| `--enable-overlap-schedule` | off | on | #36796 ran with it; interaction with PLE two-batch overlap unclear. |
| SPEC_STEPS | 3 | — | **4 is impossible** (draft=5 rejected: compress-ratio cap). 3/1/4 is max. |
| PLE storage | pinned host (~11 GB/rank) | NVMe-mmap (`MADV_RANDOM`, evictable) | frees ~11 GB → ~+1M KV tokens; SGLang TP2 impact **unmeasured** (single-Spark no-MTP datapoint: 14–15 t/s); open PR sglang#36567 implements NVMe streaming. The vLLM 181-agg run used it. |

## Throughput ceiling math (context vs concurrency)

- **1M ctx caps concurrency ~14** (mamba/GDN state pool) — aggregate tops out near MiaAI's 117 ×2 / 153–160 ×8 territory only at lower concurrency.
- **If aggregate > context**: 512K YaRN (factor 2) + 8–16 concurrent → pocharlies' measured **153–160 agg** (1.37M pool at mem 0.90, mamba-40).
- `--mamba-track-interval 64` stays (granularity, not context cap); `--chunked-prefill-size 1024` stays at 1M (4096 wedges the box at 300k+ history — QSA indexer fp32 workspace).

## Newer external measurements (for context, not directly comparable)

- **181 agg tok/s (peak 195)** on 2× Spark — vLLM, 512K, PLE-mmap'd off NVMe, MTP k=3, 9 concurrent agent sessions (prismix, 08-28). Not SGLang.
- **1M on ONE Spark** — vLLM, 989.7k-token request, 26.7 t/s decode, PLE-NVMe (sayyidfareed, 08-28/29).
- **SM120 (RTX PRO 6000) TP2** — 127.8–128.3 t/s decode, 8.8–11.6K t/s prefill (not GB10).

## Execute plan (once spark2 is power-cycled)

1. Drop caches both nodes (privileged docker — see session-log Phase 13).
2. `./start.sh serve` with the NCCL + graph-hygiene `.env` additions.
3. Baseline: 64.4/117/153-style numbers + pool size in boot log.
4. Run the A/B matrix (KV dtype first — it's the biggest lever per the −29% claim).
5. Document measured deltas in this file.
