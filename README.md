# Qwen3.8-Flash-Next on NVIDIA DGX Spark

Serving **Qwen3.8-Flash-Next** (`qwen4_exp` architecture, ~180B-param multimodal MoE, 6B active/token) on **2× NVIDIA DGX Spark** (GB10, sm_121, 128 GB unified each) with two working stacks:

1. **SGLang NVFP4, TP2, 1M context** (flagship — full quality, Docker-contained, live)
2. **llama.cpp + 4-bit GGUF** (single-Spark fallback — PLE n-grams streamed from disk, MTP spec decode)

## Highlights

- **SGLang NVFP4 TP2 @ 1M ctx is live** on 2× Spark over a direct CX7 RoCE rail: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (135 GB, experts-only quant, in-band with BF16), NEXTN 3/1/4 spec decode, PLE table host-offloaded. **~40–44 t/s single-stream, ~148–155 tok/s aggregate** at 24 concurrent, KV pool 1.25–2.06M tokens (>1M guaranteed). Fully Docker-contained (`qwen38flashnext-dspark:local`, one-command lifecycle).
- **The SM121 story**: stock/nightly SGLang still can't run this model on GB10 (TRT-LLM sparse decode silently corrupts long context; the fallback doesn't compile). The patched derivative image is the only working path today — see `docs/sglang-deployment.md`.
- **llama.cpp GGUF fallback** (1× Spark): `-ngl 999 -ot "per_layer_token_embd.weight=CPU"` streams the 51B-param PLE table from the NVMe; **MTP spec decode works** (PR #27836 + grafted head): code 27.4 → 32.1 t/s (+17%). First published DGX Spark + llama.cpp + MTP run.
- **Context ladder (llama.cpp)**: 512K with FP8 KV, 700K–1M with 4-bit KV — KV is cheap in this architecture (~24 KiB/token f16 across the 12 QSA layers).

## Quick start

- **SGLang NVFP4 TP2** (the full-quality path): follow [`docs/sglang-deployment.md`](docs/sglang-deployment.md) — clone the MiaAI recipe, set `.env` (template in `scripts/sglang/.env.example`), `./start.sh serve`.
- **llama.cpp GGUF** (single Spark): scripts in [`scripts/`](scripts/README.md) (`spark_download.sh`, `spark_build.sh`, `spark_serve*.sh`).

## Measured results

**SGLang NVFP4 TP2** (2026-08-30, `bench_serving` + timed runs):

| Config | Single-stream | Aggregate (24 conc) | E2E |
|---|---|---|---|
| NVFP4 KV | ~36 t/s | — | — |
| fp8_e4m3 KV | ~38–42 t/s | 138.6 tok/s | 15.8 s |
| **fp8 + `--speculative-attention-mode decode` (final)** | **~40–44 t/s** | **147.6–155.2 tok/s** | 12.7–14.3 s |

**llama.cpp GGUF** (1× Spark, 2026-08-29):

| Config | Code | Prose |
|---|---|---|
| plain | 27.4 t/s | 27.3 t/s |
| + MTP | 32.1 t/s (+17%) | 27.1 t/s |

## Repo structure

| Path | Contents |
|---|---|
| `docs/sglang-deployment.md` | SGLang NVFP4 TP2: full deployment runbook, .env, fixes, validation, gotchas |
| `docs/sglang-perf.md` | Performance tuning: A/B matrix, measured numbers, ceilings |
| `docs/research.md` | Model/hardware facts, runtime verdicts, memory math, sources |
| `docs/plan.md` | Plans A/B/C, serving modes, KV/quant measurements, MTP status |
| `docs/session-log.md` | Chronological build log: what was tried, measured, and learned |
| `scripts/` | llama.cpp setup/serve scripts + SGLang `.env` template (see `scripts/README.md`) |
| `assets/` | README charts (regenerate: `python scripts/gen_charts.py`) |

GLM-5.3-Flash feasibility is a separate project.

## Notes & caveats

- **Software maturity**: SGLang qwen4_exp merged Aug 26 (PR #36497) but unreleased; SM121 needs the patched image (PR #36845 still open). llama.cpp `qwen4_exp` base is merged; MTP is PR #27836. GB10 runs are community-verified, not upstream-validated.
- **UMA memory**: keep ≥8 GB headroom; booting under a full page cache can softlock the box (drop caches first — no sudo needed via the privileged-Docker one-liner in the deployment doc). mem-fraction 0.85+ risks the earlyoom SIGTERM.
- **lmcache**: evaluated and deferred — hybrid-layout bugs open for qwen4_exp, and 1M fits without KV offload.
- `--enable-torch-compile` breaks SM121 CUDA-graph capture (keep off); NEXTN steps 4 is impossible.

## License

Repo: MIT. Model weights: Qwen Community License 1.0 (see [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).
