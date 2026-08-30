# Scripts

All scripts run on a Spark (aarch64) under `~/qwen3.8-flash-next/`. Detached runs use
`setsid nohup bash <script> > <log> 2>&1 < /dev/null &` so they survive SSH disconnects.
Every script is idempotent-ish and safe to re-run.

| Script | Purpose |
|---|---|
| `spark_download.sh` | venv + `hf` CLI; downloads `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-Q4_K_XL` (111.3 GB, 4 shards) → `models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/`. |
| `spark_build.sh` | Clones llama.cpp, checks out **PR #27742** (`qwen4exp` support), builds CUDA sm_121 (`llama-cli`, `llama-server`, `llama-mtmd-cli`, `llama-gguf-split`). |
| `spark_smoke.sh` | `llama-cli` smoke test: `-ngl 0` (safe CPU/mmap) + `--no-warmup`, 64-token completion, prints timing. |
| `spark_serve.sh` | 16K ctx, 4 slots, no MTP, ~25 t/s, ~110 G used. (Pre-MTP daily config.) |
| `spark_serve_mtp.sh` | **Current serving config** — see above (PR #27836 + grafted head, 16K×1, f16 KV, `draft-mtp` n-max 3). |
| `spark_serve_1m.sh` | 1M ctx, 1 slot, **iq4_nl KV**, YaRN factor 4. ~1 G headroom (knife-edge). |
| `spark_serve_512k.sh` | 512K ctx, 1 slot, **q8_0 (FP8) KV**, YaRN factor 2. ~7 G headroom — safe long-context choice. |
| `spark_serve_800k.sh` | 800K ctx, 1 slot, YaRN factor 3.1. **q8_0 (FP8) KV does NOT fit** (swap-thrash, verified); the `_iq4nl` variant fits (~1 G headroom, verified serving). |
| `spark_serve_700k.sh` | 700K ctx, 1 slot, YaRN factor 2.7. **q8_0 (FP8) does NOT fit** (swap); **`_iq4nl` variant fits comfortably** (~4–5 G headroom, verified serving) — best big-ctx config. |
| `spark_serve_ngram.sh` | 16K×4 + `--spec-type ngram-mod` — measured **no speedup** (not engaged); kept for reference. |
| `spark_serve_mtp.sh` | **MTP server** (PR #27836 build + grafted `...-MTP-00001-of-00005.gguf`): `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 -fa on`, 16K×1 slot. Measured: code 32.1 vs 27.4 plain (+17%); prose neutral. Baseline: swap the spec-type line to `--spec-type none` (`spark_serve_mtp_plain.sh` on spark1). |

## Switching servers

```bash
ssh spark1 'pkill -f "[l]lama-server"; sleep 3; setsid nohup bash ~/qwen3.8-flash-next/spark_serve_512k.sh > ~/qwen3.8-flash-next/serve.log 2>&1 < /dev/null &'
```

## Key flags / knobs

- PLE-on-disk (the "ngrams streamed from disk" mechanism): `-ot "per_layer_token_embd.weight=CPU"` (tensor name verified in the GGUF).
- KV quantization: `--cache-type-k/v {f16,bf16,q8_0,iq4_nl,...}` (FP8 = `q8_0`).
- Context: `--ctx-size N --override-kv qwen4exp.context_length=int:N --rope-scaling yarn --rope-scale F --yarn-orig-ctx 262144` (F=2 → 512K, F=4 → 1M).
- Skip the slow warmup: `--no-warmup`.
- MTP on the GGUF stack: **available** via llama.cpp PR #27836 + grafted head — see `docs/plan.md` (MTP section).

## SGLang NVFP4 TP2 deployment scripts

| Script | Purpose |
|---|---|
| `sglang/.env.example` | The working `.env` for the SGLang NVFP4 TP2 deployment (1M ctx, fp8/NVFP4 KV, NEXTN, NCCL tuning). |
| `patch_miaai_rsync.py` | Patches the MiaAI recipe's weight rsync to exclude the root-owned HF `trees/` cache dir (rsync code-23 fix). |
| `sglang_ab.sh` | One A/B cycle: stop → drop caches (privileged docker, both nodes) → set `EXTRA_ARGS` → boot → `bench_serving` + timed single-stream. Usage: `sglang_ab.sh <label> "<EXTRA_ARGS>"`. |
| `sglang_ab_all.sh` | Runs a sequence of A/B cycles with clean attribution (base + one lever each). |
| `sglang/spark_serve_tp1_nvme.sh` | Single-Spark SGLang TP1 with PLE streamed from NVMe (sglang#36567): env `MAX_RUNNING` (4), `CTX` (1048576), `CHUNK` (1024), `MAX_TOTAL` (2097152), `MEM_FRACTION` (0.88), NVFP4 KV, YaRN 1M. |
| `sglang/apply_ple_nvme_patches.py` | Applies sglang#36567 (PLE-NVMe feature only) into the patched image build. |
| `sglang/pr36567/` | The PR's new modules (`qwen4_ple_nvme.py`, `qsa_decode.py`). |

Full SGLang runbook: `docs/sglang-deployment.md` · perf results: `docs/sglang-perf.md`.
