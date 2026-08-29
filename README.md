# Qwen3.8-Flash-Next on NVIDIA DGX Spark — llama.cpp GGUF serving

Serving **Qwen3.8-Flash-Next** (`qwen4_exp` architecture, ~180B-param multimodal MoE, 6B active/token) on **NVIDIA DGX Spark** (GB10, sm_121, 128 GB unified memory) with **llama.cpp** and the Unsloth 4-bit GGUF — including the 51B-parameter **PLE n-gram table streamed from disk** and working **MTP speculative decoding** on the GGUF stack.

## Highlights

- **GPU experts + PLE n-grams streamed from disk** — `-ngl 999 -ot "per_layer_token_embd.weight=CPU"` pins the 51B-param PLE lookup table to CPU/mmap (it's a sparse gather, 16 rows/token), keeping the ~25 GB table on the NVMe instead of the 121 GiB unified pool. ~25 t/s, ~1.5 min load, box stays safe.
- **MTP spec decode works on the GGUF stack** — llama.cpp PR [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) + a grafted MTP head (4B, 1 layer): **code 27.4 → 32.1 t/s (+17%)**, prose neutral. First published DGX Spark + llama.cpp + MTP run.
- **Context ladder (YaRN)** — 512K with FP8 (`q8_0`) KV; 700K–1M with 4-bit (`iq4_nl`) KV. KV is cheap in this architecture (~24 KiB/token f16 across the 12 QSA layers; 36 GDN layers use fixed recurrent state).
- **Full-quality path documented** — the NVFP4 checkpoint (135 GB, experts-only quant) via SGLang on 2× Spark TP2 (proven 47–70 t/s with MTP4) is analyzed in `docs/plan.md` but not built here.

## Quick start (on a DGX Spark)

```bash
# 1. Build llama.cpp with qwen4exp + MTP support (PR #27836; base qwen4exp is in master)
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git fetch origin pull/27836/head:pr27836 && git checkout pr27836
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build --config Release -j --target llama-server llama-cli

# 2. Download the model + MTP head, and graft the head onto the GGUF split
pip install huggingface_hub gguf numpy
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*"
hf download jlkivey/Qwen3.8-Flash-Next-MTP-PR27836-GGUF
python graft-mtp-shard.py <UD-Q4_K_XL-00001-of-00004.gguf> <mtp-Qwen3.8-Flash-Next-Q8_0.gguf>   # -> ...-MTP-00001-of-00005.gguf

# 3. Serve
export LLAMA_ATTN_ROT_DISABLE=1
build/bin/llama-server -m <...-MTP-00001-of-00005.gguf> \
  -ngl 999 -ot "per_layer_token_embd.weight=CPU" -fa on \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  --ctx-size 16384 --parallel 1 --no-warmup
```

The exact steps used here are captured as re-runnable scripts in [`scripts/`](scripts/README.md) (`spark_download.sh`, `spark_build.sh`, `spark_smoke.sh`, `spark_serve*.sh`).

## Measured results (single DGX Spark, 2026-08-29)

| Config | Code | Prose |
|---|---|---|
| plain (f16 KV, `-ngl 999`, PLE on disk) | 27.4 t/s | 27.3 t/s |
| **+ MTP** (`draft-mtp`, n-max 3, f16 KV) | **32.1 t/s (+17%)** | 27.1 t/s |
| MTP, q8_0 KV | ~29.3 t/s | ~31.3 t/s |

Tuning sweep: f16 KV beats q8_0 KV for code; n-max 2 ≈ 3. Prompt eval ~77–86 t/s.

Context × KV fits (1 slot): 512K `q8_0` ✅ (~7 G headroom) · 700K `iq4_nl` ✅ (~4–5 G) · 800K/1M `iq4_nl` ✅ (~1 G) · FP8 above ~512–640K swaps.

## Repo structure

| Path | Contents |
|---|---|
| `docs/research.md` | Model/hardware facts, runtime verdicts (SGLang / vLLM / llama.cpp), memory math, sources |
| `docs/plan.md` | Plan A (SGLang NVFP4 2×Spark), Plan B (GGUF, done), serving modes, MTP status, ops notes |
| `docs/session-log.md` | Chronological build log: what was tried, measured, and learned |
| `scripts/` | Re-runnable setup + serving scripts (see `scripts/README.md`) |

GLM-5.3-Flash feasibility is tracked in a separate project.

## Notes & caveats

- **Software maturity**: llama.cpp `qwen4exp` support shipped via PRs (base merged into master; MTP = PR #27836, still open). GB10 (sm_121) runs are community-verified, not upstream-validated.
- **Pair the MTP head with its patch** — community heads use incompatible tensor layouts (jlkivey's is for PR #27836).
- **UMA memory**: keep ≥8 GB headroom. A full-model GPU copy can softlock the unified pool (two auto-reboots logged in `docs/session-log.md`); file-backed (mmap) weights are safe and reclaimable.
- The NVFP4/SGLang path (Plan A) exists because the 135 GB checkpoint does not fit a single Spark — see `docs/plan.md`.

## License

Repo: MIT. Model weights: Qwen Community License 1.0 (see [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).
