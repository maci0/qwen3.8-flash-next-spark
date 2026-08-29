# Qwen3.8-Flash-Next on DGX Spark — Workspace

Goal: run **Qwen3.8-Flash-Next** (~180B-param multimodal MoE, `qwen4_exp` architecture, 6B active/token) on the **NVIDIA DGX Spark** units (`spark1` = 192.168.0.211, `spark2` = 192.168.0.212), ideally with the **PLE n-gram tables streamed from disk**.

| Doc | Contents |
|---|---|
| [`docs/research.md`](docs/research.md) | Full research: model facts, hardware facts, runtime verdicts, memory math, sources |
| [`docs/plan.md`](docs/plan.md) | Plans A/B/C, serving modes, KV/quant measurements, MTP status, ops notes |
| [`docs/session-log.md`](docs/session-log.md) | Chronological log of everything done on 2026-08-27, with measured results & lessons |
| [`scripts/README.md`](scripts/README.md) | Script inventory (what each does, how to run) |

---

## Status — 2026-08-27 (final)

| # | Item | State |
|---|------|-------|
| 1 | Research (model, hardware, runtimes, memory math) | ✅ done |
| 2 | Spark access (SSH via NVIDIA Sync key, aliases `spark1`/`spark2`) | ✅ done |
| 3 | **GGUF 4-bit via llama.cpp — serving on spark1** | ⏸ **stopped 2026-08-27 (user requested)** — server shut down, port 8080 closed, memory freed; model/build/scripts remain on `spark1` for restart |
| 4 | GPU experts + PLE n-gram table streamed from disk | ✅ **the active config** — ~25 t/s, ~110 G used / 10 G available |
| 5 | 1M context (YaRN) + quantized KV | ✅ tested: 1M with `iq4_nl` KV (knife-edge ~1 G headroom); **512K with FP8 KV recommended for daily use** |
| 6 | MTP spec decode | ✅ **working on spark1** — PR [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) build + jlkivey MTP-head GGUF grafted onto UD-Q4_K_XL; **code 32.1 vs 27.4 t/s (+17%), prose neutral** (first DGX Spark llama.cpp MTP run) |
| 7 | SGLang NVFP4 (Plan A) | ⏸ not started — the decision point |

### Live state (checked 2026-08-27 late)

- `spark1`: `llama-server` running detached (`~/qwen3.8-flash-next/spark_serve_ngram.sh` config — 16K×4 slots, `-ngl 999`, PLE on CPU/disk), **24–25 t/s** generation, 76.7 t/s prompt eval. Memory 111/121 G used, 10 G available, 0 swap. Disk 2.3 TB free.
- `spark1` artifacts: GGUF `UD-Q4_K_XL` (104 GB, 4 shards) in `models/`, llama.cpp 0.3.0-dev (PR #27742, commit `6c5afc86a`) built with CUDA sm_121.
- `spark2`: empty (`~/qwen3.8-flash-next` not yet created) — reserved for Plan A.
- Desktop: **not used for model work** (x86_64, no GPU); only hosts this workspace + scripts.

## Machines

| Host | Alias | IP | Specs (verified 2026-08-27) |
|------|-------|----|------|
| DGX Spark #1 | `spark1` | 192.168.0.211 | GB10 (sm_121), 20× aarch64, 121 GiB usable, driver 580.173.02, CUDA 13.0, 2.3 TB free NVMe |
| DGX Spark #2 | `spark2` | 192.168.0.212 | same |
| Desktop | — | 192.168.0.77 | x86_64, 32 cores, 123 GB RAM, no NVIDIA GPU |

SSH: `~/.config/NVIDIA/Sync/config/ssh_config` defines `spark1`/`spark2` with `nvsync.key`.

## Decision log

1. **"llamacpp or vllm?"** → Neither for the NVFP4 checkpoint: only **SGLang** serves `qwen4_exp` + NVFP4, and it needs **2× Spark TP2** (135 GB > 121 GiB single Spark). vLLM can't load the RadixArk FP8-PLE tensors; llama.cpp can't load NVFP4 safetensors.
2. **"ngrams streamed from disk"** → PLE n-gram tables (51B params) are a sparse lookup (16 rows/token). Achieved via **llama.cpp `-ngl 999` + `-ot "per_layer_token_embd.weight=CPU"`**: experts on GPU, PLE mmap'd from the NVMe.
3. **Single-Spark route = GGUF 4-bit** (Plan B): works at ~25 t/s. **MTP is reachable on the GGUF stack** (llama.cpp PR #27836 + community MTP-head GGUFs, +30–90% on code/structured) — but the higher-throughput path remains **Plan A: SGLang NVFP4 on 2×Spark** (proven 47–70 t/s with MTP4).
4. **1M context**: feasible with YaRN (factor 4) + quantized KV; `iq4_nl` KV fits 1M at ~1 G headroom; `q8_0` (FP8) KV fits 512K comfortably.
