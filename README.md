# Qwen3.8-Flash-Next on DGX Spark — Workspace

Goal: run **Qwen3.8-Flash-Next** (~180B-param multimodal MoE, `qwen4_exp` architecture, 6B active/token) on the **NVIDIA DGX Spark** units (`spark1` = 192.168.0.211, `spark2` = 192.168.0.212), ideally with the **PLE n-gram tables streamed from disk**.

| Doc | Contents |
|---|---|
| [`docs/research.md`](docs/research.md) | Full research: model facts, hardware facts, runtime verdicts, memory math, sources |
| [`docs/plan.md`](docs/plan.md) | Plans A/B/C, serving modes, KV/quant measurements, MTP status, ops notes |
| [`docs/session-log.md`](docs/session-log.md) | Chronological log of everything done 2026-08-27 → 08-29, with measured results & lessons |
| [`scripts/README.md`](scripts/README.md) | Script inventory (what each does, how to run) |

GLM-5.3-Flash feasibility is a **separate project**: `~/Desktop/glm-5.3-flash/` (own git repo, not pushed).

---

## Status — 2026-08-29 (final)

| # | Item | State |
|---|------|-------|
| 1 | Research (model, hardware, runtimes, memory math) | ✅ done |
| 2 | Spark access (SSH via NVIDIA Sync key, aliases `spark1`/`spark2`) | ✅ done |
| 3 | **GGUF 4-bit via llama.cpp — serving on spark1** | ✅ **live** — currently the **MTP config** (`http://192.168.0.211:8080`) |
| 4 | GPU experts + PLE n-gram table streamed from disk | ✅ `-ngl 999 -ot "per_layer_token_embd.weight=CPU"` — the PLE 51B table stays mmap'd on the NVMe |
| 5 | Context ladder (YaRN + quantized KV) | ✅ measured: 512K `q8_0` · 700K/800K/1M `iq4_nl` (FP8 tops out ~512–640K) |
| 6 | MTP spec decode | ✅ **working on spark1** — PR [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) build + jlkivey MTP head grafted onto UD-Q4_K_XL; **code 32.1 vs 27.4 t/s (+17%), prose neutral** — first DGX Spark llama.cpp MTP run |
| 7 | SGLang NVFP4 (Plan A) | ⏸ not started — the decision point |

### Live state (checked 2026-08-29)

- `spark1`: `llama-server` running detached via `spark_serve_mtp.sh` — grafted `...-MTP-00001-of-00005.gguf`, `-ngl 999`, PLE on CPU/disk, `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 -fa on`, 16K×1 slot, f16 KV. **Code ~32 t/s, prose ~27 t/s**, health OK, **113/121 G used / 8 G available**.
- Artifacts on `spark1` (`~/qwen3.8-flash-next/`): GGUF `UD-Q4_K_XL` (104 GB, 4 shards) + grafted `-MTP` (5 shards), MTP head GGUF, llama.cpp **PR #27836** build (`1d8de7c1b`), 9 serve/build scripts.
- `spark2`: empty — reserved for Plan A (SGLang TP2 NVFP4).
- Desktop: **not used for model work** (x86_64, no GPU); hosts this workspace only.

## Machines

| Host | Alias | IP | Specs |
|------|-------|----|------|
| DGX Spark #1 | `spark1` | 192.168.0.211 | GB10 (sm_121), 20× aarch64, 121 GiB usable, driver 580.173.02, CUDA 13.0, ~2.3 TB free NVMe |
| DGX Spark #2 | `spark2` | 192.168.0.212 | same |
| Desktop | — | 192.168.0.77 | x86_64, 32 cores, 123 GB RAM, no NVIDIA GPU |

SSH: `~/.config/NVIDIA/Sync/config/ssh_config` defines `spark1`/`spark2` with `nvsync.key`.

## Decision log

1. **"llamacpp or vllm?"** → Neither for the NVFP4 checkpoint: only **SGLang** serves `qwen4_exp` + NVFP4, and it needs **2× Spark TP2** (135 GB > 121 GiB single Spark). vLLM can't load the RadixArk FP8-PLE tensors; llama.cpp can't load NVFP4 safetensors.
2. **"ngrams streamed from disk"** → PLE n-gram tables (51B params) are a sparse lookup (16 rows/token). Achieved via **llama.cpp `-ngl 999` + `-ot "per_layer_token_embd.weight=CPU"`**: experts on GPU, PLE mmap'd from the NVMe.
3. **Single-Spark route = GGUF 4-bit** (Plan B): ~25–32 t/s. **MTP works on the GGUF stack** (PR #27836 + grafted head): code +17%, prose neutral. The higher-throughput path remains **Plan A: SGLang NVFP4 on 2×Spark** (proven 47–70 t/s with MTP4).
4. **Context**: YaRN + quantized KV; `iq4_nl` fits 700K–1M, `q8_0` (FP8) tops out ~512–640K; KV is only ~24 KiB/token f16 (12 QSA layers).
5. **Ops**: UMA softlocks are a recurring risk on restarts (two auto-reboots logged) — keep ≥8 G headroom.
