# Deployment Plan — Qwen3.8-Flash-Next on DGX Spark

Decision tree that came out of the research (`docs/research.md`):

```
One Spark + NVFP4 135GB  → impossible (OOM at any ctx)
Two Sparks + SGLang TP2  → full quality, 47–70 tok/s  ← endgame (Plan A)
One Spark + GGUF 4-bit   → works, mmap + PLE on SSD    ← current attempt (Plan B)
One Spark + smaller model→ Qwen3.8-27B-class NVFP4     ← fallback (Plan C)
```

## Plan A — SGLang, 2× Spark TP2, NVFP4 (full quality)

Proven by two independent community runs of this exact checkpoint. Both Sparks confirmed available (`.211`/`.212`).

1. **Build SGLang with qwen4_exp support** (both units): PR [sgl-project/sglang#36497](https://github.com/sgl-project/sglang/pull/36497) (branch `qwen4-main-squashed@73a2552`) — not in any release — plus:
   - one-line sm_121 QSA gate patch ([#36531](https://github.com/sgl-project/sglang/issues/36531));
   - flash-attn ABI stub;
   - JIT single-job: `MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=4`.
   Or use the community prebuilt `radixark/sglang-qwen38flashnext:sm121-qsa*` image.
2. **Download** `RadixArk/Qwen3.8-Flash-Next-NVFP4` (135 GB) on both units. Repo is a private candidate release — needs HF token if gated.
3. **Serve** (GB10-adapted from the card):
   ```bash
   python -m sglang.launch_server --model-path <path> --tp 2 \
     --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
     --page-size 64 --mamba-scheduler-strategy extra_buffer --mamba-track-interval 64 \
     --chunked-prefill-size 4096 --max-running-requests 8 --context-length 262144 \
     --mem-fraction-static 0.80 --allow-auto-truncate --ple-offload-embedding \
     --disable-radix-cache --max-mamba-cache-size 97 --max-total-tokens 600000 \
     --enable-torch-compile
   ```
   plus MTP spec-decode flags (`steps=3`, `draft=4`) per the community recipes. `drop_caches` before launch. Expect 41–70 tok/s, ≤8 concurrent, 83–85 °C.
4. **Verify**: OpenAI-compatible smoke test, GSM8K/AIME spot checks, throughput + memory/thermal monitoring.

Notes: 2×Spark networking over ConnectX (200G) or 10GbE; `--tp 2` requires both units reachable from each other.

## Plan B — GGUF 4-bit, llama.cpp, single Spark (DONE — serving on spark1)

Chosen because: single-Spark viable, real n-gram disk streaming via mmap, quick validation of the model on real hardware.

1. **Download** `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-Q4_K_XL` (111.3 GB, 4 shards) on `spark1` → `~/qwen3.8-flash-next/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/` — script `scripts/spark_download.sh`, log `download.log`.
2. **Build llama.cpp** with PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742), CUDA sm_121 — script `scripts/spark_build.sh`, log `build.log`. Targets: `llama-cli llama-mtmd-cli llama-server llama-gguf-split`.
3. **Run smoke test** (mmap default; PLE pages stay on disk under memory pressure):
   ```bash
   ~/qwen3.8-flash-next/llama.cpp/build/bin/llama-cli \
     -m ~/qwen3.8-flash-next/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
     -ngl 999 --ctx-size 8192 --temp 1.0 --top-p 0.95 --top-k 20 \
     -p "<your prompt>" -n 128
   ```
   (verify flags from `llama-cli --help` post-build: `-ngl`, mmap, `--spec-type draft-mtp` for the MTP draft, `--chat-template-kwargs '{"reasoning_effort":"medium"}'`.)
4. **If OK**: optionally serve via `llama-server` (OpenAI-compatible) and/or repeat on `spark2`. Then decide: keep GGUF for quick single-Spark use, or proceed to Plan A for full quality.
5. **If OOM**: drop to `UD-IQ4_XS` (93.7 GB) or a lower quant; or trim `--ctx-size`.

## Plan C — smaller model on one Spark

Qwen3.8-27B-class NVFP4 (fits one Spark; ~38 tok/s with DFlash2/DSpark draft). Only if a single Spark is a hard constraint and GGUF quality is insufficient.

### MTP (multi-token prediction) — status 2026-08-29 (CORRECTED)

**MTP IS possible on the llama.cpp/GGUF stack** — my 08-27 conclusion ("not possible, only via SGLang") was outdated within ~48h. Corrected state:

- The Unsloth GGUF still ships **without** the MTP head (converter `no_mtp = True`; verified) — that part stands.
- **llama.cpp PR #27742 (qwen4exp base) merged into master** (between 08-26 and 08-28). Master still has `supports_mtp_export = False` for qwen4exp — no MTP on stock master yet.
- **llama.cpp PR [#27836](https://github.com/ggml-org/llama.cpp/pull/27836)** ("qwen4exp: add NextN/MTP draft head `--spec-type draft-mtp`") — open, working, community-used. It loads the 1-layer 4B MTP block (31 `mtp.*` tensors, exists in the BF16 source — **not** only in NVFP4 as I wrongly claimed) and routes MTP contexts to a plain KV cache.
- **MTP-head GGUFs published** (2026-08-27/28): [jlkivey (PR#27836 graft route)](https://huggingface.co/jlkivey/Qwen3.8-Flash-Next-MTP-PR27836-GGUF), [dzannotti (`-md` sidecar)](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF), [quimmedes (`-md`)](https://huggingface.co/quimmedes/Qwen3.8-Flash-Next-MTP-GGUF), [ashbash (MLX only)](https://huggingface.co/ashbash/Qwen3.8-Flash-Next-MTP-Drafter-GGUF). **Patch and head must be paired** — the routes use incompatible tensor layouts.
- **Community-verified speedups** (llama.cpp + qwen4exp + draft-mtp, n-max 2–3): +30–90% on code/structured, ~neutral on prose. E.g. Strix Halo UD-Q4_K_XL: 20.3 → 35.8 t/s code; M5 Max UD-IQ4_XS: 38.5 → 50.4 t/s code (acceptance 0.57–0.90). Tuning: `--spec-draft-n-max 3 --spec-draft-p-min 0.75` ("stop drafting when the head is unsure").
- **Recipe sketch** (dzannotti sidecar route):
  ```bash
  LLAMA_ATTN_ROT_DISABLE=1 llama-server \
    -m <main UD-Q4_K_XL shard1> -md <MTP-Q4_K_M.gguf> -ngld 999 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
    -ngl 999 -fa on -ctk q8_0 -ctv q8_0
  ```
  Memory: head adds ~2.5 GB (Q4_K_M) → fits the 16K config (~113 G/121 G); too tight for 700K+ ctx configs.
- **No published DGX Spark + llama.cpp + MTP run yet** (as of 08-29) — we'd be first.
- **Our DGX Spark measurement (2026-08-29, UD-Q4_K_XL + jlkivey graft, PR #27836 build `1d8de7c1b`, parallel 1, f16 KV, `-fa on`)**: code 27.4 → **32.1 t/s (+17%)**; prose 27.3 → 27.1 t/s (neutral). **First published DGX Spark + llama.cpp + MTP run.** Community (Metal/ROCm) saw larger code gains (+30–90%) with q8_0 KV / mlock / their tuning. Graft verify passed (1256 tensors, 5 shards).
- **Tuning sweep (2026-08-29, same box)**: q8_0 KV + n-max 3 → code ~29.3 t/s (consistently slightly *worse* than f16's 32.1), prose ~31.3; n-max 2 ≈ n-max 3. Single-run variance ±10%. **Verdict: f16 KV + n-max 3 remains the best config** on this box.
- SGLang NVFP4 on 2×Spark TP2 remains the higher-throughput path (47–70 t/s with MTP4).

## Live runbook — Plan B attempt

| Step | Machine | Status |
|---|---|---|
| SSH access via `nvsync.key` (`spark1`/`spark2` aliases) | — | ✅ |
| venv + `hf` CLI | spark1 | ✅ |
| GGUF download (111.3 GB, UD-Q4_K_XL × 4 shards) | spark1 | ✅ (xet stalled → `HF_HUB_DISABLE_XET=1` plain-HTTP resume) |
| llama.cpp build (PR 27742, sm_121 CUDA) | spark1 | ✅ llama.cpp 0.3.0-dev (commit `6c5afc86a`) |
| Smoke test (`scripts/spark_smoke.sh`) | spark1 | ✅ PASSED — thinking trace + `4`, 7.9 t/s (CPU/mmap) |
| **Serve: GPU experts + PLE on disk** (`spark_serve.sh`) | spark1 | ✅ **live at `http://192.168.0.211:8080`** — ~25 t/s, ~110 G/10 G |
| 1M ctx (iq4_nl KV) / 512K (q8_0 KV) | spark1 | ✅ tested / script ready |
| MTP (spec decode) | — | ❌ not possible on GGUF stack (head absent) — see MTP section |
| Plan A: SGLang NVFP4 2×Spark TP2 | spark2+both | ⏳ decision point |

Full chronological record: [`docs/session-log.md`](session-log.md). Script inventory: [`scripts/README.md`](../scripts/README.md).

### Serving modes tested (2026-08-27, spark1)

| Mode | Command essence | Result |
|---|---|---|
| Full GPU offload | `-ngl 1024` | ❌ softlocked the box (OOM livelock → auto-reboot): full 111 GB GPU copy on the 121 GiB pool |
| CPU + mmap | `-ngl 0` | ✅ works, 7.9 t/s; ~110 GB file-backed RSS, fully reclaimable |
| **GPU + PLE on disk** (current) | `-ngl 999 -ot "per_layer_token_embd.weight=CPU"` | ✅ **~24.7 t/s**, load in ~1.5 min, ~110 G used / 10 G avail (PLE table mmap'd from NVMe, 16 rows/token) |

- Cold load takes ~13 min single-threaded (~295K tensors; PR #27742 loader). `--no-warmup` used.
- PLE tensor name in the GGUF: `per_layer_token_embd.weight` (dims 160×320,001,536; 51.2 G elements). Experts: `blk.N.ffn_{gate,up,down}_exps.weight`.
- **KV cache**: 4 slots × 16,384 ctx (`n_slots=4, n_ctx_slot=16384, kv_unified=true`) = 65,536 tokens capacity. **KV is ~24 KiB/token f16** — 12 QSA layers × (2 KV heads × 256 dims × 2) — NOT 2 KiB/token (that was per-layer). So: 16K×4 slots ≈ 1.6 GB; 1M ctx ≈ 23.9 GB (f16) / 12.7 GB (q8_0) / 6.4 GB (iq4_nl). The 36 GDN layers use fixed recurrent state, not KV. **Measured ctx×KV fits (single slot)**: 512K `q8_0` ✅ ~7 G headroom; **700K `q8_0` ❌ swap-thrash (121/121 + growing swap)**; **700K `iq4_nl` ✅ ~4–5 G headroom (verified serving, the most comfortable big-ctx config)**; **800K `q8_0` ❌ swap-thrash**; **800K `iq4_nl` ✅ ~1 G headroom (verified)**; 1M `iq4_nl` ✅ ~1 G headroom. FP8 ceiling is ~512–640K. The ~3 KiB/token QSA indexer cache stays f16 in this build (not quantized).
- **1M ctx config** (`spark_serve_1m.sh`): `--ctx-size 1000000 --parallel 1 --cache-type-k/v iq4_nl --rope-scaling yarn --rope-scale 4.0 --yarn-orig-ctx 262144 --override-kv qwen4exp.context_length=int:1000000`. At q8_0 KV the box ran 120/121 G used + 9 G swap (too tight); iq4_nl KV → 120/121 G used, **0 swap**, serving (verified completion). 1M leaves only ~1 G headroom — knife-edge. **512K with q8_0** (`spark_serve_512k.sh`) → ~7 G headroom (recommended for daily use). YaRN factor 4 = Qwen's static recipe for 1M; note their warning that static YaRN slightly affects short-context quality. KV types available in this build: f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1 ("turboquant" is not a llama.cpp type; q8_0 ≈ FP8).
- **Quant mix (UD-Q4_K_XL, 104.9 GB)**: Q4_K 41.8 GB (experts) + IQ4_NL 27.1 GB (PLE n-gram table) + Q5_1 24.9 GB (some experts) + Q8_0 9.6 GB (attention/small) + ~1.5 GB misc. Measured prompt eval 76.7 t/s, generation 25.7 t/s.

## Blockers / notes

- Download + build run detached on `spark1` (`~/qwen3.8-flash-next/`, logs `download.log` / `build.log`). They survive SSH disconnects (`setsid nohup`).
- If `-DCMAKE_CUDA_ARCHITECTURES=121` misbehaves, fall back to `120` (GB10 JITs PTX from sm_120). Configure step already passed with 121.
- Do not run model workloads on the desktop (x86_64, no GPU) — Sparks only.
