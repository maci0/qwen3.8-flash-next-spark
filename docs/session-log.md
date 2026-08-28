# Session Log — Qwen3.8-Flash-Next on DGX Spark (2026-08-27)

Chronological record of everything done, measured, and learned. Companion to `research.md` (facts) and `plan.md` (plans).

## Goal & constraints

Serve `RadixArk/Qwen3.8-Flash-Next-NVFP4` on the DGX Spark(s); user preference: **ngrams streamed from disk**; question: **llama.cpp or vLLM?** Then: "try a GGUF 4-bit"; then GPU experts + PLE-on-disk; then 1M ctx; then MTP.

## Phase 1 — Research (start of session)

- Fetched the RadixArk model card: NVFP4 W4A4 (experts only) of Qwen3.8-Flash-Next (`qwen4_exp`), ~180B = 125B main + 51B PLE + 4B MTP, 6B active/token, 48 layers (12× GDN×3 + QSA×1), 512 experts top-10, ctx 262K, checkpoint **135 GB** (from 360 GB BF16). Card's serve command is SGLang-only (`qwen4_exp` support required), validated on **GB300/B300** with `--tp 2`.
- Two research agents produced cross-verified findings:
  - PLE n-gram tables: 51.2B params, 128 shards, **51.2 GB FP8 on disk / 102.4 GB BF16 in RAM**; sparse lookup (16 rows/token).
  - Single Spark (121 GiB usable) cannot host NVFP4 fully (~174 GiB) → **2×Spark TP2 required** for SGLang; community runs (tonyd2wild, pocharlies, 2026-08-26/27) measured **47–70 t/s with MTP4**.
  - vLLM: recipe exists for FP8/BF16 (GB300/H200/MI355X) but **cannot load RadixArk's FP8-PLE tensors**; NVFP4+GB10 unvalidated.
  - llama.cpp: `qwen4_exp` only via **unmerged PR #27742**; GGUF-only; mmap is the only true disk-streaming path.
  - **No draft model exists** — the 1-layer 4B **MTP head is the built-in spec-decode draft**.
- **Verdict**: llama.cpp or vLLM? → Neither for NVFP4. SGLang (2×Spark) is the full-quality path; GGUF+llama.cpp is the single-Spark path.

## Phase 2 — Access & connectivity

- `192.168.0.121` (user's original) was unreachable (`No route to host`, ARP failed). User revealed the Sparks: **192.168.0.211 / .212**, reachable via the **NVIDIA Sync key** (`~/.config/NVIDIA/Sync/config/ssh_config` → aliases `spark1`/`spark2`, key `nvsync.key`).
- Verified both: GB10, sm_121 (compute 12.1), 20 cores aarch64, 121 GiB usable, driver 580.173.02, CUDA 13.0 toolkit, ~2.4 TB free, Docker 29.2.1.
- User instruction: **do not use the desktop** (no GPU) — all model work on the Sparks.

## Phase 3 — GGUF 4-bit attempt on spark1 (Plan B)

### 3.1 Download (16:45–17:0x)

- `hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*"` (111.3 GB, 4 shards) via venv (`~/.venv`).
- **Stall**: xet transfer froze at 41,050,384,124 bytes (~10 min no progress). Killed; restarted with `HF_HUB_DISABLE_XET=1` → resumed over plain HTTP at ~100 MB/s, completed (`✓ Downloaded`). Cleaned 41 GB of stale `.incomplete` temp files.
- Lesson: `pkill -f "hf download"` self-matches the SSH shell (the pattern is in its own cmdline) → kills the session. Use the `[h]` bracket trick.

### 3.2 Build (16:45–17:0x)

- `git clone llama.cpp && git fetch origin pull/27742/head:pr-27742 && checkout pr-27742`
- `cmake -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121` — sm_121 accepted, build OK in ~10 min. Binaries: `llama-cli`, `llama-server`, `llama-mtmd-cli`, `llama-gguf-split`.
- Lesson: backgrounding chains over SSH (`cmd & cmd2`) breaks cwd — use absolute paths or one chain per job (`setsid nohup` + absolute paths).

### 3.3 Smoke test — three attempts

| Attempt | Config | Result |
|---|---|---|
| 1 | `-ngl 1024` (full GPU offload) | ❌ **softlocked the box** (OOM livelock at 121/121 G → auto-reboot). Two instances were launched (watcher race), each copying 111 GB into the unified pool. |
| 2 | `-ngl 0` (CPU + mmap) | ✅ **worked** — `What is 2+2?` → thinking trace + `4`, `Generation: 7.9 t/s`. Load took ~13 min (single-threaded over ~295K tensors); box stayed responsive (~110 GB file-backed RSS is reclaimable). |
| 3 | `-ngl 999 -ot "per_layer_token_embd.weight=CPU"` | ✅ **~24.7 t/s, load ~1.5 min** — the winning config (see Phase 4). |

- Misdiagnosis along the way: a 2.5 GB `smoke_raw.log` of `\n\r` junk looked like a hang; it was llama.cpp's REPL/spinner rendering while the loader ground through ~295K tensors. `syscw` counts (1.8B write syscalls) were the spinner on a pipe, not a bug. The loader is pathologically slow single-threaded but does finish.
- `--no-warmup` used to skip the extra warmup run.

## Phase 4 — GPU experts + PLE streamed from disk (the user's ask)

- GGUF tensor analysis (manual GGUF header parser — the `gguf` PyPI reader returned 0 tensors because shard 1 is metadata-only with per-shard headers in shards 2–4):
  - PLE n-gram table = **`per_layer_token_embd.weight`** (dims 160×320,001,536 = 51.2 G elements, type IQ4_NL).
  - Experts = `blk.N.ffn_{gate,up,down}_exps.weight` (0.84 G elements each, 512 experts × 48 layers).
- Final config (`scripts/spark_serve.sh`): `-ngl 999 -ot "per_layer_token_embd.weight=CPU" --ctx-size 16384 --parallel 4 --no-warmup`.
- Results: **24.7 t/s** generation (measured 157 tok / 6.5 s; server logs: 25.0–25.7 t/s), prompt eval 76.7 t/s, load ~1.5 min, ~110 G used / 10 G available. Correct outputs incl. thinking traces.
- Memory accounting: ~86 GB CUDA (unified) + ~30 GB process RSS; the ~25 GB PLE stays as mmap'd file pages on the NVMe (16 rows/token touched).

## Phase 5 — KV cache & quantization answers

- **KV = ~24 KiB/token at f16** (12 QSA layers × 2 KV heads × 256 dims × 2). My first answer (2 KiB/token) was wrong — that's per-layer; corrected in `plan.md`.
- 16K×4 slots ≈ 1.6 GB KV; 1M ctx ≈ 23.9 GB (f16) / 12.7 GB (q8_0) / 6.4 GB (iq4_nl) + indexer cache (~3 GB f16 at 1M, not quantizable in this build).
- **Quant mix of UD-Q4_K_XL** (from GGUF header): Q4_K 41.8 GB (experts) + IQ4_NL 27.1 GB (PLE) + Q5_1 24.9 GB (some experts) + Q8_0 9.6 GB (attention/small) + ~1.5 GB misc = 104.9 GB.

## Phase 6 — 1M context (YaRN + quantized KV)

- GGUF declares `qwen4exp.context_length = 262144`, `rope.freq_base = 1e7`.
- Tried `--ctx-size 1000000 --parallel 1 --rope-scaling yarn --rope-scale 4.0 --yarn-orig-ctx 262144 --override-kv qwen4exp.context_length=int:1000000`:
  - With `--cache-type-k/v q8_0`: loaded+served but 120/121 G used + **9 G swap** — too tight.
  - With `--cache-type-k/v iq4_nl`: **120/121 G used, 0 swap, serving** (verified completion `1M context works`) — ~1 G headroom = knife-edge.
- Created `spark_serve_512k.sh` (512K ctx, q8_0 KV → ~7 G headroom) as the recommended daily long-context config.
- YaRN factor 4 = Qwen's static 1M recipe; their warning: static YaRN slightly degrades short-context quality.

## Phase 7 — MTP investigation

- `--spec-type draft-mtp` exists in the build (docs: "Use MTP heads from the main model"), but:
  - The GGUF has **no MTP head**: tensor blocks only span `blk.0–47`; no `mtp`/`nextn` metadata; `n_layer_nextn` absent.
  - The converter **deliberately drops it**: `conversion/qwen4exp.py` — "the MTP block is a separate draft head; vLLM drops it too; `supports_mtp_export = False`". So **no GGUF for this model can carry MTP today**.
  - Tried `--spec-type ngram-mod` as a stopgap: **not engaged** (0 spec lines in log), no speedup (24.4 vs 25.7 t/s).
- **MTP exists only in the NVFP4 checkpoint** (31 BF16 MTP tensors) → the only real MTP path is **SGLang `qwen4_exp` on 2×Spark TP2** (Plan A): proven 47–70 t/s with MTP4.

## Phase 8 — Model card re-check

- RadixArk card: unchanged; confirms 31 MTP tensors kept BF16 in NVFP4; SGLang-only; card's command has no MTP flags (community recipes add `steps=3, draft=4`).
- SGLang cookbook: support not in any release (build PR); **NVFP4 officially targets B200/B300/GB300 only** (GB10 = community-verified); MTP "keeps speculative acceptance high"; **thinking cannot be turned off** (`--reasoning-parser auto`); leave sampling unset (SGLang applies `generation_config.json`).
- Qwen card: official 1M YaRN recipe = `rope_theta 1e7, factor 4.0, original_max 262144` — matches what we applied in llama.cpp.

## Measured numbers (spark1)

| Metric | Value |
|---|---|
| Load time (cold, `-ngl 0`) | ~13 min |
| Load time (`-ngl 999` + PLE on disk) | ~1.5 min |
| Generation | 7.9 t/s (CPU) → **24.7–25.7 t/s** (GPU + PLE on disk) |
| Prompt eval | 76.7 t/s |
| Memory (16K×4 config) | ~110–111 G used / ~10 G available, 0 swap |
| Memory (1M×1, iq4_nl KV) | ~120 G used / ~1 G available, 0 swap |
| KV per token (f16) | ~24 KiB (12 QSA layers) |
| GGUF on disk | 104 GB (4 shards) |

## Lessons learned

1. **Unified-memory OOM is a softlock**: a full-model GPU copy (111 GB) on the 121 GiB pool livelocks the allocator → auto-reboot. File-backed (mmap) memory is reclaimable and safe; anonymous CUDA copies are not. Keep ≥5–10 GB headroom.
2. **`-ot "<tensor>=CPU"` is the lever** for per-tensor placement in llama.cpp — that's how you stream specific weights from disk.
3. **GGUF multi-file headers**: shard 1 can be metadata-only; per-shard headers carry the tensor lists. The `gguf` PyPI reader misreports 0 tensors in that layout — parse manually.
4. **`pkill -f` self-match** kills your own SSH session; use `[x]` bracket patterns.
5. **Spinner floods**: llama.cpp's loader spinner writes unbounded output when stdout is a pipe/file — don't mistake it for a hang; check `/proc/<pid>/io` (syscw) and CPU for real progress.
6. **KV is tiny in QSA architectures** — long context is memory-cheap here; the binding constraint is total weights + headroom.
7. **MTP for this model only exists in SGLang+NVFP4** — the GGUF toolchain deliberately drops the head.

## Open items / next steps

1. **Decision point**: SGLang Plan A (2×Spark NVFP4, 47–70 t/s, MTP) vs keep the GGUF server (~25 t/s).
2. Plan A first steps: clone sglang PR #36497 on both Sparks + sm_121 QSA gate patch (#36531) + flash-attn ABI stub; `MAX_JOBS=1`; download 135 GB NVFP4 ×2 (HF token if gated); serve TP2 (`drop_caches` before launch; mem-fraction 0.78–0.80; ≤8 concurrent).
3. Sync `spark_serve_512k.sh` to the Spark if 512K becomes the daily config.


## Phase 9 — 800K context test (FP8 vs iq4_nl KV)

- **800K + q8_0 (FP8) KV: does NOT fit.** Measured 121/121 G used, 0 available, **swap climbing to 5 G** — same softlock-adjacent pattern as 1M/FP8. The QSA KV alone at FP8 is ~12.75 KiB/token × 800K ≈ 10.4 GB, plus the always-f16 indexer cache (~3 KiB/token ≈ 2.4 GB) and cell metadata.
- **800K + iq4_nl KV: fits.** 120/121 G used, **0 swap**, verified serving (`800k works`). ~1 G headroom — knife-edge like 1M, but functional.
- Conclusion: 800K is reachable only with 4-bit KV on this box; FP8 tops out around 512–640K. Configs: `scripts/spark_serve_800k.sh` (q8_0) and the `_iq4nl` variant.


## Phase 10 — 700K context test

- **700K + q8_0 (FP8) KV: does NOT fit** — 121/121 used, 0 available, swap climbing (2 GB). FP8 ceiling on this box is ~512–640K.
- **700K + iq4_nl KV: fits comfortably** — 116/121 used, ~4–5 G headroom, 0 swap, verified serving (`700k works`). The best big-context config (more headroom than 800K/1M).
- Configs: `scripts/spark_serve_700k.sh` (q8_0, reference) + `_iq4nl` variant (recommended).
