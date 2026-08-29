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

- The originally-given host address was unreachable (`No route to host`, ARP failed). User revealed the two Sparks (aliases `spark1`/`spark2`), reachable via the **NVIDIA Sync key**.
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


## Phase 11 — MTP correction (2026-08-29)

My 08-27 conclusion ("MTP impossible on the GGUF stack; head exists only in NVFP4") was **wrong / outdated within 48h**, corrected via online research:

- llama.cpp PR **#27742 (qwen4exp base) merged into master** (08-26/28). Master still has no qwen4exp MTP.
- llama.cpp PR **#27836** (open) adds qwen4exp MTP draft (`--spec-type draft-mtp`): loads the 1-layer 4B MTP block (31 `mtp.*` tensors, ~7.7 GB BF16) from a grafted main GGUF or a sidecar `-md` file.
- **4 community MTP-head GGUFs published** (jlkivey, dzannotti, quimmedes, ashbash/MLX). Patch and head must be paired (incompatible layouts).
- Community-verified: **+30–90% t/s on code/structured**, ~neutral on prose; `--spec-draft-n-max 3 --spec-draft-p-min 0.75`; head ≈ +2.5 GB (Q4_K_M).
- **No published DGX Spark + llama.cpp + MTP run yet** — open opportunity.
- Lesson: the llama.cpp qwen4exp ecosystem moved in days; conclusions about unmerged-PR-era capability go stale fast. Also: I under-researched the MTP *head* existence — it's in the BF16 source and NVFP4 alike; "absent from the Unsloth GGUF" ≠ "doesn't exist".


## Phase 12 — MTP implemented & benchmarked on spark1 (2026-08-29)

After the Phase 11 correction, actually ran it:

1. Rebuilt llama.cpp on spark1 from **PR #27836** (`git fetch origin pull/27836/head:pr27836` → `1d8de7c1b`, build 10667, sm_121 CUDA).
2. Downloaded jlkivey's MTP head (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, 4.1 GB, 34 tensors) + `graft-mtp-shard.py`; grafted onto our UD-Q4_K_XL split → `...-MTP-0000{1..5}-of-00005.gguf` (head as trailing shard, `block_count 48→49`, `nextn_predict_layers=1`); `--verify` PASSED (1256 tensors).
3. Served with `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 -fa on` + our PLE-on-disk config; log confirmed "creating MTP draft context against the target model". Memory ~107/121 G.
4. **A/B on identical build/model (parallel 1, 256-token gens)**:

| | code | prose |
|---|---|---|
| plain (`--spec-type none`) | 27.4 t/s | 27.3 t/s |
| **draft-mtp** | **32.1 t/s (+17%)** | 27.1 t/s (neutral) |

MTP helps code/structured, ~neutral on prose — matches the community pattern (they saw +30–90% code on Metal/ROCm with q8_0 KV/mlock). First published DGX Spark + llama.cpp + MTP run. Scripts: `spark_serve_mtp.sh` (MTP), `spark_serve_mtp_plain.sh` (baseline; one-flag swap).

**Tuning sweep** (q8_0 KV / n-max 2 vs 3): q8_0 KV consistently lowered code speed (32.1 → ~29.3 t/s), prose within noise (±10%); n-max 2 ≈ n-max 3. **Best config stays f16 KV + n-max 3** (code +17%).


## Phase 12b — second UMA softlock & auto-reboot (2026-08-29)

After the tuning sweep, restarting the MTP server wedged the box again: ICMP responded (0.1 ms), sshd couldn't send its banner, `spark2` unaffected. Auto-rebooted after ~10 min (`up 1 min`). Trigger pattern: server restart while the page cache is full of the model + a fresh ~100 GB CUDA allocation on the 121 GiB pool. Everything survived (NVMe); MTP server relaunched cleanly (113/121 G used, 8 G avail, health OK). Recurring failure mode on UMA — keep ≥8 GB headroom and expect occasional reboot-on-restart; `drop_caches` before load would help but needs root (no sudo on maci).


## Phase 13 — SGLang TP2 deployment (goal mode) — status 2026-08-29

**Staged on `spark1` (all done):** MiaAI recipe (`~/Qwen3.8-Flash-Next-Dual-DGX-Sparks`) cloned + reviewed; `.env` configured (CX rail 10.0.1.1/10.0.1.2, `enp1s0f1np1`/`rocep1s0f1` both, 1M ctx YaRN, NVFP4 KV, mem-fraction 0.82, NEXTN 3/1/4); patched image `qwen38flashnext-dspark:local` built on both nodes; NVFP4 weights (135.25 GB, 419 files, public repo) downloaded + verified on head and rsync'd to worker at ~350 MB/s over the CX7 rail. Two local fixes: `start.sh` patched to exclude the root-owned HF `trees/` cache dir from the weight rsync; page cache can be dropped without sudo via `docker run --rm --privileged --pid=host lmsysorg/sglang:qwen38flashnext sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'` (verified on spark1: 112 GB → 0).

**Blocker:** `spark2` wedged (UMA softlock — kernel pings at 0.3 ms, sshd banner times out on both LAN and CX IPs, no auto-reboot after 25+ min). TP2 cannot boot without it → needs a **physical power cycle**.

**Resume (one command after power cycle):**
```bash
# on spark1: drop caches on both, then boot
docker run --rm --privileged --pid=host lmsysorg/sglang:qwen38flashnext sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
ssh spark2 'docker run --rm --privileged --pid=host lmsysorg/sglang:qwen38flashnext sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"'
cd ~/Qwen3.8-Flash-Next-Dual-DGX-Sparks && ./start.sh serve   # idempotent; weights already synced
```
Then validate: `:8888` health, boot-log `max_total_num_tokens` ≥ 1M, throughput, NVFP4 KV pool.


## Phase 14 — SGLang TP2 LIVE (2026-08-29/30)

**Server up on 2× DGX Spark, fully Docker-contained.** MiaAI recipe, patched SM121 image (`qwen38flashnext-dspark:local`), TP2 over the CX7 rail (10.0.1.1:26400), OpenAI API on `:8888`.

- Boot log proof: `max_total_num_tokens=2,056,576` (NVFP4 KV) / `1,332,352` (fp8 KV), `context_len=1048576` (**1M ✓**), `ple_offload_embedding=True`, NEXTN 3/1/4, chunked prefill 1024, radix cache on (64 cached tokens observed), CUDA graphs captured, FlashInfer autotune done.
- Measured single-stream (timed curl, 300–600-token gens): **NVFP4 KV ~36 t/s; fp8_e4m3 KV ~38–42 t/s**. First request after boot is slower (~28 t/s, kernel warmup).
- Blocker resolved: spark2's vLLM DeepSeek-V4 container was stopped (`docker stop vllm-ds4-0731`, reversible) to free the node; page caches dropped via privileged docker (`docker run --rm --privileged --pid=host lmsysorg/sglang:qwen38flashnext sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'`) — no sudo needed.
- Currently running: **fp8_e4m3 KV** (best measured performance). NVFP4 KV = one `.env` flip (`NVFP4_KV_CACHE=1`) for max KV headroom (2M pool).
- Remaining A/B candidates (docs/sglang-perf.md): `--enable-linear-replayssm-spec`, `--speculative-attention-mode decode`, `--sampling-backend pytorch`, dual-rail NCCL (10.0.2.x rail), `--enable-overlap-schedule`.


## Phase 15 — Performance tuning complete (2026-08-30)

A/B results on the patched SM121 image, TP2 @1M:

| Change | Aggregate output (24c) | E2E | Single-stream | Kept? |
|---|---|---|---|---|
| baseline (NVFP4 KV) | — | — | ~36 t/s | — |
| fp8_e4m3 KV | 138.6 tok/s | 15.8 s | ~38–42 t/s | ✅ |
| + `--speculative-attention-mode decode` | **147.6–155.2** | 12.7–14.3 s | **~40–44 t/s** | ✅ **FINAL** |
| dual-rail NCCL (both CX7 rails) | — | — | — | ❌ reverted (recipe preflight is single-device) |

**Final running config**: fp8_e4m3 KV + spec-attn-mode decode + NEXTN 3/1/4 + NCCL channel tuning, single CX7 rail, `qwen38flashnext-dspark:local` image on both nodes, API `http://spark1:8888`, `max_model_len=1,048,576`, pool 1,254,528 tokens. NVFP4 KV (2,056,576 pool) is one `.env` flip away (`NVFP4_KV_CACHE=1`).


## Phase 16 — Documentation pass (2026-08-30)

Full doc set for the SGLang deployment published:

- **`docs/sglang-deployment.md`** (new) — reproducible runbook: SM121 patched-image rationale, `.env` (all values), the two local fixes (rsync trees-exclude via `scripts/patch_miaai_rsync.py`, privileged-Docker page-cache drop without sudo), boot/resume, validation checklist, measured results, gotchas.
- **`scripts/sglang/.env.example`** (new) — the exact working `.env` (no secrets).
- **`README.md`** — restructured as a two-stack project (SGLang NVFP4 TP2 flagship + llama.cpp GGUF fallback) with the measured tables.
- **`docs/plan.md`** — Plan A marked DONE with outcomes.
- **`docs/sglang-perf.md`** — final A/B numbers (fp8 + spec-attn-mode decode = 147.6–155.2 agg / ~40–44 t/s single; NVFP4-KV 2.06M pool option).

Live state at close: SGLang TP2 up on both Sparks (`spark1:8888`, `max_model_len=1,048,576`, pool 1,254,528 fp8), 105/121 GB used per node, 16 GB available each.


## Phase 17 — Lever A/B series complete (2026-08-30)

Tested all remaining perf levers (each = base config + one lever, 24 concurrent, same bench):

| Lever | Aggregate (24c) | Verdict |
|---|---|---|
| `NCCL_MAX/MIN_NCHANNELS` 4→8 | 150.9 | neutral (kept) |
| `--schedule-conservativeness 0.5` | 162.1 | ✅ helps alone |
| `--sampling-backend pytorch` | 163.4 | ✅ helps alone |
| **`--enable-linear-replayssm-spec`** | **183–234** | ✅✅ **winner** |
| all three combined | 193.2 | ❌ negative interaction |
| `--enable-overlap-schedule` | fails to boot | ❌ PLE constraint confirmed |

**Final running config**: fp8 KV + `--speculative-attention-mode decode --enable-linear-replayssm-spec` + NCCL8. Pool 1,232,832 (1M ✓). Single-stream up to ~81–103 t/s; aggregate band 183–234 (run-to-run variance ±25%). Scripts `scripts/sglang_ab.sh` / `scripts/sglang_ab_all.sh` reproduce the A/B harness. Full table in `docs/sglang-perf.md`.


## Phase 18 — Softlock #4 + NCCL GID drift (2026-08-30)

- **Incident**: after the A/B benchmark marathon, `spark1` softlocked (4th UMA softlock of the session — kernel pings, sshd starved). User power-cycled it.
- **Recovery boots failed twice with NCCL errors** ("remote process exited" / "unhandled system error"). Root cause: **GID-table drift after the reboot** — the recipe hardcodes `NCCL_IB_GID_INDEX=3`; post-reboot, spark1's GID 3 was a valid link-local address but **spark2's GID 3 was empty** (asymmetric tables; the IPv4 GIDs also sat at different indices: 4 vs 5).
- **Fix**: removed `NCCL_IB_GID_INDEX` from the recipe's NCCL env entirely (`sed -i '/NCCL_IB_GID_INDEX=/d' start.sh`) — **NCCL auto-selects a usable RoCEv2 GID**. Verified: clean boot (~600 s), pool 1,264,128 (>1M), completion OK.
- **Ops rule going forward**: never pin `NCCL_IB_GID_INDEX` on this setup; the tables drift across reboots. Also: expect a ~10-min boot + possible NCCL hiccup after any host reboot (drop caches first; the GID auto-fix makes reboots safe).


## Phase 19 — Post-recovery verification (2026-08-30)

Server recovered and re-verified after the softlock + GID fix:

- **API**: `spark1:8888` up, model `Qwen3.8-Flash-Next-NVFP4`, `max_model_len=1,048,576`.
- **KV pool**: 1,264,128 tokens (>1M ✓) with the final config (fp8 KV + spec-attn-decode + replayssm-spec + NCCL8, GID pin removed).
- **Completion**: works (thinking trace + answer).
- **Memory steady-state**: spark1 105/121 GB used, spark2 104/121 used, **~16 GB available per node** (~32 GB cluster-wide), ~47 GB reclaimable buff/cache.
- Cluster stable; the NCCL auto-GID fix makes future reboots safe (drop caches first, ~10-min boot).
