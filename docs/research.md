# Research — Qwen3.8-Flash-Next on DGX Spark

Research date: 2026-08-27. Cross-verified from Hugging Face model cards/configs, SGLang/vLLM/llama.cpp docs, NVIDIA playbooks, and two independent community deployments of this exact model on DGX Spark systems (tonyd2wild, pocharlies — both dated 2026-08-26/27).

## 1. Goal & constraints

- Serve **RadixArk/Qwen3.8-Flash-Next-NVFP4** on the DGX Spark(s).
- Preference: **PLE n-gram tables streamed from disk** (memory pressure: 128 GB unified, ~121 GiB usable).
- Question asked: **llama.cpp or vLLM?**

## 2. The model

RadixArk NVFP4 = NVFP4 (W4A4) quantization of **Qwen/Qwen3.8-Flash-Next** ([card](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)), the open-weight **experimental preview of the Qwen4 architecture** (`model_type: qwen4_exp`).

| Property | Value |
|---|---|
| Total params | ~180B = 125B main + 51B PLE n-gram + 4B MTP |
| Active params | **6B/token** (512 routed experts, top-10 + 1 shared, per layer) |
| Layers | 48 = 12 × (3 × (Gated DeltaNet → MoE) → 1 × (QSA → MoE)) |
| Attention | GDN (linear) + **QSA** (Qwen Sparse Attention, micro-block, only 12/48 layers hold growing KV) |
| Gated residual | 4 branches, bottleneck rank 320 |
| PLE n-gram | 20M-entry base vocab; 8 bigram + 8 trigram hash heads; layer 2; 51.2B params |
| MTP | 1-layer (4B) multi-step predictor — the **built-in spec-decode draft** (no separate draft model exists) |
| Multimodal | text / image / video; ctx 262,144 native (extensible to 1M) |
| Source size | 360 GB BF16 |
| NVFP4 checkpoint | **135 GB** (~2.7× smaller); only routed experts quantized (294,912 tensors); attention/QSA/GDN/mHC/shared experts/routers/embeddings/LM head/vision/MTP stay BF16; PLE tables FP8 |

References: [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [LMSYS/SGLang day-0 blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next), [Qwen GitHub README](https://github.com/QwenLM/Qwen3.8-Flash-Next/).

## 3. PLE n-gram tables (the "ngrams")

- Config: `ngram_size: 3`, `ngram_vocab_size_base: 20000000`, `split_ngram_parts: 128`, `heads_per_ngram: 8`, `ple_embed_dim: 2560`, `ple_layer_ids: [2]`.
- 51.2B params = 16 heads × 20M rows × 160 values. Stored as **128 shard tensors**.
- On disk: **51.2 GB FP8** (F8_E4M3 + per-table scalar, from the `-FP8` revision) ≈ 47.7 GiB.
- In RAM if dequantized to BF16: **102.4 GB** ≈ 95.4 GiB.
- Access pattern: sparse lookup — each token reads exactly **16 rows** (2.5K values). It is a random-access store, not a GEMM weight; keeping it off-GPU costs almost nothing per step.

## 4. Memory math (single Spark: ~121 GiB usable)

| Config | Resident | Fits? |
|---|---|---|
| Full load NVFP4 (experts packed, PLE→BF16, rest BF16) | ~174 GiB | ❌ |
| NVFP4 + PLE streamed from disk | ~78 GiB + KV (4–17 GiB) | ✅ ~82–95 GiB |
| NVFP4, PLE kept FP8 (no dequant) | ~126 GiB | ❌ (over 121, and no serving path does this) |

**2× Spark TP2** (demonstrated config): ~74 GiB/rank (PLE split vocabulary-parallel) + KV → fits with 23–35 GiB headroom. This is why the model card's command (`--tp 2`, GB300/B300-validated) cannot work on a single GB10 — `--tp 1` OOMs at any context length; `--max-running-requests 36` is unrealistic (community caps at ~8, bounded by the GDN state pool).

## 5. Runtime verdicts

| Runtime | Serves `qwen4_exp`? | NVFP4 on GB10 (sm_121)? | Verdict for this model |
|---|---|---|---|
| **SGLang** | ✅ day-0 ([blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)); support is **PR [#36497](https://github.com/sgl-project/sglang/pull/36497)**, not a release; needs one-line sm_121 QSA gate patch ([#36531](https://github.com/sgl-project/sglang/issues/36531)) + flash-attn ABI stub | ✅ community-verified (2×Spark: [tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark), [pocharlies](https://github.com/pocharlies/qwen38-flash-next-dgx-spark-sglang)); 47–70 tok/s with MTP4 | **The only full-quality path.** Requires 2× Spark TP2 |
| **vLLM** | Partial: FP8/BF16 recipe + image exist ([recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)), validated GB300/H200/MI355X, **no GB10**; **cannot load RadixArk's FP8-PLE tensors** | NVFP4+qwen4_exp on GB10: no evidence | ✗ for the RadixArk checkpoint |
| **llama.cpp** | Via unmerged **PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742)** (+ [Unsloth guide](https://unsloth.ai/docs/models/qwen3.8-next)); GGUF quants only (cannot load NVFP4 safetensors) | Native sm_121 CUDA builds exist | ✅ single-Spark route: **GGUF + mmap + PLE on SSD** — the only real "ngrams streamed from disk" |

### "N-grams streamed from disk" — what it really means

- **SGLang** offloads the PLE table to *pinned host RAM* (`--ple-offload-embedding`). On a discrete-GPU box that saves VRAM; on the Spark's unified memory it is the **same 128 GB pool — no savings** → still OOM on one Spark. No disk-resident PLE option ([open request #36514](https://github.com/sgl-project/sglang/issues/36514)).
- **vLLM** `VLLM_PLE_CPU_OFFLOAD=1` uses pageable memory → OS can swap to SSD (measured ~30 tok/s on one Spark, but with a different community checkpoint, not RadixArk's).
- **llama.cpp** mmaps the GGUF → weights page in from disk on demand. Unsloth explicitly recommends offloading the PLE/Ngram layer to SSD with mmap ([guide](https://unsloth.ai/docs/models/qwen3.8-next)). 4-bit quant tables also keep PLE cheap (PLE is only ever quantized to 4-bit min even in 1-bit quants).
- llama.cpp's `--ngram` flags are context-statistics self-speculation, **unrelated** to the PLE tables.

## 6. GGUF quants (Unsloth, `unsloth/Qwen3.8-Flash-Next-GGUF`)

| Quant | Size | KLD |
|---|---|---|
| **UD-Q4_K_XL** (chosen) | **111.3 GB** | 0.0447 |
| UD-IQ4_XS | 93.7 GB | 0.0791 |
| UD-Q3_K_XL | 90.0 GB | 0.0997 |
| UD-IQ3_XXS | 82.0 GB | 0.1565 |
| UD-Q2_K_XL | 78.9 GB | 0.2133 |
| UD-IQ1_M | 74.5 GB | 0.3022 |
| UD-IQ1_S | 72.5 GB | 0.3751 |

`UD-Q4_K_XL` shards (verified via HF API): `-00001-of-00004.gguf` (10.9 MB), `-00002` (49.86 GB), `-00003` (49.38 GB), `-00004` (12.09 GB).

## 7. Hardware (DGX Spark — verified live)

- GB10: Blackwell GPU sm_121 (compute 12.1a) + 20-core Arm CPU (10× Cortex-X925 + 10× A725), **128 GB LPDDR5x unified**, ~273 GB/s, 4 TB NVMe, 240 W. [NVIDIA page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), [hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
- DGX OS (Ubuntu 24.04 ARM), driver 580.173.02, CUDA 13.0 toolkit at `/usr/local/cuda`. [release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
- Live check `spark1`/`spark2`: `aarch64`, 20 cores, 121 GiB usable, GB10, 2.4–2.5 TB free, docker 29.2.1, python 3.12, gcc 13.3.

## 8. Precedents (community, this exact model, 2026-08-26/27)

- **2× Spark TP2 SGLang NVFP4**: ~70 tok/s peak / ~47 typical (MTP4 + CUDA graphs); ~41–42 tok/s single-stream; up to ~153 tok/s @ 8 concurrent; KV pool up to 600K–1.05M tokens (~10–17 GB). [tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark), [pocharlies](https://github.com/pocharlies/qwen38-flash-next-dgx-spark-sglang)
- Known GB10 quirks: `--mem-fraction-static 0.85` OOMs during CUDA-graph capture (use 0.78–0.80); drop page cache (`sync; echo 3 > /proc/sys/vm/drop_caches`) before launch (UMA); JIT build must be single-job; sustained load runs 83–85 °C.
- Similar-size single-Spark precedents: Nemotron-3-Super-120B-A12B-NVFP4 @ 22.7–23.7 tok/s on one Spark via vLLM ([blog](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark)); gpt-oss-120b MXFP4 on Spark via SGLang.
- The card's "118.4 GB unchanged BF16" figure **already includes the PLE tables** — do not add dequantized PLE on top of it (double-counts 102.4 GB).

## 9. License

Qwen Community License 1.0 ([LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE)): display name if >100M MAU / >$20M monthly revenue; separate license required for Model-as-a-Service; internal use exempt.

## 10. Sources- [RadixArk/Qwen3.8-Flash-Next-NVFP4 model card](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) + [config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json)
- [LMSYS/SGLang day-0 blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next), [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next), [sglang#36514](https://github.com/sgl-project/sglang/issues/36514), [sglang#36531](https://github.com/sgl-project/sglang/issues/36531)
- [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next), [vLLM DGX Spark blog](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark)
- [Unsloth guide](https://unsloth.ai/docs/models/qwen3.8-next), [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742), [llama.cpp speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
- NVIDIA [DGX Spark page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), [playbooks](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/main/nvidia/sglang/README.md)
- Community: [tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark), [pocharlies](https://github.com/pocharlies/qwen38-flash-next-dgx-spark-sglang), [dolf3131](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark)


---

## 11. Model card re-check (addendum 2026-08-27)

Re-verified against the [RadixArk NVFP4 card](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), the [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next), and the [Qwen source card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next):

- **NVFP4 officially targets B200 / B300 / GB300 only** — the cookbook's "Where it runs" table lists exactly those for NVFP4; **GB10 (DGX Spark) is community-verified, not upstream-validated**.
- **MTP is the designed spec-decode mechanism**: *"a multi-step-trained MTP module, whose own full-attention layers are QSA as well, which is what keeps speculative acceptance high in practice."* The NVFP4 checkpoint keeps **all 31 MTP tensors BF16**.
- SGLang `qwen4_exp` support is **not in any tagged release** — build the model-support PR (sgl-project/sglang#36497).
- **Thinking cannot be turned off** for this model; use `--reasoning-parser auto` in SGLang to split `reasoning_content`. SGLang applies the checkpoint's own `generation_config.json` — leave sampling unset.
- Official 1M recipe (matches what was applied in llama.cpp): YaRN `rope_theta 10000000, factor 4.0, original_max_position_embeddings 262144`; SGLang form: `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 ... --json-model-override-args '{...}' --context-length 1000000`. Static YaRN slightly affects short-context quality (Qwen's own note).
- Param-count note: Qwen card says 125B + 51B + 4B MTP ≈ 180B; the SGLang cookbook says 176B total. Difference is unaccounted MTP/main overlap — not material for serving.

---

## 12. GLM-5.3-Flash on 2× DGX Spark

Moved to its own file: **[`docs/glm-5.3-flash.md`](glm-5.3-flash.md)** — specs, precision sizes, 2×Spark memory math, runtime status (SGLang/vLLM/llama.cpp), precedents, bottom line.

TL;DR: **runs on 2× Spark TP2 with 4-bit-class weights** (NVFP4 ~97 GB/node or GGUF Q4_K_XL), proven day-0 at 24.7–30.3 t/s; FP8 needs 4× Spark; single Spark only fits lossy 1-bit. Conflicts with the Qwen Plan A (both need TP2 on both nodes).
