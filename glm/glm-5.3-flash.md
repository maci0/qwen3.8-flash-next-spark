# GLM-5.3-Flash on 2× DGX Spark — Feasibility

Research date: 2026-08-27. GLM-5.3-Flash (codename *ox-alpha*) was released **2026-08-26** — the ecosystem is ~24–48 h old. Day-0/1 evidence; unverified facts flagged as such.

## 1. Model specs

| Property | Value |
|---|---|
| Total params | 320–321B (marketed 320B; HF/vLLM tag 321B: 311.65B routed experts + 9.67B rest) |
| Active params | **18B** (MoE: 288 routed experts, top-8, 1 shared; first 3 of 45 layers dense) |
| Architecture | `Glm5NextForConditionalGeneration` — hybrid: **34 KDA linear-attention + 11 NoPE sparse-MLA** (DeepSeek-sparse, KPool indexer, `index_topk=2048`, `index_kpool=4`), MLA-style KV (`kv_lora_rank=512`), mHC hyper-connections, **1 MTP draft layer** |
| Layers / hidden | 45 layers, hidden 4096, dense inter 12288, MoE inter 2048, 64 heads |
| Context | **1,048,576 native (1M)**; max output 128K (API). Nothing documented beyond 1M |
| What "Flash" is | Cost/speed-optimized **open** variant of flagship GLM-5.3 (which is API-only, no weights). Flash has an internal MTP draft head for spec decode — not a draft model itself |
| Multimodal | Yes — image + video (24-layer ViT; video via GLM-OCR-ViT; SGLang up to 240K visual tokens) |

## 2. Precisions & on-disk sizes

| Repo | Precision | Size |
|---|---|---|
| `zai-org/GLM-5.3-Flash` (default) | **FP8** (e4m3) | ~306 GiB (~328 GB) |
| `zai-org/GLM-5.3-Flash-BF16` | BF16 | ~598.5 GiB (~642 GB) |
| `unsloth/GLM-5.3-Flash-GGUF` | UD-IQ1_S / Q2_K_XL / IQ3_XXS / Q4_K_XL | 93.1 / 109 / 120 / 200 GB |
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | NVFP4 (experts only; attn/vision/embeddings BF16) | ~181 GiB (~195 GB) |
| misc | unsloth FP8, dealignai, local-inference-lab, antirez GGUF, MLX | — |

## 3. Memory math (per Spark: ~121 GiB total / ~110 GiB usable after headroom)

| Precision | Total | Per node @ TP2 | 2× Spark? | Single Spark? |
|---|---|---|---|---|
| BF16 | 642 GB | 299 GB | ❌ (needs 5×) | ❌ |
| FP8 | 328 GB | 153 GB | ❌ (needs 4×: 76.5 GB/node) | ❌ |
| **NVFP4** | ~195 GB | ~97 GB | ✅ tight (measured ~97 GB/node) | ❌ |
| GGUF Q4_K_XL | 200 GB | 100 GB | ✅ tight | ❌ |
| GGUF IQ3_XXS | 120 GB | 60 GB | ✅ comfortable | ❌ (111.8 GiB > 110) |
| GGUF Q2_K_XL | 109 GB | 54.5 GB | ✅ | ⚠️ borderline (~8 GiB slack) |
| GGUF IQ1_S | 93.1 GB | 46.5 GB | ✅ | ✅ (~23 GiB slack) |

**KV cache is nearly free** (34/45 KDA layers = no per-token KV; 11 MLA layers ≈ 12–13 KB/token BF16 / 6.4–6.6 KB/token FP8):
- 32K: ~0.4 GB · 128K: ~1.7 GB · **1M: ~13 GB (6.6 GB FP8)**. Not the binding constraint.
- KDA layers carry a per-request recurrent state (~2.2 GB for 31 slots on 2× Spark, SGLang-measured).

**GB10 overheads (community-measured):** CUDA runtime/activations ~3–8 GB; after ~97 GB of NVFP4 weights at TP2 the driver only reliably grants ~4.5 GiB KV; page cache competes with GPU allocations during load (use a cache-flusher sidecar); vLLM's multimodal front-end OOMs the 121 GB box → `--language-model-only` required on 2× Spark.

## 4. Runtime support on sm_121 (GB10)

| Engine | Status | Notes / evidence |
|---|---|---|
| **SGLang** | ✅ **only fully-verified path** | Official recipe on **2× GB10 TP2** (`lmsysorg/sglang:glm-5.3-flash-arm64`), merged PR [#36507](https://github.com/sgl-project/sglang/issues/36507). Needs `--attention-backend dsa --dsa-*-backend tilelang --kv-cache-dtype bfloat16 --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion` + TileLang smem patch (GB10 caps 101,376 B smem). |
| **vLLM** | ⚠️ community mods only | `glm5_next` not in main (PR [#53906](https://github.com/vllm-project/vllm/pull/53906) open; v0.28.0 lacks it) — use per-model images `vllm/vllm-openai:glm53-flash-arm64-cu130`. Vendor-verified H100/B200/GB200 only; stock image fails on GB10 (FlashInfer 0-RoPE sparse-MLA: "Unsupported architecture"). Community fix: NoPE zero-pad mod + `--moe-backend marlin` → **24.7–30.3 t/s TP2 MTP-5, 262K ctx** (43.4 t/s peak elsewhere); TP4 needs an 8-patch stack. |
| **llama.cpp** | ✅ arch supported, unmerged | Unsloth PR [#27754](https://github.com/ggml-org/llama.cpp/pull/27754) (KDA+DSA+kpool+mHC+MoE+vision); use `unslothai/llama.cpp` `glm5next/upstream` branch / Unsloth Desktop. GGUF quants 93–200 GB. **No published sm_121 throughput numbers.** |
| TokenSpeed / KTransformers | listed by Z.ai | no Spark evidence — unknown |

FP8/FP4 on GB10: sm_121 has FP8+FP4 tensor cores; NVFP4 weight-only works (vLLM marlin int4; SGLang flashinfer_cutlass); community unlocked FP8-e4m3 KV on sm_121. **No official NVIDIA playbook entry and no NIM** for GLM-5.3-Flash yet.

## 5. Precedents

- **GLM-5.3-Flash day-0 on 2× Spark**: 24.7 code / 30.3 structured / 19.6 prose t/s (MTP-5, TP2) — [NVIDIA forum](https://forums.developer.nvidia.com/t/glm-5-3-flash-running-on-2x-dgx-spark-sm-121-day-0-24-7-30-3-tok-s-with-mtp-5-two-silent-gb10-gotchas-worth-knowing/381433); 43.4 t/s peak — [forum](https://forums.developer.nvidia.com/t/glm-5-3-flash-on-2x-nvidia-dgx-spark-43-4-tok-s-peak-checkpoint/381429); TP4/1M on 4× Spark 35.7–63.8 t/s — [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark).
- GLM-4.7 (355B NVFP4) on 2× Spark, 64K, ~17.5 t/s vLLM TP2 — [level1techs](https://forum.level1techs.com/t/full-glm-4-7-355b-nvfp4-at-64k-on-two-dgx-sparks-working-recipe/252212).
- GLM-5.2 (753B) on 4× Spark, 1M ctx, ~24.7 t/s — [bird/GLM-spark](https://github.com/bird/GLM-spark).
- Unsloth: 1-bit needs 100 GB, 3-bit needs 128 GB devices ("like a Mac or DGX Spark") — [unsloth](https://unsloth.ai/docs/models/glm-5.3-flash).

## 6. Bottom line

- **2× DGX Spark (TP2): YES** — NVFP4 (~97 GB/node, measured) or GGUF Q4_K_XL (100 GB/node); IQ3/Q2/IQ1 more comfortable. Proven day-0 at 24.7–30.3 t/s (262K ctx). Binding constraint: **per-node memory** → 4-bit-class weights mandatory (FP8 needs 4× Spark).
- **Single Spark: NO for useful quality** — only 1-bit IQ1_S (93 GB) fits with headroom (retains ~71% of top-1% accuracy per Unsloth).
- Engine maturity: SGLang tilelang is the only cleanly-verified sm_121 path; vLLM needs community mods and loses multimodal on 2× Spark; llama.cpp is unmerged-PR territory.
- **Conflict with Qwen Plan A**: GLM TP2 occupies both Sparks — mutually exclusive with the Qwen SGLang NVFP4 TP2 plan; can coexist with the single-Spark Qwen GGUF server only by dedicating one Spark per model.

## 7. Sources

- [zai-org/GLM-5.3-Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash) · [config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json)
- [Z.ai Flash docs](https://docs.z.ai/guides/llm/glm-5.3-flash) · [Z.ai GLM-5.3 docs](https://docs.z.ai/guides/llm/glm-5.3)
- [vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) · [vLLM PR #53906](https://github.com/vllm-project/vllm/pull/53906)
- [SGLang PR #36507](https://github.com/sgl-project/sglang/issues/36507) · [llama.cpp PR #27754](https://github.com/ggml-org/llama.cpp/pull/27754)
- [LibertAI NVFP4 card](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) (+ [discussion](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/discussions/4))
- [unsloth guide](https://unsloth.ai/docs/models/glm-5.3-flash) · [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- [kingjones30 2×Spark repo](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark) · [tonyd2wild 4×Spark repo](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)
- [NVIDIA forum: day-0 2×Spark](https://forums.developer.nvidia.com/t/glm-5-3-flash-running-on-2x-dgx-spark-sm-121-day-0-24-7-30-3-tok-s-with-mtp-5-two-silent-gb10-gotchas-worth-knowing/381433) · [43.4 t/s](https://forums.developer.nvidia.com/t/glm-5-3-flash-on-2x-nvidia-dgx-spark-43-4-tok-s-peak-checkpoint/381429) · [ox-alpha release](https://forums.developer.nvidia.com/t/glm-5-3-flash-weights-released-ox-alpha/381345)
- [bird/GLM-spark (GLM-5.2)](https://github.com/bird/GLM-spark) · [glm5.app Spark analysis](https://glm5.app/blog/glm-5-3-flash-dgx-spark)
