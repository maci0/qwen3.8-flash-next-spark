#!/usr/bin/env python3
"""Apply sglang#36567 (Qwen4 PLE streaming from NVMe) inside the image.

Scope: the PLE-from-NVMe feature only (TP1, FP8 PLE, mmap or io_uring backend).
The PR's SM121 QSA-decode kernel is intentionally NOT applied — the DSpark image
already carries its own SM121 decode fixes (sm121_varlen + #36806 + NVFP4-KV).

Patches:
1. Copy qwen4_ple_nvme.py into sglang/srt/models/.
2. environ.py — add the 6 SGLANG_QWEN4_PLE_NVME_* env vars.
3. qwen4_exp.py — NVMePLEEmbedding branch in the PLE init, prefetch overlap,
   and the load-path skip for NVMe-backed shards.
"""

import pathlib

SRT = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt")
MARKER = "SGLANG_QWEN4_PLE_NVME_PATH"


def patch(path: pathlib.Path, replacements: list[tuple[str, str]], marker: str = MARKER):
    s = path.read_text()
    if marker in s:
        print(f"{path.name}: already patched")
        return
    for anchor, replacement in replacements:
        count = s.count(anchor)
        assert count == 1, (
            f"{path.name}: anchor matched {count} times (want 1):\n{anchor[:120]}"
        )
        s = s.replace(anchor, replacement, 1)
    path.write_text(s)
    print(f"{path.name}: patched")


# 1. new module -------------------------------------------------------------
SRC = pathlib.Path("/tmp/qwen4_ple_nvme.py")
DST = SRT / "models" / "qwen4_ple_nvme.py"
if not DST.exists():
    DST.write_text(SRC.read_text())
    print("qwen4_ple_nvme.py: installed")
else:
    print("qwen4_ple_nvme.py: already present")

# 2. environ.py -------------------------------------------------------------
ENV = SRT / "environ.py"
patch(
    ENV,
    [
        (
            "    SGLANG_ENABLE_QWEN4_PLE_FUSION = EnvBool(True)\n",
            "    SGLANG_ENABLE_QWEN4_PLE_FUSION = EnvBool(True)\n"
            "    # Stream Qwen4's PLE n-gram table from a local safetensors snapshot\n"
            "    # instead of allocating it in host or device memory. Empty = off.\n"
            "    SGLANG_QWEN4_PLE_NVME_PATH = EnvStr(\"\")\n"
            "    SGLANG_QWEN4_PLE_NVME_BACKEND = EnvStr(\"io_uring\")\n"
            "    SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH = EnvInt(512)\n"
            "    SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES = EnvInt(4096)\n"
            "    SGLANG_QWEN4_PLE_NVME_CACHE_PAGES = EnvInt(0)\n"
            "    SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL = EnvInt(1000)\n",
        ),
    ],
)

# 3. qwen4_exp.py -----------------------------------------------------------
MODEL = SRT / "models" / "qwen4_exp.py"

patch(
    MODEL,
    [
        # import
        (
            "from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration\n",
            "from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration\n"
            "from sglang.srt.models.qwen4_ple_nvme import (\n"
            "    NVMePLEEmbedding,\n"
            "    is_nvme_ple_embedding,\n"
            ")\n",
        ),
        # PLE init -> NVMe branch
        (
            "        self.ngram_embedding = VocabParallelEmbedding(\n"
            "            padded_vocab_size,\n"
            "            self.head_dim_per_ngram,\n"
            "            params_dtype=(\n"
            "                torch.float8_e4m3fn\n"
            "                if (quant_config is not None and quant_config.get_name() == \"fp8\")\n"
            "                or getattr(config, \"ple_embedding_dtype\", None) == \"float8_e4m3fn\"\n"
            "                else torch.bfloat16\n"
            "            ),\n"
            "            output_dtype=torch.bfloat16,\n"
            "            use_attn_tp_group=self.use_attn_tp_ngram,\n"
            "        )\n"
            "        self.ngram_embedding.register_buffer(\n"
            "            \"weight_scale\", torch.ones(1, dtype=torch.bfloat16), persistent=True\n"
            "        )\n",
            "        nvme_snapshot = envs.SGLANG_QWEN4_PLE_NVME_PATH.get()\n"
            "        if nvme_snapshot:\n"
            "            if get_parallel().tp_size != 1:\n"
            "                raise NotImplementedError(\"Qwen4 NVMe PLE currently supports TP1 only\")\n"
            "            self.ngram_embedding = NVMePLEEmbedding(\n"
            "                nvme_snapshot,\n"
            "                num_embeddings=padded_vocab_size,\n"
            "                embedding_dim=self.head_dim_per_ngram,\n"
            "                expected_shards=int(config.split_ngram_parts),\n"
            "            )\n"
            "        else:\n"
            "            self.ngram_embedding = VocabParallelEmbedding(\n"
            "                padded_vocab_size,\n"
            "                self.head_dim_per_ngram,\n"
            "                params_dtype=(\n"
            "                    torch.float8_e4m3fn\n"
            "                    if (quant_config is not None and quant_config.get_name() == \"fp8\")\n"
            "                    or getattr(config, \"ple_embedding_dtype\", None) == \"float8_e4m3fn\"\n"
            "                    else torch.bfloat16\n"
            "                ),\n"
            "                output_dtype=torch.bfloat16,\n"
            "                use_attn_tp_group=self.use_attn_tp_ngram,\n"
            "            )\n"
            "            self.ngram_embedding.register_buffer(\n"
            "                \"weight_scale\", torch.ones(1, dtype=torch.bfloat16), persistent=True\n"
            "            )\n",
        ),
        # skip host-pin when NVMe-backed
        (
            "        if config.ple_offload_embedding:\n"
            "            self.ple_embedding.ngram_embedding = Qwen4ExpPinnedHostEmbedding(\n"
            "                self.ple_embedding.ngram_embedding\n"
            "            )\n",
            "        if config.ple_offload_embedding and not is_nvme_ple_embedding(\n"
            "            self.ple_embedding.ngram_embedding\n"
            "        ):\n"
            "            self.ple_embedding.ngram_embedding = Qwen4ExpPinnedHostEmbedding(\n"
            "                self.ple_embedding.ngram_embedding\n"
            "            )\n",
        ),
        # prefetch stream
        (
            "        self._prefetch_stream = (\n"
            "            torch.cuda.Stream() if config.ple_offload_embedding else None\n"
            "        )\n",
            "        self._prefetch_stream = (\n"
            "            torch.cuda.Stream()\n"
            "            if config.ple_offload_embedding\n"
            "            or is_nvme_ple_embedding(self.ple_embedding.ngram_embedding)\n"
            "            else None\n"
            "        )\n",
        ),
        # start_prefetch
        (
            "        stream = self._prefetch_stream\n"
            "        stream.wait_stream(torch.cuda.current_stream())\n"
            "        lookup_ids.record_stream(stream)\n"
            "        with torch.cuda.stream(stream):\n"
            "            offloaded_embedding.gather(lookup_ids, out=output_view)\n"
            "        self._prefetch_state = prefetched, semantic_tokens, physical_tokens\n",
            "        stream = self._prefetch_stream\n"
            "        pending_nvme = None\n"
            "        if is_nvme_ple_embedding(offloaded_embedding):\n"
            "            pending_nvme = offloaded_embedding.start_gather(lookup_ids)\n"
            "        else:\n"
            "            stream.wait_stream(torch.cuda.current_stream())\n"
            "            lookup_ids.record_stream(stream)\n"
            "            with torch.cuda.stream(stream):\n"
            "                offloaded_embedding.gather(lookup_ids, out=output_view)\n"
            "        self._prefetch_state = (\n"
            "            prefetched,\n"
            "            semantic_tokens,\n"
            "            physical_tokens,\n"
            "            pending_nvme,\n"
            "        )\n",
        ),
        # consume prefetched
        (
            "        embeddings, semantic_tokens, physical_tokens = self._prefetch_state\n",
            "        embeddings, semantic_tokens, physical_tokens, pending_nvme = (\n"
            "            self._prefetch_state\n"
            "        )\n"
            "        if pending_nvme is not None:\n"
            "            output_view = embeddings.view(\n"
            "                embeddings.shape[0], self.ple_embedding.ngram_heads, -1\n"
            "            )\n"
            "            self.ple_embedding.ngram_embedding.finish_gather(\n"
            "                pending_nvme,\n"
            "                embeddings.device,\n"
            "                out=output_view,\n"
            "                stream=self._prefetch_stream,\n"
            "            )\n",
        ),
        # load path: skip NVMe shards
        (
            "            emb = ple_mod.ngram_embedding\n"
            "            if (\n",
            "            emb = ple_mod.ngram_embedding\n"
            "            if is_nvme_ple_embedding(emb):\n"
            "                loaded_shard_params.add(f\"{mod_prefix}.ngram_embedding.weight\")\n"
            "                return True\n"
            "            if (\n",
        ),
    ],
)

print("ALL PLE-NVMe PATCHES APPLIED")
