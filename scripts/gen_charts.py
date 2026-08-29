#!/usr/bin/env python3
"""Generate the README charts from measured data (2026-08-29, single DGX Spark).

Outputs PNGs into ../assets/. Re-run after any new measurements.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "assets"

# ---------------------------------------------------------------- throughput
configs = ["plain\n(f16 KV)", "MTP\n(f16, n3)", "MTP\n(q8_0, n3)", "MTP\n(q8_0, n2)"]
code = [27.4, 32.1, 29.3, 28.8]
prose = [27.3, 27.1, 31.3, 29.9]

x = np.arange(len(configs))
w = 0.36
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
b1 = ax.bar(x - w / 2, code, w, label="code", color="#4C72B0")
b2 = ax.bar(x + w / 2, prose, w, label="prose", color="#DD8452")
for b in (*b1, *b2):
    ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.set_ylabel("tokens / s")
ax.set_title("Qwen3.8-Flash-Next (UD-Q4_K_XL) on 1× DGX Spark — generation throughput\n(GPU experts + PLE on disk; PR #27836 build; 256-token gens, single stream)")
ax.set_xticks(x, configs)
ax.axhline(32.1, color="#4C72B0", ls="--", lw=0.8, alpha=0.5)
ax.legend(loc="upper right")
ax.set_ylim(0, 38)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/throughput.png")
plt.close(fig)

# ---------------------------------------------------------------- KV vs context
ctx = np.array([0.5, 0.7, 0.8, 1.0])  # M tokens
kv = {  # GB at full context, per-token incl. 3 KiB f16 indexer
    "f16":   ctx * 27.0,
    "q8_0 (FP8)": ctx * 15.75,
    "iq4_nl": ctx * 9.4,
}
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
for name, gb in kv.items():
    ax.plot(ctx, gb, marker="o", label=name, lw=2)
ax.axhline(12.8, color="red", ls="--", lw=1.2, label="~KV budget on 121 GiB box (108 GB weights + headroom)")
ax.set_xlabel("context length (M tokens)")
ax.set_ylabel("KV cache size (GB)")
ax.set_title("Qwen3.8-Flash-Next KV cache vs context — 12 QSA layers, 2 KV heads × 256\n(KV only; the ~3 KiB/token indexer cache stays f16)")
ax.set_xticks(ctx, ["512K", "700K", "800K", "1M"])
ax.set_ylim(0, 30)
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/kv_vs_ctx.png")
plt.close(fig)

# ---------------------------------------------------------------- quant mix
labels = ["Q4_K (experts)", "IQ4_NL (PLE n-gram)", "Q5_1 (some experts)", "Q8_0 (attention/small)", "misc"]
gb = [41.8, 27.1, 24.9, 9.6, 1.5]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
bars = ax.barh(labels[::-1], gb[::-1], color=["#55A868", "#4C72B0", "#DD8452", "#8172B3", "#937860"])
for b in bars:
    ax.annotate(f"{b.get_width():.1f} GB", (b.get_width(), b.get_y() + b.get_height() / 2),
                va="center", ha="left", fontsize=9, xytext=(3, 0), textcoords="offset points")
ax.set_xlabel("GB")
ax.set_title("Unsloth UD-Q4_K_XL GGUF — per-tensor quant mix (104.9 GB total)")
ax.set_xlim(0, 48)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/quant_mix.png")
plt.close(fig)

print("charts written to", OUT)
