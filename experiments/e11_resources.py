"""E11 — Resource model: per-stage operations, memory and platform estimates (N = 8192).

* validates the RAM model against the ELF of the thesis firmware (static buffers)
* per-stage cycle breakdown, legacy vs optimized pipeline, on every platform
* latency, energy, real-time factor, SRAM share
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fdedge.config import CODE_DIR
from fdedge.data import FeatureStore, blocked_split
from fdedge.experiment import Run
from fdedge.models import make_model
from fdedge.plotting import PAGE_W, SERIES, save
from fdedge.resources import (OP_CLASSES, estimate, flops, load_platforms, ops_rf, ops_softmax, pipeline_memory,
                              pipeline_ops, rf_flash_bytes, softmax_flash_bytes, total_ops)
from tools.elf_memory import symbol_sizes

run = Run("e11_resources", __doc__)
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
N = cfg.signal["window"]
fs_hz = float(cfg[ds]["fs"])
band = cfg[ds]["shaft_band_hz"]
plats = load_platforms()

st = FeatureStore(cfg, ds, N)
tr, va, te = blocked_split(st, 0)
rf = make_model("rf", C, cfg).fit(tr.X, tr.y)
D = len(make_model("softmax", C, cfg).features)
path = rf.mean_path_length(te.X)
n_nodes, n_trees = rf.n_nodes(), len(rf.clf.estimators_)
classifiers = {"RF": (ops_rf(n_trees, path, C), rf_flash_bytes(n_nodes, n_trees, "q16"), rf_flash_bytes(n_nodes, n_trees, "legacy")),
               "Softmax": (ops_softmax(D, C), softmax_flash_bytes(D, C, 1) + 4 * C, softmax_flash_bytes(D, C))}

# ---- 1. validate memory model on the thesis ELF ---------------------------
val = {}
elf = CODE_DIR / "legacy" / "firmware" / "rf_esp32" / "build" / "firmware.elf"
if elf.exists():
    syms = symbol_sizes(elf, "xtensa-esp32-elf", r"fft_real|fft_imag|extract_freq_features.*::Y|tree\d_nodes|forest_trees")
    measured_bufs = sum(v for k, v in syms.items() if "fft" in k or "::Y" in k) + 4 * 8192   # + malloc(window)
    measured_model = sum(v for k, v in syms.items() if "tree" in k)
    model_mem = pipeline_memory(8192, "legacy", 16, 0)
    val = {"elf": str(elf), "symbols": syms, "measured_buffers_bytes": measured_bufs,
           "model_buffers_bytes": model_mem.ram_total - 4 * 16, "measured_rf_bytes": measured_model,
           "note": "window buffer (32 KiB) is heap-allocated by malloc() in the thesis firmware"}
    run.json(val, "elf_validation")
    print("ELF validation:", val["measured_buffers_bytes"], "vs model", val["model_buffers_bytes"])

# ---- 2. per stage, per platform --------------------------------------------
rows, mem_rows = [], []
for model, (cops, flash_opt, flash_leg) in classifiers.items():
    for variant in ["legacy", "optimized"]:
        stages = pipeline_ops(N, fs_hz, variant, cops, band)
        mem = pipeline_memory(N, variant, 19, flash_leg if variant == "legacy" else flash_opt)
        for k, v in {**{f"ram:{a}": b for a, b in mem.ram.items()}, **{f"flash:{a}": b for a, b in mem.flash.items()}}.items():
            mem_rows.append({"model": model, "variant": variant, "component": k, "bytes": v})
        for pname, p in plats.items():
            e = estimate(stages, p)
            for s, ms in e["stage_ms"].items():
                rows.append({"model": model, "variant": variant, "platform": pname, "stage": s,
                             "ops": flops(stages[s]), "ms": ms})
            rows.append({"model": model, "variant": variant, "platform": pname, "stage": "TOTAL",
                         "ops": flops(total_ops(stages)), "ms": e["latency_ms"], "energy_mj": e["energy_mj"],
                         "rtf": e["latency_ms"] / (1e3 * N / fs_hz), "ram_bytes": mem.ram_total,
                         "flash_bytes": mem.flash_total,
                         "sram_share": mem.ram_total / (p["sram_kb"] * 1024)})
df = pd.DataFrame(rows)
run.table(df, "stages", latex=False)
tot = df[df.stage == "TOTAL"].drop(columns=["stage"])
run.table(tot, "platform_summary")
run.table(pd.DataFrame(mem_rows), "memory_breakdown")
opc = []
for variant in ["legacy", "optimized"]:
    stages = pipeline_ops(N, fs_hz, variant, classifiers["RF"][0], band)
    for s, c in stages.items():
        opc.append({"variant": variant, "stage": s, **{k: c.get(k, 0) for k in OP_CLASSES}})
run.table(pd.DataFrame(opc), "op_counts")

# ---- figure: stage share of the per-window latency (100 % stacked), RF -----
stages_order = ["time_feat", "window", "fft", "magnitude", "freq_feat", "order_feat", "classifier"]
fig, ax = plt.subplots(figsize=(PAGE_W, 2.1))
labels, y = [], 0
for pname in plats:
    for variant in ["legacy", "optimized"]:
        s = df[(df.model == "RF") & (df.variant == variant) & (df.platform == pname)].set_index("stage")
        total = float(s.loc["TOTAL", "ms"])
        left = 0.0
        for i, stg in enumerate(stages_order):
            v = 100.0 * float(s.loc[stg, "ms"]) / total if stg in s.index else 0.0
            ax.barh(y, v, left=left, color=SERIES[i], edgecolor="white", linewidth=0.8, height=0.72,
                    label=stg if y == 0 else None)
            left += v
        ax.text(101.5, y, f"{total:.1f} ms", va="center", fontsize=6.3, color="#0b0b0b")
        labels.append(f"{pname} ({'legacy' if variant == 'legacy' else 'optimized'})")
        y += 1
ax.set_yticks(range(len(labels)), labels, fontsize=6.3)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of estimated per-window latency (%), N = 8192; total at right")
ax.legend(ncol=7, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.9)
ax.grid(axis="y", visible=False)
save(fig, run.fig_dir, run.fig_name("stage_breakdown"))
print(tot.to_string(index=False))
run.done()
