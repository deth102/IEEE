"""E06 — Design-space exploration over the window length N.

For each N: blocked macro-F1 (F19, RF and Softmax), operation count, RAM/flash of
the legacy and optimized pipelines, estimated latency/energy on each platform,
real-time factor (latency / window duration) and whether it fits in SRAM.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fdedge.data import FeatureStore, blocked_split
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.metrics import scores
from fdedge.models import make_model
from fdedge.plotting import COL_W, PAGE_W, save, style
from fdedge.resources import (estimate, flops, load_platforms, ops_rf, ops_softmax, pipeline_memory, pipeline_ops,
                              rf_flash_bytes, softmax_flash_bytes, total_ops)

run = Run("e06_window_sweep", __doc__)
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
fs_hz = float(cfg[ds]["fs"])
band = cfg[ds]["shaft_band_hz"]
plats = load_platforms()
Ns = cfg["window_sweep"]

if run.plot_only:
    acc_s, res = run.load_table("accuracy"), run.load_table("resources")
else:
    acc_rows, res_rows = [], []
    for N in Ns:
        st = FeatureStore(cfg, ds, N)
        paths, nodes, trees = [], [], 0
        for f in run.folds:
            tr, va, te = blocked_split(st, f)
            for kind in ["rf", "softmax"]:
                m = make_model(kind, C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
                acc_rows.append({"N": N, "fold": f, "model": "RF" if kind == "rf" else "Softmax",
                                 **scores(te.y, m.predict(te.X), C)})
                if kind == "rf":
                    paths.append(m.mean_path_length(te.X))
                    nodes.append(m.n_nodes())
                    trees = len(m.clf.estimators_)
        D = len(make_model("softmax", C, cfg).features)
        n_nodes = int(np.mean(nodes))
        clf = {"RF": (ops_rf(trees, float(np.mean(paths)), C),
                      {"legacy": rf_flash_bytes(n_nodes, trees, "legacy"), "optimized": rf_flash_bytes(n_nodes, trees)}),
               "Softmax": (ops_softmax(D, C), {"legacy": softmax_flash_bytes(D, C), "optimized": softmax_flash_bytes(D, C)})}
        for model, (cops, cflash_by_variant) in clf.items():
            for variant in ["legacy", "optimized"]:
                cflash = cflash_by_variant[variant]
                stages = pipeline_ops(N, fs_hz, variant, cops, band)
                mem = pipeline_memory(N, variant, 19, cflash)
                for pname, p in plats.items():
                    e = estimate(stages, p)
                    res_rows.append({"N": N, "model": model, "variant": variant, "platform": pname,
                                     "arith_ops": flops(total_ops(stages)), "ram_bytes": mem.ram_total,
                                     "flash_bytes": mem.flash_total, "latency_ms": e["latency_ms"],
                                     "energy_mj": e["energy_mj"], "window_ms": 1e3 * N / fs_hz,
                                     "rtf": e["latency_ms"] / (1e3 * N / fs_hz),
                                     "fits_sram": mem.ram_total <= 0.6 * p["sram_kb"] * 1024})
    acc = pd.DataFrame(acc_rows)
    res = pd.DataFrame(res_rows)
    run.table(acc, "accuracy_raw", latex=False)
    acc_s = summarize(acc, ["N", "model"])
    run.table(acc_s, "accuracy")
    run.table(res, "resources")


# ---- figure: accuracy / RAM / latency vs N (three panels, one measure each) -
fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.0))
fig.subplots_adjust(wspace=0.42)
ax = axes[0]
for i, m in enumerate(["RF", "Softmax"]):
    s = acc_s[acc_s.model == m]
    ax.errorbar(s["N"], s["f1_mean"], yerr=s["f1_std"], capsize=1.5, elinewidth=0.6, label=m, **style(i))
ax.set_xscale("log", base=2)
ax.set_xlabel("Window length N")
ax.set_ylabel("Blocked macro-F1")
ax.legend()
ax = axes[1]
for i, v in enumerate(["legacy", "optimized"]):
    s = res[(res.model == "RF") & (res.variant == v) & (res.platform == "esp32")]
    ax.plot(s["N"], s["ram_bytes"] / 1024, label=v, **style(i + 2))
ax.axhline(plats["esp32"]["sram_kb"], color="#898781", lw=0.8, ls=":")
ax.text(Ns[0], plats["esp32"]["sram_kb"] * 1.08, "ESP32 usable DRAM", fontsize=6, color="#52514e")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Window length N")
ax.set_ylabel("RAM (KiB)")
ax.legend()
ax = axes[2]
for i, pname in enumerate(plats):
    s = res[(res.model == "RF") & (res.variant == "optimized") & (res.platform == pname)]
    ax.plot(s["N"], s["latency_ms"], label=pname, **style(i))
ax.plot(Ns, [1e3 * n / fs_hz for n in Ns], color="#0b0b0b", lw=0.9, ls="--", label="window duration")
# measured on the ESP32 board (bench_mcu): tables in DRAM (filled) and in flash (open)
hw = cfg.path("results_dir") / "hw"
meas = {"DRAM": [], "flash": []}
for n in Ns:
    for key, env in (("DRAM", "esp32dev_ramtables"), ("flash", "esp32dev")):
        f = hw / (f"bench_n{n}_{env}.log" if n != 8192 else f"bench_esp32_{env}.log")
        if f.exists():
            for line in f.read_text(errors="ignore").splitlines():
                if line.startswith("STAGE,optimized,total_features,"):
                    meas[key].append((n, int(line.split(",")[3]) / 240e3))
for key, mk in (("DRAM", "o"), ("flash", "o")):
    if meas[key]:
        x, y = zip(*meas[key])
        ax.plot(x, y, lw=0, marker=mk, ms=5, markerfacecolor=(style(0)["color"] if key == "DRAM" else "white"),
                markeredgecolor="#0b0b0b", markeredgewidth=0.8, label=f"ESP32 measured ({key} tables)")
if any(meas.values()):
    run.table(pd.DataFrame([(k, n, t) for k, v in meas.items() for n, t in v], columns=["tables", "N", "ms"]),
              "esp32_measured", latex=False)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Window length N")
ax.set_ylabel("Est. latency (ms)")
ax.legend(fontsize=5.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.6, handlelength=1.6)
save(fig, run.fig_dir, run.fig_name("design_space"))
print(acc_s.to_string(index=False))
print(res[(res.platform == "esp32")].to_string(index=False))
run.done()
