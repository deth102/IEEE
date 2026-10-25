"""E01 — Evaluation protocol and feature set.

Shows (i) how the leaky random-split protocol of the conference paper inflates
accuracy and (ii) the gain of the shaft-order features under the leak-free
blocked protocol.

Protocols
  random-32k : conference setting reproduced (N = 32768, 50 % overlap, random split)
  random-8k  : thesis firmware window (N = 8192, 75 % overlap, random split)
  blocked-8k : N = 8192, 5-fold time-blocked CV (proposed)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fdedge.data import FeatureStore
from fdedge.evaluation import blocked_cv, random_protocol, summarize
from fdedge.experiment import Run
from fdedge.plotting import COL_W, PAGE_W, heatmap, save, style

run = Run("e01_protocol", __doc__)
cfg, ds = run.cfg, run.dataset
kinds, fsets = ["rf", "softmax"], ["F16", "F19"]
if cfg.get("features", "F19") not in fsets:            # variant config (e.g. F22 with bearing orders)
    fsets.append(cfg["features"])
classes = list(cfg[ds]["classes"])
N = cfg.signal["window"]

frames = []
if ds == "kaist":
    st32 = FeatureStore(cfg, ds, 32768, hop=16384)
    frames.append(random_protocol(st32, kinds, fsets, run.seeds).assign(protocol="random, N=32768"))
st_r = FeatureStore(cfg, ds, N, hop=N // 4)
frames.append(random_protocol(st_r, kinds, fsets, run.seeds).assign(protocol=f"random, N={N}"))
st = FeatureStore(cfg, ds, N)
df_b, cms = blocked_cv(st, kinds, fsets, run.folds)
frames.append(df_b.assign(protocol=f"blocked, N={N}"))
df = pd.concat(frames, ignore_index=True)
run.table(df, "raw", latex=False)
summ = summarize(df, ["protocol", "model", "features"], ("f1", "acc", "bacc"))
run.table(summ, "summary")

# ---- figure: macro-F1 per protocol --------------------------------------
protocols = list(dict.fromkeys(df["protocol"]))
combos = [(m, f) for m in ["RF", "Softmax"] for f in fsets]
fig, ax = plt.subplots(figsize=(COL_W, 2.1))
w = 0.8 / len(combos)
for i, (m, f) in enumerate(combos):
    s = summ[(summ.model == m) & (summ.features == f)].set_index("protocol").reindex(protocols)
    x = np.arange(len(protocols)) + (i - (len(combos) - 1) / 2) * w
    st_ = style(i)
    ax.bar(x, s["f1_mean"], w * 0.9, yerr=s["f1_std"], color=st_["color"], label=f"{m}-{f}",
           edgecolor="white", linewidth=0.6, error_kw=dict(lw=0.7, capsize=1.5, ecolor="#52514e"),
           hatch=None if f == "F19" else "////")
ax.set_xticks(range(len(protocols)), [p.replace(", ", "\n") for p in protocols])
ax.set_ylabel("Macro-F1")
ax.set_ylim(0.5, 1.02)
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.2, columnspacing=0.8, handlelength=1.4)
save(fig, run.fig_dir, run.fig_name("f1_by_protocol"))

# ---- figure: blocked confusion matrices ---------------------------------
fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.9))
for ax, (k, f) in zip(axes, [(k, f) for k in kinds for f in fsets]):
    heatmap(ax, cms[(k, f)], classes, classes)
    ax.set_title(f"{'RF' if k == 'rf' else 'Softmax'} — {f}")
    ax.set_xlabel("Predicted")
    ax.tick_params(axis="x", rotation=45)
axes[0].set_ylabel("True")
for ax in axes[1:]:
    ax.set_yticklabels([])
save(fig, run.fig_dir, run.fig_name("confusion_blocked"))
run.json({k: v for k, v in {f"{a}_{b}": cms[(a, b)] for a, b in cms}.items()}, "confusion")
print(summ.to_string(index=False))
run.done()
