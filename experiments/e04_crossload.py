"""E04 — Generalisation to unseen loads.

(a) single-source: train on one load, test on every load (diagonal = blocked test block)
(b) leave-one-load-out (LOLO): train on all other loads, test on the held-out load
"""
import matplotlib.pyplot as plt
import numpy as np

from fdedge.data import FeatureStore, loads_of
from fdedge.evaluation import cross_load, summarize
from fdedge.experiment import Run
from fdedge.plotting import COL_W, PAGE_W, heatmap, save

run = Run("e04_crossload", __doc__)
cfg, ds = run.cfg, run.dataset
loads = loads_of(cfg, ds)
st = FeatureStore(cfg, ds, cfg.signal["window"])
fsets = ["F16", "F19"]
if run.plot_only:
    s1, s2 = run.load_table("single_summary"), run.load_table("lolo_summary")
    unseen = run.load_table("overview").query("setting == 'single-source, unseen'")
else:
    single = cross_load(st, ["rf", "softmax"], fsets, run.folds, multi=False)
    lolo = cross_load(st, ["rf", "softmax"], fsets, run.folds, multi=True)
    run.table(single, "single_raw", latex=False)
    run.table(lolo, "lolo_raw", latex=False)
    s1 = summarize(single, ["model", "features", "source", "target", "seen"])
    s2 = summarize(lolo[~lolo.seen], ["model", "features", "source", "target"])
    run.table(s1, "single_summary")
    run.table(s2, "lolo_summary")
    unseen = summarize(single[~single.seen], ["model", "features"]).assign(setting="single-source, unseen")
    seen = summarize(single[single.seen], ["model", "features"]).assign(setting="single-source, seen (blocked)")
    lol = summarize(lolo[~lolo.seen], ["model", "features"]).assign(setting="leave-one-load-out")
    run.table(__import__("pandas").concat([seen, unseen, lol]), "overview")

fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.95))
for ax, (m, f) in zip(axes, [(m, f) for m in ["RF", "Softmax"] for f in fsets]):
    M = np.full((len(loads), len(loads)), np.nan)
    for i, a in enumerate(loads):
        for j, b in enumerate(loads):
            v = s1[(s1.model == m) & (s1.features == f) & (s1.source == a) & (s1.target == b)]["f1_mean"]
            M[i, j] = v.iloc[0] if len(v) else np.nan
    heatmap(ax, M, loads, loads)
    ax.set_title(f"{m} — {f}")
    ax.set_xlabel("Test load")
axes[0].set_ylabel("Training load")
for ax in axes[1:]:
    ax.set_yticklabels([])
save(fig, run.fig_dir, run.fig_name("crossload_matrix"))

# narrow version for the manuscript (proposed feature set only), set beside the few-shot curves
fig, axes = plt.subplots(2, 1, figsize=(2.1, 1.95), sharex=True)
for ax, m in zip(axes, ["RF", "Softmax"]):
    M = np.full((len(loads), len(loads)), np.nan)
    for i, a in enumerate(loads):
        for j, b in enumerate(loads):
            v = s1[(s1.model == m) & (s1.features == "F19") & (s1.source == a) & (s1.target == b)]["f1_mean"]
            M[i, j] = v.iloc[0] if len(v) else np.nan
    heatmap(ax, M, loads, loads)
    ax.set_ylabel(f"{m}\ntraining load")
axes[1].set_xlabel("Test load")
save(fig, run.fig_dir, run.fig_name("crossload_matrix_f19"))
print(s2.to_string(index=False))
print(unseen.to_string(index=False))
run.done()
