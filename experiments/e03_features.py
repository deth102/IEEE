"""E03 — Feature ablation and permutation importance (blocked protocol).

Feature sets: T6 (time), S10 (spectral), O3 (order), F16 (conference),
F13nr (F16 without affine duplicates), F16nr (F13nr + order), F19 (proposed).
Also reports the on-device cost of each group (operations for N = 8192).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fdedge.data import FeatureStore, blocked_split
from fdedge.evaluation import blocked_cv, summarize
from fdedge.experiment import Run
from fdedge.features import FEATURE_NAMES, FEATURE_SETS
from fdedge.metrics import scores
from fdedge.models import make_model
from fdedge.plotting import COL_W, save, style
from fdedge.resources import flops, ops_fft, ops_freq_features, ops_magnitude, ops_order_features, ops_time_features, ops_window

run = Run("e03_features", __doc__)
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
N = cfg.signal["window"]
fs_hz = cfg[ds]["fs"]
st = FeatureStore(cfg, ds, N)
sets = ["T6", "S10", "O3", "F16", "F13nr", "F16nr", "F19"]
df, _ = blocked_cv(st, ["rf", "softmax"], sets, run.folds)
summ = summarize(df, ["features", "model"], ("f1", "acc"))

# operation cost of each feature group (optimized pipeline)
spec = ops_window(N, "opt") + ops_fft(N, "opt") + ops_magnitude(N, "opt")
cost = {"T6": flops(ops_time_features(N, "opt")),
        "S10": flops(spec + ops_freq_features(N, "opt")),
        "O3": flops(spec + ops_order_features(N, fs_hz, cfg[ds]["shaft_band_hz"]))}
cost["F16"] = cost["T6"] + cost["S10"]
cost["F13nr"] = cost["F16"]
cost["F16nr"] = cost["F19"] = cost["F16"] + flops(ops_order_features(N, fs_hz, cfg[ds]["shaft_band_hz"]))
summ["arith_ops"] = summ["features"].map(cost)
summ["n_features"] = summ["features"].map(lambda s: len(FEATURE_SETS[s]))
run.table(summ, "summary")

# permutation importance of F19 (RF and Softmax), on the blocked test blocks
rng = np.random.default_rng(0)
imp_rows = []
for f in run.folds:
    tr, va, te = blocked_split(st, f)
    for kind in ["rf", "softmax"]:
        m = make_model(kind, C, cfg, seed=f, features="F19").fit(tr.X, tr.y, va.X, va.y)
        base = scores(te.y, m.predict(te.X), C)["f1"]
        for j in FEATURE_SETS["F19"]:
            drops = []
            for r in range(3):
                Xp = te.X.copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                drops.append(base - scores(te.y, m.predict(Xp), C)["f1"])
            imp_rows.append({"fold": f, "model": "RF" if kind == "rf" else "Softmax",
                             "feature": FEATURE_NAMES[j], "f1_drop": float(np.mean(drops))})
imp = pd.DataFrame(imp_rows).groupby(["model", "feature"], sort=False)["f1_drop"].agg(["mean", "std"]).reset_index()
run.table(imp, "permutation_importance")

fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.3), gridspec_kw={"width_ratios": [1, 1.2]})
ax = axes[0]
w = 0.38
for i, m in enumerate(["RF", "Softmax"]):
    s = summ[summ.model == m].set_index("features").reindex(sets)
    ax.bar(np.arange(len(sets)) + (i - 0.5) * w, s["f1_mean"], w * 0.92, yerr=s["f1_std"], label=m,
           color=style(i)["color"], edgecolor="white", error_kw=dict(lw=0.6, capsize=1.2, ecolor="#52514e"))
ax.set_xticks(range(len(sets)), sets, rotation=30)
ax.set_ylabel("Blocked macro-F1")
ax.set_ylim(0.3, 1.02)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax = axes[1]
order = imp[imp.model == "RF"].sort_values("mean", ascending=True)["feature"].tolist()
y = np.arange(len(order))
for i, m in enumerate(["RF", "Softmax"]):
    s = imp[imp.model == m].set_index("feature").reindex(order)
    ax.barh(y + (i - 0.5) * w, s["mean"], w * 0.92, xerr=s["std"], color=style(i)["color"], label=m,
            edgecolor="white", error_kw=dict(lw=0.5, capsize=1, ecolor="#52514e"))
ax.set_yticks(y, order, fontsize=6)
ax.set_xlabel("Macro-F1 drop when permuted")
save(fig, run.fig_dir, run.fig_name("ablation_importance"))
print(summ.to_string(index=False))
run.done()
