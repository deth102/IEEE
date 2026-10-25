"""E02 — Hyper-parameter validation curves under the blocked protocol (F19).

RF: number of trees, maximum depth, maximum leaf nodes (+ flash size of the
compact node format). Softmax: learning rate and L2. Scores are macro-F1 on the
validation block of every fold (the test block is never touched).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fdedge.data import FeatureStore, blocked_split
from fdedge.metrics import scores
from fdedge.models import make_model
from fdedge.plotting import PAGE_W, save, style
from fdedge.resources import rf_flash_bytes
from fdedge.experiment import Run

run = Run("e02_hyperparams", __doc__)
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
st = FeatureStore(cfg, ds, cfg.signal["window"])
base_rf = dict(cfg["rf"])

grids = {
    "n_estimators": [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 10, 12, 16],
    "max_leaf_nodes": [4, 6, 8, 12, 16, 22, 32, 48, 64, 128],
}
sm_grids = {"lr": [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0], "l2": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}

splits = [blocked_split(st, f) for f in run.folds]
rows = []
for param, values in grids.items():
    for v in values:
        for f, (tr, va, _) in zip(run.folds, splits):
            m = make_model("rf", C, cfg, seed=f, **{param: v}).fit(tr.X, tr.y)
            rows.append({"model": "RF", "param": param, "value": v, "fold": f, "nodes": m.n_nodes(),
                         "flash_bytes": rf_flash_bytes(m.n_nodes(), len(m.clf.estimators_)),
                         **scores(va.y, m.predict(va.X), C)})
for param, values in sm_grids.items():
    for v in values:
        for f, (tr, va, _) in zip(run.folds, splits):
            m = make_model("softmax", C, cfg, seed=f, **{param: v}).fit(tr.X, tr.y, va.X, va.y)
            rows.append({"model": "Softmax", "param": param, "value": v, "fold": f, **scores(va.y, m.predict(va.X), C)})
df = pd.DataFrame(rows)
run.table(df, "raw", latex=False)
summ = df.groupby(["model", "param", "value"]).agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                                                    nodes=("nodes", "mean"), flash_bytes=("flash_bytes", "mean")).reset_index()
run.table(summ, "summary")

fig, axes = plt.subplots(1, 5, figsize=(PAGE_W, 1.7))
panels = [("RF", "n_estimators", "Trees", True), ("RF", "max_depth", "Max depth", False),
          ("RF", "max_leaf_nodes", "Max leaf nodes", True), ("Softmax", "lr", "Learning rate", True),
          ("Softmax", "l2", "L2 (0 plotted at 1e-7)", True)]
for ax, (m, p, label, logx) in zip(axes, panels):
    s = summ[(summ.model == m) & (summ.param == p)].sort_values("value")
    x = s["value"].replace(0.0, 1e-7) if p == "l2" else s["value"]
    st_ = style(0 if m == "RF" else 1)
    ax.errorbar(x, s["f1_mean"], yerr=s["f1_std"], capsize=1.5, elinewidth=0.6, **st_)
    sel = cfg["rf"].get(p, cfg["softmax"].get(p))
    if sel is not None:
        ax.axvline(1e-7 if (p == "l2" and sel == 0) else sel, color="#898781", lw=0.8, ls=":")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(label)
    ax.set_title(m, fontsize=7)
axes[0].set_ylabel("Validation macro-F1")
save(fig, run.fig_dir, run.fig_name("validation_curves"))

# accuracy vs model size (all RF grid points)
fig, ax = plt.subplots(figsize=(3.45, 2.0))
for i, p in enumerate(grids):
    s = summ[(summ.model == "RF") & (summ.param == p)]
    ax.plot(s["flash_bytes"] / 1024, s["f1_mean"], lw=0, label=p, **{**style(i), "linestyle": "none"})
ax.set_xscale("log")
ax.set_xlabel("RF flash footprint (KiB, compact 8-B nodes)")
ax.set_ylabel("Validation macro-F1")
ax.legend(fontsize=6.5)
save(fig, run.fig_dir, run.fig_name("rf_size_tradeoff"))
print(summ.to_string(index=False))
run.done()
