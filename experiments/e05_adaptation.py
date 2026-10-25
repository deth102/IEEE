"""E05 — Adaptation to an unseen load.

Methods: none | HBR (healthy-baseline recalibration, unsupervised, M healthy
windows) | HBR-shift | DFS (oracle, all unlabeled target windows) | few-shot with
k labelled windows/class (Softmax: on-device fine-tuning of W, b; RF: host
retraining with the extra windows). Single-source and leave-one-load-out.
Also: HBR vs number of calibration windows, and on-device fine-tuning cost.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fdedge.adaptation import align, moments, pick_calibration
from fdedge.data import FeatureStore, blocked_split, loads_of, target_split
from fdedge.evaluation import adaptation, summarize
from fdedge.experiment import Run
from fdedge.features import FEATURE_SETS
from fdedge.metrics import scores
from fdedge.models import make_model
from fdedge.plotting import COL_W, save, style
from fdedge.resources import flops, load_platforms, ops_softmax_finetune, cycles

run = Run("e05_adaptation", __doc__)
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
st = FeatureStore(cfg, ds, cfg.signal["window"])
ks = cfg.adaptation["fewshot_k"]
seeds = run.seeds[:3]
frames = []
for multi in (False, True):
    d = adaptation(st, ["rf", "softmax"], run.folds, calib_m=16, fewshot_k=ks, seeds=seeds, multi=multi)
    frames.append(d.assign(setting="LOLO" if multi else "single-source"))
df = pd.concat(frames, ignore_index=True)
run.table(df, "raw", latex=False)
summ = summarize(df, ["setting", "model", "method", "k"])
run.table(summ, "summary")

# HBR vs calibration windows (single-source, Softmax + RF)
normal = list(cfg[ds]["classes"]).index("Normal")
loads = loads_of(cfg, ds)
rows = []
for f in run.folds:
    for src in loads:
        tr, va, _ = blocked_split(st, f, loads=[src])
        mdl = {k: make_model(k, C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y) for k in ["rf", "softmax"]}
        muS, sdS = moments(tr.X[tr.y == normal])
        for tgt in [t for t in loads if t != src]:
            T = target_split(st, [tgt])
            for M in cfg.adaptation["calib_windows"]:
                cal, rest = pick_calibration(T, normal, M)
                muT, sdT = moments(T.X[cal])
                for k, m in mdl.items():
                    rows.append({"fold": f, "source": src, "target": tgt, "M": M, "model": "RF" if k == "rf" else "Softmax",
                                 **scores(T.y[rest], m.predict(align(T.X[rest], muS, sdS, muT, sdT)), C)})
hbr = pd.DataFrame(rows)
run.table(summarize(hbr, ["model", "M"]), "hbr_vs_M")

# on-device fine-tuning cost (FD_SM_D = 19, C classes, k windows/class, 200 epochs)
plats = load_platforms()
cost_rows = []
for k in ks:
    ops = ops_softmax_finetune(len(FEATURE_SETS["F19"]), C, k * C, 200)
    for pname, p in plats.items():
        cost_rows.append({"k": k, "platform": pname, "arith_ops": flops(ops),
                          "est_ms": cycles(ops, p) / (p["f_mhz"] * 1e3),
                          "support_ram_bytes": 4 * (k * C * len(FEATURE_SETS["F19"]) + 2 * C * (len(FEATURE_SETS["F19"]) + 1))})
run.table(pd.DataFrame(cost_rows), "finetune_cost")

# ---- figure: F1 vs k -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.1), sharey=True)
for ax, setting in zip(axes, ["single-source", "LOLO"]):
    s = summ[summ.setting == setting]
    for i, (m, meth) in enumerate([("RF", "few-shot (host retrain)"), ("Softmax", "few-shot (on-device FT)")]):
        base = s[(s.model == m) & (s.method == "none")]["f1_mean"].iloc[0]
        c = s[(s.model == m) & (s.method == meth)].sort_values("k")
        x = [0] + c["k"].tolist()
        y = [base] + c["f1_mean"].tolist()
        e = [s[(s.model == m) & (s.method == "none")]["f1_std"].iloc[0]] + c["f1_std"].tolist()
        ax.errorbar(x, y, yerr=e, capsize=1.5, elinewidth=0.6, label=f"{m}: {meth.split('(')[1][:-1]}", **style(i))
        hb = s[(s.model == m) & (s.method == "HBR")]["f1_mean"].iloc[0]
        ax.axhline(hb, color=style(i)["color"], lw=0.8, ls=":", alpha=0.9)
    ax.set_xlabel("Labelled target windows per class (k)")
    ax.set_title(setting)
    ax.set_xticks([0] + list(ks))
axes[0].set_ylabel("Macro-F1 on unseen load")
axes[0].legend(loc="lower right", fontsize=6.5)
axes[1].text(0.98, 0.04, "dotted: HBR (M = 16 healthy windows)", transform=axes[1].transAxes, ha="right",
             fontsize=6, color="#52514e")
save(fig, run.fig_dir, run.fig_name("fewshot_curve"))
print(summ[summ.k.isin([0, 5])].to_string(index=False))
run.done()
