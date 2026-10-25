"""E09 — 1-D CNN baselines under the same protocols, with MCU cost.

CNN input: log-magnitude spectrum of 2048-sample windows (earlier CNN code).
For a like-for-like decision time with the N = 8192 feature pipeline, CNN
predictions of the 4 consecutive windows covering 8192 samples are majority-voted.
Reported: blocked macro-F1 (per window and per 0.32-s decision), single-source
cross-load F1, parameters, MACs, peak activation RAM and estimated latency.
"""
import numpy as np
import pandas as pd

from fdedge.cnn import CNN_LABEL, cnn_cost, cnn_ops, predict_cnn, train_cnn
from fdedge.data import FeatureStore, blocked_split, loads_of, target_split
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.metrics import majority_vote, scores
from fdedge.resources import (estimate, load_platforms, ops_fft, ops_magnitude, ops_rf, ops_softmax, ops_window,
                              pipeline_ops, _c)

run = Run("e09_cnn", __doc__, extra_args=[
    (("--cnn-window",), {"type": int, "default": None, "help": "CNN input window (default: config)"}),
    (("--cnn-folds",), {"type": int, "default": None, "help": "blocked folds for the CNNs (default: config)"})])
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
cc = dict(cfg["cnn"])
if run.args.cnn_window or run.args.cnn_folds:          # variant: own output files, e.g. e09_cnn_w8192_f5
    cc["window"] = run.args.cnn_window or cc["window"]
    cc["folds"] = run.args.cnn_folds or cc["folds"]
    run.tag = f"{run.tag}_w{cc['window']}_f{cc['folds']}"
Nc = cc["window"]
N = cfg.signal["window"]
per_decision = N // Nc
fs_hz = float(cfg[ds]["fs"])
st = FeatureStore(cfg, ds, Nc, kind="cnn")
folds = run.folds[: (1 if run.quick else cc["folds"])]
models = cc["models"] if not run.quick else ["mobilelite", "edgelite"]
loads = loads_of(cfg, ds)
ep = 5 if run.quick else cc["epochs"]

if run.plot_only:
    summ = run.load_table("accuracy")
else:
    rows = []
    for f in folds:
        tr, va, te = blocked_split(st, f)
        for name in models:
            m = train_cnn(name, C, tr.X, tr.y, va.X, va.y, epochs=ep, batch=cc["batch_size"], lr=cc["lr"],
                          patience=cc["patience"], seed=f)
            p = predict_cnn(m, te.X)
            pv, idx = majority_vote(p, te.rec, te.start, per_decision, C)
            rows.append({"fold": f, "model": CNN_LABEL[name], "setting": "blocked (per window)", **scores(te.y, p, C)})
            rows.append({"fold": f, "model": CNN_LABEL[name], "setting": f"blocked (vote {per_decision} = {N} samples)",
                         **scores(te.y[idx], pv, C)})
            print(f"fold {f} {name}: F1 {rows[-2]['f1']:.3f} / voted {rows[-1]['f1']:.3f}")
        for src in loads:
            tr, va, _ = blocked_split(st, f, loads=[src])
            for name in models:
                m = train_cnn(name, C, tr.X, tr.y, va.X, va.y, epochs=ep, batch=cc["batch_size"], lr=cc["lr"],
                              patience=cc["patience"], seed=f)
                for tgt in [t for t in loads if t != src]:
                    T = target_split(st, [tgt])
                    rows.append({"fold": f, "model": CNN_LABEL[name], "setting": "single-source, unseen load",
                                 "source": src, "target": tgt, **scores(T.y, predict_cnn(m, T.X), C)})
    df = pd.DataFrame(rows)
    run.table(df, "raw", latex=False)
    summ = summarize(df, ["model", "setting"])
    run.table(summ, "accuracy")

# ---- cost per 0.32-s decision on each platform ------------------------------
plats = load_platforms()
cost_rows = []
pre = ops_window(Nc, "opt") + ops_fft(Nc, "opt") + ops_magnitude(Nc, "opt") + _c(trans=Nc // 2, flop=2 * Nc // 2)
for name in cfg["cnn"]["models"]:
    c = cnn_cost(name, C, Nc // 2)
    stages = {"preproc": _c(**{k: v * per_decision for k, v in pre.items()}),
              "cnn": _c(**{k: v * per_decision for k, v in cnn_ops(c["macs"]).items()})}
    for pname, p in plats.items():
        e = estimate(stages, p)
        cost_rows.append({"model": CNN_LABEL[name], "platform": pname, "params": c["params"],
                          "macs_per_decision": c["macs"] * per_decision,
                          "flash_int8_bytes": c["params"], "flash_f32_bytes": 4 * c["params"],
                          "act_ram_f32_bytes": 4 * c["peak_act"] + 4 * Nc,
                          "latency_ms": e["latency_ms"], "energy_mj": e["energy_mj"]})
# feature pipeline for reference (same decision time)
for label, cl in [("RF (F19)", ops_rf(10, 40.0, C)), ("Softmax (F19)", ops_softmax(19, C))]:
    stages = pipeline_ops(N, fs_hz, "optimized", cl, cfg[ds]["shaft_band_hz"])
    for pname, p in plats.items():
        e = estimate(stages, p)
        cost_rows.append({"model": label, "platform": pname, "latency_ms": e["latency_ms"], "energy_mj": e["energy_mj"],
                          "act_ram_f32_bytes": 4 * N + 4 * 19})
run.table(pd.DataFrame(cost_rows), "cost")

# ---- figure: accuracy vs estimated ESP32 latency per 0.32-s decision ----------
import matplotlib.pyplot as plt  # noqa: E402
from fdedge.plotting import COL_W, save, style  # noqa: E402

cost_df = pd.DataFrame(cost_rows) if not run.plot_only else run.load_table("cost")
e01 = pd.read_csv(run.tab_dir / "e01_protocol_summary.csv")
pts = []
for m in ["RF", "Softmax"]:
    r = e01[(e01.protocol.str.startswith("blocked")) & (e01.model == m) & (e01.features == "F19")].iloc[0]
    c = cost_df[(cost_df.model == f"{m} (F19)") & (cost_df.platform == "esp32")].iloc[0]
    pts.append((f"{m} (F19)", c.latency_ms, r.f1_mean, r.f1_std))
for name in cfg["cnn"]["models"]:
    lab = CNN_LABEL[name]
    r = summ[(summ.model == lab) & summ.setting.str.startswith("blocked (vote")]
    c = cost_df[(cost_df.model == lab) & (cost_df.platform == "esp32")]
    if len(r) and len(c):
        pts.append((lab, c.iloc[0].latency_ms, r.iloc[0].f1_mean, r.iloc[0].f1_std))
fig, ax = plt.subplots(figsize=(COL_W, 2.2))
for i, (lab, x, y, e) in enumerate(pts):
    st_ = style(i)
    ax.errorbar(x, y, yerr=e, fmt=st_["marker"], color=st_["color"], ms=5, capsize=1.5, elinewidth=0.6, label=lab,
                alpha=0.9)
    if lab == "Softmax (F19)":
        continue                                  # coincides with RF (same pipeline cost); labelled jointly
    text = "RF / Softmax (F19)" if lab == "RF (F19)" else lab
    ax.annotate(text, (x, y), xytext=(x * 1.3, y - 0.022), fontsize=6, color="#0b0b0b")
ax.axvline(1e3 * N / fs_hz, color="#898781", lw=0.8, ls=":")
ax.text(1e3 * N / fs_hz * 1.1, ax.get_ylim()[0] + 0.01, "real-time limit (320 ms)", fontsize=6, color="#52514e", rotation=90,
        va="bottom")
ax.set_xscale("log")
ax.set_xlabel("Estimated ESP32 latency per decision (ms, log)")
ax.set_ylabel("Blocked macro-F1")
save(fig, run.fig_dir, run.fig_name("accuracy_vs_cost"))

print(summ.to_string(index=False))
run.done()
