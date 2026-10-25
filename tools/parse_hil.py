"""Summarise the hardware-in-the-loop logs of firmware/hil_esp32.

usage: python -m tools.parse_hil results/hw/hil_clean.log results/hw/hil_aug.log
Writes results/tables/hil_summary.csv and results/tables/hil_calibration.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from fdedge.config import load_config

COLS = ["mode", "rec", "win", "label", "rf", "sm", "py_rf", "py_sm", "proc_us",
        "c_time", "c_win", "c_fft", "c_freq", "c_order", "c_cls"]
F_CPU = 240e6


def parse(path: Path):
    wins, cal, info = [], [], {}
    info["adapt"], info["feat"], info["raw"] = [], [], {}
    cur_raw = None
    for line in path.read_text(errors="ignore").splitlines():
        p = line.strip().split(",")
        if p[0] == "WIN" and len(p) == 16:
            wins.append([p[1]] + [int(v) for v in p[2:]])
        elif p[0] == "ADAPTWIN" and len(p) == 7:
            info["adapt"].append([int(v) for v in p[1:]])
        elif p[0] == "ADAPT" and len(p) == 4:
            info["adapt_cost"] = (int(p[1]), int(p[2]), float(p[3]))
        elif p[0] == "FEAT":
            info["feat"].append([int(v) for v in p[1:4]] + [float(v) for v in p[4:]])
        elif p[0] == "RAWA":
            cur_raw = int(p[1]); info["raw"][cur_raw] = (int(p[2]), [])
        elif p[0] == "R" and cur_raw is not None:
            info["raw"][cur_raw][1].extend(int(v) for v in p[1:])
        elif p[0] == "CAL" and len(p) == 4:
            cal.append((int(p[1]), float(p[2]), float(p[3])))
        elif p[0] in ("FIT", "RATE", "ORDER"):
            info[p[0]] = p[1:]
        elif p[0] == "SUM" and len(p) == 4:
            info.setdefault("overruns", 0)
            info["overruns"] += int(p[3])
        elif p[0] == "INFO" and p[1].startswith("hil_esp32"):
            info["model"] = line.split("model=")[1].split(",")[0]
    return pd.DataFrame(wins, columns=COLS), pd.DataFrame(cal, columns=["dac", "adc_mean", "adc_std"]), info


def f1(y, p, C):
    return float(f1_score(y, p, labels=list(range(C)), average="macro", zero_division=0))


def rf_host_retrain(cfg, model, feats, C):
    """RF retrained on the host with the analog support windows (win 0) added, weight 50."""
    from fdedge.data import FeatureStore, blocked_split, class_weights
    from fdedge.models import make_model
    from tools.hil_models import _augment
    F = np.array(feats)
    rec, win, lab, X = F[:, 0].astype(int), F[:, 1].astype(int), F[:, 2].astype(int), F[:, 3:].astype(np.float32)
    st = FeatureStore(cfg, "kaist", cfg.signal["window"])
    tr, va, _ = blocked_split(st, 0)
    Xs, ys = (tr.X, tr.y) if model == "clean" else _augment(cfg, st, tr)
    sup, qry = win == 0, win != 0
    base = make_model("rf", C, cfg, seed=0).fit(Xs, ys)
    m = make_model("rf", C, cfg, seed=0)
    w = np.concatenate([class_weights(ys, C), np.full(int(sup.sum()), 50.0)])
    m.fit(np.vstack([Xs, X[sup]]), np.concatenate([ys, lab[sup]]), sample_weight=w)
    return {"rf_acc_before": accuracy_score(lab[qry], base.predict(X[qry])), "rf_f1_before": f1(lab[qry], base.predict(X[qry]), C),
            "rf_acc": accuracy_score(lab[qry], m.predict(X[qry])), "rf_f1": f1(lab[qry], m.predict(X[qry]), C)}


_MODELS = {}


def python_finetune_check(cfg, model, feats, adapt, C):
    """Repeat the on-device softmax update in Python on the logged features and compare the predictions of all windows."""
    import copy
    from tools.hil_models import train_models
    if not _MODELS:
        _MODELS.update(train_models(cfg, export_aug=False))
    F = np.array(feats)
    win, lab, X = F[:, 1].astype(int), F[:, 2].astype(int), F[:, 3:]
    assert (F[:, 0].astype(int) == adapt.rec.to_numpy()).all() and (win == adapt.win.to_numpy()).all()
    sm = copy.deepcopy(_MODELS[model][1])
    assert (sm.predict(X) == adapt.before.to_numpy()).all()          # same head before the update
    sm.finetune(X[win == 0], lab[win == 0], lr=0.1, epochs=200, l2_anchor=1e-3)
    py = sm.predict(X)
    z = np.abs(sm._z(X[win == 0]))
    return {"ft_agree_python": int((py == adapt.after.to_numpy()).sum()), "ft_windows": len(py),
            "sm_f1_python_ft": f1(lab[win != 0], py[win != 0], C), "support_max_abs_z": float(z.max())}


def spectra(cfg, raws):
    """Smoothed power spectra of one window per class: stored signal and the same recording after the DAC-ADC loop.

    Returns (f [Hz], {class index: [psd_stored, psd_loop]}); also used by tools.visio_figs."""
    from fdedge.data import blocked_ranges, list_kaist, load_signal
    from fdedge.features import hamming
    recs = list_kaist(cfg)
    N, fs = cfg.signal["window"], float(cfg.kaist["fs"])
    raw = raws.get("clean") or next(iter(raws.values()))
    f = np.arange(N // 2) * fs / N
    psd = {}
    for c in sorted(raw):
        ri, q = raw[c]
        x = np.asarray(load_signal(recs[ri], cfg), dtype=np.float64)
        lo = blocked_ranges(len(x), cfg.protocol["n_folds"], 0)[2][0][0]
        orig = x[lo + N: lo + 2 * N]
        ana = np.asarray(q[:N], dtype=np.float64) * (np.abs(orig).max() / max(np.abs(q[:N]).max(), 1))
        psd[c] = []
        for sig in (orig, ana):
            Y = np.abs(np.fft.rfft((sig - sig.mean()) * hamming(N)))[: N // 2] ** 2
            psd[c].append(np.convolve(Y, np.ones(64) / 64, mode="same"))     # 200-Hz smoothing of the power
    return f, psd


def plot_spectra(cfg, raws):
    """Magnitude spectra of one window per class: stored signal vs. the same recording after the DAC-ADC loop."""
    import matplotlib.pyplot as plt
    from fdedge.plotting import COL_W, PAGE_W, save, setup, style
    setup()
    classes = cfg.kaist["classes"]
    f, psd = spectra(cfg, raws)
    # all classes (page width) and a two-class version for one column of the manuscript
    for name, sel, size in (("hil_spectra", sorted(psd), (PAGE_W, 1.7)), ("hil_spectra_2", sorted(psd)[:2], (COL_W, 1.6))):
        fig, axes = plt.subplots(1, len(sel), figsize=size, sharey=True)
        for ax, c in zip(np.atleast_1d(axes), sel):
            for i, lab in enumerate(["stored", "DAC-ADC loop"]):
                ax.plot(f / 1e3, 10 * np.log10(psd[c][i] / psd[c][0].max() + 1e-12), lw=0.9, color=style(i)["color"],
                        linestyle=style(i)["linestyle"], label=lab)
            ax.set_ylim(-45, 3)
            ax.set_title(classes[c], fontsize=7)
            ax.set_xlabel("Frequency (kHz)")
        np.atleast_1d(axes)[0].set_ylabel("Smoothed PSD (dB)")
        np.atleast_1d(axes)[0].legend(fontsize=6)
        save(fig, cfg.path("results_dir") / "figures", name)
    # level change of the smoothed spectrum caused by the loop (band edges of the smoothing kernel excluded)
    rows = []
    for c in sorted(psd):
        d = 10 * np.log10(psd[c][1] / psd[c][0])[64:-64]
        o = 10 * np.log10(psd[c][0] / psd[c][0].max())[64:-64]
        rows.append({"class": classes[c], "median_db": float(np.median(d)), "share_above_10db": float((d > 10).mean()),
                     "stored_level_median_db": float(np.median(o)), "stored_level_where_above_10db": float(np.median(o[d > 10]))})
    pd.DataFrame(rows).to_csv(cfg.path("results_dir") / "tables" / "hil_spectrum_shift.csv", index=False)


def main(paths):
    cfg = load_config()
    C = len(cfg.kaist["classes"])
    rows = []
    raws = {}
    cal_out = None
    for path in paths:
        df, cal, info = parse(Path(path))
        model = info.get("model", Path(path).stem)
        for mode, g in df.groupby("mode"):
            r = {"model": model, "phase": "digital replay" if mode == "D" else "analog DAC-ADC loop",
                 "windows": len(g), "overruns": info.get("overruns", 0),
                 "rf_acc": accuracy_score(g.label, g.rf), "rf_f1": f1(g.label, g.rf, C),
                 "sm_acc": accuracy_score(g.label, g.sm), "sm_f1": f1(g.label, g.sm, C),
                 "proc_ms_mean": g.proc_us.mean() / 1e3, "proc_ms_max": g.proc_us.max() / 1e3,
                 "cls_cycles_median": float(np.median(g.c_cls))}
            if mode == "D":
                r["rf_agree_python"] = float((g.rf == g.py_rf).mean())
                r["sm_agree_python"] = float((g.sm == g.py_sm).mean())
                r["py_rf_acc"] = accuracy_score(g.label, g.py_rf)
                r["py_sm_acc"] = accuracy_score(g.label, g.py_sm)
            for c in ["c_time", "c_win", "c_fft", "c_freq", "c_order"]:
                r[f"{c}_ms"] = float(np.median(g[c])) / F_CPU * 1e3
            rows.append(r)
        if info["adapt"]:
            a = pd.DataFrame(info["adapt"], columns=["rec", "win", "label", "support", "before", "after"])
            q = a[a.support == 0]
            n_sup, cyc, ms = info.get("adapt_cost", (0, 0, 0.0))
            r = {"model": model, "phase": "analog + on-device FT (softmax)", "windows": len(q),
                 "support_windows": n_sup, "ft_ms": ms,
                 "sm_acc_before": accuracy_score(q.label, q.before), "sm_f1_before": f1(q.label, q.before, C),
                 "sm_acc": accuracy_score(q.label, q.after), "sm_f1": f1(q.label, q.after, C)}
            if info["feat"]:
                r.update(rf_host_retrain(cfg, model, info["feat"], C))
                r.update(python_finetune_check(cfg, model, info["feat"], a, C))
            rows.append(r)
            info["raw_dump"] = info["raw"]
            raws[model] = info["raw"]
        if "RATE" in info:
            rows[-1]["adc_rate_hz"] = float(info["RATE"][2])
        if "FIT" in info:
            rows[-1]["fit_slope"] = float(info["FIT"][0])
            rows[-1]["fit_intercept"] = float(info["FIT"][1])
        cal_out = cal
    out = pd.DataFrame(rows)
    tab = cfg.path("results_dir") / "tables"
    out.to_csv(tab / "hil_summary.csv", index=False)
    if cal_out is not None:
        cal_out.to_csv(tab / "hil_calibration.csv", index=False)
    if raws:
        plot_spectra(cfg, raws)
    pd.set_option("display.width", 250)
    print(out.T.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
