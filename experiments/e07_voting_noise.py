"""E07 — Decision-level majority voting and robustness to additive noise.

Voting: majority over V consecutive windows of the same recording
(decision latency = V x N / fs). Noise: white Gaussian noise added to the raw
test-block signal at a given SNR (w.r.t. the recording's power) before feature
extraction. Models are trained on clean data, and additionally ("-aug") on clean
+ noise-augmented training windows (SNR drawn from {30, 20, 10} dB), which costs
nothing on the device.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fdedge.data import FeatureStore, blocked_ranges, blocked_split, load_signal, window_view
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.features import extract_features
from fdedge.metrics import majority_vote, scores
from fdedge.models import make_model
from fdedge.plotting import COL_W, save, style

run = Run("e07_voting_noise", __doc__)
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
N = cfg.signal["window"]
fs_hz = float(cfg[ds]["fs"])
band = cfg[ds]["shaft_band_hz"]
st = FeatureStore(cfg, ds, N)
K = cfg.protocol["n_folds"]
rng = np.random.default_rng(0)
AUG_SNR = [30.0, 20.0, 10.0]


def noisy_copy(x, snr_db, lo=0, hi=None):
    hi = len(x) if hi is None else hi
    seg = x[lo:hi]
    sigma = np.sqrt(np.mean(seg.astype(np.float64) ** 2) / 10 ** (snr_db / 10))
    out = x.copy()
    out[lo:hi] = seg + rng.normal(0, sigma, size=seg.shape).astype(np.float32)
    return out


def augmented(split):
    """Clean + one noisy copy per AUG_SNR level of every training window."""
    Xs, ys = [split.X], [split.y]
    for ri in np.unique(split.rec):
        rec = st.recs[ri]
        x = np.asarray(load_signal(rec, cfg), dtype=np.float32)
        sel = np.nonzero(split.rec == ri)[0]
        for snr in AUG_SNR:
            Xs.append(extract_features(window_view(noisy_copy(x, snr), N, split.start[sel]), fs_hz, band_hz=band))
            ys.append(split.y[sel])
    return np.vstack(Xs), np.concatenate(ys)

if run.plot_only:
    vs, ns = run.load_table("voting"), run.load_table("noise")
    ns["snr_db"] = ns["snr_db"].map(lambda v: v if v == "clean" else int(float(v)))
else:
    vote_rows, noise_rows = [], []
    for f in run.folds:
        tr, va, te = blocked_split(st, f)
        Xaug, yaug = augmented(tr)
        for fs_name, aug in [("F16", False), ("F19", False), ("F19", True)]:
            for kind in ["rf", "softmax"]:
                m = make_model(kind, C, cfg, seed=f, features=fs_name)
                m = m.fit(Xaug, yaug, va.X, va.y) if aug else m.fit(tr.X, tr.y, va.X, va.y)
                p = m.predict(te.X)
                label = ("RF" if kind == "rf" else "Softmax") + ("-aug" if aug else "")
                for V in cfg["voting"]:
                    pv, idx = majority_vote(p, te.rec, te.start, V, C)
                    vote_rows.append({"fold": f, "model": label, "features": fs_name, "V": V,
                                      "decision_s": V * N / fs_hz, **scores(te.y[idx], pv, C)})
                # noise: recompute test-block features from noisy raw windows
                for snr in cfg["snr_db"]:
                    if snr is None:
                        Xn = te.X
                    else:
                        Xn = np.empty_like(te.X)
                        for ri in np.unique(te.rec):
                            rec = st.recs[ri]
                            x = np.asarray(load_signal(rec, cfg), dtype=np.float32)
                            lo, hi = blocked_ranges(rec.length, K, f)[2][0]
                            noisy = noisy_copy(x, snr, lo, hi)
                            sel = np.nonzero(te.rec == ri)[0]
                            Xn[sel] = extract_features(window_view(noisy, N, te.start[sel]), fs_hz, band_hz=band)
                    noise_rows.append({"fold": f, "model": label, "features": fs_name,
                                       "snr_db": "clean" if snr is None else snr, **scores(te.y, m.predict(Xn), C)})
    vote = pd.DataFrame(vote_rows)
    noise = pd.DataFrame(noise_rows)
    vs = summarize(vote, ["model", "features", "V", "decision_s"])
    ns = summarize(noise, ["model", "features", "snr_db"])
    run.table(vs, "voting")
    run.table(ns, "noise")


fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 2.25))
combos = [(m, f) for m in ["RF", "Softmax"] for f in ["F16", "F19"]] + [("RF-aug", "F19"), ("Softmax-aug", "F19")]
for i, (m, f) in enumerate(combos):
    s = vs[(vs.model == m) & (vs.features == f)]
    axes[0].errorbar(s["decision_s"], s["f1_mean"], yerr=s["f1_std"], capsize=1.2, elinewidth=0.5,
                     label=f"{m}-{f}", **style(i))
    s = ns[(ns.model == m) & (ns.features == f)]
    order = [v for v in cfg["snr_db"] if v is not None][::-1] + ["clean"]
    s = s.set_index("snr_db").reindex(order)
    axes[1].errorbar(range(len(order)), s["f1_mean"], yerr=s["f1_std"], capsize=1.2, elinewidth=0.5, **style(i))
axes[0].set_xlabel("Decision time (s) = V·N/fs")
axes[0].set_ylabel("Macro-F1")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=6,
           fontsize=6, columnspacing=0.8, handlelength=1.6)
axes[1].set_xticks(range(len(order)), [str(o) for o in order])
axes[1].set_xlabel("Test SNR (dB)")
save(fig, run.fig_dir, run.fig_name("voting_noise"))
print(vs[vs.features == "F19"].to_string(index=False))
print(ns[ns.features == "F19"].to_string(index=False))
print(ns.to_string(index=False))
run.done()
