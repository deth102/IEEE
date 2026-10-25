"""Models used by the HIL experiments.

clean : exactly the model exported by e12 (tag kaist_n8192): fold-0 blocked training part, seed 0
aug   : same data + one AWGN copy of every training window at 30, 20 and 10 dB SNR (as in e07);
        exported to firmware/models/kaist_n8192_aug and verified with the host golden test.
"""
from __future__ import annotations

import numpy as np

from fdedge.config import load_config
from fdedge.data import FeatureStore, blocked_split, load_signal, window_view
from fdedge.export_c import export_model, global_scale, to_int16
from fdedge.features import extract_features
from fdedge.models import make_model

AUG_SNR = (30.0, 20.0, 10.0)


def _augment(cfg, st, split, seed=0):
    rng = np.random.default_rng(seed)
    N, fs, band = st.n, float(cfg.kaist["fs"]), cfg.kaist["shaft_band_hz"]
    Xs, ys = [split.X], [split.y]
    for ri in np.unique(split.rec):
        x = np.asarray(load_signal(st.recs[ri], cfg), dtype=np.float32)
        sel = np.nonzero(split.rec == ri)[0]
        p = np.mean(x.astype(np.float64) ** 2)
        for snr in AUG_SNR:
            noisy = x + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), size=x.shape).astype(np.float32)
            Xs.append(extract_features(window_view(noisy, N, split.start[sel]), fs, band_hz=band))
            ys.append(split.y[sel])
    return np.vstack(Xs), np.concatenate(ys)


def train_models(cfg=None, st=None, export_aug: bool = True):
    cfg = cfg or load_config()
    st = st or FeatureStore(cfg, "kaist", cfg.signal["window"])
    C = len(cfg.kaist["classes"])
    tr, va, te = blocked_split(st, 0)
    rf = make_model("rf", C, cfg, seed=0).fit(tr.X, tr.y)
    sm = make_model("softmax", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
    Xa, ya = _augment(cfg, st, tr)
    rfa = make_model("rf", C, cfg, seed=0).fit(Xa, ya)
    sma = make_model("softmax", C, cfg, seed=0).fit(Xa, ya, va.X, va.y)
    if export_aug:
        N = st.n
        scale = global_scale(max(float(np.max(np.abs(load_signal(r, cfg)))) for r in st.recs))
        rng = np.random.default_rng(N)
        idx = np.array([rng.choice(np.nonzero(te.y == c)[0]) for c in range(C)])
        W = np.stack([to_int16(window_view(load_signal(st.recs[te.rec[i]], cfg), N, np.array([te.start[i]]))[0], scale)
                      for i in idx])
        export_model(cfg.path("firmware_models_dir") / f"kaist_n{N}_aug", n=N, fs=float(cfg.kaist["fs"]),
                     band_hz=cfg.kaist["shaft_band_hz"], class_names=cfg.kaist["classes"], rf=rfa, softmax=sma,
                     X_train=Xa, test_windows_i16=W, scale=scale, test_labels=te.y[idx],
                     extra_info={"trained_on": "blocked fold 0 train + AWGN copies at 30/20/10 dB"})
    return {"clean": (rf, sm), "aug": (rfa, sma)}
