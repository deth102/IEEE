"""E14 — Checks requested in review.

(a) Leave-one-severity-out: the fault classes are trained on one severity and tested on the held-out
    severity (recording-disjoint); Normal (one recording per load) is split by time blocks.
(b) Few-shot adaptation with time-separated support: support drawn from the first time block of each
    target recording, query = blocks 2..5. Compared with support drawn anywhere (as in E05) on the same
    query, and with a model trained on all labelled windows of the first block (target-trained reference).
(c) Simple baselines on the same features: small MLP and linear SVM (blocked CV and unseen load).
(d) Replay subset (KAIST): the deployed fold-0 models on the 135 replay windows vs the full fold-0 test block.
"""
import copy

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fdedge.adaptation import pick_fewshot
from fdedge.data import (FeatureStore, Split, block_bounds, blocked_ranges, blocked_split, class_weights,
                         load_signal, loads_of, target_split, window_view)
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.features import extract_features
from fdedge.metrics import scores
from fdedge.models import make_model, resolve_features

run = Run("e14_review_checks", __doc__)
cfg, ds = run.cfg, run.dataset
classes = list(cfg[ds]["classes"])
C = len(classes)
NORMAL = classes.index("Normal")
N = cfg.signal["window"]
st = FeatureStore(cfg, ds, N)
FSETS = ("F16", "F19") + ((cfg["features"],) if cfg.get("features", "F19") not in ("F16", "F19") else ())
K = cfg.protocol["n_folds"]
EV = int(N * cfg.signal["eval_hop_ratio"])
loads = loads_of(cfg, ds)


def windows(rec_idx, ranges, stride):
    r = st.recs[rec_idx]
    X, S = st.select(r, ranges, stride)
    return X, np.full(len(S), r.cls), np.full(len(S), rec_idx), S


def collect(parts):
    parts = [p for p in parts if len(p[1])]
    return Split(*(np.concatenate([p[i] for p in parts]) for i in range(4)))


def fit_eval(kind, fs_name, tr, va, te, seed):
    m = make_model(kind, C, cfg, seed=seed, features=fs_name).fit(tr.X, tr.y, va.X, va.y)
    return scores(te.y, m.predict(te.X), C)


# --------------------------------------------------------------------------- (a) severity
sev_of = {}
for c in range(C):
    if c != NORMAL:
        sev_of[c] = sorted({r.severity for r in st.recs if r.cls == c})
n_sev = min(len(v) for v in sev_of.values())
rows = []
for h in range(n_sev):                       # index of the held-out severity within each fault class
    for f in run.folds:
        tr_p, va_p, te_p = [], [], []
        for ri, r in enumerate(st.recs):
            tr_r, va_r, te_r = blocked_ranges(r.length, K, f)
            if r.cls == NORMAL:
                tr_p.append(windows(ri, tr_r, st.hop)); va_p.append(windows(ri, va_r, EV)); te_p.append(windows(ri, te_r, EV))
            elif r.severity == sev_of[r.cls][h]:
                te_p.append(windows(ri, [(0, r.length)], EV))
            else:
                tr_p.append(windows(ri, tr_r, st.hop)); va_p.append(windows(ri, va_r, EV))
        tr, va, te = collect(tr_p), collect(va_p), collect(te_p)
        for fs_name in FSETS:
            for kind in ("rf", "softmax"):
                rows.append({"held_out": "low" if h == 0 and n_sev == 2 else ("high" if n_sev == 2 else f"sev{h}"),
                             "fold": f, "features": fs_name, "model": "RF" if kind == "rf" else "Softmax",
                             **fit_eval(kind, fs_name, tr, va, te, f)})
sev = pd.DataFrame(rows)
run.table(sev, "severity_raw", latex=False)
sev_s = summarize(sev, ["held_out", "features", "model"])
run.table(sev_s, "severity")
print(sev_s.to_string(index=False))
all_sev = summarize(sev, ["features", "model"])
run.table(all_sev, "severity_pooled")
print(all_sev.to_string(index=False))

# --------------------------------------------------------------------------- (b) time-separated few-shot
def target_blocks(tgt):
    """Support pool = first time block, query = blocks 2..5 (non-overlapping windows) of every target recording."""
    pool_p, q_p = [], []
    for ri, r in enumerate(st.recs):
        if r.load != tgt:
            continue
        b = block_bounds(r.length, K)
        pool_p.append(windows(ri, [(b[0], b[1])], EV))
        q_p.append(windows(ri, [(b[1], b[K])], EV))
    return collect(pool_p), collect(q_p)


def adapt(model, kind, tr, S_X, S_y, seed):
    if kind == "softmax":
        m = copy.deepcopy(model)
        return m.finetune(S_X, S_y, lr=0.1, epochs=200, l2_anchor=1e-3)
    m = make_model(kind, C, cfg, seed=seed)
    w = np.concatenate([class_weights(tr.y, C), np.full(len(S_y), 50.0)])
    return m.fit(np.vstack([tr.X, S_X]), np.concatenate([tr.y, S_y]), sample_weight=w)


rows = []
ks = cfg.adaptation["fewshot_k"]
for multi in (False, True):
    sources = [[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]
    for src in sources:
        for f in run.folds:
            tr, va, _ = blocked_split(st, f, loads=src)
            models = {k: make_model(k, C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y) for k in ("rf", "softmax")}
            for tgt in [L for L in loads if L not in src]:
                pool, Q = target_blocks(tgt)
                T = target_split(st, [tgt])
                base = {"setting": "LOLO" if multi else "single-source", "source": "+".join(src), "target": tgt, "fold": f}
                for kind, m in models.items():
                    lab = "RF" if kind == "rf" else "Softmax"
                    rows.append({**base, "model": lab, "support": "none", "k": 0, **scores(Q.y, m.predict(Q.X), C)})
                    ref = make_model(kind, C, cfg, seed=f).fit(pool.X, pool.y)
                    rows.append({**base, "model": lab, "support": "target-trained (block 1)", "k": 0,
                                 **scores(Q.y, ref.predict(Q.X), C)})
                for kk in ks:
                    for s in range(3):
                        sup_t, _ = pick_fewshot(pool, kk, C, seed=1000 * f + s)      # time-separated
                        # support drawn anywhere in the target recordings (as in E05); support windows are
                        # removed from the query so both variants are scored on (almost) the same windows
                        sup_any_all, _ = pick_fewshot(T, kk, C, seed=1000 * f + s)
                        anykey = set(zip(T.rec[sup_any_all].tolist(), T.start[sup_any_all].tolist()))
                        q_mask_any = np.array([(a, b) not in anykey for a, b in zip(Q.rec, Q.start)])
                        for kind, m in models.items():
                            lab = "RF" if kind == "rf" else "Softmax"
                            mt = adapt(m, kind, tr, pool.X[sup_t], pool.y[sup_t], s)
                            rows.append({**base, "model": lab, "support": "time-separated", "k": kk, "seed": s,
                                         **scores(Q.y, mt.predict(Q.X), C)})
                            ma = adapt(m, kind, tr, T.X[sup_any_all], T.y[sup_any_all], s)
                            rows.append({**base, "model": lab, "support": "anywhere (E05)", "k": kk, "seed": s,
                                         **scores(Q.y[q_mask_any], ma.predict(Q.X[q_mask_any]), C)})
fs = pd.DataFrame(rows)
run.table(fs, "fewshot_time_raw", latex=False)
fs_s = summarize(fs, ["setting", "model", "support", "k"])
run.table(fs_s, "fewshot_time")
print(fs_s[fs_s.k.isin([0, 1, 5])].to_string(index=False))

# --------------------------------------------------------------------------- (c) MLP and linear SVM
def simple(kind, fs_name, seed):
    idx = resolve_features(fs_name, cfg)
    if kind == "MLP":
        est = MLPClassifier(hidden_layer_sizes=(32,), max_iter=600, early_stopping=True, random_state=seed)
    else:
        est = LinearSVC(C=1.0, class_weight="balanced", max_iter=20000, random_state=seed)
    return idx, make_pipeline(StandardScaler(), est)


rows = []
for kind in ("MLP", "LinearSVM"):
    for fs_name in FSETS:
        for f in run.folds:
            tr, va, te = blocked_split(st, f)
            idx, m = simple(kind, fs_name, f)
            m.fit(tr.X[:, idx], tr.y)
            rows.append({"model": kind, "features": fs_name, "setting": "blocked", "fold": f,
                         **scores(te.y, m.predict(te.X[:, idx]), C)})
            for src in loads:
                trs, vas, _ = blocked_split(st, f, loads=[src])
                idx, m = simple(kind, fs_name, f)
                m.fit(trs.X[:, idx], trs.y)
                for tgt in [L for L in loads if L != src]:
                    T = target_split(st, [tgt])
                    rows.append({"model": kind, "features": fs_name, "setting": "unseen load", "fold": f,
                                 "source": src, "target": tgt, **scores(T.y, m.predict(T.X[:, idx]), C)})
bl = pd.DataFrame(rows)
run.table(bl, "baselines_raw", latex=False)
bl_s = summarize(bl, ["model", "features", "setting"])
run.table(bl_s, "baselines")
print(bl_s.to_string(index=False))

# --------------------------------------------------------------------------- (d) replay subset
if ds == "kaist":
    from tools.hil_models import train_models
    rf, sm = train_models(cfg, st, export_aug=False)["clean"]
    _, _, te = blocked_split(st, 0)
    Xs, ys, first_wins = [], [], []
    for ri, r in enumerate(st.recs):
        lo, hi = blocked_ranges(r.length, K, 0)[2][0]
        x = np.asarray(load_signal(r, cfg), dtype=np.float32)
        starts = lo + N * np.arange(5)
        Xs.append(extract_features(window_view(x, N, starts), r.fs, band_hz=cfg[ds]["shaft_band_hz"]))
        ys.append(np.full(5, r.cls))
    Xr, yr = np.vstack(Xs), np.concatenate(ys)
    rows = []
    for lab, m in (("RF", rf), ("Softmax", sm)):
        pr = m.predict(Xr)
        rows.append({"model": lab, "set": "135 replay windows (start of fold-0 test block)", "windows": len(yr),
                     "errors": int((pr != yr).sum()), **scores(yr, pr, C)})
        pt = m.predict(te.X)
        rows.append({"model": lab, "set": "full fold-0 test block", "windows": len(te.y),
                     "errors": int((pt != te.y).sum()), **scores(te.y, pt, C)})
        wrong = [(classes[a], classes[b]) for a, b in zip(yr[pr != yr], pr[pr != yr])]
        print(lab, "replay errors (true -> predicted):", wrong)
    rep = pd.DataFrame(rows)
    run.table(rep, "replay_subset", latex=False)
    print(rep.to_string(index=False))
run.done()
