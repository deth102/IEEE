"""Reusable evaluation loops (blocked CV, cross-load, adaptation)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .adaptation import align, moments, pick_calibration, pick_fewshot
from .data import FeatureStore, Split, blocked_split, class_weights, loads_of, random_split, target_split
from .metrics import confusion, scores
from .models import MODEL_LABEL, make_model


def fit(kind, tr: Split, va: Split, cfg, n_classes, seed=0, features=None, **kw):
    m = make_model(kind, n_classes, cfg, seed=seed, features=features, **kw)
    return m.fit(tr.X, tr.y, va.X, va.y)


def blocked_cv(store: FeatureStore, kinds, feature_sets, folds, loads=None, n_classes=None, **model_kw):
    """Blocked K-fold CV. Returns (rows DataFrame, {(kind, fs): mean confusion})."""
    C = n_classes or len(store.cfg[store.dataset]["classes"])
    rows, cms = [], {}
    for f in folds:
        tr, va, te = blocked_split(store, f, loads=loads)
        for fs in feature_sets:
            for k in kinds:
                m = fit(k, tr, va, store.cfg, C, seed=f, features=fs, **model_kw)
                p = m.predict(te.X)
                rows.append({"fold": f, "model": MODEL_LABEL[k], "features": fs, **scores(te.y, p, C)})
                cms[(k, fs)] = cms.get((k, fs), 0) + confusion(te.y, p, C) / len(folds)
    return pd.DataFrame(rows), cms


def random_protocol(store: FeatureStore, kinds, feature_sets, seeds):
    C = len(store.cfg[store.dataset]["classes"])
    rows = []
    for s in seeds:
        tr, va, te = random_split(store, seed=s)
        for fs in feature_sets:
            for k in kinds:
                m = fit(k, tr, va, store.cfg, C, seed=s, features=fs)
                rows.append({"fold": s, "model": MODEL_LABEL[k], "features": fs, **scores(te.y, m.predict(te.X), C)})
    return pd.DataFrame(rows)


def source_sets(loads, multi: bool):
    if not multi:
        return [[L] for L in loads]
    return [[L for L in loads if L != t] for t in loads]


def cross_load(store: FeatureStore, kinds, feature_sets, folds, multi=False):
    """Train on source load(s) (blocked train/val of each fold), test on each target load.

    For a target in the source set the blocked test block is used (within-domain).
    """
    cfg = store.cfg
    C = len(cfg[store.dataset]["classes"])
    loads = loads_of(cfg, store.dataset)
    targets = {L: target_split(store, [L]) for L in loads}
    rows = []
    for src in source_sets(loads, multi):
        for f in folds:
            tr, va, te = blocked_split(store, f, loads=src)
            for fs in feature_sets:
                for k in kinds:
                    m = fit(k, tr, va, cfg, C, seed=f, features=fs)
                    for tgt in loads:
                        T = te if tgt in src else targets[tgt]
                        if tgt in src and len(src) > 1:
                            T = Split(*(a[np.isin(te.rec, [i for i, r in enumerate(store.recs) if r.load == tgt])]
                                        for a in (te.X, te.y, te.rec, te.start)))
                        rows.append({"source": "+".join(src), "target": tgt, "fold": f, "model": MODEL_LABEL[k],
                                     "features": fs, "seen": tgt in src, **scores(T.y, m.predict(T.X), C)})
    return pd.DataFrame(rows)


def adaptation(store: FeatureStore, kinds, folds, calib_m=16, fewshot_k=(1, 2, 5, 10), seeds=(0, 1, 2),
               multi=False, features=None, ft_lr=0.1, ft_epochs=200, ft_anchor=1e-3, rf_target_weight=50.0):
    """Adaptation study on every (source -> unseen target) pair."""
    cfg = store.cfg
    C = len(cfg[store.dataset]["classes"])
    normal = list(cfg[store.dataset]["classes"]).index("Normal")
    loads = loads_of(cfg, store.dataset)
    rows = []
    for src in source_sets(loads, multi):
        for f in folds:
            tr, va, _ = blocked_split(store, f, loads=src)
            models = {k: fit(k, tr, va, cfg, C, seed=f, features=features) for k in kinds}
            H = tr.X[tr.y == normal]
            muS, sdS = moments(H)
            for tgt in [L for L in loads if L not in src]:
                T = target_split(store, [tgt])
                base = {"source": "+".join(src), "target": tgt, "fold": f}
                # unsupervised: HBR (healthy windows only) and DFS reference (all target windows)
                cal, rest = pick_calibration(T, normal, calib_m)
                muT, sdT = moments(T.X[cal])
                muA, sdA = moments(T.X)
                for k, m in models.items():
                    lab = MODEL_LABEL[k]
                    rows.append({**base, "model": lab, "method": "none", "k": 0,
                                 **scores(T.y[rest], m.predict(T.X[rest]), C)})
                    rows.append({**base, "model": lab, "method": "HBR", "k": 0,
                                 **scores(T.y[rest], m.predict(align(T.X[rest], muS, sdS, muT, sdT)), C)})
                    rows.append({**base, "model": lab, "method": "HBR-shift", "k": 0,
                                 **scores(T.y[rest], m.predict(align(T.X[rest], muS, sdS, muT, sdT, scale=False)), C)})
                    rows.append({**base, "model": lab, "method": "DFS (oracle)", "k": 0,
                                 **scores(T.y, m.predict(align(T.X, *moments(tr.X), muA, sdA)), C)})
                # supervised few-shot
                for kk in fewshot_k:
                    for s in seeds:
                        sup, q = pick_fewshot(T, kk, C, seed=1000 * f + s)
                        for k in kinds:
                            if k == "softmax":
                                m = copy.deepcopy(models[k])
                                m.finetune(T.X[sup], T.y[sup], lr=ft_lr, epochs=ft_epochs, l2_anchor=ft_anchor)
                                method = "few-shot (on-device FT)"
                            else:
                                m = make_model(k, C, cfg, seed=s, features=features)
                                Xa = np.vstack([tr.X, T.X[sup]])
                                ya = np.concatenate([tr.y, T.y[sup]])
                                w = np.concatenate([class_weights(tr.y, C), np.full(len(sup), rf_target_weight)])
                                m.fit(Xa, ya, sample_weight=w)
                                method = "few-shot (host retrain)"
                            rows.append({**base, "model": MODEL_LABEL[k], "method": method, "k": kk, "seed": s,
                                         **scores(T.y[q], m.predict(T.X[q]), C)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by, metrics=("f1", "acc")) -> pd.DataFrame:
    g = df.groupby(by, sort=False)
    out = g[list(metrics)].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()
