"""E16 — Further checks of the few-shot load adaptation (review round 2).

(a) Matched test set: the unadapted, HBR and DFS rows of Table III scored on the same windows as the
    few-shot rows (blocks 2..5 of every target recording; support / calibration from block 1).
(b) Cross-severity support: at the target load, the fault-class support windows come from the recordings
    of one fault severity (A) and the test windows from the recordings of the other severity (B), so the
    adapted model never sees a fault recording it is tested on. Normal has one recording per load; its
    support is block 1 and its test windows blocks 2..5, as before. Compared on the same test windows with
    no adaptation and with support from block 1 of the tested recordings (the protocol of Table III).
(c) Adaptation hyper-parameters chosen without target data: inner leave-one-load-out over the source loads
    only (train on one non-target load, adapt to another non-target load: support block 1, score blocks
    2..5), grid over step size, anchor strength and epochs (softmax) and support weight (RF). The selected
    values are then applied to the real target load and compared with the fixed values (0.1, 1e-3, 200, 50).
"""
import copy
import itertools

import numpy as np
import pandas as pd

from fdedge.adaptation import align, moments, pick_calibration, pick_fewshot
from fdedge.data import FeatureStore, Split, block_bounds, blocked_split, class_weights, loads_of, target_split
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.metrics import scores
from fdedge.models import make_model

run = Run("e16_adaptation_checks", __doc__)
cfg, ds = run.cfg, run.dataset
classes = list(cfg[ds]["classes"])
C = len(classes)
NORMAL = classes.index("Normal")
N = cfg.signal["window"]
st = FeatureStore(cfg, ds, N)
K = cfg.protocol["n_folds"]
EV = int(N * cfg.signal["eval_hop_ratio"])
loads = loads_of(cfg, ds)
SEEDS = range(3)
FIXED = dict(lr=0.1, l2_anchor=1e-3, epochs=200, rf_w=50.0)


def part(rec_ok, load, first: bool) -> Split:
    """Block 1 (first=True) or blocks 2..K (first=False) of the recordings at ``load`` accepted by ``rec_ok``."""
    ps = []
    for ri, r in enumerate(st.recs):
        if r.load != load or not rec_ok(r):
            continue
        b = block_bounds(r.length, K)
        X, S = st.select(r, [(b[0], b[1])] if first else [(b[1], b[K])], EV)
        if len(S):
            ps.append((X, np.full(len(S), r.cls), np.full(len(S), ri), S))
    return Split(*(np.concatenate([p[i] for p in ps]) for i in range(4)))


ALL = lambda r: True  # noqa: E731


def adapt(model, kind, tr, SX, Sy, seed, hp=FIXED):
    if kind == "softmax":
        return copy.deepcopy(model).finetune(SX, Sy, lr=hp["lr"], epochs=hp["epochs"], l2_anchor=hp["l2_anchor"])
    m = make_model(kind, C, cfg, seed=seed)
    w = np.concatenate([class_weights(tr.y, C), np.full(len(Sy), hp["rf_w"])])
    return m.fit(np.vstack([tr.X, SX]), np.concatenate([tr.y, Sy]), sample_weight=w)


def source_sets(multi):
    return [[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]


LAB = {"rf": "RF", "softmax": "Softmax"}
models_cache = {}


def source_models(src, f):
    key = (tuple(src), f)
    if key not in models_cache:
        tr, va, _ = blocked_split(st, f, loads=src)
        models_cache[key] = (tr, {k: make_model(k, C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y) for k in LAB})
    return models_cache[key]


# --------------------------------------------------------------------------- (a) matched test set
rows = []
for multi in (False, True):
    for src in source_sets(multi):
        for f in run.folds:
            tr, models = source_models(src, f)
            muS, sdS = moments(tr.X[tr.y == NORMAL])
            for tgt in [L for L in loads if L not in src]:
                pool, Q = part(ALL, tgt, True), part(ALL, tgt, False)
                T = target_split(st, [tgt])
                cal, _ = pick_calibration(pool, NORMAL, 16)            # first 16 healthy windows (block 1)
                muT, sdT = moments(pool.X[cal])
                muA, sdA = moments(T.X)
                base = {"setting": "LOLO" if multi else "single-source", "source": "+".join(src), "target": tgt, "fold": f}
                for k, m in models.items():
                    for meth, Xq in (("none", Q.X), ("HBR", align(Q.X, muS, sdS, muT, sdT)),
                                     ("HBR-shift", align(Q.X, muS, sdS, muT, sdT, scale=False)),
                                     ("DFS (oracle)", align(Q.X, *moments(tr.X), muA, sdA))):
                        rows.append({**base, "model": LAB[k], "method": meth, **scores(Q.y, m.predict(Xq), C)})
mt = pd.DataFrame(rows)
run.table(mt, "matched_raw", latex=False)
mt_s = summarize(mt, ["setting", "model", "method"])
run.table(mt_s, "matched")
print(mt_s.to_string(index=False))

# --------------------------------------------------------------------------- (b) cross-severity support
sev = {c: sorted({r.severity for r in st.recs if r.cls == c}) for c in range(C) if c != NORMAL}
n_sev = min(len(v) for v in sev.values())
pairs = [(a, b) for a in range(n_sev) for b in range(n_sev) if a != b]
names = (["low", "high"] if n_sev == 2 else [f"sev{i}" for i in range(n_sev)])
ks = cfg.adaptation["fewshot_k"]
rows = []
for multi in (False, True):
    for src in source_sets(multi):
        for f in run.folds:
            tr, models = source_models(src, f)
            for tgt in [L for L in loads if L not in src]:
                for a, b in pairs:
                    is_b = lambda r, b=b: r.cls == NORMAL or r.severity == sev[r.cls][b]      # noqa: E731
                    is_a = lambda r, a=a: r.cls == NORMAL or r.severity == sev[r.cls][a]      # noqa: E731
                    Q = part(is_b, tgt, False)                                                 # test: severity B
                    pools = {"same recordings (block 1 of B)": part(is_b, tgt, True),
                             "other severity (block 1 of A)": part(is_a, tgt, True)}
                    base = {"setting": "LOLO" if multi else "single-source", "source": "+".join(src), "target": tgt,
                            "fold": f, "support_sev": names[a], "test_sev": names[b]}
                    for k, m in models.items():
                        rows.append({**base, "model": LAB[k], "support": "none", "k": 0, **scores(Q.y, m.predict(Q.X), C)})
                    for kk in ks:
                        for s in SEEDS:
                            for pname, pool in pools.items():
                                sup, _ = pick_fewshot(pool, kk, C, seed=1000 * f + s)
                                for k, m in models.items():
                                    ma = adapt(m, k, tr, pool.X[sup], pool.y[sup], s)
                                    rows.append({**base, "model": LAB[k], "support": pname, "k": kk, "seed": s,
                                                 **scores(Q.y, ma.predict(Q.X), C)})
cs = pd.DataFrame(rows)
run.table(cs, "cross_severity_raw", latex=False)
cs_s = summarize(cs, ["setting", "model", "support", "k"])
run.table(cs_s, "cross_severity")
print(cs_s[cs_s.k.isin([0, 1, 5])].to_string(index=False))
run.table(summarize(cs, ["setting", "model", "support", "k", "support_sev", "test_sev"]), "cross_severity_by_pair", latex=False)

# --------------------------------------------------------------------------- (c) hyper-parameters from source loads only
GRID_SM = [dict(lr=lr, l2_anchor=l2, epochs=ep) for lr, l2, ep in
           itertools.product([0.03, 0.1, 0.3, 1.0], [1e-4, 1e-3, 1e-2, 1e-1], [50, 100, 200, 400])]
GRID_RF = [dict(rf_w=w) for w in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]]
QK = 5


def score_task(src, tgt, kind, hp, folds):
    """Mean macro-F1 of adapting the models of ``src`` to ``tgt`` (support block 1, query blocks 2..K)."""
    pool, Q = part(ALL, tgt, True), part(ALL, tgt, False)
    out = []
    for f in folds:
        tr, models = source_models(src, f)
        for s in SEEDS:
            sup, _ = pick_fewshot(pool, QK, C, seed=1000 * f + s)
            out.append(scores(Q.y, adapt(models[kind], kind, tr, pool.X[sup], pool.y[sup], s, {**FIXED, **hp}).predict(Q.X), C)["f1"])
    return float(np.mean(out))


rows, chosen = [], []
for kind, grid in (("softmax", GRID_SM), ("rf", GRID_RF)):
    for t in loads:
        inner = [(a, b) for a in loads for b in loads if a != b and t not in (a, b)]
        inner_f1 = [np.mean([score_task([a], b, kind, hp, run.folds) for a, b in inner]) for hp in grid]
        best = grid[int(np.argmax(inner_f1))]
        fixed_rank = None
        fixed_hp = {k: FIXED[k] for k in grid[0]}
        if fixed_hp in grid:
            order = np.argsort(inner_f1)[::-1]
            fixed_rank = int(np.nonzero(order == grid.index(fixed_hp))[0][0]) + 1
        chosen.append({"model": LAB[kind], "target": t, "inner_tasks": len(inner), **{f"sel_{k}": v for k, v in best.items()},
                       "inner_f1_selected": float(np.max(inner_f1)),
                       "inner_f1_fixed": float(inner_f1[grid.index(fixed_hp)]) if fixed_hp in grid else np.nan,
                       "fixed_rank": fixed_rank, "grid_size": len(grid)})
        for multi in (False, True):
            for src in [s for s in source_sets(multi) if t not in s]:
                for name, hp in (("fixed", fixed_hp), ("inner-LOLO selected", best)):
                    rows.append({"setting": "LOLO" if multi else "single-source", "model": LAB[kind], "target": t,
                                 "source": "+".join(src), "hp": name, "f1": score_task(src, t, kind, hp, run.folds)})
                # outer sensitivity over the whole grid (reported only as a range)
                vals = [score_task(src, t, kind, hp, run.folds) for hp in grid]
                rows.append({"setting": "LOLO" if multi else "single-source", "model": LAB[kind], "target": t,
                             "source": "+".join(src), "hp": "grid min", "f1": float(np.min(vals))})
                rows.append({"setting": "LOLO" if multi else "single-source", "model": LAB[kind], "target": t,
                             "source": "+".join(src), "hp": "grid max", "f1": float(np.max(vals))})
ch = pd.DataFrame(chosen)
run.table(ch, "hp_selected", latex=False)
print(ch.to_string(index=False))
hs = pd.DataFrame(rows)
run.table(hs, "hp_outer_raw", latex=False)
hs_s = hs.groupby(["setting", "model", "hp"], sort=False)["f1"].agg(["mean", "std"]).reset_index()
run.table(hs_s, "hp_outer")
print(hs_s.to_string(index=False))
run.done()
