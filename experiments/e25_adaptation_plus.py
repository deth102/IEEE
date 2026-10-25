"""E25 — Class-mean alignment before the few-shot update, selected without target test data.

The support windows of a new load carry labels for every class. Besides retuning the heads, their class means
can map the new load back into the source feature space: per feature j, x' = a_j x + c_j, fitted by least
squares between the target class means (from the support windows only) and the source class means (training
windows). The map costs 2 x D floats and one small regression per feature on the MCU.

Candidates (fixed before any outer evaluation):
  align   none | shift (a = 1) | cma1 (slope shrunk halfway to 1) | cma0 (plain slope);  a clipped to [0.25, 4]
  softmax lr in {0.1, 0.3} x epochs in {200, 400}, anchor 1e-3 (16 configurations, baseline: none, 0.1, 200)
  RF      retrained by the gateway on source + aligned support windows, support weight 50 (4 configurations)
Selection: inner leave-one-load-out over the non-target loads only (train on one non-target load, adapt to
another; support block 1, score blocks 2..5), separately for q = 1 and q = 5 and for each head, as in E16 (c).
Outer evaluation: the selected configuration on the real target load (single-source and LOLO, five folds,
three support draws), next to the fixed baseline; every candidate's outer score is stored as well, so that no
configuration is reported selectively. Cross-severity support (E16 (b)) is rerun with the configuration that
the inner search selects for q = 5.
"""
import copy
import itertools

import numpy as np
import pandas as pd

from fdedge.adaptation import pick_fewshot
from fdedge.data import FeatureStore, Split, block_bounds, blocked_split, class_weights, loads_of
from fdedge.experiment import Run
from fdedge.metrics import scores
from fdedge.models import make_model

run = Run("e25_adaptation_plus", __doc__)
cfg, ds = run.cfg, run.dataset
classes = list(cfg[ds]["classes"])
C = len(classes)
NORMAL = classes.index("Normal")
N = cfg.signal["window"]
st = FeatureStore(cfg, ds, N)
K = cfg.protocol["n_folds"]
EV = int(N * cfg.signal["eval_hop_ratio"])
loads = loads_of(cfg, ds)
SEEDS = range(2 if run.quick else 3)
QS = (1, 5)
ALIGNS = ("none", "shift", "cma1", "cma0")
GRID_SM = [dict(align=a, lr=lr, epochs=ep) for a, lr, ep in itertools.product(ALIGNS, (0.1, 0.3), (200, 400))]
GRID_RF = [dict(align=a) for a in ALIGNS]
BASE_SM, BASE_RF = dict(align="none", lr=0.1, epochs=200), dict(align="none")


def part(rec_ok, load, first):
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
cache = {}


def source(src, f):
    key = (tuple(src), f)
    if key not in cache:
        tr, va, _ = blocked_split(st, f, loads=src)
        mu_c = np.stack([tr.X[tr.y == c].mean(axis=0) for c in range(C)])
        cache[key] = (tr, mu_c, {k: make_model(k, C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y) for k in ("rf", "softmax")})
    return cache[key]


def class_map(SX, Sy, mu_src, mode):
    """Per-feature affine map from the target support class means onto the source class means."""
    D = SX.shape[1]
    if mode == "none":
        return np.ones(D), np.zeros(D)
    cls = np.unique(Sy)
    mt = np.stack([SX[Sy == c].mean(axis=0) for c in cls])
    ms = mu_src[cls]
    xm, ym = mt.mean(axis=0), ms.mean(axis=0)
    if mode == "shift":
        a = np.ones(D)
    else:
        sxx = ((mt - xm) ** 2).sum(axis=0)
        sxy = ((mt - xm) * (ms - ym)).sum(axis=0)
        slope = np.where(sxx > 1e-12 * (1 + xm ** 2), sxy / np.maximum(sxx, 1e-300), 1.0)
        a = (slope + 1.0) / 2.0 if mode == "cma1" else slope
        a = np.clip(a, 0.25, 4.0)
    return a, ym - a * xm


def adapt_predict(kind, tr, mu_src, model, SX, Sy, QX, hp, seed):
    a, c = class_map(SX, Sy, mu_src, hp["align"])
    S2, Q2 = SX * a + c, QX * a + c
    if kind == "softmax":
        m = copy.deepcopy(model).finetune(S2, Sy, lr=hp["lr"], epochs=hp["epochs"], l2_anchor=1e-3)
    else:
        m = make_model("rf", C, cfg, seed=seed)
        w = np.concatenate([class_weights(tr.y, C), np.full(len(Sy), 50.0)])
        m.fit(np.vstack([tr.X, S2]), np.concatenate([tr.y, Sy]), sample_weight=w)
    return m.predict(Q2)


def evaluate(src, tgt, kind, grid, q, folds, pool=None, Q=None):
    """Macro-F1 of every configuration in ``grid``, mean over folds and support draws."""
    pool = part(ALL, tgt, True) if pool is None else pool
    Q = part(ALL, tgt, False) if Q is None else Q
    out = np.zeros(len(grid))
    n = 0
    for f in folds:
        tr, mu_c, models = source(src, f)
        for s in SEEDS:
            sup, _ = pick_fewshot(pool, q, C, seed=1000 * f + s)
            for i, hp in enumerate(grid):
                out[i] += scores(Q.y, adapt_predict(kind, tr, mu_c, models[kind], pool.X[sup], pool.y[sup], Q.X, hp, s), C)["f1"]
            n += 1
    return out / n


def sources(multi):
    return [[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]


# --------------------------------------------------------------------------- inner selection, outer evaluation
sel_rows, rows = [], []
selected = {}
for kind, grid, base in (("softmax", GRID_SM, BASE_SM), ("rf", GRID_RF, BASE_RF)):
    for q in QS:
        for t in loads:
            inner = [(a, b) for a in loads for b in loads if a != b and t not in (a, b)]
            inner_f1 = np.mean([evaluate([a], b, kind, grid, q, run.folds) for a, b in inner], axis=0)
            best = grid[int(np.argmax(inner_f1))]
            selected[(kind, q, t)] = best
            sel_rows.append({"model": kind, "q": q, "target": t, **{f"sel_{k}": v for k, v in best.items()},
                             "inner_f1_selected": float(inner_f1.max()), "inner_f1_baseline": float(inner_f1[grid.index(base)])})
            print(f"  {kind} q={q} target {t}: selected {best}", flush=True)
            for multi in (False, True):
                for src in [s for s in sources(multi) if t not in s]:
                    for f in run.folds:
                        vals = evaluate(src, t, kind, grid, q, [f])
                        b_ = {"setting": "LOLO" if multi else "single-source", "model": kind, "q": q, "source": "+".join(src),
                              "target": t, "fold": f}
                        rows.append({**b_, "config": "baseline (fixed)", "f1": float(vals[grid.index(base)])})
                        rows.append({**b_, "config": "selected (inner-LOLO)", "f1": float(vals[grid.index(best)])})
                        for hp, v in zip(grid, vals):
                            rows.append({**b_, "config": "candidate " + ",".join(f"{k}={v_}" for k, v_ in hp.items()), "f1": float(v)})
sel = pd.DataFrame(sel_rows)
run.table(sel, "selected", latex=False)
print(sel.to_string(index=False))
raw = pd.DataFrame(rows)
run.table(raw, "outer_raw", latex=False)
main = raw[raw.config.isin(["baseline (fixed)", "selected (inner-LOLO)"])]
summ = main.groupby(["setting", "model", "q", "config"], sort=False)["f1"].agg(["mean", "std"]).reset_index()
run.table(summ, "summary")
print(summ.to_string(index=False))
pairs_ = main[main.setting == "single-source"].groupby(["model", "q", "config", "source", "target"])["f1"].mean().reset_index()
run.table(pairs_, "by_pair", latex=False)
cand = raw[raw.config.str.startswith("candidate")].groupby(["setting", "model", "q", "config"])["f1"].mean().reset_index()
run.table(cand, "candidates", latex=False)

# --------------------------------------------------------------------------- cross-severity support with the selection for q = 5
sev = {c: sorted({r.severity for r in st.recs if r.cls == c}) for c in range(C) if c != NORMAL}
n_sev = min(len(v) for v in sev.values())
pairs = [(a, b) for a in range(n_sev) for b in range(n_sev) if a != b]
rows = []
for multi in (False, True):
    for src in sources(multi):
        for tgt in [L for L in loads if L not in src]:
            for a, b in pairs:
                is_b = lambda r, b=b: r.cls == NORMAL or r.severity == sev[r.cls][b]      # noqa: E731
                is_a = lambda r, a=a: r.cls == NORMAL or r.severity == sev[r.cls][a]      # noqa: E731
                Q, pool = part(is_b, tgt, False), part(is_a, tgt, True)
                for kind, grid, base in (("softmax", GRID_SM, BASE_SM), ("rf", GRID_RF, BASE_RF)):
                    best = selected[(kind, 5, tgt)]
                    for f in run.folds:
                        v = evaluate(src, tgt, kind, [base, best], 5, [f], pool=pool, Q=Q)
                        b_ = {"setting": "LOLO" if multi else "single-source", "model": kind, "source": "+".join(src),
                              "target": tgt, "fold": f, "support_sev": a, "test_sev": b}
                        rows.append({**b_, "config": "baseline (fixed)", "f1": float(v[0])})
                        rows.append({**b_, "config": "selected (inner-LOLO)", "f1": float(v[1])})
cs = pd.DataFrame(rows)
run.table(cs, "cross_severity_raw", latex=False)
cs_s = cs.groupby(["setting", "model", "config"], sort=False)["f1"].agg(["mean", "std"]).reset_index()
run.table(cs_s, "cross_severity")
print(cs_s.to_string(index=False))
run.done()
