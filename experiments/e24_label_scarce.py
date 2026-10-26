"""E24 — Retuning under the labels a plant can supply at a new load.

Right after a load change, a plant can label healthy operation; fault labels arrive only as faults are confirmed.
Support from block 1 of the target recordings, test on blocks 2..5 (as E14 (b)); every method starts from the same
source models. Metrics: macro-F1 over all classes and the fault-detection accuracy (balanced accuracy of normal
versus any fault).
  none            source softmax head
  HBR-shift       the 16 first healthy windows map onto the source healthy mean (label-free baseline)
  healthy only    HBR-shift, then the anchored softmax update on the 16 aligned healthy windows
  partial         anchored update with q = 5 labels for normal and half of the fault classes (three draws)
  all classes     anchored update with q = 5 labels for every class (three draws; as Table IV)
"""
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from fdedge.adaptation import align, moments, pick_calibration, pick_fewshot
from fdedge.data import FeatureStore, Split, block_bounds, blocked_split, loads_of
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.metrics import scores
from fdedge.models import make_model

run = Run("e24_label_scarce", __doc__)
cfg, ds = run.cfg, run.dataset
classes = list(cfg[ds]["classes"])
C = len(classes)
NORMAL = classes.index("Normal")
N = cfg.signal["window"]
st = FeatureStore(cfg, ds, N)
K = cfg.protocol["n_folds"]
EV = int(N * cfg.signal["eval_hop_ratio"])
loads = loads_of(cfg, ds)


def part(load, first):
    ps = []
    for ri, r in enumerate(st.recs):
        if r.load != load:
            continue
        b = block_bounds(r.length, K)
        X, S = st.select(r, [(b[0], b[1])] if first else [(b[1], b[K])], EV)
        ps.append((X, np.full(len(S), r.cls), np.full(len(S), ri), S))
    return Split(*(np.concatenate([p[i] for p in ps]) for i in range(4)))


def upd(sm, X, y):
    return copy.deepcopy(sm).finetune(X, y, **cfg.adaptation["update"])


rows = []
for multi in (False, True):
    sources = [[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]
    for src in sources:
        for f in run.folds:
            tr, va, _ = blocked_split(st, f, loads=src)
            sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
            muS, sdS = moments(tr.X[tr.y == NORMAL])
            for tgt in [L for L in loads if L not in src]:
                pool, Q = part(tgt, True), part(tgt, False)
                base = {"setting": "LOLO" if multi else "single-source", "source": "+".join(src), "target": tgt, "fold": f}

                def rec(meth, m, Xq, s=0):
                    p = m.predict(Xq)
                    rows.append({**base, "method": meth, "seed": s, **scores(Q.y, p, C),
                                 "detection": float(balanced_accuracy_score(Q.y != NORMAL, p != NORMAL))})

                cal, _ = pick_calibration(pool, NORMAL, 16)
                muT, sdT = moments(pool.X[cal])
                Pa = align(pool.X, muS, sdS, muT, sdT, scale=False)
                Qa = align(Q.X, muS, sdS, muT, sdT, scale=False)
                rec("none", sm, Q.X)
                rec("HBR-shift", sm, Qa)
                rec("healthy only", upd(sm, Pa[cal], pool.y[cal]), Qa)
                # the node retunes only when its own healthy windows are misread (fewer than 90 % predicted normal)
                if np.mean(sm.predict(pool.X[cal]) == NORMAL) < 0.9:
                    rec("healthy only, when needed", upd(sm, Pa[cal], pool.y[cal]), Qa)
                else:
                    rec("healthy only, when needed", sm, Q.X)
                faults = [c for c in range(C) if c != NORMAL]
                for s in range(3):
                    sup, _ = pick_fewshot(pool, 5, C, seed=1000 * f + s)
                    rng = np.random.default_rng(1000 * f + s)
                    labeled = {NORMAL, *rng.choice(faults, size=len(faults) // 2, replace=False).tolist()}
                    supp = np.array([i for i in sup if pool.y[i] in labeled])
                    rec("partial", upd(sm, pool.X[supp], pool.y[supp]), Q.X, s)
                    rec("all classes", upd(sm, pool.X[sup], pool.y[sup]), Q.X, s)
        print(f"  {'LOLO' if multi else 'single'} {src} done", flush=True)

df = pd.DataFrame(rows)
run.table(df, "raw", latex=False)
s = df.groupby(["setting", "method"], sort=False)[["f1", "detection"]].agg(["mean", "std"]).reset_index()
s.columns = ["_".join(c).strip("_") for c in s.columns]
run.table(s, "summary")
pd.set_option("display.width", 200)
print(s.round(3).to_string(index=False))
run.done()
