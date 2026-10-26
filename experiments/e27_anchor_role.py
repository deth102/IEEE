"""E27 — Role of the L2-SP anchor in long updates (Supplementary Note S8).

Support: q = 5 labelled windows per class from block 1 of the target recordings (three draws), test: blocks 2..5
(as E14 (b)). The softmax head is updated with the step and epochs of the configuration (adaptation.update) and with
a long schedule (step 1.0, 2000 epochs), each without the anchor (lambda = 0) and with the anchor of the
configuration. Reports the macro-F1 and the relative drift ||W - W0|| / ||W0|| of the weights.
"""
import copy

import numpy as np
import pandas as pd

from fdedge.adaptation import pick_fewshot
from fdedge.data import FeatureStore, Split, block_bounds, blocked_split, loads_of
from fdedge.experiment import Run
from fdedge.metrics import scores
from fdedge.models import make_model

run = Run("e27_anchor_role", __doc__)
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
N = cfg.signal["window"]
K = cfg.protocol["n_folds"]
EV = int(N * cfg.signal["eval_hop_ratio"])
loads = loads_of(cfg, ds)
st = FeatureStore(cfg, ds, N)
UPD = cfg.adaptation["update"]
SCHEDULES = {"selected": (UPD["lr"], UPD["epochs"]), "long (step 1.0, 2000 epochs)": (1.0, 2000)}


def target_blocks(tgt):
    pool_p, q_p = [], []
    for ri, r in enumerate(st.recs):
        if r.load != tgt:
            continue
        b = block_bounds(r.length, K)
        for lst, rng in ((pool_p, [(b[0], b[1])]), (q_p, [(b[1], b[K])])):
            X, S = st.select(r, rng, EV)
            lst.append((X, np.full(len(S), r.cls), np.full(len(S), ri), S))
    mk = lambda ps: Split(*(np.concatenate([p[i] for p in ps]) for i in range(4)))  # noqa: E731
    return mk(pool_p), mk(q_p)


rows = []
for multi in (False, True):
    for src in ([[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]):
        for f in run.folds:
            tr, va, _ = blocked_split(st, f, loads=src)
            sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
            for tgt in [L for L in loads if L not in src]:
                pool, Q = target_blocks(tgt)
                for s in range(3):
                    sup, _ = pick_fewshot(pool, 5, C, seed=1000 * f + s)
                    for sched, (lr, ep) in SCHEDULES.items():
                        for lam in (0.0, UPD["l2_anchor"]):
                            m = copy.deepcopy(sm).finetune(pool.X[sup], pool.y[sup], lr=lr, epochs=ep, l2_anchor=lam)
                            rows.append({"setting": "LOLO" if multi else "single-source", "source": "+".join(src),
                                         "target": tgt, "fold": f, "seed": s, "schedule": sched, "anchor": lam,
                                         "drift": float(np.linalg.norm(m.W - sm.W) / np.linalg.norm(sm.W)),
                                         **scores(Q.y, m.predict(Q.X), C)})
df = pd.DataFrame(rows)
run.table(df, "raw", latex=False)
s = df.groupby(["setting", "schedule", "anchor"], sort=False)[["f1", "drift"]].mean().reset_index()
run.table(s, "summary")
print(s.to_string(index=False))
run.done()
