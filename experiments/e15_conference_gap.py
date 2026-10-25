"""E15 — Why the 99.52 % accuracy of the conference paper [11] is not reproduced.

[11] reports test accuracy 99.52 % (RF, 10 trees, depth 7, 22 leaves) after a random 70/15/15
split of "non-overlapping 8192-sample windows ... 26,112,000 labeled samples across all machine
conditions and load levels". The legacy feature file used for training had 4175 rows.

(a) Data accounting: which subset of the KAIST x-channel recordings and which (N, hop) reproduce
    26,112,000 samples and 4175 windows.
(b) Accuracy of the conference pipeline (F16 or the 10 spectral features S10, random stratified
    70/15/15 split, accuracy as in [11]) on candidate subsets / windowings, with scikit-learn RF
    and with the legacy pure-Python RF of the thesis code (legacy/ml_python/randomforest/Model.py).
(c) Normal <-> unbalance confusion of each variant ([11] shows none).
"""
import importlib.util

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fdedge.config import CODE_DIR
from fdedge.data import FeatureStore, _gather
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.features import FEATURE_SETS
from fdedge.metrics import confusion, scores
from fdedge.models import make_model

run = Run("e15_conference_gap", __doc__)
cfg = run.cfg
classes = list(cfg.kaist["classes"])
C = len(classes)
NORMAL, UNB, MIS = classes.index("Normal"), classes.index("Unbalance"), classes.index("Misalign")
loads = list(cfg.kaist["loads"])

# --------------------------------------------------------------------------- (a) accounting
st8 = FeatureStore(cfg, "kaist", 8192, hop=4096)
lengths = pd.DataFrame([{"rec": r.name, "load": r.load, "label": r.label, "severity": r.severity,
                         "samples": r.length, "seconds": r.length / r.fs} for r in st8.recs])
run.table(lengths, "lengths", latex=False)
acc_rows = []
subsets = {"all loads": loads, **{L: [L] for L in loads}}
for name, ls in subsets.items():
    sub = lengths[lengths.load.isin(ls)]
    row = {"subset": name, "recordings": len(sub), "samples": int(sub.samples.sum()),
           "seconds": float(sub.seconds.sum())}
    for n, h in [(8192, 8192), (8192, 4096), (8192, 2048), (16384, 8192), (32768, 16384), (32768, 32768)]:
        row[f"win N={n} hop={h}"] = int(sum((L - n) // h + 1 for L in sub.samples))
    acc_rows.append(row)
acct = pd.DataFrame(acc_rows)
run.table(acct, "accounting", latex=False)
print(acct.T.to_string())
print("26,112,000 samples in [11]; legacy feature file: 4175 rows")

# --------------------------------------------------------------------------- legacy RF
spec = importlib.util.spec_from_file_location("legacy_rf", CODE_DIR / "legacy" / "ml_python" / "randomforest" / "Model.py")
legacy_rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy_rf)


class LegacyRF:
    """Thesis RF (pure Python, no class weights, all features per split, 10 random thresholds)."""

    def __init__(self, n_trees, seed, features):
        self.idx = list(features)
        self.m = legacy_rf.RandomForest(n_trees=n_trees, max_depth=7, max_leaf_nodes=22, random_state=seed)

    def fit(self, X, y, *_):
        self.m.fit(X[:, self.idx].astype(np.float64).tolist(), y.tolist())
        return self

    def predict(self, X):
        return np.asarray(self.m.predict(X[:, self.idx].astype(np.float64)))


# --------------------------------------------------------------------------- (b) conference pipeline
stores = {(8192, 4096): st8, (32768, 16384): FeatureStore(cfg, "kaist", 32768, hop=16384),
          (8192, 2048): FeatureStore(cfg, "kaist", 8192, hop=2048)}


def windows(n, hop, ls):
    """All windows with the given hop (hop = n: non-overlapping) of the recordings at loads ``ls``."""
    base = stores[(n, n // 2)] if (n, n // 2) in stores and hop % (n // 2) == 0 else stores[(n, hop)]
    return _gather(base, lambda r: r.load in set(ls), "all", 0, 1, hop)


variants = [
    # (label, N, hop, loads)
    ("all loads, N=32768, 50% overlap", 32768, 16384, loads),
    ("all loads, N=8192, no overlap", 8192, 8192, loads),
    ("all loads, N=8192, 75% overlap", 8192, 2048, loads),
    *[(f"{L} only, N=8192, no overlap", 8192, 8192, [L]) for L in loads],
    *[(f"{L} only, N=32768, 50% overlap", 32768, 16384, [L]) for L in loads],
]
models = [("RF (sklearn, class-balanced)", "rf"), ("Softmax", "softmax")]
rows = []
for label, n, hop, ls in variants:
    W = windows(n, hop, ls)
    for fs_name in ("F16", "S10"):
        for seed in run.seeds:
            idx = np.arange(len(W.y))
            tr, tmp = train_test_split(idx, test_size=0.30, stratify=W.y, random_state=42 + seed)
            va, te = train_test_split(tmp, test_size=0.50, stratify=W.y[tmp], random_state=42 + seed)
            fits = [(lab, make_model(k, C, cfg, seed=seed, features=fs_name)) for lab, k in models]
            if n == 32768 and len(ls) == len(loads):
                fits += [("RF (legacy code, 10 trees)", LegacyRF(10, 42 + seed, FEATURE_SETS[fs_name])),
                         ("RF (legacy code, 1 tree)", LegacyRF(1, 42 + seed, FEATURE_SETS[fs_name]))]
            for lab, m in fits:
                m.fit(W.X[tr], W.y[tr], W.X[va], W.y[va])
                p = m.predict(W.X[te])
                cm = confusion(W.y[te], p, C)
                rows.append({"variant": label, "windows": len(W.y), "features": fs_name, "model": lab, "seed": seed,
                             **scores(W.y[te], p, C), "normal->unb": cm[NORMAL, UNB], "unb->normal": cm[UNB, NORMAL],
                             "unb->mis": cm[UNB, MIS]})
        print(f"done: {label} {fs_name}")
df = pd.DataFrame(rows)
run.table(df, "raw", latex=False)
summ = summarize(df, ["variant", "features", "model"], ("acc", "f1", "normal->unb", "unb->normal", "unb->mis"))
run.table(summ, "summary")
pd.set_option("display.width", 250)
print(summ.to_string(index=False))
run.done()
