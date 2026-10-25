"""E23 — design of the self-check statistic on the MCU budget (scenarios, models and threshold rule of E17/E21).

Candidates (z: standardized features of the softmax head, d: head disagreement, D = dim z, C classes):
  T2 + d            log(1 + z'P z / D) + d,             P = inv(cov z),               memory D^2,        D^2 MAC
  CM + d            log(1 + min_c (z-mu_c)'Pw (z-mu_c) / D) + d, Pw = inv(within-class cov)  C D + D^2, C D^2
  CMpred + d        same, class of the softmax prediction only                                C D + D^2, D^2
  CM                class-conditional distance alone
  kNN + d           log(1 + d_5(z)^2 / D) + d, 5th nearest source training window             n D
  proto + d         nearest of 16 k-means prototypes per class (log(1 + d^2/D) + d)          16 C D
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from fdedge.data import FeatureStore, Split, blocked_ranges, blocked_split, loads_of, target_split
from fdedge.experiment import Run
from fdedge.models import make_model

run = Run("e23_gate_design", __doc__)
cfg, ds = run.cfg, run.dataset
classes = list(cfg[ds]["classes"])
C = len(classes)
NORMAL = classes.index("Normal")
N = cfg.signal["window"]
st = FeatureStore(cfg, ds, N)
K = cfg.protocol["n_folds"]
EV = int(N * cfg.signal["eval_hop_ratio"])
loads = loads_of(cfg, ds)
W = 10 if ds == "kaist" else 3
NAMES = ["T2 + d", "CM + d", "CMpred + d", "CM", "kNN + d", "proto + d"]


class Gate:
    def __init__(self, rf, sm, Xtr, ytr, seed=0):
        self.rf, self.sm = rf, sm
        Z = sm._z(Xtr)
        self.d = Z.shape[1]
        I = 1e-3 * np.eye(self.d)
        self.P = np.linalg.inv(np.cov(Z, rowvar=False) + I)
        self.mu = np.stack([Z[ytr == c].mean(0) for c in range(C)])
        self.Pw = np.linalg.inv(np.cov(Z - self.mu[ytr], rowvar=False) + I)
        self.nn = NearestNeighbors(n_neighbors=5).fit(Z)
        protos = [KMeans(16, n_init=3, random_state=seed).fit(Z[ytr == c]).cluster_centers_ for c in range(C)]
        self.protos = np.vstack(protos)

    def scores(self, X):
        p = self.sm.predict_proba(X)
        Z = self.sm._z(X)
        d = (self.rf.predict(X) != p.argmax(1)).astype(float)
        D = self.d
        t2 = np.log1p(np.einsum("ij,jk,ik->i", Z, self.P, Z) / D)
        dc = np.stack([np.einsum("ij,jk,ik->i", Z - m, self.Pw, Z - m) for m in self.mu], axis=1)
        cm = np.log1p(dc.min(1) / D)
        cmp = np.log1p(dc[np.arange(len(Z)), p.argmax(1)] / D)
        knn = np.log1p(self.nn.kneighbors(Z)[0][:, -1] ** 2 / D)
        pr = np.log1p((((Z[:, None, :] - self.protos[None]) ** 2).sum(-1)).min(1) / D)
        return {"T2 + d": t2 + d, "CM + d": cm + d, "CMpred + d": cmp + d, "CM": cm, "kNN + d": knn + d, "proto + d": pr + d}


def sliding(split, s, w):
    vals = []
    for r in np.unique(split.rec):
        ii = np.nonzero(split.rec == r)[0]
        ii = ii[np.argsort(split.start[ii])]
        if len(ii) >= w:
            vals.append(np.convolve(s[ii], np.ones(w) / w, mode="valid"))
    return np.concatenate(vals) if vals else np.zeros(0)


def evaluate(g, va, te, sh, base, rows, w=W):
    sv, si, ss = g.scores(va.X), g.scores(te.X), g.scores(sh.X)
    for name in NAMES:
        gv, gi, gs = sliding(va, sv[name], w), sliding(te, si[name], w), sliding(sh, ss[name], w)
        tau = np.quantile(gv, 0.99)
        y = np.r_[np.zeros(len(gi)), np.ones(len(gs))]
        rows.append({**base, "score": name, "auroc": roc_auc_score(y, np.r_[gi, gs]), "alarm_rate": float(np.mean(gs > tau)),
                     "false_alarm_rate": float(np.mean(gi > tau))})


rows = []
for multi in (False, True):
    sources = [[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]
    for src in sources:
        for f in run.folds:
            tr, va, te = blocked_split(st, f, loads=src)
            rf = make_model("rf", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
            sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
            g = Gate(rf, sm, tr.X, tr.y, seed=f)
            for tgt in [L for L in loads if L not in src]:
                evaluate(g, va, te, target_split(st, [tgt]), {"scenario": "load", "setting": "LOLO" if multi else "single-source"}, rows)
sev = {c: sorted({r.severity for r in st.recs if r.cls == c}) for c in range(C) if c != NORMAL}
for h in range(min(len(v) for v in sev.values())):
    for f in run.folds:
        tr_p, va_p, te_p, sh_p = [], [], [], []
        for ri, r in enumerate(st.recs):
            tr_r, va_r, te_r = blocked_ranges(r.length, K, f)
            add = lambda lst, rng, stride, ri=ri, r=r: lst.append(  # noqa: E731
                (lambda X, S: (X, np.full(len(S), r.cls), np.full(len(S), ri), S))(*st.select(r, rng, stride)))
            if r.cls != NORMAL and r.severity == sev[r.cls][h]:
                add(sh_p, [(0, r.length)], EV)
            else:
                add(tr_p, tr_r, st.hop); add(va_p, va_r, EV); add(te_p, te_r, EV)
        mk = lambda ps: Split(*(np.concatenate([p[i] for p in ps if len(p[1])]) for i in range(4)))  # noqa: E731
        tr, va, te, sh = mk(tr_p), mk(va_p), mk(te_p), mk(sh_p)
        rf = make_model("rf", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
        sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
        evaluate(Gate(rf, sm, tr.X, tr.y, seed=f), va, te, sh, {"scenario": "severity", "setting": "unseen severity"}, rows)
df = pd.DataFrame(rows)
run.table(df, "raw", latex=False)
s = df.groupby(["scenario", "setting", "score"], sort=False)[["auroc", "alarm_rate", "false_alarm_rate"]].mean().reset_index()
run.table(s, "summary")
pd.set_option("display.width", 250)
for col in ("auroc", "alarm_rate", "false_alarm_rate"):
    print(col)
    print((100 * s.pivot_table(index=["scenario", "setting"], columns="score", values=col, sort=False)[NAMES]).round(1).to_string())
run.done()
