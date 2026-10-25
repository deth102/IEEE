"""E21 — the self-check gate against stronger shift detectors (same models, windows, trailing mean and threshold rule).

Scores per window (higher = more novel), each averaged over the trailing window W and thresholded at the 99th
percentile of the source validation blocks, as the gate of E17:
  KESTREL        log(1 + min_p (z - p)'Pw (z - p) / D) + 1[RF != softmax], 8 prototypes p per class (fdedge.gate)
                                                                        memory 8 C D + 2 D^2
  T2 + d         log(1 + z'P z / D) + 1[RF != softmax] (shared covariance; the earlier gate)    D^2
  T2             log(1 + z'P z / D)                                     D^2
  Dis+Ent        (1[RF != softmax] + H(p)/log C) / 2                    0
  1 - max p      maximum-probability baseline (Hendrycks & Gimpel)      0
  Energy         -log sum_c exp(logit_c) (Liu et al., NeurIPS 2020)     0
  kNN            distance to the 5th nearest source training window in z (Sun et al., ICML 2022)   n_train x D
  Class Maha.    min_c (z - mu_c)' P (z - mu_c) (Lee et al., NeurIPS 2018)                      C D + D^2
  PCA-SPE        squared residual of z outside the principal subspace (95 % variance), the Q statistic of MSPC
  IForest        isolation forest on z (Liu et al., ICDM 2008), 100 trees
Scenarios as in E17: unseen load (single-source, LOLO), unseen severity, and (KAIST) the ESP32 DAC-ADC path.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from fdedge.data import FeatureStore, Split, blocked_ranges, blocked_split, loads_of, target_split
from fdedge.experiment import Run
from fdedge.gate import ProtoGate
from fdedge.models import make_model

run = Run("e21_detectors", __doc__)
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
NAMES = ["KESTREL", "T2 + d", "T2", "Dis+Ent", "1 - max p", "Energy", "kNN", "Class Maha.", "PCA-SPE", "IForest"]


class Detectors:
    def __init__(self, rf, sm, Xtr, ytr, seed=0):
        self.rf, self.sm = rf, sm
        Z = sm._z(Xtr)
        self.d = Z.shape[1]
        self.P = np.linalg.inv(np.cov(Z, rowvar=False) + 1e-3 * np.eye(self.d))
        self.mu_c = np.stack([Z[ytr == c].mean(0) for c in range(C)])
        Zc = Z - self.mu_c[ytr]
        self.Pc = np.linalg.inv(np.cov(Zc, rowvar=False) + 1e-3 * np.eye(self.d))
        self.nn = NearestNeighbors(n_neighbors=5).fit(Z)
        ev, V = np.linalg.eigh(np.cov(Z, rowvar=False))
        order = np.argsort(ev)[::-1]
        ev, V = ev[order], V[:, order]
        k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
        self.V = V[:, :k]
        self.iso = IsolationForest(n_estimators=100, random_state=seed).fit(Z)
        self.pg = ProtoGate(sm, Xtr, ytr, C, seed=seed)

    def scores(self, X):
        p = self.sm.predict_proba(X)
        logits = self.sm._z(X) @ self.sm.W.T + self.sm.b
        Z = self.sm._z(X)
        d = (self.rf.predict(X) != p.argmax(1)).astype(float)
        h = -(p * np.log(p + 1e-12)).sum(1) / np.log(C)
        t2 = np.log1p(np.einsum("ij,jk,ik->i", Z, self.P, Z) / self.d)
        m = logits.max(1, keepdims=True)
        energy = -(m[:, 0] + np.log(np.exp(logits - m).sum(1)))
        knn = self.nn.kneighbors(Z)[0][:, -1]
        cm = np.min(np.stack([np.einsum("ij,jk,ik->i", Z - mu, self.Pc, Z - mu) for mu in self.mu_c]), axis=0)
        R = Z - (Z @ self.V) @ self.V.T
        spe = (R ** 2).sum(1)
        iso = -self.iso.score_samples(Z)
        return {"KESTREL": self.pg.scores(self.rf, self.sm, X), "T2 + d": t2 + d, "T2": t2, "Dis+Ent": 0.5 * (d + h), "1 - max p": 1 - p.max(1), "Energy": energy,
                "kNN": knn, "Class Maha.": np.log1p(cm / self.d), "PCA-SPE": np.log1p(spe), "IForest": iso}


def sliding(split, s, w):
    vals = []
    for r in np.unique(split.rec):
        ii = np.nonzero(split.rec == r)[0]
        ii = ii[np.argsort(split.start[ii])]
        if len(ii) >= w:
            vals.append(np.convolve(s[ii], np.ones(w) / w, mode="valid"))
    return np.concatenate(vals) if vals else np.zeros(0)


def evaluate(det, va, te, sh, base, rows, w=W):
    sv, si, ss = det.scores(va.X), det.scores(te.X), det.scores(sh.X)
    for name in NAMES:
        gv, gi, gs = sliding(va, sv[name], w), sliding(te, si[name], w), sliding(sh, ss[name], w)
        if not (len(gv) and len(gi) and len(gs)):
            continue
        tau = np.quantile(gv, 0.99)
        y = np.r_[np.zeros(len(gi)), np.ones(len(gs))]
        rows.append({**base, "score": name, "W": w, "auroc": roc_auc_score(y, np.r_[gi, gs]),
                     "alarm_rate": float(np.mean(gs > tau)), "false_alarm_rate": float(np.mean(gi > tau))})


rows = []
for multi in (False, True):
    sources = [[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]
    for src in sources:
        for f in run.folds:
            tr, va, te = blocked_split(st, f, loads=src)
            rf = make_model("rf", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
            sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
            det = Detectors(rf, sm, tr.X, tr.y, seed=f)
            for tgt in [L for L in loads if L not in src]:
                evaluate(det, va, te, target_split(st, [tgt]),
                         {"scenario": "load", "setting": "LOLO" if multi else "single-source", "fold": f}, rows)
        print(f"  load {'LOLO' if multi else 'single'} {src} done", flush=True)

sev = {c: sorted({r.severity for r in st.recs if r.cls == c}) for c in range(C) if c != NORMAL}
n_sev = min(len(v) for v in sev.values())
for h in range(n_sev):
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
        evaluate(Detectors(rf, sm, tr.X, tr.y, seed=f), va, te, sh, {"scenario": "severity", "setting": "unseen severity",
                                                                      "fold": f}, rows)
print("  severity done", flush=True)

if ds == "kaist":
    from tools.hil_models import _augment, train_models
    models = train_models(cfg, st, export_aug=False)
    tr, va, te = blocked_split(st, 0)
    e19 = cfg.path("results_dir") / "tables" / "e19_onchip_analog_windows.csv"
    if e19.exists():                                   # every fold-0 test window through the board's DAC-ADC loop
        A = pd.read_csv(e19)
        for tag in ("clean", "aug"):
            a = A[(A.model == tag) & (A["pass"] == 1)]
            Fa = a[[f"f{j}" for j in range(19)]].to_numpy(np.float32)
            an = Split(Fa, a.label.to_numpy(), a.rec.to_numpy(), a.win.to_numpy())
            rf, sm = models[tag]
            Xtr, ytr = (_augment(cfg, st, tr) if tag == "aug" else (tr.X, tr.y))
            evaluate(Detectors(rf, sm, Xtr, ytr), va, te, an, {"scenario": "analog", "setting": f"DAC-ADC, {tag} model",
                                                                "fold": 0}, rows, w=4)

df = pd.DataFrame(rows)
run.table(df, "raw", latex=False)
summ = df.groupby(["scenario", "setting", "score"], sort=False)[["auroc", "alarm_rate", "false_alarm_rate"]].mean().reset_index()
run.table(summ, "summary")
pv = summ.pivot_table(index=["scenario", "setting"], columns="score", values="auroc", sort=False)[NAMES]
pd.set_option("display.width", 250)
print((100 * pv).round(1).to_string())
run.done()
