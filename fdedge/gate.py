"""KESTREL self-check gate (block S), as on the MCU (firmware/lib/fdcore/src/fd_gate.c).

Per window, with z the standardized features of the softmax head and y_hat its prediction,
    s_t = log(1 + min_{p in prototypes(y_hat)} (z - p)' Pw (z - p) / D) + 1[RF != softmax],
where every class keeps a few prototypes (k-means centres of its training windows) and Pw is the inverse of the
within-class covariance (+ ridge). The gate statistic is the trailing mean of s_t; the threshold tau is the 99th
percentile of that mean on the source validation blocks.
Re-referencing with labelled support windows: each support window joins the prototypes of its class, and the class
means and the within-class scatter are updated exactly, as if every support window were added `weight` times to the
training set. The node stores C k prototypes, the class means and counts, the scatter and Pw (about 4.5 KiB at D = 19).
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.cluster import KMeans


class ProtoGate:
    def __init__(self, sm, X, y, n_classes: int, k: int = 8, ridge: float = 1e-3, seed: int = 0):
        Z = sm._z(X)
        y = np.asarray(y)
        self.C, self.d, self.k, self.ridge = n_classes, Z.shape[1], k, ridge
        self.n_c = np.array([(y == c).sum() for c in range(n_classes)], dtype=np.float64)
        self.mu = np.stack([Z[y == c].mean(0) for c in range(n_classes)])
        R = Z - self.mu[y]
        self.Sw = R.T @ R
        self.n = float(len(Z))
        self.Pw = np.linalg.inv(self.Sw / (self.n - 1) + ridge * np.eye(self.d))
        self.protos = np.vstack([KMeans(k, n_init=3, random_state=seed).fit(Z[y == c]).cluster_centers_
                                 for c in range(n_classes)])
        self.pcls = np.repeat(np.arange(n_classes), k)

    def distance(self, Z: np.ndarray, cls: np.ndarray) -> np.ndarray:
        """min over the prototypes of class cls[i] of (z_i - p)' Pw (z_i - p)."""
        out = np.full(len(Z), np.inf)
        for c in range(self.C):
            ii = np.nonzero(cls == c)[0]
            if not len(ii):
                continue
            for p in self.protos[self.pcls == c]:
                R = Z[ii] - p
                out[ii] = np.minimum(out[ii], np.einsum("ij,jk,ik->i", R, self.Pw, R))
        return out

    def scores(self, rf, sm, X) -> np.ndarray:
        p = sm.predict_proba(X)
        yh = p.argmax(1)
        return np.log1p(self.distance(sm._z(X), yh) / self.d) + (rf.predict(X) != yh)

    def rereferenced(self, sm, Xs, ys, weight: float = 50.0) -> "ProtoGate":
        g = copy.deepcopy(self)
        Zs = sm._z(Xs)
        ys = np.asarray(ys)
        w = float(weight)
        for c in range(self.C):
            zc = Zs[ys == c]
            if not len(zc):
                continue
            n_new = g.n_c[c] + w * len(zc)
            mu_new = (g.n_c[c] * g.mu[c] + w * zc.sum(0)) / n_new
            g.Sw = g.Sw + g.n_c[c] * np.outer(g.mu[c], g.mu[c]) + w * zc.T @ zc - n_new * np.outer(mu_new, mu_new)
            g.mu[c], g.n_c[c] = mu_new, n_new
        g.n = self.n + w * len(Zs)
        g.Pw = np.linalg.inv(g.Sw / (g.n - 1) + self.ridge * np.eye(self.d))
        g.protos = np.vstack([self.protos, Zs])
        g.pcls = np.concatenate([self.pcls, ys])
        return g


def trailing(rec: np.ndarray, s: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last w windows of the same recording (stream order); NaN until w are seen."""
    g = np.full(len(s), np.nan)
    for r in np.unique(rec):
        ii = np.nonzero(rec == r)[0]
        if len(ii) >= w:
            g[ii[w - 1:]] = np.convolve(s[ii], np.ones(w) / w, mode="valid")
    return g


def threshold(*gs) -> float:
    """99th percentile of the trailing statistic (source validation blocks, optionally plus the support pool)."""
    v = np.concatenate([g[np.isfinite(g)] for g in gs])
    return float(np.quantile(v, 0.99))
