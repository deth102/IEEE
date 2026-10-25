"""Low-cost adaptation to an unseen operating condition (load).

HBR  (healthy-baseline recalibration, proposed, on-device):
     after a load change the machine is assumed healthy for a short period; the
     node collects M windows, estimates per-feature mean/std of the *healthy*
     state at the new load and maps features back to the source healthy
     statistics:  x' = (x - mu_T,H) / sd_T,H * sd_S,H + mu_S,H
     Cost on the MCU: 2 x D floats of state + Welford update per window.

DFS  (domain feature standardisation, reference): same mapping using the
     statistics of *all* unlabeled target windows (all health states mixed).
     Not realisable on a single machine (it needs every fault present) and is
     reported only as an optimistic upper reference.

Few-shot: k labelled target windows per class; Softmax is fine-tuned by SGD
     (feasible on-device), RF is re-trained on the host with the extra windows.
"""
from __future__ import annotations

import numpy as np


def moments(X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    return X.mean(axis=0), X.std(axis=0) + 1e-12


def align(X, mu_src, sd_src, mu_tgt, sd_tgt, scale: bool = True):
    X = np.asarray(X, dtype=np.float64)
    if scale:
        return ((X - mu_tgt) / sd_tgt * sd_src + mu_src).astype(np.float32)
    return (X - mu_tgt + mu_src).astype(np.float32)


class Welford:
    """Streaming mean/variance — the exact update used in fd_adapt.c."""

    def __init__(self, d: int):
        self.n, self.mean, self.m2 = 0, np.zeros(d), np.zeros(d)

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self):
        return np.sqrt(self.m2 / max(self.n, 1)) + 1e-12


def pick_calibration(split, normal_cls: int, m: int):
    """First ``m`` healthy windows (time order, per recording round-robin) of a target split.

    Returns (calibration indices, remaining indices).
    """
    ii = np.nonzero(split.y == normal_cls)[0]
    ii = ii[np.lexsort((split.start[ii], split.rec[ii]))]
    recs = np.unique(split.rec[ii])
    per_rec = [ii[split.rec[ii] == r] for r in recs]
    cal = []
    j = 0
    while len(cal) < m and any(j < len(p) for p in per_rec):
        for p in per_rec:
            if j < len(p) and len(cal) < m:
                cal.append(p[j])
        j += 1
    cal = np.array(sorted(cal), dtype=np.int64)
    rest = np.setdiff1d(np.arange(len(split.y)), cal)
    return cal, rest


def pick_fewshot(split, k: int, n_classes: int, seed: int):
    """k random labelled windows per class; returns (support idx, query idx)."""
    rng = np.random.default_rng(seed)
    sup = []
    for c in range(n_classes):
        ii = np.nonzero(split.y == c)[0]
        if len(ii):
            sup.extend(rng.choice(ii, size=min(k, len(ii)), replace=False).tolist())
    sup = np.array(sorted(sup), dtype=np.int64)
    return sup, np.setdiff1d(np.arange(len(split.y)), sup)
