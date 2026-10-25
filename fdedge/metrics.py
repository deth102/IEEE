"""Evaluation metrics and decision-level majority voting."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def scores(y_true, y_pred, n_classes: int) -> dict:
    labels = list(range(n_classes))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }


def confusion(y_true, y_pred, n_classes: int, normalize: bool = True) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(np.float64)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return cm


def majority_vote(y_pred: np.ndarray, rec: np.ndarray, start: np.ndarray, v: int, n_classes: int):
    """Vote over ``v`` consecutive windows of the same recording.

    Returns (y_voted, idx_of_last_window) for every complete group of ``v`` windows
    (non-overlapping groups, in temporal order).
    """
    out_pred, out_idx = [], []
    for r in np.unique(rec):
        ii = np.nonzero(rec == r)[0]
        ii = ii[np.argsort(start[ii])]
        for g in range(0, len(ii) - v + 1, v):
            grp = ii[g:g + v]
            votes = np.bincount(y_pred[grp], minlength=n_classes)
            out_pred.append(int(np.argmax(votes)))
            out_idx.append(grp[-1])
    return np.asarray(out_pred, dtype=np.int64), np.asarray(out_idx, dtype=np.int64)


def mean_std(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)
