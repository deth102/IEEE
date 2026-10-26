"""Search space and vectorised workers of the hyper-parameter search (E26).

Every configuration of the method is scored on the same tasks: the source models of one (setting, source,
target, fold), the same support draws, the same validation and test windows. The softmax update is run for
all (lr, anchor) pairs at once and read out at every epoch checkpoint, with the arithmetic of
``SoftmaxRegression.finetune``.
"""
from __future__ import annotations

import itertools

import numpy as np

from .adaptation import pick_fewshot
from .data import class_weights
from .models import make_model

# --------------------------------------------------------------------------- search space
BASE_SM = [dict(l2=v) for v in (0.0, 1e-4, 1e-3, 1e-2)]
# equal flash budget (about 430 nodes, as the deployed 10 trees x 22 leaves): trees x depth x leaves
BASE_RF = [dict(n_estimators=t, max_depth=d, max_leaf_nodes=l)
           for t, d, l in ((5, 10, 44), (7, 10, 32), (10, 7, 22), (10, 10, 22), (14, 8, 16), (20, 8, 11))]
LRS = (0.03, 0.1, 0.3, 1.0)
ANCHORS = (1e-4, 1e-3, 1e-2, 1e-1)
EPOCHS = (100, 200, 400, 800)
RF_W = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
QS = (1, 5)
SEEDS = (0, 1, 2)
HP_PAIRS = list(itertools.product(LRS, ANCHORS))          # lr-major, as in finetune_grid

SM_LAB = [f"l2={b['l2']:g}" for b in BASE_SM]
RF_LAB = [f"{b['n_estimators']}x{b['max_depth']}x{b['max_leaf_nodes']}" for b in BASE_RF]


def sm_adapt(lr, anchor, epochs):
    return f"lr={lr:g},a={anchor:g},ep={epochs}"


def rf_adapt(w):
    return f"w={w:g}"


DEFAULT = {"softmax": ("l2=0.0001", sm_adapt(0.1, 1e-3, 200)), "rf": ("10x7x22", rf_adapt(50.0))}   # values before E26
ROW_COLS = ["setting", "source", "target", "fold", "task", "q", "seed", "head", "base", "adapt",
            "support_sev", "test_sev", "val_f1", "test_f1"]


# --------------------------------------------------------------------------- vectorised pieces
def macro_f1(y, preds, n_classes):
    """Macro-F1 of every row of ``preds`` (labels 0..C-1, zero_division=0, as fdedge.metrics.scores)."""
    preds = np.atleast_2d(preds)
    H, C = preds.shape[0], n_classes
    idx = (np.arange(H)[:, None] * C * C + np.asarray(y)[None, :] * C + preds).ravel()
    cm = np.bincount(idx, minlength=H * C * C).reshape(H, C, C)          # [h, true, predicted]
    tp = np.diagonal(cm, axis1=1, axis2=2)
    den = cm.sum(axis=1) + cm.sum(axis=2)                                 # 2 tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean(axis=1)


def finetune_grid(model, X, y, evals, lrs=LRS, anchors=ANCHORS, checkpoints=EPOCHS):
    """``SoftmaxRegression.finetune`` for every (lr, anchor) pair at once.

    Returns {epochs: [predictions on each array of ``evals``, shape (H, n)]}, H = len(lrs) * len(anchors).
    """
    lr = np.repeat(np.asarray(lrs, dtype=np.float64), len(anchors))[:, None, None]
    an = np.tile(np.asarray(anchors, dtype=np.float64), len(lrs))[:, None, None]
    Z = model._z(X)
    Y = np.eye(model.n_classes)[y]
    Ze = [model._z(E) for E in evals]
    W0, b0 = model.W[None], model.b[None]
    W, b = np.repeat(W0, len(lr), axis=0), np.repeat(b0, len(lr), axis=0)
    out = {}
    for ep in range(1, max(checkpoints) + 1):
        L = Z @ W.transpose(0, 2, 1) + b[:, None, :]
        L -= L.max(axis=2, keepdims=True)
        P = np.exp(L)
        P /= P.sum(axis=2, keepdims=True)
        G = (P - Y) / len(y)
        W = W - lr * (G.transpose(0, 2, 1) @ Z + an * (W - W0))
        b = b - lr[:, :, 0] * (G.sum(axis=1) + an[:, :, 0] * (b - b0))
        if ep in checkpoints:
            out[ep] = [np.argmax(z @ W.transpose(0, 2, 1) + b[:, None, :], axis=2) for z in Ze]
    return out


def rf_model(cfg, n_classes, seed, base):
    m = make_model("rf", n_classes, cfg, seed=seed, **base)
    m.clf.n_jobs = 1                                   # the search parallelises over tasks
    return m


# --------------------------------------------------------------------------- one task
def run_task(cfg, n_classes, meta, fold, tr, va, pool, Q, sev):
    """Every configuration on one (setting, source, target, fold); returns tuples in ROW_COLS order.

    task "load": support from block 1 of the target (q = 1 and 5), validation = the rest of block 1,
    test = blocks 2..K. task "severity": support from block 1 of severity A, test = blocks 2..K of severity B.
    """
    C = n_classes
    rows = []
    nan = float("nan")
    sm = [make_model("softmax", C, cfg, seed=fold, **b).fit(tr.X, tr.y, va.X, va.y) for b in BASE_SM]
    rf = [rf_model(cfg, C, fold, b).fit(tr.X, tr.y) for b in BASE_RF]

    def add(task, q, s, head, base, adapt, a, b, v, t):
        rows.append((*meta, task, q, s, head, base, adapt, a, b, float(v), float(t)))

    tests = [("load", Q, "", "")] + [("severity", qb, a, b) for a, b, _, qb in sev]
    for task, T, a, b in tests:                                            # no update
        for i, m in enumerate(sm):
            add(task, 0, -1, "softmax", SM_LAB[i], "none", a, b, nan, macro_f1(T.y, m.predict(T.X), C)[0])
        for i, m in enumerate(rf):
            add(task, 0, -1, "rf", RF_LAB[i], "none", a, b, nan, macro_f1(T.y, m.predict(T.X), C)[0])

    jobs = [("load", q, s, pool, Q, "", "") for q in QS for s in SEEDS]
    jobs += [("severity", 5, s, pa, qb, a, b) for a, b, pa, qb in sev for s in SEEDS]
    for task, q, s, P, T, a, b in jobs:
        sup, rest = pick_fewshot(P, q, C, seed=1000 * fold + s)
        SX, Sy = P.X[sup], P.y[sup]
        has_val = task == "load"
        evals = [T.X] + ([P.X[rest]] if has_val else [])
        for i, m in enumerate(sm):
            for ep, preds in finetune_grid(m, SX, Sy, evals).items():
                ft = macro_f1(T.y, preds[0], C)
                fv = macro_f1(P.y[rest], preds[1], C) if has_val else np.full(len(ft), nan)
                for h, (lr, an) in enumerate(HP_PAIRS):
                    add(task, q, s, "softmax", SM_LAB[i], sm_adapt(lr, an, ep), a, b, fv[h], ft[h])
        for i, base in enumerate(BASE_RF):
            for w in RF_W:
                m = rf_model(cfg, C, s, base)
                wt = np.concatenate([class_weights(tr.y, C), np.full(len(Sy), w)])
                m.fit(np.vstack([tr.X, SX]), np.concatenate([tr.y, Sy]), sample_weight=wt)
                ft = macro_f1(T.y, m.predict(T.X), C)[0]
                fv = macro_f1(P.y[rest], m.predict(P.X[rest]), C)[0] if has_val else nan
                add(task, q, s, "rf", RF_LAB[i], rf_adapt(w), a, b, fv, ft)
    return rows
