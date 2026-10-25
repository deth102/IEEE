"""Lightweight classifiers used on the edge node.

* ``RandomForest``  : scikit-learn training, library-free C inference (exported by
                      ``export_c.py``). Trees are flattened into pre-order arrays.
* ``SoftmaxRegression`` : multinomial logistic regression with built-in feature
                      standardisation, trained by full-batch gradient descent (the
                      same algorithm can run on-device for few-shot fine-tuning).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .data import class_weights


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------


class RandomForest:
    name = "RF"

    def __init__(self, n_classes: int, seed: int = 0, n_estimators=10, max_depth=7, max_leaf_nodes=22,
                 class_weight="balanced_subsample", features=None, **_):
        self.n_classes = n_classes
        self.features = None if features is None else list(features)
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          max_leaf_nodes=max_leaf_nodes, class_weight=class_weight,
                                          random_state=seed, n_jobs=-1)

    def _x(self, X):
        return X if self.features is None else X[:, self.features]

    def fit(self, X, y, Xv=None, yv=None, sample_weight=None):
        self.clf.fit(self._x(X), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        p = np.zeros((len(X), self.n_classes))
        p[:, self.clf.classes_] = self.clf.predict_proba(self._x(X))
        return p

    def predict(self, X):
        # Majority vote over hard tree decisions, exactly as on the MCU.
        votes = np.zeros((len(X), self.n_classes), dtype=np.int32)
        Xs = self._x(X).astype(np.float32)
        for est in self.clf.estimators_:
            leaf_cls = np.argmax(est.tree_.value[:, 0, :], axis=1)
            cls = self.clf.classes_[leaf_cls[est.apply(Xs)]]
            np.add.at(votes, (np.arange(len(X)), cls), 1)
        return np.argmax(votes, axis=1)

    # -- structure for export / resource analysis --------------------------
    def flat_trees(self):
        """Pre-order arrays per tree: feature (-1 = leaf), threshold, right child, leaf class.

        In pre-order the left child of node i is always i + 1, so only the right
        child index is stored.
        """
        out = []
        feat_map = self.features
        for est in self.clf.estimators_:
            t = est.tree_
            order, stack = [], [0]
            while stack:
                n = stack.pop()
                order.append(n)
                if t.children_left[n] != -1:
                    stack.append(t.children_right[n])
                    stack.append(t.children_left[n])
            pos = {n: i for i, n in enumerate(order)}
            feat, thr, right, leaf = [], [], [], []
            for n in order:
                if t.children_left[n] == -1:
                    feat.append(-1); thr.append(0.0); right.append(-1)
                    leaf.append(int(self.clf.classes_[np.argmax(t.value[n, 0])]))
                else:
                    f = int(t.feature[n])
                    feat.append(f if feat_map is None else int(feat_map[f]))
                    # sklearn tests float32(x) <= thr64. Use the largest float32 <= thr64
                    # so that the float32 comparison on the MCU gives identical decisions.
                    t32 = np.float32(t.threshold[n])
                    if float(t32) > t.threshold[n]:
                        t32 = np.nextafter(t32, np.float32(-np.inf))
                    thr.append(float(t32))
                    right.append(pos[t.children_right[n]])
                    leaf.append(-1)
            out.append(dict(feature=np.array(feat), threshold=np.array(thr, dtype=np.float32),
                            right=np.array(right), leaf=np.array(leaf)))
        return out

    def n_nodes(self) -> int:
        return int(sum(e.tree_.node_count for e in self.clf.estimators_))

    def mean_path_length(self, X) -> float:
        """Average number of internal nodes visited per tree (comparisons)."""
        Xs = self._x(X).astype(np.float32)
        tot = 0.0
        for est in self.clf.estimators_:
            tot += est.decision_path(Xs).sum(axis=1).mean() - 1.0
        return float(tot)


# ---------------------------------------------------------------------------
# Softmax regression
# ---------------------------------------------------------------------------


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    name = "Softmax"

    def __init__(self, n_classes: int, seed: int = 0, lr=0.5, l2=1e-4, epochs=2000, patience=50,
                 features=None, **_):
        self.n_classes, self.lr, self.l2, self.epochs, self.patience = n_classes, lr, l2, epochs, patience
        self.features = None if features is None else list(features)
        self.W = self.b = self.mu = self.sd = None

    def _x(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X if self.features is None else X[:, self.features]

    def _z(self, X):
        return (self._x(X) - self.mu) / self.sd

    def fit(self, X, y, Xv=None, yv=None, sample_weight=None):
        Xr = self._x(X)
        self.mu = Xr.mean(axis=0)
        self.sd = Xr.std(axis=0) + 1e-8
        Z = (Xr - self.mu) / self.sd
        n, d = Z.shape
        C = self.n_classes
        w = class_weights(y, C) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        w = w / w.sum()
        Y = np.eye(C)[y]
        W = np.zeros((C, d))
        b = np.zeros(C)
        Zv = None if Xv is None else (self._x(Xv) - self.mu) / self.sd
        best, best_state, wait = np.inf, (W.copy(), b.copy()), 0
        for _ in range(self.epochs):
            P = _softmax(Z @ W.T + b)
            G = (P - Y) * w[:, None]
            W -= self.lr * (G.T @ Z + 2 * self.l2 * W)
            b -= self.lr * G.sum(axis=0)
            if Zv is not None:
                Pv = _softmax(Zv @ W.T + b)
                loss = -np.mean(np.log(Pv[np.arange(len(yv)), yv] + 1e-12))
                if loss < best - 1e-7:
                    best, best_state, wait = loss, (W.copy(), b.copy()), 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if Zv is not None:
            W, b = best_state
        self.W, self.b = W, b
        return self

    def predict_proba(self, X):
        return _softmax(self._z(X) @ self.W.T + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def finetune(self, X, y, lr=0.05, epochs=100, l2_anchor=1e-2):
        """Few-shot on-device style update: SGD on (W, b) anchored to the source weights."""
        W0, b0 = self.W.copy(), self.b.copy()
        Z = self._z(X)
        Y = np.eye(self.n_classes)[y]
        W, b = self.W.copy(), self.b.copy()
        for _ in range(epochs):
            P = _softmax(Z @ W.T + b)
            G = (P - Y) / len(y)
            W -= lr * (G.T @ Z + l2_anchor * (W - W0))
            b -= lr * (G.sum(axis=0) + l2_anchor * (b - b0))
        self.W, self.b = W, b
        return self


# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    kind: str                 # "rf" | "softmax"
    params: dict


def resolve_features(features, cfg):
    from .features import FEATURE_SETS
    if features is None:
        features = cfg.get("features", "F19")
    return list(FEATURE_SETS[features]) if isinstance(features, str) else list(features)


def make_model(kind: str, n_classes: int, cfg, seed: int = 0, **override):
    override["features"] = resolve_features(override.get("features"), cfg)
    if kind == "rf":
        p = {**cfg["rf"], **override}
        return RandomForest(n_classes, seed=seed, **p)
    if kind == "softmax":
        p = {**cfg["softmax"], **override}
        return SoftmaxRegression(n_classes, seed=seed, **p)
    raise ValueError(kind)


MODEL_KINDS = ("rf", "softmax")
MODEL_LABEL = {"rf": "RF", "softmax": "Softmax"}
