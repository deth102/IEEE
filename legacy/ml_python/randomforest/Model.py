# Model.py
import random
from collections import Counter
import math
import numpy as np

# ======================================================
# Tree Node (C-friendly)
# ======================================================
class TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = None   # class label for leaf


# ======================================================
# Decision Tree (Fast CART - simplified)
# ======================================================
class DecisionTree:
    def __init__(
        self,
        max_depth=10,
        min_samples=2,
        max_leaf_nodes=50,
        max_features=None,
        max_thresholds=10,
        rng=None
    ):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.max_thresholds = max_thresholds

        # RNG riêng (để reproducible & C-friendly)
        self.rng = rng if rng is not None else random

        self.leaf_count = 0
        self.root = None

    # -----------------------------
    # Public API
    # -----------------------------
    def fit(self, X, y):
        self.leaf_count = 0
        self.root = self._build(X, y, depth=0)

    def predict(self, x):
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    # -----------------------------
    # Internal
    # -----------------------------
    def _build(self, X, y, depth):
        node = TreeNode()

        # stopping conditions
        if (
            depth >= self.max_depth or
            len(X) < self.min_samples or
            len(set(y)) == 1 or
            self.leaf_count >= self.max_leaf_nodes
        ):
            node.value = self._majority(y)
            self.leaf_count += 1
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None:
            node.value = self._majority(y)
            self.leaf_count += 1
            return node

        node.feature = feature
        node.threshold = threshold

        Xl, yl, Xr, yr = self._split(X, y, feature, threshold)

        node.left = self._build(Xl, yl, depth + 1)
        node.right = self._build(Xr, yr, depth + 1)
        return node

    def _best_split(self, X, y):
        best_gini = 1.0
        best_f = None
        best_t = None

        n_features = len(X[0])

        # feature subsampling (Random Forest standard)
        if self.max_features is None or self.max_features >= n_features:
            features = range(n_features)
        else:
            features = self.rng.sample(range(n_features), self.max_features)

        for f in features:
            values = list(set(x[f] for x in X))
            if len(values) > self.max_thresholds:
                values = self.rng.sample(values, self.max_thresholds)

            for t in values:
                yl, yr = [], []
                for xi, yi in zip(X, y):
                    if xi[f] <= t:
                        yl.append(yi)
                    else:
                        yr.append(yi)

                if not yl or not yr:
                    continue

                g = self._gini_index(yl, yr)
                if g < best_gini:
                    best_gini = g
                    best_f = f
                    best_t = t

        return best_f, best_t

    def _split(self, X, y, f, t):
        Xl, yl, Xr, yr = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[f] <= t:
                Xl.append(xi)
                yl.append(yi)
            else:
                Xr.append(xi)
                yr.append(yi)
        return Xl, yl, Xr, yr

    def _gini(self, labels):
        total = len(labels)
        counts = Counter(labels)
        g = 1.0
        for c in counts.values():
            p = c / total
            g -= p * p
        return g

    def _gini_index(self, yl, yr):
        total = len(yl) + len(yr)
        return (len(yl) / total) * self._gini(yl) + \
               (len(yr) / total) * self._gini(yr)

    def _majority(self, y):
        return Counter(y).most_common(1)[0][0]


# ======================================================
# Random Forest (Incremental / Reproducible)
# ======================================================
class RandomForest:
    def __init__(
        self,
        n_trees=100,
        max_depth=10,
        min_samples=2,
        max_leaf_nodes=50,
        max_features=None,
        max_thresholds=10,
        random_state=None
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.max_thresholds = max_thresholds

        self.random_state = random_state
        self._rng = random.Random(random_state)

        self.trees = []

    # ===== FIT =====
    def fit(self, X, y, n_new_trees=None):
        if n_new_trees is None:
            self.trees = []
            n_new_trees = self.n_trees

        n = len(X)

        for _ in range(n_new_trees):
            idx = [self._rng.randint(0, n - 1) for _ in range(n)]
            Xb = [X[i] for i in idx]
            yb = [y[i] for i in idx]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                max_leaf_nodes=self.max_leaf_nodes,
                max_features=self.max_features,
                max_thresholds=self.max_thresholds,
                rng=self._rng
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)

    # ===== PREDICT LABEL =====
    def predict_one(self, x):
        votes = [t.predict(x) for t in self.trees]
        return Counter(votes).most_common(1)[0][0]

    def predict(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            return self.predict_one(X.tolist())

        return [self.predict_one(x.tolist()) for x in X]

    # ===== PREDICT PROBA (BẮT BUỘC CHO ROC/AUC) =====
    def predict_proba(self, X):
        """
        X: (n_features,) hoặc (n_samples, n_features)
        return:
            - single sample: (n_classes,)
            - batch: (n_samples, n_classes)
        """
        X = np.asarray(X)

        # lấy tất cả nhãn xuất hiện trong forest
        labels = set()
        for t in self.trees:
            for leaf_label in [t.predict(x) for x in X[:min(len(X), 10)]]:
                labels.add(leaf_label)

        labels = sorted(list(labels))
        n_classes = len(labels)
        label_to_idx = {c: i for i, c in enumerate(labels)}

        def proba_one(x):
            votes = [t.predict(x) for t in self.trees]
            counts = Counter(votes)
            proba = np.zeros(n_classes)
            for cls, cnt in counts.items():
                proba[label_to_idx[cls]] = cnt / len(self.trees)
            return proba

        if X.ndim == 1:
            return proba_one(X.tolist())

        return np.array([proba_one(x.tolist()) for x in X])

