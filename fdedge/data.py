"""Dataset access, feature caching and evaluation protocols.

Protocols
---------
* ``random``  : legacy protocol of the conference paper — overlapping windows
                shuffled and split at random (train/test windows share samples).
* ``blocked`` : each recording is cut into K time-contiguous blocks; block f is
                the test set, block f+1 the validation set, the rest is used for
                training. No window crosses a block boundary -> no leakage.
* cross-load  : train on the blocked train/val part of the source load(s) and
                test on complete recordings of an unseen target load.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .features import cnn_input, extract_features

# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------


@dataclass
class Recording:
    dataset: str
    name: str
    path: Path
    load: str
    label: str
    cls: int
    severity: str
    fs: float
    var: str | None = None          # variable name inside a .mat file (CWRU)
    length: int = 0
    meta: dict = field(default_factory=dict)


_KAIST_RE = re.compile(r"^(?P<load>\d+Nm)_(?P<fault>Normal|BPFI|BPFO|Misalign|Unbalance)(?:_(?P<sev>[^.]+))?\.csv$")


def list_kaist(cfg: Config) -> list[Recording]:
    classes = list(cfg.kaist["classes"])
    recs = []
    for p in sorted(cfg.path("kaist_dir").glob("*.csv")):
        m = _KAIST_RE.match(p.name)
        if not m:
            continue
        recs.append(Recording("kaist", p.stem, p, m["load"], m["fault"], classes.index(m["fault"]),
                              m["sev"] or "-", float(cfg.kaist["fs"])))
    if not recs:
        raise FileNotFoundError(f"No KAIST csv files in {cfg.path('kaist_dir')}")
    return recs


# CWRU 48 kHz drive-end records. File names in the local archive are not
# consistent, so the record id (variable prefix) is fixed explicitly.
_CWRU_TABLE = {
    # file stem            var           load   class   size
    "Normal_0":        ("X097_DE_time", "0hp", "Normal", "-"),
    "Normal_1":        ("X098_DE_time", "1hp", "Normal", "-"),
    "Normal_2hp":      ("X099_DE_time", "2hp", "Normal", "-"),
    "Normal_3":        ("X100_DE_time", "3hp", "Normal", "-"),
    "IR_0_7":          ("X109_DE_time", "0hp", "IR", "007"),
    "IR_1_7":          ("X110_DE_time", "1hp", "IR", "007"),
    "IR_2hp_7mils":    ("X111_DE_time", "2hp", "IR", "007"),
    "IR_3_7":          ("X112_DE_time", "3hp", "IR", "007"),
    "IR_0_14":         ("X173_DE_time", "0hp", "IR", "014"),
    "IR_1_14":         ("X175_DE_time", "1hp", "IR", "014"),
    "IR_2hp_14mils":   ("X176_DE_time", "2hp", "IR", "014"),
    "IR_3_14":         ("X177_DE_time", "3hp", "IR", "014"),
    "IR_0_21":         ("X213_DE_time", "0hp", "IR", "021"),
    "IR_1_21":         ("X214_DE_time", "1hp", "IR", "021"),
    "IR_2hp_21mils":   ("X215_DE_time", "2hp", "IR", "021"),
    "IR_3_21":         ("X217_DE_time", "3hp", "IR", "021"),
    "Ball_0_7":        ("X122_DE_time", "0hp", "Ball", "007"),
    "Ball_1_7":        ("X123_DE_time", "1hp", "Ball", "007"),
    "Ball_2hp_7mils":  ("X124_DE_time", "2hp", "Ball", "007"),
    "Ball_3_7":        ("X125_DE_time", "3hp", "Ball", "007"),
    "Ball_0_14":       ("X189_DE_time", "0hp", "Ball", "014"),
    "Ball_1_14":       ("X190_DE_time", "1hp", "Ball", "014"),
    "Ball_2hp_14mils": ("X191_DE_time", "2hp", "Ball", "014"),
    "Ball_3_14":       ("X192_DE_time", "3hp", "Ball", "014"),
    "Ball_0_21":       ("X226_DE_time", "0hp", "Ball", "021"),
    "Ball_1_21":       ("X227_DE_time", "1hp", "Ball", "021"),
    "Ball_2hp_21mils": ("X228_DE_time", "2hp", "Ball", "021"),
    "Ball_3_21":       ("X229_DE_time", "3hp", "Ball", "021"),
    "OR_0_7":          ("X135_DE_time", "0hp", "OR", "007"),
    "OR_1_7":          ("X136_DE_time", "1hp", "OR", "007"),
    "OR_2hp_7mils":    ("X137_DE_time", "2hp", "OR", "007"),
    "OR_3_7":          ("X138_DE_time", "3hp", "OR", "007"),
    "OR_0_14":         ("X201_DE_time", "0hp", "OR", "014"),
    "OR_1_14":         ("X202_DE_time", "1hp", "OR", "014"),
    "OR_2hp_14mils":   ("X203_DE_time", "2hp", "OR", "014"),
    "OR_3_14":         ("X204_DE_time", "3hp", "OR", "014"),
    "OR_0_21":         ("X238_DE_time", "0hp", "OR", "021"),
    "OR_1_21":         ("X239_DE_time", "1hp", "OR", "021"),
    "OR_2hp_21mils":   ("X240_DE_time", "2hp", "OR", "021"),
    "OR_3_21":         ("X241_DE_time", "3hp", "OR", "021"),
}


def list_cwru(cfg: Config) -> list[Recording]:
    classes = list(cfg.cwru["classes"])
    root = cfg.path("cwru_dir")
    recs = []
    for stem, (var, load, label, size) in _CWRU_TABLE.items():
        p = root / f"{stem}.mat"
        if not p.exists():
            raise FileNotFoundError(p)
        recs.append(Recording("cwru", stem, p, load, label, classes.index(label), size,
                              float(cfg.cwru["fs"]), var=var))
    return recs


def list_recordings(cfg: Config, dataset: str) -> list[Recording]:
    return {"kaist": list_kaist, "cwru": list_cwru}[dataset](cfg)


def class_names(cfg: Config, dataset: str) -> list[str]:
    return list(cfg[dataset]["classes"])


def loads_of(cfg: Config, dataset: str) -> list[str]:
    return list(cfg[dataset]["loads"])


# ---------------------------------------------------------------------------
# Raw signals (cached as float32 .npy)
# ---------------------------------------------------------------------------


def _read_csv_column(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip().split(",")[0]
    try:
        float(first)
        skip = 0
    except ValueError:
        skip = 1                                   # 4 Nm files carry a header line ("Xa")
    df = pd.read_csv(path, header=None, skiprows=skip, usecols=[0], dtype=np.float64, engine="c")
    return df.iloc[:, 0].to_numpy(dtype=np.float32)


def load_signal(rec: Recording, cfg: Config) -> np.ndarray:
    cache = cfg.path("cache_dir") / "signals" / rec.dataset / f"{rec.name}.npy"
    if cache.exists():
        x = np.load(cache, mmap_mode="r")
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        if rec.dataset == "kaist":
            x = _read_csv_column(rec.path)
        else:
            import scipy.io as sio
            x = np.asarray(sio.loadmat(rec.path)[rec.var], dtype=np.float32).ravel()
        np.save(cache, x)
        x = np.load(cache, mmap_mode="r")
    rec.length = int(x.shape[0])
    return x


def window_view(x: np.ndarray, n: int, starts: np.ndarray) -> np.ndarray:
    idx = starts[:, None] + np.arange(n)[None, :]
    return np.asarray(x[idx], dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature store: features of every window (fixed hop) of every recording
# ---------------------------------------------------------------------------


class FeatureStore:
    """Caches per-recording features (or CNN spectra) computed on a hop grid."""

    def __init__(self, cfg: Config, dataset: str, n: int, hop: int | None = None, kind: str = "feat"):
        self.cfg, self.dataset, self.n = cfg, dataset, int(n)
        self.hop = int(hop or max(1, int(n * cfg.signal["train_hop_ratio"])))
        self.kind = kind
        self.recs = list_recordings(cfg, dataset)
        self.data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._load_or_build()

    @property
    def env(self) -> dict | None:
        """Envelope (bearing-order) settings of this dataset, if enabled by the config."""
        if self.kind != "feat" or not self.cfg.get("use_envelope", False):
            return None
        return self.cfg[self.dataset].get("envelope")

    @property
    def cache_file(self) -> Path:
        sub = f"{self.kind}_v2"
        if self.env:
            import hashlib
            import json
            sub = f"{self.kind}_env_" + hashlib.md5(json.dumps(self.env, sort_keys=True).encode()).hexdigest()[:8]
        return self.cfg.path("cache_dir") / sub / f"{self.dataset}_N{self.n}_h{self.hop}.npz"

    def _load_or_build(self):
        f = self.cache_file
        if f.exists():
            z = np.load(f)
            for r in self.recs:
                self.data[r.name] = (z[f"{r.name}__X"], z[f"{r.name}__S"])
                r.length = int(z[f"{r.name}__L"])
            return
        f.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for r in self.recs:
            x = load_signal(r, self.cfg)
            starts = np.arange(0, r.length - self.n + 1, self.hop, dtype=np.int64)
            X = self.compute(window_view(x, self.n, starts), r.fs)
            self.data[r.name] = (X, starts)
            arrays.update({f"{r.name}__X": X, f"{r.name}__S": starts, f"{r.name}__L": np.array(r.length)})
        np.savez(f, **arrays)

    def compute(self, windows: np.ndarray, fs: float) -> np.ndarray:
        if self.kind == "feat":
            return extract_features(windows, fs, band_hz=self.cfg[self.dataset]["shaft_band_hz"], env=self.env)
        return cnn_input(windows)

    # -- selection -------------------------------------------------------
    def select(self, rec: Recording, ranges: list[tuple[int, int]], stride_samples: int):
        """Windows of ``rec`` lying fully inside one of ``ranges``, spaced by ``stride_samples``."""
        X, S = self.data[rec.name]
        keep = []
        for lo, hi in ranges:
            idx = np.nonzero((S >= lo) & (S + self.n <= hi))[0]
            if idx.size:
                step = max(1, stride_samples // self.hop)
                keep.append(idx[::step])
        if not keep:
            return X[:0], S[:0]
        idx = np.concatenate(keep)
        return X[idx], S[idx]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    rec: np.ndarray          # recording index (into store.recs) of every window
    start: np.ndarray        # window start sample

    def __len__(self):
        return len(self.y)


def _concat(parts) -> Split:
    if not parts:
        raise ValueError("empty split")
    return Split(*(np.concatenate([p[i] for p in parts]) for i in range(4)))


def block_bounds(length: int, k: int) -> list[int]:
    return [int(round(i * length / k)) for i in range(k + 1)]


def blocked_ranges(length: int, k: int, fold: int):
    b = block_bounds(length, k)
    test = [(b[fold], b[fold + 1])]
    v = (fold + 1) % k
    val = [(b[v], b[v + 1])]
    train_blocks = [i for i in range(k) if i not in (fold, v)]
    train, cur = [], None                          # merge adjacent train blocks
    for i in train_blocks:
        if cur and cur[1] == b[i]:
            cur = (cur[0], b[i + 1])
        else:
            if cur:
                train.append(cur)
            cur = (b[i], b[i + 1])
    train.append(cur)
    return train, val, test


def _gather(store: FeatureStore, rec_filter, which: str, fold: int, k: int, stride: int) -> Split:
    parts = []
    for ri, r in enumerate(store.recs):
        if not rec_filter(r):
            continue
        if which == "all":
            ranges = [(0, r.length)]
        else:
            tr, va, te = blocked_ranges(r.length, k, fold)
            ranges = {"train": tr, "val": va, "test": te}[which]
        X, S = store.select(r, ranges, stride)
        if len(S):
            parts.append((X, np.full(len(S), r.cls), np.full(len(S), ri), S))
    return _concat(parts)


def blocked_split(store: FeatureStore, fold: int, loads=None, k: int | None = None):
    """Train/val/test of a blocked fold, optionally restricted to ``loads``."""
    cfg = store.cfg
    k = k or cfg.protocol["n_folds"]
    tr_stride = store.hop
    ev_stride = int(store.n * cfg.signal["eval_hop_ratio"])
    f = (lambda r: True) if loads is None else (lambda r: r.load in set(loads))
    return (_gather(store, f, "train", fold, k, tr_stride),
            _gather(store, f, "val", fold, k, ev_stride),
            _gather(store, f, "test", fold, k, ev_stride))


def target_split(store: FeatureStore, loads, which: str = "all", fold: int = 0) -> Split:
    """All (non-overlapping) windows of recordings at ``loads`` — the unseen target domain."""
    ev_stride = int(store.n * store.cfg.signal["eval_hop_ratio"])
    return _gather(store, lambda r: r.load in set(loads), which, fold, store.cfg.protocol["n_folds"], ev_stride)


def random_split(store: FeatureStore, seed: int, ratios=(0.70, 0.15, 0.15)):
    """Legacy protocol: every (overlapping) window shuffled and split at random."""
    all_ = _gather(store, lambda r: True, "all", 0, 1, store.hop)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_))
    n_tr = int(ratios[0] * len(idx))
    n_va = int(ratios[1] * len(idx))
    pick = lambda ii: Split(all_.X[ii], all_.y[ii], all_.rec[ii], all_.start[ii])  # noqa: E731
    return pick(idx[:n_tr]), pick(idx[n_tr:n_tr + n_va]), pick(idx[n_tr + n_va:])


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return w[y]
