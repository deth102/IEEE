"""Analytical resource model of the on-device diagnosis pipeline.

Two pipeline variants are modelled operation-by-operation after the C code:

* ``legacy``    — the graduation-thesis firmware (ArduinoFFT complex FFT on a
                  real signal, separate feature loops with per-bin divisions,
                  float copy of the window, separate magnitude buffer).
* ``optimized`` — ``firmware/lib/fdcore`` (int16 acquisition buffers, in-place
                  real FFT via an N/2 complex FFT, magnitude fused into the split
                  step, three fused feature passes, constant tables in flash).

Memory figures are exact functions of the buffer layout (validated against the
ELF of the thesis firmware in experiment e11). Latency/energy use the per-class
effective cycle costs of ``configs/platforms.yaml``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log2

import numpy as np
import yaml

from .config import CODE_DIR

OP_CLASSES = ("flop", "div", "sqrt", "trans", "mem", "int")


def _c(**kw) -> Counter:
    return Counter({k: float(v) for k, v in kw.items() if v})


# ---------------------------------------------------------------------------
# Operation counts per stage
# ---------------------------------------------------------------------------


def ops_time_features(n: int, variant: str) -> Counter:
    if variant == "legacy":             # int16 -> float copy of the window, then a separate statistics loop
        return _c(flop=6 * n, int=n, mem=3 * n, div=3, sqrt=1)
    # int16 -> float, scale, sum, sum of squares, max, min (fused with the copy)
    return _c(flop=6 * n, int=n, mem=2 * n, div=3, sqrt=1)


def ops_window(n: int, variant: str) -> Counter:
    if variant == "legacy":             # ArduinoFFT recomputes cos() for N/2 weights
        return _c(flop=n + 3 * (n // 2), trans=n // 2, div=n // 2, mem=2 * n)
    return _c(flop=n, mem=2 * n, int=n // 2)


def ops_fft(n: int, variant: str) -> Counter:
    if variant == "legacy":             # complex radix-2 FFT of size N, imag = 0
        m, stages = n, int(log2(n))
        bfly = (m // 2) * stages
        return _c(flop=10 * bfly + 6 * (m - 1), mem=8 * bfly + 3 * m, int=2 * m + bfly,
                  sqrt=2 * stages, div=stages)
    m = n // 2                           # packed real FFT: N/2-point complex FFT
    stages = int(log2(m))
    bfly = (m // 2) * stages
    split = m // 2                       # pairs (k, M-k)
    return _c(flop=10 * bfly + 22 * split, mem=10 * bfly + 3 * m + 10 * split,
              int=2 * m + bfly + 2 * split)


def ops_magnitude(n: int, variant: str) -> Counter:
    if variant == "legacy":             # complexToMagnitude over all N bins + copy of N/2
        return _c(flop=3 * n, sqrt=n, mem=3 * n + n)
    k = n // 2                           # fused into the split step
    return _c(flop=3 * k, sqrt=k, mem=k)


def ops_freq_features(n: int, variant: str) -> Counter:
    k = n // 2
    if variant == "legacy":
        return _c(flop=33 * k, div=5 * k, trans=k, mem=9 * k, int=4 * k, sqrt=4)
    return _c(flop=24 * k, trans=k, mem=3 * k, int=3 * k, div=4, sqrt=4)


def ops_order_features(n: int, fs: float, band_hz) -> Counter:
    width = max(1, int(band_hz[1] * n / fs) - int(band_hz[0] * n / fs) + 1)
    return _c(flop=width + 3, mem=width + 3, div=1)


def ops_envelope_features(n: int, decim: int, n_orders: int = 3, harmonics: int = 2, search_bins: int = 3,
                          sections: int = 2) -> Counter:
    """Bearing-order stage: biquad band-pass (fused into the int16 -> float pass), square and block mean,
    mean removal and Hann window on M = N/decim points, M-point real FFT with magnitude, order search."""
    m = n // decim
    filt = _c(flop=sections * 9 * n + 2 * n, int=n, mem=n)
    prep = _c(flop=4 * m, mem=3 * m, int=m)
    peaks = n_orders * harmonics * search_bins
    feat = _c(flop=m // 2 + peaks + 2 * n_orders, mem=m // 2 + peaks, div=n_orders, trans=n_orders, int=peaks)
    tot = Counter()
    for part in (filt, prep, ops_fft(m, "optimized"), ops_magnitude(m, "optimized"), feat):
        tot.update(part)
    return tot


def ops_rf(n_trees: int, mean_path: float, n_classes: int) -> Counter:
    visits = mean_path + n_trees         # internal comparisons + leaf reads
    return _c(flop=mean_path, mem=3 * visits, int=2 * visits + n_classes)


def ops_softmax(n_features: int, n_classes: int) -> Counter:
    d, c = n_features, n_classes
    return _c(flop=2 * d + 2 * c * d + c, mem=2 * d + c * d + c)


def ops_softmax_finetune(n_features: int, n_classes: int, n_samples: int, epochs: int) -> Counter:
    d, c, s = n_features, n_classes, n_samples
    per_epoch = _c(flop=s * (4 * c * d + 3 * c) + 4 * c * d, trans=s * c, div=s, mem=s * (2 * c * d) + 2 * c * d)
    return Counter({k: v * epochs for k, v in per_epoch.items()})


def pipeline_ops(n: int, fs: float, variant: str, classifier: Counter, band_hz=None) -> dict[str, Counter]:
    stages = {
        "time_feat": ops_time_features(n, variant),
        "window": ops_window(n, variant),
        "fft": ops_fft(n, variant),
        "magnitude": ops_magnitude(n, variant),
        "freq_feat": ops_freq_features(n, variant),
    }
    if band_hz is not None:
        stages["order_feat"] = ops_order_features(n, fs, band_hz)
    stages["classifier"] = classifier
    return stages


def total_ops(stages: dict[str, Counter]) -> Counter:
    tot = Counter()
    for c in stages.values():
        tot.update(c)
    return tot


def flops(counter: Counter) -> float:
    """Arithmetic operations (flop + div + sqrt + trans)."""
    return float(sum(counter.get(k, 0) for k in ("flop", "div", "sqrt", "trans")))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class Memory:
    ram: dict = field(default_factory=dict)
    flash: dict = field(default_factory=dict)

    @property
    def ram_total(self) -> int:
        return int(sum(self.ram.values()))

    @property
    def flash_total(self) -> int:
        return int(sum(self.flash.values()))


def pipeline_memory(n: int, variant: str, n_features: int, model_flash: int,
                    streaming: bool = True, model_ram: int = 0) -> Memory:
    k = n // 2
    if variant == "legacy":
        ram = {"window_f32": 4 * n, "fft_real": 4 * n, "fft_imag": 4 * n, "magnitude": 4 * k,
               "features": 4 * n_features}
        flash = {"model": model_flash}
    else:
        ram = {"acq_int16": (2 if streaming else 1) * 2 * n, "work_f32": 4 * n, "features": 4 * n_features}
        flash = {"model": model_flash, "twiddle_tab": 4 * n, "hamming_tab": 2 * n}
    if model_ram:
        ram["model_ram"] = model_ram
    return Memory(ram, flash)


RF_NODE_BYTES = {"legacy": 20, "compact": 8, "q16": 6}


def rf_flash_bytes(n_nodes: int, n_trees: int, fmt: str = "compact", n_features: int = 19) -> int:
    table = 8 * n_trees                   # pointer + node count per tree
    extra = 8 * n_features if fmt == "q16" else 0   # per-feature scale/offset
    return RF_NODE_BYTES[fmt] * n_nodes + table + extra


def softmax_flash_bytes(n_features: int, n_classes: int, weight_bytes: int = 4) -> int:
    return weight_bytes * n_classes * n_features + 4 * n_classes + 8 * n_features


# ---------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------


def load_platforms(path=None) -> dict:
    with open(path or CODE_DIR / "configs" / "platforms.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["platforms"]


def cycles(counter: Counter, platform: dict) -> float:
    cy = platform["cycles"]
    return float(sum(counter.get(k, 0.0) * cy[k] for k in OP_CLASSES))


def estimate(stages: dict[str, Counter], platform: dict) -> dict:
    per = {s: cycles(c, platform) for s, c in stages.items()}
    tot = sum(per.values())
    t_ms = tot / (platform["f_mhz"] * 1e3)
    return {"cycles": tot, "latency_ms": t_ms, "energy_mj": platform["power_w"] * t_ms,
            "stage_ms": {s: v / (platform["f_mhz"] * 1e3) for s, v in per.items()}}


def calibrate(rows: list[tuple[Counter, float]]) -> dict:
    """Fit effective cycles per op class from measured (op counts, cycles) pairs (NNLS)."""
    from scipy.optimize import nnls
    A = np.array([[r[0].get(k, 0.0) for k in OP_CLASSES] for r in rows])
    b = np.array([r[1] for r in rows])
    used = A.sum(axis=0) > 0
    sol = np.zeros(len(OP_CLASSES))
    sol[used], _ = nnls(A[:, used], b)
    pred = A @ sol
    return {"cycles": dict(zip(OP_CLASSES, sol.tolist())),
            "mape": float(np.mean(np.abs(pred - b) / np.maximum(b, 1)))}
