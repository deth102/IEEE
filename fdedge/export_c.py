"""Code generation: trained models -> C headers for firmware/lib/fdcore.

Generates in ``firmware/models/<tag>/``:
  fd_model_config.h  compile-time constants (N, fs, band bins, sizes)
  fd_model_data.h    RF nodes (float + int16), softmax parameters, FFT/window tables
  fd_testvec.h       int16 test windows + expected features/predictions (golden test)
  model_info.json    metadata (feature set, sizes, training protocol)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES, band_bins, hamming

# ---------------------------------------------------------------------------
# int16 quantisation of RF thresholds (shared with the compression experiment)
# ---------------------------------------------------------------------------


def q16_params(X_train: np.ndarray):
    lo = X_train.min(axis=0).astype(np.float64)
    hi = X_train.max(axis=0).astype(np.float64)
    rng = np.maximum(hi - lo, 1e-12)
    off = lo - 0.05 * rng
    inv_step = 65535.0 / (1.10 * rng)
    return off.astype(np.float32), inv_step.astype(np.float32)


def q16(x, off, inv_step):
    u = np.floor((np.asarray(x, np.float32) - off) * inv_step) - 32768.0
    return np.clip(u, -32768, 32767).astype(np.int16)


def rf_predict_q16(trees, X, off, inv_step, n_classes):
    Q = q16(X, off, inv_step)
    nf = len(off)
    votes = np.zeros((len(X), n_classes), dtype=np.int32)
    for t in trees:
        qthr = {i: q16(np.full(nf, t["threshold"][i]), off, inv_step)[t["feature"][i]]
                for i in range(len(t["feature"])) if t["feature"][i] >= 0}
        for r in range(len(X)):
            i = 0
            while t["feature"][i] >= 0:
                i = i + 1 if Q[r, t["feature"][i]] <= qthr[i] else t["right"][i]
            votes[r, t["leaf"][i]] += 1
    return np.argmax(votes, axis=1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def cfloat(v) -> str:
    """C float literal that round-trips float32."""
    s = f"{float(np.float32(v)):.9g}"
    if not any(ch in s for ch in ".eEn"):
        s += ".0"
    return s + "f"


def _floats(a, per_line=8) -> str:
    a = np.asarray(a, dtype=np.float32).ravel()
    items = [cfloat(v) for v in a]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "    " + ",\n    ".join(lines)


def _ints(a, per_line=16) -> str:
    a = np.asarray(a).ravel()
    items = [str(int(v)) for v in a]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "    " + ",\n    ".join(lines)


def global_scale(max_abs: float) -> float:
    """int16 full scale of the (emulated) ADC for the whole dataset."""
    return float(max_abs) / 32767.0


def to_int16(x, scale: float) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, np.float64) / scale), -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# main export
# ---------------------------------------------------------------------------


def envelope_constants(n: int, fs: float, env: dict):
    """Biquads (per band: 2 sections x b0 b1 b2 a1 a2) and the envelope-bin search ranges of fd_envelope.c."""
    from scipy.signal import butter
    decim = int(env["decim"])
    m = n // decim
    k2 = m // 2
    res = (fs / decim) / (2 * k2)
    sos, bins = [], []
    for band in env["bands_hz"]:
        s = butter(2, band, btype="bandpass", fs=fs, output="sos")
        sos.extend(float(v) for sec in s for v in (sec[0], sec[1], sec[2], sec[4], sec[5]))
        for c in env["orders"]:
            for h in range(1, int(env.get("harmonics", 2)) + 1):
                bins.append(max(1, int(np.floor(h * c * env["speed_hz"][0] / res))))
                bins.append(min(k2 - 1, int(np.ceil(h * c * env["speed_hz"][1] / res))))
    return np.array(sos), np.array(bins, dtype=np.int16), decim


def export_model(out_dir: Path, *, n: int, fs: float, band_hz, class_names, rf, softmax,
                 X_train: np.ndarray, test_windows_i16: np.ndarray | None = None, scale: float = 1.0,
                 test_labels=None, extra_info: dict | None = None, env: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C = len(class_names)
    lo, hi = band_bins(n, fs, band_hz)
    nf = X_train.shape[1]                       # 19, or 19 + 3 per envelope band
    n_env = len(env["bands_hz"]) if env else 0
    assert nf == N_FEATURES + 3 * n_env, (nf, n_env)

    # ---- RF ------------------------------------------------------------
    trees = rf.flat_trees()
    roots, nodes, nodes_q = [], [], []
    off, inv_step = q16_params(X_train)
    for t in trees:
        base = len(nodes)
        roots.append(base)
        for i, f in enumerate(t["feature"]):
            if f >= 0:
                thr = float(t["threshold"][i])
                qthr = int(q16(np.full(nf, thr, np.float32), off, inv_step)[f])
                nodes.append((int(f), base + int(t["right"][i]), thr))
                nodes_q.append((int(f), base + int(t["right"][i]), qthr))
            else:
                leaf = -int(t["leaf"][i]) - 1
                nodes.append((leaf, -1, 0.0))
                nodes_q.append((leaf, -1, 0))
    if len(nodes) > 32767:
        raise ValueError("too many RF nodes for int16 indices")

    # ---- softmax -------------------------------------------------------
    idx = softmax.features if softmax.features is not None else list(range(nf))
    D = len(idx)
    env_cfg, env_data = "", ""
    if env:
        sos, bins, decim = envelope_constants(n, fs, env)
        env_cfg = f"#define FD_ENV_BANDS {n_env}\n#define FD_ENV_DECIM {decim}\n"
        env_data = f"""
/* Squared envelope, bands {list(env["bands_hz"])} Hz: biquads (b0 b1 b2 a1 a2) and order search bins */
const float fd_env_sos[FD_ENV_BANDS * 10] = {{
{_floats(sos, 5)}
}};
const int16_t fd_env_bins[FD_ENV_BANDS * 12] = {{
{_ints(bins, 12)}
}};
"""

    cfg_h = f"""/* Generated by fdedge.export_c — do not edit. */
#ifndef FD_MODEL_CONFIG_H
#define FD_MODEL_CONFIG_H
#define FD_N {n}
#define FD_FS {float(fs):.1f}f
#define FD_BAND_LO {lo}
#define FD_BAND_HI {hi}
#define FD_N_CLASSES {C}
#define FD_RF_N_TREES {len(trees)}
#define FD_RF_N_NODES {len(nodes)}
#define FD_SM_D {D}
#define FD_SCALE {cfloat(scale)}
{env_cfg}#endif
"""
    rf_rows = ",\n".join(f"    {{{f}, {r}, {cfloat(thr)}}}" for f, r, thr in nodes)
    rfq_rows = ",\n".join(f"    {{{f}, {r}, {q}}}" for f, r, q in nodes_q)
    names = ", ".join(f'"{c}"' for c in class_names)
    k = np.arange(n // 2)
    data_h = f"""/* Generated by fdedge.export_c — included once by fd_model.c. */
#include "fd_core.h"

const char *const fd_class_names[FD_N_CLASSES] = {{{names}}};

/* Random forest: {len(trees)} trees, {len(nodes)} nodes, pre-order (left child = i + 1) */
const uint16_t fd_rf_roots[FD_RF_N_TREES] = {{
{_ints(roots)}
}};
const fd_rf_node_t fd_rf_nodes[FD_RF_N_NODES] = {{
{rf_rows}
}};
const fd_rf_node_q16_t fd_rf_nodes_q16[FD_RF_N_NODES] = {{
{rfq_rows}
}};
const float fd_rf_q_off[{nf}] = {{
{_floats(off)}
}};
const float fd_rf_q_inv_step[{nf}] = {{
{_floats(inv_step)}
}};

/* Softmax regression on features {idx} */
const uint8_t fd_sm_idx[FD_SM_D] = {{
{_ints(idx)}
}};
const float fd_sm_mu[FD_SM_D] = {{
{_floats(softmax.mu)}
}};
const float fd_sm_inv_sd[FD_SM_D] = {{
{_floats(1.0 / softmax.sd)}
}};
const float fd_sm_W[FD_N_CLASSES * FD_SM_D] = {{
{_floats(softmax.W)}
}};
const float fd_sm_b[FD_N_CLASSES] = {{
{_floats(softmax.b)}
}};

/* Tables: W_N^k = cos(2 pi k / N) - j sin(2 pi k / N), k = 0..N/2-1; Hamming half window */
const float fd_tw_cos[FD_N / 2] FD_TABLE_ATTR = {{
{_floats(np.cos(2 * np.pi * k / n))}
}};
const float fd_tw_sin[FD_N / 2] FD_TABLE_ATTR = {{
{_floats(np.sin(2 * np.pi * k / n))}
}};
const float fd_hamming_half[FD_N / 2] FD_TABLE_ATTR = {{
{_floats(hamming(n)[: n // 2])}
}};
{env_data}"""
    (out_dir / "fd_model_config.h").write_text(cfg_h, encoding="utf-8")
    (out_dir / "fd_model_data.h").write_text(data_h, encoding="utf-8")

    from .features import ENV_NAMES
    names = FEATURE_NAMES + [f"{e}{j}" for j in range(n_env) for e in ENV_NAMES]
    info = {"n": n, "fs": fs, "band_bins": [lo, hi], "classes": list(class_names),
            "rf_trees": len(trees), "rf_nodes": len(nodes), "softmax_features": [names[i] for i in idx],
            "scale": scale, "envelope": env, **(extra_info or {})}

    # ---- golden test vectors ------------------------------------------
    if test_windows_i16 is not None:
        from .features import extract_features
        W = np.asarray(test_windows_i16, dtype=np.int16)
        feats = extract_features(W.astype(np.float32) * np.float32(scale), fs, band_hz=band_hz, env=env)
        rf_pred = rf.predict(feats)
        sm_pred = softmax.predict(feats)
        rfq_pred = rf_predict_q16(trees, feats, off, inv_step, C)
        labels = np.asarray(test_labels if test_labels is not None else np.full(len(W), -1))
        tv = f"""/* Generated golden test vectors — include in ONE translation unit only. */
#ifndef FD_TESTVEC_H
#define FD_TESTVEC_H
#include <stdint.h>
#define FD_TV_COUNT {len(W)}
static const int16_t fd_tv_windows[FD_TV_COUNT][FD_N] = {{
""" + ",\n".join("  {\n" + _ints(w, 24) + "\n  }" for w in W) + f"""
}};
static const float fd_tv_features[FD_TV_COUNT][{nf}] = {{
""" + ",\n".join("  {\n" + _floats(f) + "\n  }" for f in feats) + f"""
}};
static const int8_t fd_tv_label[FD_TV_COUNT] = {{{", ".join(str(int(v)) for v in labels)}}};
static const int8_t fd_tv_rf[FD_TV_COUNT] = {{{", ".join(str(int(v)) for v in rf_pred)}}};
static const int8_t fd_tv_rf_q16[FD_TV_COUNT] = {{{", ".join(str(int(v)) for v in rfq_pred)}}};
static const int8_t fd_tv_softmax[FD_TV_COUNT] = {{{", ".join(str(int(v)) for v in sm_pred)}}};
#endif
"""
        (out_dir / "fd_testvec.h").write_text(tv, encoding="utf-8")
        info["test_vectors"] = int(len(W))
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
