"""E08 — Model compression for the MCU.

RF: node formats (legacy 20 B struct of the thesis, compact 8 B pre-order,
int16-quantised 6 B) and the accuracy impact of int16 thresholds; forest size
grid (trees x depth). Softmax: float32 vs int8 (per-class symmetric) weights.
Input quantisation: 16-bit ADC emulation (int16 full scale of the dataset).
"""
import numpy as np
import pandas as pd

from fdedge.data import FeatureStore, blocked_split, load_signal, window_view
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.export_c import global_scale, q16_params, rf_predict_q16, to_int16
from fdedge.features import extract_features
from fdedge.metrics import scores
from fdedge.models import make_model
from fdedge.resources import RF_NODE_BYTES, rf_flash_bytes, softmax_flash_bytes

run = Run("e08_compression", __doc__)
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
N = cfg.signal["window"]
fs_hz = float(cfg[ds]["fs"])
band = cfg[ds]["shaft_band_hz"]
st = FeatureStore(cfg, ds, N)
max_abs = max(float(np.max(np.abs(load_signal(r, cfg)))) for r in st.recs)
scale = global_scale(max_abs)

rows, grid_rows = [], []
for f in run.folds:
    tr, va, te = blocked_split(st, f)
    # 16-bit input quantisation of the test windows
    Xq = np.empty_like(te.X)
    for ri in np.unique(te.rec):
        sel = np.nonzero(te.rec == ri)[0]
        w = window_view(load_signal(st.recs[ri], cfg), N, te.start[sel])
        Xq[sel] = extract_features(to_int16(w, scale).astype(np.float32) * np.float32(scale), fs_hz, band_hz=band)

    rf = make_model("rf", C, cfg, seed=f).fit(tr.X, tr.y)
    trees = rf.flat_trees()
    off, inv = q16_params(tr.X)
    n_nodes, n_trees = rf.n_nodes(), len(trees)
    rows += [
        {"fold": f, "model": "RF", "variant": "float32 thresholds, float input", "flash_bytes":
            rf_flash_bytes(n_nodes, n_trees, "compact"), **scores(te.y, rf.predict(te.X), C)},
        {"fold": f, "model": "RF", "variant": "float32 thresholds, int16 ADC input", "flash_bytes":
            rf_flash_bytes(n_nodes, n_trees, "compact"), **scores(te.y, rf.predict(Xq), C)},
        {"fold": f, "model": "RF", "variant": "int16 thresholds (q16), int16 ADC input", "flash_bytes":
            rf_flash_bytes(n_nodes, n_trees, "q16"), **scores(te.y, rf_predict_q16(trees, Xq, off, inv, C), C)},
        {"fold": f, "model": "RF", "variant": "thesis 20-B node struct", "flash_bytes":
            rf_flash_bytes(n_nodes, n_trees, "legacy"), **scores(te.y, rf.predict(te.X), C)},
    ]
    sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
    D = len(sm.features)
    rows.append({"fold": f, "model": "Softmax", "variant": "float32 weights", "flash_bytes": softmax_flash_bytes(D, C),
                 **scores(te.y, sm.predict(Xq), C)})
    W0 = sm.W.copy()
    s = np.abs(W0).max(axis=1, keepdims=True) / 127.0
    sm.W = np.round(W0 / s).clip(-127, 127) * s
    rows.append({"fold": f, "model": "Softmax", "variant": "int8 weights", "flash_bytes": softmax_flash_bytes(D, C, 1) + 4 * C,
                 **scores(te.y, sm.predict(Xq), C)})
    # forest size grid
    for t in [1, 3, 5, 10, 20]:
        for d in [4, 5, 6, 7, 8]:
            m = make_model("rf", C, cfg, seed=f, n_estimators=t, max_depth=d).fit(tr.X, tr.y)
            grid_rows.append({"fold": f, "trees": t, "depth": d, "nodes": m.n_nodes(),
                              "flash_q16": rf_flash_bytes(m.n_nodes(), t, "q16"),
                              "mean_path": m.mean_path_length(te.X), **scores(te.y, m.predict(te.X), C)})

df = pd.DataFrame(rows)
fmt = summarize(df, ["model", "variant"]).merge(
    df.groupby(["model", "variant"], sort=False)["flash_bytes"].mean().round().reset_index(), on=["model", "variant"])
run.table(fmt, "formats")
g = pd.DataFrame(grid_rows)
run.table(g.groupby(["trees", "depth"]).agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), nodes=("nodes", "mean"),
                                            flash_q16=("flash_q16", "mean"), mean_path=("mean_path", "mean")).reset_index(),
          "forest_grid")
print(summarize(df, ["model", "variant"]).to_string(index=False))
print("node bytes:", RF_NODE_BYTES)
run.done()
