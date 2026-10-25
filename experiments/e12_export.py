"""E12 — Export deployable models to C and verify them bit-for-bit on the host.

Models are trained on the blocked training part of fold 0 (all loads); test
vectors come from the fold-0 test block (never seen in training). Tags:
  <ds>_n<N>          5 windows (one per class)   -> bench_mcu / edge node
  <ds>_n<N>_golden   50 random test windows      -> host golden test only
The host test compiles fdcore with gcc and compares features/predictions with Python.
"""
import numpy as np
import pandas as pd

from fdedge.data import FeatureStore, blocked_split, load_signal, window_view
from fdedge.experiment import Run
from fdedge.export_c import export_model, global_scale, to_int16
from fdedge.features import FEATURE_NAMES
from fdedge.models import make_model
from tools.build_host import build_and_run

run = Run("e12_export", __doc__)
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
fs_hz = float(cfg[ds]["fs"])
band = cfg[ds]["shaft_band_hz"]
out_root = cfg.path("firmware_models_dir")
Ns = [cfg.signal["window"]] if run.quick else sorted({2048, 4096, cfg.signal["window"], 16384})

st0 = FeatureStore(cfg, ds, cfg.signal["window"])
scale = global_scale(max(float(np.max(np.abs(load_signal(r, cfg)))) for r in st0.recs))
rows = []
for N in Ns:
    st = FeatureStore(cfg, ds, N)
    tr, va, te = blocked_split(st, 0)
    rf = make_model("rf", C, cfg, seed=0).fit(tr.X, tr.y)
    sm = make_model("softmax", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
    rng = np.random.default_rng(N)

    def windows(idx):
        return np.stack([to_int16(window_view(load_signal(st.recs[te.rec[i]], cfg), N, np.array([te.start[i]]))[0], scale)
                         for i in idx])

    one_per_class = np.array([rng.choice(np.nonzero(te.y == c)[0]) for c in range(C)])
    n_gold = 12 if run.quick else 50
    golden = rng.choice(len(te.y), size=min(n_gold, len(te.y)), replace=False)
    sfx = "_quick" if run.quick else ""                   # smoke tests never overwrite deployed models
    for tag, idx in [(f"{ds}_n{N}{sfx}", one_per_class), (f"{ds}_n{N}_golden{sfx}", golden)]:
        info = export_model(out_root / tag, n=N, fs=fs_hz, band_hz=band, class_names=cfg[ds]["classes"], rf=rf,
                            softmax=sm, X_train=tr.X, test_windows_i16=windows(idx), scale=scale,
                            test_labels=te.y[idx], extra_info={"trained_on": "blocked fold 0 train, all loads"})
        res = build_and_run(tag)
        err = np.array(res["max_rel_err"])
        rows.append({"tag": tag, "N": N, "windows": res["n"], "rf_nodes": info["rf_nodes"],
                     "rf_agree": res["rf_agree"], "rf_q16_agree": res["rf_q16_agree"],
                     "softmax_agree": res["softmax_agree"], "rf_label_acc": res["rf_label_acc"],
                     "softmax_label_acc": res["softmax_label_acc"],
                     "max_rel_err_opt": float(err[1:].max()), "max_rel_err_mean_t": float(err[0]),
                     "max_rel_err_legacy": float(np.max(res["max_rel_err_legacy"][1:])),
                     "worst_feature_legacy": FEATURE_NAMES[1 + int(np.argmax(res["max_rel_err_legacy"][1:]))],
                     "host_us_opt": res["host_us_optimized"], "host_us_legacy": res["host_us_legacy"]})
        print(rows[-1])
df = pd.DataFrame(rows)
run.table(df, "golden", float_fmt="%.3g")
run.done()
