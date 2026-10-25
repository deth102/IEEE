"""E18 — Bearing-order (envelope) features for bearing faults (CWRU).

(a) Band selection: the band-pass of the squared envelope is chosen among four octave bands by the macro-F1 on
    the validation blocks (time-blocked CV, both heads), as the RF hyper-parameters are.
(b) Time-blocked test macro-F1 of F16, F19, the bearing orders alone (E3), F16 + E3 and F22 (= F19 + E3) with
    the selected band, F22 for every candidate band (sensitivity), and per-class recall (F19 vs F22).
(c) Cost of the bearing-order stage on the MCU: operations, buffer and ESP32 latency by the calibrated model.
"""
import numpy as np
import pandas as pd

from fdedge.data import FeatureStore, blocked_split
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.features import FEATURE_SETS, env_band_idx
from fdedge.metrics import confusion, scores
from fdedge.models import make_model
from fdedge.resources import estimate, flops, load_platforms, ops_envelope_features

run = Run("e18_envelope", __doc__)
cfg, ds = run.cfg, run.dataset
cfg["use_envelope"] = True                       # all candidate bands are computed and cached once
env = cfg[ds]["envelope"]
bands = [tuple(b) for b in env["bands_hz"]]
classes = list(cfg[ds]["classes"])
C = len(classes)
N = cfg.signal["window"]
st = FeatureStore(cfg, ds, N)
F16, F19 = FEATURE_SETS["F16"], FEATURE_SETS["F19"]


def band_label(b):
    return f"{b[0] / 1e3:g}-{b[1] / 1e3:g} kHz"


sets = {"F16": F16, "F19": F19}
for j, b in enumerate(bands):
    e3 = env_band_idx(j)
    sets.update({f"E3 [{band_label(b)}]": e3, f"F16+E3 [{band_label(b)}]": F16 + e3,
                 f"F22 [{band_label(b)}]": F19 + e3})

rows, cms = [], {}
for f in run.folds:
    tr, va, te = blocked_split(st, f)
    for name, idx in sets.items():
        for kind in ("rf", "softmax"):
            m = make_model(kind, C, cfg, seed=f, features=idx).fit(tr.X, tr.y, va.X, va.y)
            pv, pt = m.predict(va.X), m.predict(te.X)
            lab = "RF" if kind == "rf" else "Softmax"
            rows.append({"fold": f, "features": name, "model": lab, "split": "validation", **scores(va.y, pv, C)})
            rows.append({"fold": f, "features": name, "model": lab, "split": "test", **scores(te.y, pt, C)})
            cms.setdefault((name, lab), []).append(confusion(te.y, pt, C))
    print(f"fold {f} done")
df = pd.DataFrame(rows)
run.table(df, "raw", latex=False)

# (a) band selection on the validation blocks (F22, both heads)
val = df[(df.split == "validation") & df.features.str.startswith("F22")]
sel = val.groupby("features").f1.mean().rename("val_f1").reset_index()
sel["band"] = sel.features.str.extract(r"\[(.*)\]")[0]
best = sel.loc[sel.val_f1.idxmax(), "band"]
run.table(sel[["band", "val_f1"]], "band_selection")
print(sel[["band", "val_f1"]].to_string(index=False), "\nselected band:", best)

# (b) test results
test = df[df.split == "test"]
summ = summarize(test, ["features", "model"], ("f1", "acc", "bacc"))
run.table(summ, "summary_all")
main_sets = ["F16", "F19", f"E3 [{best}]", f"F16+E3 [{best}]", f"F22 [{best}]"]
main = summ[summ.features.isin(main_sets)].copy()
main["features"] = pd.Categorical(main.features, main_sets)
run.table(main.sort_values(["features", "model"]), "summary")
print(main.sort_values(["features", "model"]).to_string(index=False))
recall = []
for name in ("F19", f"F22 [{best}]"):
    for lab in ("RF", "Softmax"):
        cm = np.mean(cms[(name, lab)], axis=0)
        recall.append({"features": name, "model": lab, **{c: float(cm[i, i]) for i, c in enumerate(classes)}})
rec = pd.DataFrame(recall)
run.table(rec, "recall")
print(rec.to_string(index=False))

# (c) cost of the bearing-order stage
j = [band_label(b) for b in bands].index(best)
ops = ops_envelope_features(N, int(env["decim"]), len(env["orders"]), int(env.get("harmonics", 2)))
plat = load_platforms()
cost = []
for p in ("esp32", "esp32s3", "stm32f446", "stm32h743", "rp2040"):
    e = estimate({"envelope": ops}, plat[p])
    cost.append({"platform": p, "band": best, "arith_ops": flops(ops), "latency_ms": e["latency_ms"],
                 "buffer_bytes": 4 * (N // int(env["decim"]))})
cost = pd.DataFrame(cost)
run.table(cost, "cost")
print(cost.to_string(index=False))
run.json({"selected_band": best, "band_hz": bands[j], "env": env}, "selection")
run.done()
