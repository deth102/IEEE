"""Data behind the KESTREL figures that are not plain result tables (run from 03_code: python -m tools.fig_data).

Writes results/tables/fig_*.csv, which tools.visio_figs draws in Visio:
  fig_staircase.csv     one KESTREL model along increasingly demanding tests (KAIST)
  fig_gate_trace.csv    gate statistic on a healthy machine whose load changes, before/after adaptation
  fig_gate_meta.csv     thresholds and event times of the trace
  fig_gate_roc.csv      window-level ROC of the gate and of the max-probability baseline
  fig_cost.csv          blocked macro-F1, ESP32 latency and RAM of KESTREL and the CNN baselines
  fig_crossload3d.csv   train load x test load macro-F1 of the softmax head before/after adaptation (q = 5)
"""
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from fdedge.adaptation import pick_fewshot
from fdedge.config import load_config, results_dir
from fdedge.data import FeatureStore, Split, block_bounds, blocked_split, class_weights
from fdedge.models import make_model

cfg = load_config()
TAB = results_dir(cfg, "tables")
C, K, EV, WIN = 5, 5, 8192, 10
SEC = 8192 / 25600.0


def t(name):
    return pd.read_csv(TAB / f"{name}.csv")


# --------------------------------------------------------------------------- staircase
e01 = t("e01_protocol_summary")
blk = e01[(e01.protocol == "blocked, N=8192") & (e01.features == "F19")].set_index("model").f1_mean
rnd = e01[(e01.protocol == "random, N=8192") & (e01.features == "F19")].set_index("model").f1_mean
mt = t("e16_adaptation_checks_matched")
none = mt[(mt.setting == "single-source") & (mt.method == "none")].set_index("model").f1_mean
fs = t("e14_review_checks_fewshot_time")
k5 = fs[(fs.setting == "single-source") & (fs.support == "time-separated") & (fs.k == 5)].set_index("model").f1_mean
cs = t("e16_adaptation_checks_cross_severity")
cs = cs[(cs.setting == "single-source") & (cs.support == "other severity (block 1 of A)") & (cs.k == 5)].set_index("model").f1_mean
rows = [("Random split\n(overlapping)", rnd["RF"], rnd["Softmax"]),
        ("Time-blocked\n(seen load)", blk["RF"], blk["Softmax"]),
        ("Unseen load", none["RF"], none["Softmax"]),
        ("Unseen load\n+ R, *q* = 5", k5["RF"], k5["Softmax"]),
        ("Support from other\nseverity (R)", cs["RF"], cs["Softmax"]),
        ("ESP32 replay\nat 25.6 kHz", 0.962, 0.962),          # results/hw (tools.parse_hil): 131/135 correct
        ("ESP32 DAC–ADC\n+ on-chip R", 0.721, 0.832)]        # noise-augmented model, 82 test windows
pd.DataFrame(rows, columns=["step", "rf", "sm"]).to_csv(TAB / "fig_staircase.csv", index=False)

# --------------------------------------------------------------------------- gate trace and ROC
st = FeatureStore(cfg, "kaist", 8192)
src, tgt, f = "0Nm", "2Nm", 0
tr, va, te = blocked_split(st, f, loads=[src])
rf = make_model("rf", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)


def ref_P(model, X):
    Z = model._z(X)
    return np.linalg.inv(np.cov(Z, rowvar=False) + 1e-3 * np.eye(Z.shape[1]))


def gate(rfm, smm, P, X, std):
    p = smm.predict_proba(X)
    Z = std._z(X)
    return np.log1p(np.einsum("ij,jk,ik->i", Z, P, Z) / Z.shape[1]) + (rfm.predict(X) != p.argmax(1))


def trail(s, w=WIN):
    return np.convolve(s, np.ones(w) / w, mode="valid")


def per_rec(split, scores):
    return np.concatenate([trail(scores[split.rec == r]) for r in np.unique(split.rec) if (split.rec == r).sum() >= WIN])


def rec_windows(name, lo_blk, hi_blk):
    r = next(r for r in st.recs if r.name == name)
    b = block_bounds(r.length, K)
    return st.select(r, [(b[lo_blk], b[hi_blk])], EV)[0]


P0 = ref_P(sm, tr.X)
tau = float(np.quantile(per_rec(va, gate(rf, sm, P0, va.X, sm)), 0.99))
ps = []
for ri, r in enumerate(st.recs):
    if r.load == tgt:
        b = block_bounds(r.length, K)
        X, S = st.select(r, [(b[0], b[1])], EV)
        ps.append((X, np.full(len(S), r.cls), np.full(len(S), ri), S))
pool = Split(*(np.concatenate([p[i] for p in ps]) for i in range(4)))
sup, _ = pick_fewshot(pool, 5, C, seed=0)
sma = copy.deepcopy(sm).finetune(pool.X[sup], pool.y[sup], lr=0.1, epochs=200, l2_anchor=1e-3)
w = np.concatenate([class_weights(tr.y, C), np.full(len(sup), 50.0)])
rfa = make_model("rf", C, cfg, seed=0).fit(np.vstack([tr.X, pool.X[sup]]), np.concatenate([tr.y, pool.y[sup]]), sample_weight=w)
P1 = ref_P(sm, np.vstack([tr.X] + [pool.X[sup]] * 50))
tau1 = float(np.quantile(np.r_[per_rec(va, gate(rfa, sma, P1, va.X, sm)), trail(gate(rfa, sma, P1, pool.X, sm))], 0.99))

h_src, h_tgt1, h_tgt2 = rec_windows(f"{src}_Normal", 0, 1), rec_windows(f"{tgt}_Normal", 0, 1), rec_windows(f"{tgt}_Normal", 1, 5)[:120]
g1 = trail(np.r_[gate(rf, sm, P0, h_src, sm), gate(rf, sm, P0, h_tgt1, sm)])
g2 = trail(gate(rfa, sma, P1, h_tgt2, sm))
t1 = (np.arange(len(g1)) + WIN) * SEC
t2 = t1[-1] + 4.0 + (np.arange(len(g2)) + WIN) * SEC
tc = len(h_src) * SEC
alarm = float(t1[np.argmax((g1 > tau) & (t1 > tc))])
pd.concat([pd.DataFrame({"phase": "before", "t": t1, "g": g1}), pd.DataFrame({"phase": "after", "t": t2, "g": g2})]
          ).to_csv(TAB / "fig_gate_trace.csv", index=False)
pd.DataFrame([{"tau": tau, "tau_after": tau1, "t_change": tc, "t_alarm": alarm, "t_adapt0": float(t1[-1]),
               "t_adapt1": float(t2[0])}]).to_csv(TAB / "fig_gate_meta.csv", index=False)


def roc(si, ss):
    fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(si)), np.ones(len(ss))], np.r_[si, ss])
    keep = np.unique(np.r_[0, np.linspace(0, len(fpr) - 1, 120).astype(int), len(fpr) - 1])
    return fpr[keep], tpr[keep]


roc_rows = []
T = pool
target_all = Split(*(np.concatenate([p[i] for p in ps]) for i in range(4)))
from fdedge.data import target_split  # noqa: E402
T = target_split(st, [tgt])
cases = [("unseen load (0→2 N·m)", rf, sm, P0, te.X, T.X)]
from tools.hil_models import train_models  # noqa: E402
from tools.parse_hil import parse  # noqa: E402
rf0, sm0 = train_models(cfg, st, export_aug=False)["clean"]
tr0, va0, te0 = blocked_split(st, 0)
_, _, info = parse(cfg.path("results_dir") / "hw" / "hil_clean.log")
A = np.array(info["feat"])[:, 3:].astype(np.float32)
cases.append(("ESP32 DAC–ADC path", rf0, sm0, ref_P(sm0, tr0.X), te0.X, A))
for lab, rfm, smm, P, Xi, Xs in cases:
    for score in ("gate", "maxp"):
        if score == "gate":
            si, ss = gate(rfm, smm, P, Xi, smm), gate(rfm, smm, P, Xs, smm)
        else:
            si, ss = 1 - smm.predict_proba(Xi).max(1), 1 - smm.predict_proba(Xs).max(1)
        fpr, tpr = roc(si, ss)
        roc_rows += [{"case": lab, "score": score, "fpr": a, "tpr": b} for a, b in zip(fpr, tpr)]
pd.DataFrame(roc_rows).to_csv(TAB / "fig_gate_roc.csv", index=False)

# --------------------------------------------------------------------------- cost
cnn = t("e09_cnn_w8192_f5_accuracy").set_index(["model", "setting"]).f1_mean      # CNNs on the same 8192-sample windows
cost = t("e09_cnn_w8192_f5_cost")
cost = cost[cost.platform == "esp32"].set_index("model")
pts = [("KESTREL, tables in RAM", 13.6, blk["RF"], 112, True), ("KESTREL, tables in flash", 20.9, blk["RF"], 64, True)]
for m, lab in (("CNN-EdgeLite", "EdgeLite"), ("CNN-MobileLite", "MobileLite"), ("CNN-Baseline", "CNN 34.4 M")):
    pts.append((lab, float(cost.loc[m, "latency_ms"]), float(cnn[(m, "blocked (vote 1 = 8192 samples)")]),
                float(cost.loc[m, "act_ram_f32_bytes"]) / 1024, False))
pd.DataFrame(pts, columns=["label", "latency_ms", "f1", "ram_kib", "measured"]).to_csv(TAB / "fig_cost.csv", index=False)

# --------------------------------------------------------------------------- 3-D cross-load matrix
seen = t("e04_crossload_single_summary")
seen = seen[(seen.model == "Softmax") & (seen.features == "F19") & (seen.seen)].set_index("source").f1_mean
m16 = t("e16_adaptation_checks_matched_raw")
m16 = m16[(m16.setting == "single-source") & (m16.model == "Softmax") & (m16.method == "none")].groupby(["source", "target"]).f1.mean()
a14 = t("e14_review_checks_fewshot_time_raw")
a14 = a14[(a14.setting == "single-source") & (a14.model == "Softmax") & (a14.support == "time-separated") & (a14.k == 5)]
a14 = a14.groupby(["source", "target"]).f1.mean()
rows = []
for panel, cells in (("before", m16), ("after", a14)):
    for s_ in ("0Nm", "2Nm", "4Nm"):
        for t_ in ("0Nm", "2Nm", "4Nm"):
            rows.append({"panel": panel, "source": s_, "target": t_, "f1": seen[s_] if s_ == t_ else cells[(s_, t_)]})
pd.DataFrame(rows).to_csv(TAB / "fig_crossload3d.csv", index=False)
print("tau", round(tau, 3), "tau'", round(tau1, 3), "change", round(tc, 2), "alarm", round(alarm, 2))
