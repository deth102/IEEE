"""Figure data of the target draft (target/main_target.tex) that differ from tools.fig_data:
the KESTREL gate is the prototype gate (fdedge.gate), and the lifecycle trace is the one measured on the ESP32 (E19).

Writes results/tables/fig_gate_trace_target.csv, fig_gate_meta_target.csv, fig_gate_roc_target.csv.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from fdedge.config import load_config, results_dir
from fdedge.data import FeatureStore, Split, blocked_split, target_split
from fdedge.gate import ProtoGate
from fdedge.models import make_model

cfg = load_config()
TAB = results_dir(cfg, "tables")
C = 5

# --------------------------------------------------------------------------- trace measured on the board (E19 trace)
tr_b = pd.read_csv(TAB / "e19_onchip_trace.csv")
tr_b = tr_b[tr_b.g >= 0][["phase", "t", "g"]]
tr_b.to_csv(TAB / "fig_gate_trace_target.csv", index=False)
meta = pd.read_csv(TAB / "e19_onchip_trace_meta.csv")
meta[["tau", "tau_after", "t_change", "t_alarm", "t_adapt0", "t_adapt1"]].to_csv(TAB / "fig_gate_meta_target.csv", index=False)

# --------------------------------------------------------------------------- window-level ROC of the prototype gate
st = FeatureStore(cfg, "kaist", 8192)


def roc(si, ss):
    fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(si)), np.ones(len(ss))], np.r_[si, ss])
    keep = np.unique(np.r_[0, np.linspace(0, len(fpr) - 1, 120).astype(int), len(fpr) - 1])
    return fpr[keep], tpr[keep]


tr, va, te = blocked_split(st, 0, loads=["0Nm"])
rf = make_model("rf", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
sm = make_model("softmax", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
cases = [("unseen load (0→2 N·m)", rf, sm, ProtoGate(sm, tr.X, tr.y, C), te.X, target_split(st, ["2Nm"]).X)]
from tools.hil_models import train_models  # noqa: E402
rf0, sm0 = train_models(cfg, st, export_aug=False)["clean"]
tr0, va0, te0 = blocked_split(st, 0)
if (TAB / "e19_onchip_analog_windows.csv").exists():       # every fold-0 test window through the DAC-ADC loop
    A = pd.read_csv(TAB / "e19_onchip_analog_windows.csv")
    A = A[(A.model == "clean") & (A["pass"] == 1)]
    Xa = A[[f"f{j}" for j in range(19)]].to_numpy(np.float32)
else:                                                      # earlier 109-window run (hil_esp32)
    from tools.parse_hil import parse
    Xa = np.array(parse(cfg.path("results_dir") / "hw" / "hil_clean.log")[2]["feat"])[:, 3:].astype(np.float32)
cases.append(("ESP32 DAC–ADC path", rf0, sm0, ProtoGate(sm0, tr0.X, tr0.y, C), te0.X, Xa))
rows = []
for lab, rfm, smm, g, Xi, Xs in cases:
    for score in ("gate", "maxp"):
        if score == "gate":
            si, ss = g.scores(rfm, smm, Xi), g.scores(rfm, smm, Xs)
        else:
            si, ss = 1 - smm.predict_proba(Xi).max(1), 1 - smm.predict_proba(Xs).max(1)
        fpr, tpr = roc(si, ss)
        rows += [{"case": lab, "score": score, "fpr": a, "tpr": b} for a, b in zip(fpr, tpr)]
pd.DataFrame(rows).to_csv(TAB / "fig_gate_roc_target.csv", index=False)
print("target figure data written")
