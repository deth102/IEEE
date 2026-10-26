"""Replay of the on-chip update on the board's analog features (E19 analog campaign), k labelled windows per recording.

Support: the first k windows of every recording in pass 1 (features computed by the board from the DAC-ADC loop);
query: the pass-2 windows of the same recordings without the k replayed support segments. The update is the on-chip
update of the configuration (float64), applied to the softmax head of the model that ran on the board. With k = 1 the
replay repeats the board's own update (agreement with the board's predictions is reported).

usage: python -m tools.analog_replay        (writes results/tables/analog_replay.csv)
"""
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from fdedge.config import load_config, results_dir
from tools.hil_models import train_models

cfg = load_config()
TAB = results_dir(cfg, "tables")
A = pd.read_csv(TAB / "e19_onchip_analog_windows.csv")
models = train_models(cfg, export_aug=False)
FEAT = [f"f{j}" for j in range(19)]
rows = []
for tag in ("aug", "clean"):
    sm = models[tag][1]
    a = A[A.model == tag]
    p1, p2 = a[a["pass"] == 1], a[a["pass"] == 2]
    for k in (1, 2, 5):
        sup, q = p1[p1.win < k], p2[p2.win >= k]
        m = copy.deepcopy(sm).finetune(sup[FEAT].to_numpy(np.float64), sup.label.to_numpy(), **cfg.adaptation["update"])
        pred = m.predict(q[FEAT].to_numpy(np.float64))
        rows.append({"model": tag, "k": k, "support": len(sup), "query": len(q),
                     "f1_sm": f1_score(q.label, pred, labels=list(range(5)), average="macro", zero_division=0),
                     "agree_board": float(np.mean(pred == q.sm.to_numpy())) if k == 1 else np.nan})
df = pd.DataFrame(rows)
df.to_csv(TAB / "analog_replay.csv", index=False)
print(df.round(4).to_string(index=False))
