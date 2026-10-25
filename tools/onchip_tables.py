"""Summaries of the on-chip campaigns (E19 KAIST, E20 CWRU) for the paper and Supplementary Note S9.

usage: python -m tools.onchip_tables      prints the numbers and LaTeX rows; writes results/tables/onchip_summary.csv
"""
import numpy as np
import pandas as pd

from fdedge.config import load_config

cfg = load_config()
T = cfg.path("results_dir") / "tables"
rows = {}


def pct(v):
    return f"{100 * v:.1f}"


lk = T / "e19_onchip_lifecycle_pairs.csv"
if lk.exists():
    d = pd.read_csv(lk)
    print("KAIST lifecycle pairs:", len(d))
    for r in d.itertuples():
        print(f"{r.source.replace('Nm', '')}$\\rightarrow${r.target.replace('Nm', '')} & {pct(r.auroc)} & {pct(r.alarm_rate_shifted)} & "
              f"{pct(r.false_alarm_rate)} & {r.delay_windows_median:.0f} & {pct(r.f1_rf_block1_before)} / {pct(r.f1_sm_block1_before)} & "
              f"{pct(r.f1_rf_after)} / {pct(r.f1_sm_after)} & {pct(r.alarm_rate_after)} & {pct(min(r.agree_rf_after, r.agree_sm_after, r.agree_alarm))} \\\\")
    m = d.mean(numeric_only=True)
    rows["kaist_lifecycle"] = m.to_dict() | {"windows": int(d.windows.sum()), "pairs": len(d)}
    print("mean:", {k: round(v, 4) for k, v in m.items()})
    w = pd.read_csv(T / "e19_onchip_lifecycle_windows.csv")
    print("gate us median", w.c_gate.median() / 240, "cls us median", w.c_cls.median() / 240, "us max", w.us.max())
    rows["kaist_lifecycle"] |= {"gate_us": w.c_gate.median() / 240, "cls_us": w.c_cls.median() / 240, "us_max": w.us.max()}

lc = T / "e20_cwru_onchip_cwru_env_lifecycle_pairs.csv"
if lc.exists():
    d = pd.read_csv(lc)
    print("CWRU lifecycle pairs:", len(d))
    for r in d.itertuples():
        print(f"{r.source}$\\rightarrow${r.target} & {pct(r.auroc)} & {pct(r.alarm_rate_shifted)} & {pct(r.false_alarm_rate)} & "
              f"{pct(r.f1_rf_before)} / {pct(r.f1_sm_before)} & {pct(r.f1_rf_after)} / {pct(r.f1_sm_after)} & "
              f"{pct(r.alarm_rate_after)} & {pct(r.agree_sm_after_python)} \\\\")
    m = d.mean(numeric_only=True)
    rows["cwru_lifecycle"] = m.to_dict() | {"windows": int(d.windows.sum()), "pairs": len(d)}
    print("mean:", {k: round(v, 4) for k, v in m.items()})

for name, f in (("kaist_equiv", "e19_onchip_equiv_summary.csv"), ("cwru_equiv", "e20_cwru_onchip_cwru_env_equiv_summary.csv"),
                ("analog", "e19_onchip_analog.csv")):
    if (T / f).exists():
        d = pd.read_csv(T / f)
        print(name, d.to_dict("records"))
        rows[name] = d.to_dict("records")
pd.DataFrame([{"part": k, "values": v} for k, v in rows.items()]).to_csv(T / "onchip_summary.csv", index=False)
