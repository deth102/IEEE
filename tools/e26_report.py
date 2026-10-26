"""Selection and report of the hyper-parameter search (E26).

The selection reads validation scores only (see experiments/e26_hparam_search.py):
  1. base models whose in-distribution validation macro-F1 lies within 0.5 points of the best one;
  2. among them, the highest mean unseen-load validation macro-F1 over {single-source, LOLO} x {q = 1, 5}.
Done per dataset and for one set shared by both datasets. Test scores are then read for the selected set, for
the values used before E26 (DEFAULT) and, as an upper bound only, for the best configuration of every cell on test.

usage: python -m tools.e26_report
"""
from __future__ import annotations

import pandas as pd

from fdedge.config import load_config, results_dir
from fdedge.hpsearch import DEFAULT

TAGS = {"KAIST": "e26_hparam_search", "CWRU": "e26_hparam_search_cwru_env"}
TOL = 0.005
CELLS = [("single-source", 0), ("single-source", 1), ("single-source", 5), ("LOLO", 0), ("LOLO", 1), ("LOLO", 5)]


def load_tables(tab):
    return {ds: {k: pd.read_csv(tab / f"{tag}_{k}.csv") for k in ("indist", "load", "load_by_pair", "severity")}
            for ds, tag in TAGS.items()}


def criterion(t, head):
    ind = t["indist"][t["indist"]["head"] == head].set_index("base")["val_f1"]
    ok = set(ind.index[ind >= ind.max() - TOL])
    ld = t["load"][(t["load"]["head"] == head) & (t["load"]["adapt"] != "none")]
    crit = ld.groupby(["base", "adapt"])["val_f1"].mean()                 # equal weight per (setting, q) cell
    return crit[[b in ok for b in crit.index.get_level_values("base")]]


def cell(t, head, base, adapt, setting, q):
    ld = t["load"]
    a = "none" if q == 0 else adapt
    r = ld[(ld["head"] == head) & (ld["base"] == base) & (ld["adapt"] == a) & (ld["setting"] == setting) & (ld["q"] == q)]
    return float(r["test_f1"].iloc[0]), float(r["test_sd"].iloc[0])


def oracle(t, head, setting, q):
    ld = t["load"]
    r = ld[(ld["head"] == head) & (ld["setting"] == setting) & (ld["q"] == q)]
    return float(r["test_f1"].max())


def main():
    cfg = load_config()
    tab = results_dir(cfg, "tables")
    T = load_tables(tab)
    choice = {}
    for head in ("rf", "softmax"):
        crits = {ds: criterion(t, head) for ds, t in T.items()}
        for ds, c in crits.items():
            choice[(ds, head)] = c.idxmax()
        both = pd.concat(crits, axis=1).dropna().mean(axis=1)
        choice[("shared", head)] = both.idxmax()
        for ds, c in crits.items():
            d = DEFAULT[head]
            rank = int((c > c.get(d, float("-inf"))).sum()) + 1 if d in c.index else None
            print(f"{ds} {head}: selected {choice[(ds, head)]} (val {100 * c.max():.1f}); "
                  f"previous {d} val {100 * c.get(d, float('nan')):.1f}, rank {rank}/{len(c)}")
        print(f"shared {head}: {choice[('shared', head)]}")

    rows = []
    for ds, t in T.items():
        for head in ("rf", "softmax"):
            sets = {"previous": DEFAULT[head], "selected (dataset)": choice[(ds, head)], "selected (shared)": choice[("shared", head)]}
            ind = t["indist"][t["indist"]["head"] == head].set_index("base")
            sev = t["severity"]
            for name, (base, adapt) in sets.items():
                r = {"dataset": ds, "head": head, "set": name, "base": base, "adapt": adapt,
                     "indist": 100 * ind.loc[base, "test_f1"], "flash_B": ind.loc[base, "flash_bytes"]}
                for setting, q in CELLS:
                    m, s = cell(t, head, base, adapt, setting, q)
                    r[f"{setting} q={q}"] = 100 * m
                    r[f"{setting} q={q} sd"] = 100 * s
                for setting in ("single-source", "LOLO"):
                    x = sev[(sev["head"] == head) & (sev["base"] == base) & (sev["adapt"] == adapt) & (sev["setting"] == setting)]
                    r[f"sev {setting}"] = 100 * float(x["test_f1"].iloc[0])
                if ds == "KAIST":
                    bp = t["load_by_pair"]
                    x = bp[(bp["head"] == head) & (bp["base"] == base) & (bp["adapt"] == adapt) & (bp["q"] == 5)]
                    r["worst pair q=5"] = 100 * float(x["test_f1"].min())
                    r["2Nm->4Nm q=5"] = 100 * float(x[(x["source"] == "2Nm") & (x["target"] == "4Nm")]["test_f1"].iloc[0])
                rows.append(r)
            r = {"dataset": ds, "head": head, "set": "best on test (upper bound)",
                 "indist": 100 * ind["test_f1"].max()}
            for setting, q in CELLS:
                r[f"{setting} q={q}"] = 100 * oracle(t, head, setting, q)
            rows.append(r)
    rep = pd.DataFrame(rows)
    rep.to_csv(tab / "e26_report.csv", index=False)
    show = ["dataset", "head", "set", "base", "adapt", "indist"] + [f"{s} q={q}" for s, q in CELLS] + \
           ["sev single-source", "sev LOLO", "worst pair q=5", "2Nm->4Nm q=5"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.1f}".format):
        print(rep[show].to_string(index=False))


if __name__ == "__main__":
    main()
