"""E26 — One hyper-parameter set for the whole method, selected on validation data only.

Search space (fdedge/hpsearch.py, fixed before any score was read; the current values are part of it):
  softmax head    l2 in {0, 1e-4, 1e-3, 1e-2} (lr 0.5, at most 2000 epochs, early stopping as before)
  random forest   six shapes of equal flash budget (about 430 nodes, as the deployed 10 trees x 22 leaves):
                  trees x depth x leaves = 5x10x44, 7x10x32, 10x7x22, 10x10x22, 14x8x16, 20x8x11
  on-chip update  lr in {0.03, 0.1, 0.3, 1} x anchor in {1e-4, 1e-3, 1e-2, 1e-1} x epochs in {100, 200, 400, 800}
  gateway RF      support weight in {5, 10, 20, 50, 100, 200}
Validation data (never the test blocks):
  in-distribution  the validation block of every blocked fold, as in E02;
  unseen load      block 1 of the target recordings minus the support windows (the support is drawn from
                   block 1, as in Table V); single-source and LOLO, q = 1 and 5, five folds, three draws.
Test data: the test block of every blocked fold (in-distribution), blocks 2..5 of the target recordings
(unseen load), and the cross-severity split of E16 (b) with q = 5. Every configuration is scored on test as
well, so that the selected one can be compared with the best possible one. The selection (tools/e26_report.py)
reads validation scores only:
  1. base models whose in-distribution validation macro-F1 lies within 0.5 points of the best one;
  2. among them, the configuration with the highest mean unseen-load validation macro-F1 over
     {single-source, LOLO} x {q = 1, q = 5}; per dataset, and one set shared by both datasets.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fdedge.data import FeatureStore, Split, block_bounds, blocked_split, loads_of
from fdedge.experiment import Run
from fdedge.hpsearch import BASE_RF, BASE_SM, RF_LAB, ROW_COLS, SM_LAB, macro_f1, rf_model, run_task
from fdedge.models import make_model
from fdedge.resources import rf_flash_bytes

run = Run("e26_hparam_search", __doc__, extra_args=[(("--jobs",), dict(type=int, default=8))])
cfg, ds = run.cfg, run.dataset
classes = list(cfg[ds]["classes"])
C = len(classes)
NORMAL = classes.index("Normal")
N = cfg.signal["window"]
st = FeatureStore(cfg, ds, N)
K = cfg.protocol["n_folds"]
EV = int(N * cfg.signal["eval_hop_ratio"])
loads = loads_of(cfg, ds)


def part(rec_ok, load, first):
    ps = []
    for ri, r in enumerate(st.recs):
        if r.load != load or not rec_ok(r):
            continue
        b = block_bounds(r.length, K)
        X, S = st.select(r, [(b[0], b[1])] if first else [(b[1], b[K])], EV)
        if len(S):
            ps.append((X, np.full(len(S), r.cls), np.full(len(S), ri), S))
    return Split(*(np.concatenate([p[i] for p in ps]) for i in range(4)))


# --------------------------------------------------------------------------- in-distribution (blocked CV)
rows = []
for f in run.folds:
    tr, va, te = blocked_split(st, f)
    for lab, b in zip(SM_LAB, BASE_SM):
        m = make_model("softmax", C, cfg, seed=f, **b).fit(tr.X, tr.y, va.X, va.y)
        rows.append({"head": "softmax", "base": lab, "fold": f, "val_f1": macro_f1(va.y, m.predict(va.X), C)[0],
                     "test_f1": macro_f1(te.y, m.predict(te.X), C)[0], "flash_bytes": np.nan})
    for lab, b in zip(RF_LAB, BASE_RF):
        m = rf_model(cfg, C, f, b).fit(tr.X, tr.y)
        rows.append({"head": "rf", "base": lab, "fold": f, "val_f1": macro_f1(va.y, m.predict(va.X), C)[0],
                     "test_f1": macro_f1(te.y, m.predict(te.X), C)[0],
                     "flash_bytes": rf_flash_bytes(m.n_nodes(), len(m.clf.estimators_))})
ind = pd.DataFrame(rows)
run.table(ind, "indist_raw", latex=False)
ind_s = ind.groupby(["head", "base"], sort=False)[["val_f1", "test_f1", "flash_bytes"]].mean().reset_index()
run.table(ind_s, "indist", latex=False)
print(ind_s.to_string(index=False), flush=True)

# --------------------------------------------------------------------------- unseen load and cross-severity
sev = {c: sorted({r.severity for r in st.recs if r.cls == c}) for c in range(C) if c != NORMAL}
n_sev = min(len(v) for v in sev.values())
sev_pairs = [(a, b) for a in range(n_sev) for b in range(n_sev) if a != b]
tasks = []
for multi in (False, True):
    for src in ([[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]):
        for tgt in [L for L in loads if L not in src]:
            pool, Q = part(lambda r: True, tgt, True), part(lambda r: True, tgt, False)
            sv = []
            for a, b in sev_pairs:
                is_a = lambda r, a=a: r.cls == NORMAL or r.severity == sev[r.cls][a]      # noqa: E731
                is_b = lambda r, b=b: r.cls == NORMAL or r.severity == sev[r.cls][b]      # noqa: E731
                sv.append((a, b, part(is_a, tgt, True), part(is_b, tgt, False)))
            for f in run.folds:
                tr, va, _ = blocked_split(st, f, loads=src)
                tasks.append((("LOLO" if multi else "single-source", "+".join(src), tgt, f), f, tr, va, pool, Q, sv))
print(f"[{run.tag}] {len(tasks)} tasks on {run.args.jobs} workers", flush=True)
out = Parallel(n_jobs=run.args.jobs, verbose=5)(delayed(run_task)(cfg, C, *t) for t in tasks)
raw = pd.DataFrame([r for rs in out for r in rs], columns=ROW_COLS)
raw.to_csv(run.tab_dir / f"{run.tag}_raw.csv.gz", index=False, compression="gzip")

keys = ["setting", "head", "base", "adapt"]
ld = raw[raw.task == "load"]
load_s = ld.groupby(keys + ["q"], sort=False).agg(val_f1=("val_f1", "mean"), test_f1=("test_f1", "mean"),
                                                  test_sd=("test_f1", "std"), n=("test_f1", "size")).reset_index()
run.table(load_s, "load", latex=False)
pairs = ld[ld.setting == "single-source"].groupby(["head", "base", "adapt", "q", "source", "target"], sort=False)
run.table(pairs["test_f1"].mean().reset_index(), "load_by_pair", latex=False)
sv_ = raw[raw.task == "severity"]
sev_s = sv_.groupby(keys + ["q"], sort=False).agg(test_f1=("test_f1", "mean"), test_sd=("test_f1", "std"),
                                                  n=("test_f1", "size")).reset_index()
run.table(sev_s, "severity", latex=False)
run.table(sv_.groupby(keys + ["q", "support_sev", "test_sev"], sort=False)["test_f1"].mean().reset_index(),
          "severity_by_direction", latex=False)
run.done()
