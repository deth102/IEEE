"""E17 — KESTREL self-check gate (block S).

The node runs both heads on every window. The KESTREL gate (fdedge.gate.ProtoGate) scores each window by the
within-class Mahalanobis distance of the standardized features to the nearest prototype of the predicted class plus
the head disagreement d_t = 1[RF != softmax]; the gate statistic g is the mean over the last W windows. An alarm is
raised when g exceeds a threshold tau chosen on the source validation blocks (99th percentile, i.e. ~1 % false alarms
per position); no target data are used to set tau. Ablations and baselines: T2 (shared covariance) with and without
disagreement, disagreement + normalized entropy, disagreement, entropy, 1 - max softmax probability.

Scenarios
  load       : source test blocks (in-distribution) vs every window of an unseen load (single-source and LOLO)
  severity   : models trained with one severity per fault class, tested on the held-out severity
  verify     : after few-shot adaptation (q = 5, support block 1), alarm rate on blocks 2..5 of the target load
  analog     : (KAIST) deployed fold-0 models, digital fold-0 test block vs windows that passed the ESP32 DAC-ADC loop
Metrics: AUROC of g (sliding means) in-distribution vs shifted; alarm rate on shifted positions; false alarms per
hour on in-distribution positions; detection delay = windows from the first shifted window to the first alarm.
"""
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fdedge.adaptation import pick_fewshot
from fdedge.config import load_config
from fdedge.data import FeatureStore, Split, block_bounds, blocked_ranges, blocked_split, class_weights, loads_of, target_split
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.gate import ProtoGate
from fdedge.models import make_model

run = Run("e17_selfcheck_gate", __doc__)
cfg, ds = run.cfg, run.dataset
K_PROTO = cfg.get("gate", {}).get("k", 8)                # prototypes per class (configs/gate_k4.yaml: 4)
classes = list(cfg[ds]["classes"])
C = len(classes)
NORMAL = classes.index("Normal")
N = cfg.signal["window"]
st = FeatureStore(cfg, ds, N)
K = cfg.protocol["n_folds"]
EV = int(N * cfg.signal["eval_hop_ratio"])
loads = loads_of(cfg, ds)
W_LIST = (5, 10, 20) if ds == "kaist" else (2, 3, 5)
W_V = 10 if ds == "kaist" else 3          # trailing window of the verify step (CWRU blocks hold ~10 windows)
WIN_S = N / float(cfg[ds]["fs"])
GATE = "KESTREL"
SCORES = ("KESTREL", "T2 (Mahalanobis)", "T2 diag", "T2 + disagreement", "disagreement + entropy", "disagreement", "entropy", "1 - max prob")


class Ref:
    """Source statistics kept on the node: softmax standardisation (mu, sd) and the inverse covariance of z."""

    def __init__(self, sm, Xtr):
        Z = sm._z(Xtr)
        self.d = Z.shape[1]
        self.P = np.linalg.inv(np.cov(Z, rowvar=False) + 1e-3 * np.eye(self.d))


def window_scores(rf, sm, X, ref, pg):
    p = sm.predict_proba(X)
    d = (rf.predict(X) != p.argmax(1)).astype(float)
    h = -(p * np.log(p + 1e-12)).sum(1) / np.log(C)
    Z = sm._z(X)
    t2 = np.einsum("ij,jk,ik->i", Z, ref.P, Z) / ref.d
    t2d = (Z ** 2).mean(1)
    lt2 = np.log1p(t2)
    return {"KESTREL": pg.scores(rf, sm, X), "T2 (Mahalanobis)": lt2, "T2 diag": np.log1p(t2d), "T2 + disagreement": lt2 + d,
            "disagreement + entropy": 0.5 * (d + h), "disagreement": d, "entropy": h, "1 - max prob": 1 - p.max(1)}


def sliding(split: Split, s: np.ndarray, w: int):
    """Trailing means of length w over each recording in time order; returns (values, recording ids)."""
    vals, recs = [], []
    for r in np.unique(split.rec):
        ii = np.nonzero(split.rec == r)[0]
        ii = ii[np.argsort(split.start[ii])]
        if len(ii) < w:
            continue
        c = np.convolve(s[ii], np.ones(w) / w, mode="valid")
        vals.append(c); recs.append(np.full(len(c), r))
    return (np.concatenate(vals), np.concatenate(recs)) if vals else (np.zeros(0), np.zeros(0))


def delay(split: Split, s: np.ndarray, w: int, tau: float):
    """Windows after the start of each shifted recording until the first alarm (w-1 windows are needed to fill)."""
    out = []
    for r in np.unique(split.rec):
        ii = np.nonzero(split.rec == r)[0]
        ii = ii[np.argsort(split.start[ii])]
        if len(ii) < w:
            continue
        c = np.convolve(s[ii], np.ones(w) / w, mode="valid")
        hit = np.nonzero(c > tau)[0]
        out.append(hit[0] + w if len(hit) else np.nan)
    return np.array(out, dtype=float)


def evaluate(rf, sm, ref, pg, va, te, sh, base, rows, w_list=W_LIST):
    sv, st_, ss = (window_scores(rf, sm, va.X, ref, pg), window_scores(rf, sm, te.X, ref, pg),
                   window_scores(rf, sm, sh.X, ref, pg))
    for name in SCORES:
        for w in w_list:
            gv, _ = sliding(va, sv[name], w)
            gi, _ = sliding(te, st_[name], w)
            gs, _ = sliding(sh, ss[name], w)
            if not len(gi) or not len(gs) or not len(gv):
                continue
            tau = np.quantile(gv, 0.99)
            y = np.r_[np.zeros(len(gi)), np.ones(len(gs))]
            auc = roc_auc_score(y, np.r_[gi, gs]) if np.ptp(np.r_[gi, gs]) > 0 else 0.5
            dl = delay(sh, ss[name], w, tau)
            rows.append({**base, "score": name, "W": w, "auroc": auc, "alarm_rate": float(np.mean(gs > tau)),
                         "false_alarm_rate": float(np.mean(gi > tau)),
                         "false_alarms_per_h": float(np.mean(gi > tau)) * 3600 / WIN_S,
                         "delay_windows": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan,
                         "detected_recordings": float(np.mean(np.isfinite(dl)))})


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


rows, ver = [], []
# --------------------------------------------------------------------------- load shift (+ verify after adaptation)
for multi in (False, True):
    sources = [[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]
    for src in sources:
        for f in run.folds:
            tr, va, te = blocked_split(st, f, loads=src)
            rf = make_model("rf", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
            sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
            for tgt in [L for L in loads if L not in src]:
                base = {"scenario": "load", "setting": "LOLO" if multi else "single-source", "source": "+".join(src),
                        "target": tgt, "fold": f}
                ref, pg = Ref(sm, tr.X), ProtoGate(sm, tr.X, tr.y, C, k=K_PROTO, seed=f)
                evaluate(rf, sm, ref, pg, va, te, target_split(st, [tgt]), base, rows)
                # verify: after q = 5 adaptation the gate should fall back to its in-distribution level
                pool, Q = part(lambda r: True, tgt, True), part(lambda r: True, tgt, False)
                sv = window_scores(rf, sm, va.X, ref, pg)[GATE]
                gv, _ = sliding(va, sv, W_V)
                tau = np.quantile(gv, 0.99)
                before, _ = sliding(Q, window_scores(rf, sm, Q.X, ref, pg)[GATE], W_V)
                for s in range(3):
                    sup, _ = pick_fewshot(pool, 5, C, seed=1000 * f + s)
                    sma = copy.deepcopy(sm).finetune(pool.X[sup], pool.y[sup], **cfg.adaptation["update"])
                    w = np.concatenate([class_weights(tr.y, C), np.full(len(sup), cfg.adaptation["rf_support_weight"])])
                    rfa = make_model("rf", C, cfg, seed=s).fit(np.vstack([tr.X, pool.X[sup]]),
                                                               np.concatenate([tr.y, pool.y[sup]]), sample_weight=w)
                    # the node re-references its statistics with the labelled support windows (weight 50, as the RF)
                    ref2 = Ref(sm, np.vstack([tr.X] + [pool.X[sup]] * 50))
                    pg2 = pg.rereferenced(sm, pool.X[sup], pool.y[sup], 50)
                    gv2, _ = sliding(va, window_scores(rfa, sma, va.X, ref2, pg2)[GATE], W_V)
                    tau2 = np.quantile(np.r_[gv2, sliding(pool, window_scores(rfa, sma, pool.X, ref2, pg2)[GATE], W_V)[0]], 0.99)
                    after, _ = sliding(Q, window_scores(rfa, sma, Q.X, ref2, pg2)[GATE], W_V)
                    after = after * 0 + (after > tau2) + 0.0
                    ver.append({**base, "seed": s, "alarm_before": float(np.mean(before > tau)),
                                "alarm_after": float(np.mean(after))})

# --------------------------------------------------------------------------- unseen severity
sev = {c: sorted({r.severity for r in st.recs if r.cls == c}) for c in range(C) if c != NORMAL}
n_sev = min(len(v) for v in sev.values())
for h in range(n_sev):
    for f in run.folds:
        tr_p, va_p, te_p, sh_p = [], [], [], []
        for ri, r in enumerate(st.recs):
            tr_r, va_r, te_r = blocked_ranges(r.length, K, f)
            add = lambda lst, rng, stride, ri=ri, r=r: lst.append(  # noqa: E731
                (lambda X, S: (X, np.full(len(S), r.cls), np.full(len(S), ri), S))(*st.select(r, rng, stride)))
            if r.cls != NORMAL and r.severity == sev[r.cls][h]:
                add(sh_p, [(0, r.length)], EV)
            else:
                add(tr_p, tr_r, st.hop); add(va_p, va_r, EV); add(te_p, te_r, EV)
        mk = lambda ps: Split(*(np.concatenate([p[i] for p in ps if len(p[1])]) for i in range(4)))  # noqa: E731
        tr, va, te, sh = mk(tr_p), mk(va_p), mk(te_p), mk(sh_p)
        rf = make_model("rf", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
        sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
        evaluate(rf, sm, Ref(sm, tr.X), ProtoGate(sm, tr.X, tr.y, C, k=K_PROTO, seed=f), va, te, sh,
                 {"scenario": "severity", "setting": f"held-out severity {h}", "fold": f}, rows)

# --------------------------------------------------------------------------- analog path (KAIST, deployed models)
if ds == "kaist":
    import pandas as _pd
    from tools.hil_models import _augment, train_models
    from tools.parse_hil import parse
    models = train_models(cfg, st, export_aug=False)
    tr, va, te = blocked_split(st, 0)
    e19 = cfg.path("results_dir") / "tables" / "e19_onchip_analog_windows.csv"
    for tag in ("clean", "aug"):
        if e19.exists():                 # every fold-0 test window through the board's DAC-ADC loop (E19)
            A = _pd.read_csv(e19)
            a = A[(A.model == tag) & (A["pass"] == 1)]
            an = Split(a[[f"f{j}" for j in range(19)]].to_numpy(np.float32), a.label.to_numpy(), a.rec.to_numpy(),
                       a.win.to_numpy())
        else:
            _, _, info = parse(cfg.path("results_dir") / "hw" / f"hil_{tag}.log")
            F = np.array(info["feat"])
            an = Split(F[:, 3:].astype(np.float32), F[:, 2].astype(int), F[:, 0].astype(int), F[:, 1].astype(int))
        rf, sm = models[tag]
        trx, trY = (tr.X, tr.y) if tag == "clean" else _augment(cfg, st, tr)
        evaluate(rf, sm, Ref(sm, trx), ProtoGate(sm, trx, trY, C, k=K_PROTO), va, te, an,
                 {"scenario": "analog", "setting": f"DAC-ADC, {tag} model", "fold": 0}, rows, w_list=(1, 2, 4))

df = pd.DataFrame(rows)
run.table(df, "raw", latex=False)
summ = df.groupby(["scenario", "setting", "score", "W"], sort=False)[
    ["auroc", "alarm_rate", "false_alarm_rate", "false_alarms_per_h", "delay_windows", "detected_recordings"]].mean().reset_index()
run.table(summ, "summary")
pd.set_option("display.width", 250)
print(summ[(summ.W == W_V) | (summ.scenario == "analog")].to_string(index=False))
if ver:
    v = pd.DataFrame(ver)
    run.table(v, "verify_raw", latex=False)
    vs = v.groupby("setting")[["alarm_before", "alarm_after"]].mean().reset_index()
    run.table(vs, "verify")
    print(vs.to_string(index=False))
run.done()
