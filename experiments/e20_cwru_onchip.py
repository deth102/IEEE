"""E20 — KESTREL with bearing orders (F22) on the ESP32: CWRU, firmware/node_esp32 env node_cwru.

  --part export : fold-0 models (all loads) -> firmware/models/cwru_n8192_env (constants of the squared-envelope
                  stage and golden windows), then build with  pio run -e node_cwru -t upload
  --part equiv  : every non-overlapping CWRU window through the deployed models on the board: features (incl. the
                  bearing orders), heads and gate vs. the Python reference, stage times of the envelope stage
  --part lifecycle : the monitor-alert-collect-adapt-verify loop on the chip for all twelve single-source load pairs
                  (fold 0, q = 5 support windows from block 1, verification on blocks 2-5)
Run with --config configs/cwru_env.yaml --dataset cwru.
"""
import copy
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from fdedge.adaptation import pick_fewshot
from fdedge.data import FeatureStore, Split, block_bounds, blocked_split, class_weights, load_signal, loads_of, target_split
from fdedge.experiment import Run
from fdedge.export_c import export_model, global_scale, to_int16
from fdedge.features import extract_features
from fdedge.models import make_model
from fdedge.gate import ProtoGate
from tools.node_host import F_FEATURES, F_RESET, F_SUPPORT, Node

run = Run("e20_cwru_onchip", __doc__, extra_args=[
    (("--part",), {"default": "equiv", "choices": ["export", "equiv", "lifecycle", "all"]}),
    (("--port",), {"default": "COM3"})])
cfg, ds = run.cfg, run.dataset
assert ds == "cwru" and cfg.get("use_envelope"), "run with --dataset cwru --config configs/cwru_env.yaml"
st = FeatureStore(cfg, ds, cfg.signal["window"])
ENV = dict(st.env)
C, N, K = len(cfg[ds]["classes"]), st.n, cfg.protocol["n_folds"]
FS, BAND = float(cfg[ds]["fs"]), cfg[ds]["shaft_band_hz"]
EV = int(N * cfg.signal["eval_hop_ratio"])
W_GATE = 3                                   # CWRU: shorter recordings (as in Table III)
loads = loads_of(cfg, ds)
SIG = {r.name: load_signal(r, cfg) for r in st.recs}
SCALE = global_scale(max(float(np.max(np.abs(SIG[r.name]))) for r in st.recs))
HW = cfg.path("results_dir") / "hw"
CPU = 240e6


def i16(split, idx=None):
    ii = range(len(split.y)) if idx is None else idx
    return np.stack([to_int16(np.asarray(SIG[st.recs[split.rec[i]].name][split.start[i]:split.start[i] + N]), SCALE)
                     for i in ii])


def qfeat(W):
    return extract_features(W.astype(np.float32) * np.float32(SCALE), FS, band_hz=BAND, env=ENV)


def metas(split, extra=0, idx=None):
    ii = list(range(len(split.y))) if idx is None else list(idx)
    out, prev = [], None
    for k, i in enumerate(ii):
        r = int(split.rec[i])
        out.append((r, k, int(split.y[i]), extra | (F_RESET if r != prev else 0)))
        prev = r
    return out


def gate_s(rf, sm, gate, X):
    return gate.scores(rf, sm, X)


def trailing(rec, s, w):
    g = np.full(len(s), np.nan)
    for r in np.unique(rec):
        ii = np.nonzero(rec == r)[0]
        if len(ii) >= w:
            g[ii[w - 1:]] = np.convolve(s[ii], np.ones(w) / w, mode="valid")
    return g


def tau_of(*gs):
    v = np.concatenate([g[np.isfinite(g)] for g in gs])
    return float(np.quantile(v, 0.99))


def f1(y, p):
    return float(f1_score(y, p, labels=list(range(C)), average="macro", zero_division=0))


def board_g(rows):
    return np.array([r["g"] if r["g"] >= 0 else np.nan for r in rows])


def part(load, first):
    ps = []
    for ri, r in enumerate(st.recs):
        if r.load != load:
            continue
        b = block_bounds(r.length, K)
        X, S = st.select(r, [(b[0], b[1])] if first else [(b[1], b[K])], EV)
        ps.append((X, np.full(len(S), r.cls), np.full(len(S), ri), S))
    return Split(*(np.concatenate([p[i] for p in ps]) for i in range(4)))


def do_export():
    tr, va, te = blocked_split(st, 0)
    rf = make_model("rf", C, cfg, seed=0).fit(tr.X, tr.y)
    sm = make_model("softmax", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
    rng = np.random.default_rng(N)
    idx = np.array([rng.choice(np.nonzero(te.y == c)[0]) for c in range(C)])
    info = export_model(cfg.path("firmware_models_dir") / f"cwru_n{N}_env", n=N, fs=FS, band_hz=BAND,
                        class_names=cfg[ds]["classes"], rf=rf, softmax=sm, X_train=tr.X, test_windows_i16=i16(te, idx),
                        scale=SCALE, test_labels=te.y[idx], env=ENV,
                        extra_info={"trained_on": "blocked fold 0 train, all loads, F22"})
    print(info)


def do_equiv(node):
    tr, va, te = blocked_split(st, 0)
    rf = make_model("rf", C, cfg, seed=0).fit(tr.X, tr.y)
    sm = make_model("softmax", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
    gs = ProtoGate(sm, tr.X, tr.y, C)
    tau = tau_of(trailing(va.rec, gate_s(rf, sm, gs, va.X), W_GATE))
    node.upload_model(rf, sm, gs, SCALE, tau, W_GATE)
    allw = target_split(st, loads)
    test_keys = set(zip(te.rec.tolist(), te.start.tolist()))
    rows = []
    for r in np.unique(allw.rec):
        ii = np.nonzero(allw.rec == r)[0]
        W = i16(allw, ii)
        b = node.batch(W, metas(allw, F_FEATURES, idx=ii))
        F = qfeat(W)
        Fb = np.stack([x["feat"] for x in b])
        rel = np.max(np.abs(Fb - F) / (np.abs(F) + 1e-3 * np.abs(F).max(axis=0)), axis=1)
        env_abs = np.max(np.abs(Fb[:, 19:] - F[:, 19:]), axis=1)
        py_rf, py_sm = rf.predict(F), sm.predict(F)
        s_py = gate_s(rf, sm, gs, F)
        for k, i in enumerate(ii):
            rows.append({"rec": int(r), "start": int(allw.start[i]), "label": int(allw.y[i]),
                         "test": (int(r), int(allw.start[i])) in test_keys, "rf": b[k]["rf"], "sm": b[k]["sm"],
                         "py_rf": int(py_rf[k]), "py_sm": int(py_sm[k]), "s": b[k]["s"], "py_s": float(s_py[k]),
                         "feat_rel_max": float(rel[k]), "env_abs_max": float(env_abs[k]), "us": b[k]["us"],
                         "c_feat": b[k]["c_feat"], "c_env": b[k]["c_env"], "c_cls": b[k]["c_cls"], "c_gate": b[k]["c_gate"]})
        print(f"    rec {r}: {len(rows)}/{len(allw.y)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run.tab_dir / f"{run.tag}_equiv_windows.csv", index=False)
    tst = df[df.test]
    summ = {"windows": len(df), "agree_rf": float((df.rf == df.py_rf).mean()), "agree_sm": float((df.sm == df.py_sm).mean()),
            "feat_rel_diff_max": float(df.feat_rel_max.max()), "env_abs_diff_max": float(df.env_abs_max.max()),
            "s_abs_diff_max": float((df.s - df.py_s).abs().max()), "test_windows": len(tst),
            "f1_rf_test": f1(tst.label, tst.rf), "f1_sm_test": f1(tst.label, tst.sm),
            "feat_ms_median": float(df.c_feat.median() / CPU * 1e3), "env_ms_median": float(df.c_env.median() / CPU * 1e3),
            "env_ms_max": float(df.c_env.max() / CPU * 1e3), "us_max": float(df.us.max()),
            "cls_us_median": float(df.c_cls.median() / CPU * 1e6), "gate_us_median": float(df.c_gate.median() / CPU * 1e6)}
    pd.DataFrame([summ]).to_csv(run.tab_dir / f"{run.tag}_equiv_summary.csv", index=False)
    print(summ, flush=True)


def do_lifecycle(node):
    out = []
    for src in loads:
        for tgt in [L for L in loads if L != src]:
            t0 = time.time()
            tr, va, te = blocked_split(st, 0, loads=[src])
            rf = make_model("rf", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
            sm = make_model("softmax", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
            gs = ProtoGate(sm, tr.X, tr.y, C)
            tau = tau_of(trailing(va.rec, gate_s(rf, sm, gs, va.X), W_GATE))
            pool, Q = part(tgt, True), part(tgt, False)
            sup, _ = pick_fewshot(pool, 5, C, seed=0)
            node.upload_model(rf, sm, gs, SCALE, tau, W_GATE)
            Wte, Wpool, WQ = i16(te), i16(pool), i16(Q)
            b_te, b_pool = node.batch(Wte, metas(te)), node.batch(Wpool, metas(pool))
            b_sup = node.batch(Wpool[sup], metas(pool, F_SUPPORT | F_FEATURES, idx=sup))
            b_q0 = node.batch(WQ, metas(Q))                  # verification windows before the update
            a = node.adapt(lr=0.1, epochs=200, l2=1e-3, weight=50.0, reref=True)
            Fsup = np.stack([r["feat"] for r in b_sup])
            sma = copy.deepcopy(sm)
            D = len(sm.mu)
            sma.W, sma.b = a["smw"][:C * D].reshape(C, D), a["smw"][C * D:]
            g2 = gs.rereferenced(sm, Fsup, pool.y[sup], 50)
            g2.Pw = a["P"].reshape(D, D)
            w = np.concatenate([class_weights(tr.y, C), np.full(len(sup), 50.0)])
            rfa = make_model("rf", C, cfg, seed=0).fit(np.vstack([tr.X, Fsup]), np.concatenate([tr.y, pool.y[sup]]),
                                                        sample_weight=w)
            tau2 = tau_of(trailing(va.rec, gate_s(rfa, sma, g2, va.X), W_GATE),
                          trailing(pool.rec, gate_s(rfa, sma, g2, pool.X), W_GATE))
            node.upload_forest(rfa)
            node.set_tau(tau2)
            b_q = node.batch(WQ, metas(Q))
            g_te, g_pool, g_q = board_g(b_te), board_g(b_pool), board_g(b_q)
            ok_i, ok_s = np.isfinite(g_te), np.isfinite(g_pool)
            auc = roc_auc_score(np.r_[np.zeros(ok_i.sum()), np.ones(ok_s.sum())], np.r_[g_te[ok_i], g_pool[ok_s]])
            FQ = qfeat(WQ)
            Fpool = qfeat(Wpool)
            sma_py = copy.deepcopy(sm).finetune(Fpool[sup], pool.y[sup], lr=0.1, epochs=200, l2_anchor=1e-3)
            row = {"source": src, "target": tgt, "auroc": auc,
                   "alarm_rate_shifted": float(np.nanmean(np.where(ok_s, g_pool > tau, np.nan))),
                   "false_alarm_rate": float(np.nanmean(np.where(ok_i, g_te > tau, np.nan))),
                   "f1_rf_before": f1(Q.y, [r["rf"] for r in b_q0]), "f1_sm_before": f1(Q.y, [r["sm"] for r in b_q0]),
                   "f1_rf_after": f1(Q.y, [r["rf"] for r in b_q]), "f1_sm_after": f1(Q.y, [r["sm"] for r in b_q]),
                   "agree_sm_after_python": float(np.mean(sma_py.predict(FQ) == np.array([r["sm"] for r in b_q]))),
                   "alarm_rate_after": float(np.nanmean(np.where(np.isfinite(g_q), g_q > tau2, np.nan))),
                   "update_ms": a["cycles_update"] / CPU * 1e3, "reref_ms": a["cycles_reref"] / CPU * 1e3,
                   "windows": len(Wte) + len(Wpool) + len(sup) + 2 * len(WQ), "sec": time.time() - t0}
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            out.append(row)
            pd.DataFrame(out).to_csv(run.tab_dir / f"{run.tag}_lifecycle_pairs.csv", index=False)


parts = ["equiv", "lifecycle"] if run.args.part == "all" else [run.args.part]
for p in parts:
    if p == "export":
        do_export()
        continue
    with open(HW / f"e20_cwru_{p}.log", "w", encoding="utf-8") as log, Node(run.args.port, log=log) as node:
        log.write(node.hello + "\n")
        {"equiv": do_equiv, "lifecycle": do_lifecycle}[p](node)
run.done()
