"""E19 — the complete KESTREL node on the ESP32 (firmware/node_esp32), driven by the host that plays the gateway.

Parts (--part, run in this order by --part all):
  lifecycle : the monitor-alert-collect-adapt-verify loop on the chip for all six single-source load pairs (fold 0):
              gate on the source test block and on block 1 of the target load, q = 5 labelled support windows from
              block 1, on-chip softmax update and gate re-referencing, forest and tau' refreshed by the gateway,
              verification on blocks 2-5. Every decision is the board's; a float64 replay in Python checks each one.
  trace     : the lifecycle of Fig. 6a (healthy machine, load 0 -> 2 N.m) replayed in real time (timer at 25.6 kHz)
  analog    : DAC -> ADC loop: gate and on-chip update with every fold-0 test window of all 27 recordings, the
              noise-augmented and the clean model; a second pass through the loop scores the updated node
  equiv     : every non-overlapping KAIST window through the deployed fold-0 models (features, heads, gate)
Board logs: results/hw/e19_<part>.log; tables: results/tables/e19_onchip_<part>_*.csv
"""
import copy
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from fdedge.adaptation import pick_fewshot
from fdedge.data import FeatureStore, Split, block_bounds, blocked_split, class_weights, load_signal, loads_of, target_split
from fdedge.experiment import Run
from fdedge.export_c import global_scale, to_int16
from fdedge.features import extract_features
from fdedge.models import make_model
from fdedge.gate import ProtoGate
from tools.node_host import F_FEATURES, F_RESET, F_SUPPORT, Node

run = Run("e19_onchip", __doc__, extra_args=[
    (("--part",), {"default": "all", "choices": ["all", "lifecycle", "trace", "analog", "equiv"]}),
    (("--port",), {"default": "COM3"})])
cfg = run.cfg
ds = "kaist"
st = FeatureStore(cfg, ds, cfg.signal["window"])
C, N, K = len(cfg[ds]["classes"]), st.n, cfg.protocol["n_folds"]
FS, BAND = float(cfg[ds]["fs"]), cfg[ds]["shaft_band_hz"]
EV = int(N * cfg.signal["eval_hop_ratio"])
W_GATE = 10
loads = loads_of(cfg, ds)
SIG = {r.name: load_signal(r, cfg) for r in st.recs}
SCALE = global_scale(max(float(np.max(np.abs(SIG[r.name]))) for r in st.recs))
HW = cfg.path("results_dir") / "hw"
HW.mkdir(parents=True, exist_ok=True)
CPU = 240e6
UPD = cfg.adaptation["update"]                    # on-chip update: lr, epochs, l2_anchor
RF_W, GATE_W = cfg.adaptation["rf_support_weight"], cfg.adaptation["gate_support_weight"]


def board_adapt(node):
    return node.adapt(lr=UPD["lr"], epochs=UPD["epochs"], l2=UPD["l2_anchor"], weight=GATE_W, reref=True)


# --------------------------------------------------------------------------- helpers
def i16(split: Split, idx=None) -> np.ndarray:
    ii = range(len(split.y)) if idx is None else idx
    return np.stack([to_int16(np.asarray(SIG[st.recs[split.rec[i]].name][split.start[i]:split.start[i] + N]), SCALE)
                     for i in ii])


def qfeat(W: np.ndarray) -> np.ndarray:
    """Python features of the int16 windows the board receives."""
    return extract_features(W.astype(np.float32) * np.float32(SCALE), FS, band_hz=BAND)


def metas(split: Split, extra=0, idx=None):
    ii = list(range(len(split.y))) if idx is None else list(idx)
    out, prev = [], None
    for k, i in enumerate(ii):
        r = int(split.rec[i])
        out.append((r, k, int(split.y[i]), extra | (F_RESET if r != prev else 0)))
        prev = r
    return out


def part(load, first: bool) -> Split:
    """Block 1 (first) or blocks 2..5 of every recording at `load`, non-overlapping windows."""
    ps = []
    for ri, r in enumerate(st.recs):
        if r.load != load:
            continue
        b = block_bounds(r.length, K)
        X, S = st.select(r, [(b[0], b[1])] if first else [(b[1], b[K])], EV)
        ps.append((X, np.full(len(S), r.cls), np.full(len(S), ri), S))
    return Split(*(np.concatenate([p[i] for p in ps]) for i in range(4)))


def gate_s(rf, sm, gate, X):
    return gate.scores(rf, sm, X)


def trailing(rec: np.ndarray, s: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last w windows of the same recording (stream order); NaN until w are seen."""
    g = np.full(len(s), np.nan)
    for r in np.unique(rec):
        ii = np.nonzero(rec == r)[0]
        if len(ii) >= w:
            g[ii[w - 1:]] = np.convolve(s[ii], np.ones(w) / w, mode="valid")
    return g


def tau_of(*gs) -> float:
    v = np.concatenate([g[np.isfinite(g)] for g in gs])
    return float(np.quantile(v, 0.99))


def f1(y, p):
    return float(f1_score(y, p, labels=list(range(C)), average="macro", zero_division=0))


def board_g(rows):
    return np.array([r["g"] if r["g"] >= 0 else np.nan for r in rows])


def delays(rec, g, tau):
    out = []
    for r in np.unique(rec):
        ii = np.nonzero(rec == r)[0]
        hit = np.nonzero(g[ii] > tau)[0]
        out.append(hit[0] + 1 if len(hit) else np.nan)
    return np.array(out, dtype=float)


def softmax_from_board(sm, smw):
    m = copy.deepcopy(sm)
    D = len(sm.mu)
    m.W = smw[:C * D].reshape(C, D).astype(np.float64)
    m.b = smw[C * D:].astype(np.float64)
    return m


def retrain_rf(tr, Xs, ys, seed=0):
    w = np.concatenate([class_weights(tr.y, C), np.full(len(ys), RF_W)])
    return make_model("rf", C, cfg, seed=seed).fit(np.vstack([tr.X, Xs]), np.concatenate([tr.y, ys]), sample_weight=w)


def progress(k, n):
    print(f"    {k}/{n} windows", flush=True)


# --------------------------------------------------------------------------- lifecycle (six load pairs)
def do_lifecycle(node):
    rows_out, win_out = [], []
    for src in loads:
        for tgt in [L for L in loads if L != src]:
            t0 = time.time()
            print(f"  {src} -> {tgt}", flush=True)
            tr, va, te = blocked_split(st, 0, loads=[src])
            rf = make_model("rf", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
            sm = make_model("softmax", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
            gs = ProtoGate(sm, tr.X, tr.y, C)
            tau = tau_of(trailing(va.rec, gate_s(rf, sm, gs, va.X), W_GATE))
            pool, Q = part(tgt, True), part(tgt, False)
            sup, _ = pick_fewshot(pool, 5, C, seed=0)
            node.upload_model(rf, sm, gs, SCALE, tau, W_GATE)
            Wte, Wpool, WQ = i16(te), i16(pool), i16(Q)
            b_te = node.batch(Wte, metas(te))
            b_pool = node.batch(Wpool, metas(pool))
            b_sup = node.batch(Wpool[sup], metas(pool, F_SUPPORT | F_FEATURES, idx=sup))
            a = board_adapt(node)
            Fsup = np.stack([r["feat"] for r in b_sup])
            sma = softmax_from_board(sm, a["smw"])
            g2 = gs.rereferenced(sm, Fsup, pool.y[sup], GATE_W)  # the board's re-referenced gate
            g2.Pw = a["P"].reshape(gs.d, gs.d)
            rfa = retrain_rf(tr, Fsup, pool.y[sup])
            tau2 = tau_of(trailing(va.rec, gate_s(rfa, sma, g2, va.X), W_GATE),
                          trailing(pool.rec, gate_s(rfa, sma, g2, pool.X), W_GATE))
            node.upload_forest(rfa)
            node.set_tau(tau2)
            b_q = node.batch(WQ, metas(Q), progress=progress)

            # ---- board metrics
            g_te, g_pool, g_q = board_g(b_te), board_g(b_pool), board_g(b_q)
            ok_i, ok_s = np.isfinite(g_te), np.isfinite(g_pool)
            auc = roc_auc_score(np.r_[np.zeros(ok_i.sum()), np.ones(ok_s.sum())], np.r_[g_te[ok_i], g_pool[ok_s]])
            dl = delays(pool.rec, g_pool, tau)
            rf_q, sm_q = np.array([r["rf"] for r in b_q]), np.array([r["sm"] for r in b_q])
            rf_b, sm_b = np.array([r["rf"] for r in b_pool]), np.array([r["sm"] for r in b_pool])

            # ---- float64 replay in Python (independent adaptation on the Python features of the same windows)
            Fte, Fpool, FQ = qfeat(Wte), qfeat(Wpool), qfeat(WQ)
            py_g_te = trailing(te.rec, gate_s(rf, sm, gs, Fte), W_GATE)
            py_g_pool = trailing(pool.rec, gate_s(rf, sm, gs, Fpool), W_GATE)
            sma_py = copy.deepcopy(sm).finetune(Fpool[sup], pool.y[sup], **UPD)
            rfa_py = retrain_rf(tr, Fpool[sup], pool.y[sup])
            gs2_py = gs.rereferenced(sm, Fpool[sup], pool.y[sup], GATE_W)
            tau2_py = tau_of(trailing(va.rec, gate_s(rfa_py, sma_py, gs2_py, va.X), W_GATE),
                             trailing(pool.rec, gate_s(rfa_py, sma_py, gs2_py, pool.X), W_GATE))
            py_g_q = trailing(Q.rec, gate_s(rfa_py, sma_py, gs2_py, FQ), W_GATE)
            py_rf_q, py_sm_q = rfa_py.predict(FQ), sma_py.predict(FQ)
            alarm_te, alarm_pool, alarm_q = g_te > tau, g_pool > tau, g_q > tau2
            py_alarm = np.r_[py_g_te > tau, py_g_pool > tau, py_g_q > tau2_py]
            board_alarm = np.r_[alarm_te, alarm_pool, alarm_q]
            row = {
                "source": src, "target": tgt, "windows": len(Wte) + len(Wpool) + len(sup) + len(WQ),
                "tau": tau, "tau_after": tau2, "tau_after_python": tau2_py,
                "auroc": auc, "alarm_rate_shifted": float(np.nanmean(np.where(ok_s, alarm_pool, np.nan))),
                "false_alarm_rate": float(np.nanmean(np.where(ok_i, alarm_te, np.nan))),
                "delay_windows_median": float(np.nanmedian(dl)), "detected_recordings": float(np.mean(np.isfinite(dl))),
                "f1_rf_block1_before": f1(pool.y, rf_b), "f1_sm_block1_before": f1(pool.y, sm_b),
                "f1_rf_after": f1(Q.y, rf_q), "f1_sm_after": f1(Q.y, sm_q),
                "f1_rf_after_python": f1(Q.y, py_rf_q), "f1_sm_after_python": f1(Q.y, py_sm_q),
                "agree_rf_after": float(np.mean(rf_q == py_rf_q)), "agree_sm_after": float(np.mean(sm_q == py_sm_q)),
                "alarm_rate_after": float(np.nanmean(np.where(np.isfinite(g_q), alarm_q, np.nan))),
                "agree_alarm": float(np.mean(board_alarm == py_alarm)),
                "update_ms": a["cycles_update"] / CPU * 1e3, "reref_ms": a["cycles_reref"] / CPU * 1e3,
                "support": a["n"], "sec": time.time() - t0}
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            rows_out.append(row)
            for name, sp, bb, g in (("source_test", te, b_te, g_te), ("target_block1", pool, b_pool, g_pool),
                                    ("target_blocks2_5", Q, b_q, g_q)):
                for i, r in enumerate(bb):
                    win_out.append({"source": src, "target": tgt, "set": name, "rec": int(sp.rec[i]), "start": int(sp.start[i]),
                                    "label": int(sp.y[i]), "rf": r["rf"], "sm": r["sm"], "s": r["s"], "g": g[i],
                                    "alarm": r["alarm"], "us": r["us"], "c_cls": r["c_cls"], "c_gate": r["c_gate"]})
            pd.DataFrame(rows_out).to_csv(run.tab_dir / f"{run.tag}_lifecycle_pairs.csv", index=False)
    pd.DataFrame(win_out).to_csv(run.tab_dir / f"{run.tag}_lifecycle_windows.csv", index=False)
    return pd.DataFrame(rows_out)


# --------------------------------------------------------------------------- real-time trace (Fig. 6a)
def rec_split(name, lo_blk, hi_blk, limit=None) -> Split:
    ri = next(i for i, r in enumerate(st.recs) if r.name == name)
    r = st.recs[ri]
    b = block_bounds(r.length, K)
    X, S = st.select(r, [(b[lo_blk], b[hi_blk])], EV)
    if limit:
        X, S = X[:limit], S[:limit]
    return Split(X, np.full(len(S), r.cls), np.full(len(S), ri), S)


def do_trace(node):
    src, tgt = "0Nm", "2Nm"
    tr, va, te = blocked_split(st, 0, loads=[src])
    rf = make_model("rf", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
    sm = make_model("softmax", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
    gs = ProtoGate(sm, tr.X, tr.y, C)
    tau = tau_of(trailing(va.rec, gate_s(rf, sm, gs, va.X), W_GATE))
    pool = part(tgt, True)
    sup, _ = pick_fewshot(pool, 5, C, seed=0)
    h1, h2 = rec_split(f"{src}_Normal", 0, 1), rec_split(f"{tgt}_Normal", 0, 1)
    h3 = rec_split(f"{tgt}_Normal", 1, 5, limit=120)
    node.upload_model(rf, sm, gs, SCALE, tau, W_GATE)
    seq1 = np.vstack([i16(h1), i16(h2)])            # one continuous stream: the load changes mid-stream
    m1 = [(0, k, 0, F_RESET if k == 0 else 0) for k in range(len(seq1))]
    t0 = time.time()
    r1, info1 = node.stream(seq1, m1)
    t_stream1 = time.time() - t0
    b_sup = node.batch(i16(pool, sup), metas(pool, F_SUPPORT | F_FEATURES, idx=sup))
    a = board_adapt(node)
    sma = softmax_from_board(sm, a["smw"])
    Fsup = np.stack([r["feat"] for r in b_sup])
    g2 = gs.rereferenced(sm, Fsup, pool.y[sup], GATE_W)
    g2.Pw = a["P"].reshape(gs.d, gs.d)
    rfa = retrain_rf(tr, Fsup, pool.y[sup])
    tau2 = tau_of(trailing(va.rec, gate_s(rfa, sma, g2, va.X), W_GATE),
                  trailing(pool.rec, gate_s(rfa, sma, g2, pool.X), W_GATE))
    node.upload_forest(rfa)
    node.set_tau(tau2)
    seq2 = i16(h3)
    t0 = time.time()
    r2, info2 = node.stream(seq2, [(1, k, 0, F_RESET if k == 0 else 0) for k in range(len(seq2))])
    t_stream2 = time.time() - t0
    sec = N / FS
    rows = [{"phase": "before", "k": i, "t": (i + 1) * sec, "g": r["g"], "s": r["s"], "alarm": r["alarm"], "rf": r["rf"],
             "sm": r["sm"], "us": r["us"]} for i, r in enumerate(r1)]
    t_adapt0 = len(r1) * sec
    rows += [{"phase": "after", "k": i, "t": t_adapt0 + 4.0 + (i + 1) * sec, "g": r["g"], "s": r["s"], "alarm": r["alarm"],
              "rf": r["rf"], "sm": r["sm"], "us": r["us"]} for i, r in enumerate(r2)]
    df = pd.DataFrame(rows)
    t_change = len(h1) * sec
    before = df[df.phase == "before"]
    hit = before[(before.t > t_change) & (before.alarm == 1)]
    meta = {"tau": tau, "tau_after": tau2, "t_change": t_change, "t_alarm": float(hit.t.iloc[0]) if len(hit) else np.nan,
            "t_adapt0": t_adapt0, "t_adapt1": t_adapt0 + 4.0, "update_ms": a["cycles_update"] / CPU * 1e3,
            "reref_ms": a["cycles_reref"] / CPU * 1e3, "overruns": info1["overruns"] + info2["overruns"],
            "underruns": info1["underruns"] + info2["underruns"], "windows": len(r1) + len(r2),
            "wall_s_stream1": t_stream1, "signal_s_stream1": len(seq1) * sec, "wall_s_stream2": t_stream2,
            "signal_s_stream2": len(seq2) * sec, "us_max": float(df.us.max()), "us_median": float(df.us.median()),
            "false_alarms_after": int((df[df.phase == "after"].alarm == 1).sum())}
    df.to_csv(run.tab_dir / f"{run.tag}_trace.csv", index=False)
    pd.DataFrame([meta]).to_csv(run.tab_dir / f"{run.tag}_trace_meta.csv", index=False)
    print(meta, flush=True)


# --------------------------------------------------------------------------- analog loop
def do_analog(node):
    from tools.hil_models import _augment, train_models
    models = train_models(cfg, st, export_aug=False)
    tr, va, te = blocked_split(st, 0)
    cal = node.calibrate()
    (HW / "e19_analog_calibration.log").write_text("\n".join(cal) + "\n", encoding="utf-8")
    W_A = 4
    out, wins = [], []
    for tag in ("aug", "clean"):
        rf, sm = models[tag]
        Xtr, ytr = (_augment(cfg, st, tr) if tag == "aug" else (tr.X, tr.y))
        gs = ProtoGate(sm, Xtr, ytr, C)
        tau = tau_of(trailing(va.rec, gate_s(rf, sm, gs, va.X), W_A))
        node.upload_model(rf, sm, gs, SCALE, tau, W_A)
        # digital in-distribution reference: the same fold-0 test windows, batch
        b_dig = node.batch(i16(te), metas(te), progress=progress)
        g_dig = board_g(b_dig)

        def play(pass_id, flags):
            rows = []
            for ri, r in enumerate(st.recs):
                b = block_bounds(r.length, K)
                lo, hi = b[0], b[1]                     # fold-0 test block
                n_win = (hi - lo) // N
                q = to_int16(np.asarray(SIG[r.name][lo:lo + n_win * N]), SCALE)
                rr, info = node.analog(q, ri, r.cls, flags, peak=int(np.max(np.abs(q))), span=80.0)
                for x in rr:
                    x.update({"pass": pass_id, "rec": ri, "label": r.cls, "overruns": info["overruns"]})
                rows += rr
                print(f"    {tag} pass {pass_id}: {r.name} {len(rr)} windows", flush=True)
            return rows

        node.clear_support()
        p1 = play(1, F_RESET | F_SUPPORT | F_FEATURES)
        a = board_adapt(node)
        sup = [x for x in p1 if x["win"] == 0]
        sma = softmax_from_board(sm, a["smw"])
        Fsup = np.stack([x["feat"] for x in sup])
        ysup = np.array([x["label"] for x in sup])
        g2 = gs.rereferenced(sm, Fsup, ysup, GATE_W)
        g2.Pw = a["P"].reshape(gs.d, gs.d)
        w = np.concatenate([class_weights(ytr, C), np.full(len(ysup), RF_W)])
        rfa = make_model("rf", C, cfg, seed=0).fit(np.vstack([Xtr, Fsup]), np.concatenate([ytr, ysup]), sample_weight=w)
        tau2 = tau_of(trailing(va.rec, gate_s(rfa, sma, g2, va.X), W_A))
        node.upload_forest(rfa)
        node.set_tau(tau2)
        p2 = play(2, F_RESET | F_FEATURES)
        # ---- metrics: pass 1 before adaptation (all windows), pass 2 after adaptation (first window of each
        # recording excluded: it replays the support segment)
        y1 = np.array([x["label"] for x in p1])
        q2 = [x for x in p2 if x["win"] != 0]
        y2 = np.array([x["label"] for x in q2])
        g1 = board_g(p1)
        ok_d, ok_a = np.isfinite(g_dig), np.isfinite(g1)
        auc = roc_auc_score(np.r_[np.zeros(ok_d.sum()), np.ones(ok_a.sum())], np.r_[g_dig[ok_d], g1[ok_a]])
        rec1 = np.array([x["rec"] for x in p1])
        dl = delays(rec1, g1, tau)
        # Python replay of the update on the board's own analog features
        F1 = np.stack([x["feat"] for x in p1])
        sma_py = copy.deepcopy(sm).finetune(Fsup, ysup, **UPD)
        F2 = np.stack([x["feat"] for x in q2])
        g2 = board_g(q2)
        row = {"model": tag, "windows_pass1": len(p1), "windows_pass2_scored": len(q2), "support": a["n"],
               "overruns_max_per_recording": int(max(x["overruns"] for x in p1 + p2)),
               "tau": tau, "tau_after": tau2, "auroc_digital_vs_analog": auc,
               "alarm_rate_analog": float(np.nanmean(np.where(ok_a, g1 > tau, np.nan))),
               "false_alarm_rate_digital": float(np.nanmean(np.where(ok_d, g_dig > tau, np.nan))),
               "detected_recordings": float(np.mean(np.isfinite(dl))), "delay_windows_median": float(np.nanmedian(dl)),
               "f1_rf_digital": f1(te.y, np.array([x["rf"] for x in b_dig])),
               "f1_sm_digital": f1(te.y, np.array([x["sm"] for x in b_dig])),
               "f1_rf_before": f1(y1, np.array([x["rf"] for x in p1])), "f1_sm_before": f1(y1, np.array([x["sm"] for x in p1])),
               "f1_rf_after": f1(y2, np.array([x["rf"] for x in q2])), "f1_sm_after": f1(y2, np.array([x["sm"] for x in q2])),
               "agree_sm_after_python": float(np.mean(sma_py.predict(F2) == np.array([x["sm"] for x in q2]))),
               "agree_rf_before_python": float(np.mean(rf.predict(F1) == np.array([x["rf"] for x in p1]))),
               "alarm_rate_after": float(np.nanmean(np.where(np.isfinite(g2), g2 > tau2, np.nan))),
               "update_ms": a["cycles_update"] / CPU * 1e3, "reref_ms": a["cycles_reref"] / CPU * 1e3}
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        out.append(row)
        for x in p1 + p2:
            wins.append({k: x[k] for k in ("pass", "rec", "win", "label", "rf", "sm", "s", "g", "alarm", "us")} |
                        {"model": tag, **{f"f{j}": float(v) for j, v in enumerate(x["feat"])}})
        pd.DataFrame(out).to_csv(run.tab_dir / f"{run.tag}_analog.csv", index=False)
    pd.DataFrame(wins).to_csv(run.tab_dir / f"{run.tag}_analog_windows.csv", index=False)


# --------------------------------------------------------------------------- equivalence on every window
def do_equiv(node):
    tr, va, te = blocked_split(st, 0)
    rf = make_model("rf", C, cfg, seed=0).fit(tr.X, tr.y)
    sm = make_model("softmax", C, cfg, seed=0).fit(tr.X, tr.y, va.X, va.y)
    gs = ProtoGate(sm, tr.X, tr.y, C)
    tau = tau_of(trailing(va.rec, gate_s(rf, sm, gs, va.X), W_GATE))
    node.upload_model(rf, sm, gs, SCALE, tau, W_GATE)
    allw = target_split(st, loads)
    test_keys = set(zip(te.rec.tolist(), te.start.tolist()))
    rows, n = [], len(allw.y)
    for r in np.unique(allw.rec):                   # per recording: bounded memory
        ii = np.nonzero(allw.rec == r)[0]
        W = i16(allw, ii)
        b = node.batch(W, metas(allw, F_FEATURES, idx=ii))
        F = qfeat(W)
        Fb = np.stack([x["feat"] for x in b])
        s_py = gate_s(rf, sm, gs, F)
        g_py = trailing(np.zeros(len(ii)), s_py, W_GATE)
        rel = np.max(np.abs(Fb - F) / (np.abs(F) + 1e-3 * np.abs(F).max(axis=0)), axis=1)
        py_rf, py_sm = rf.predict(F), sm.predict(F)
        for k, i in enumerate(ii):
            rows.append({"rec": int(r), "start": int(allw.start[i]), "label": int(allw.y[i]),
                         "test": (int(r), int(allw.start[i])) in test_keys, "rf": b[k]["rf"], "sm": b[k]["sm"],
                         "py_rf": int(py_rf[k]), "py_sm": int(py_sm[k]),
                         "s": b[k]["s"], "py_s": float(s_py[k]), "g": b[k]["g"], "py_g": float(g_py[k]),
                         "alarm": b[k]["alarm"], "feat_rel_max": float(rel[k]), "us": b[k]["us"],
                         "c_feat": b[k]["c_feat"], "c_cls": b[k]["c_cls"], "c_gate": b[k]["c_gate"]})
        print(f"    rec {r}: {len(rows)}/{n} windows", flush=True)
        pd.DataFrame(rows).to_csv(run.tab_dir / f"{run.tag}_equiv_windows.csv", index=False)
    df = pd.DataFrame(rows)
    tst = df[df.test]
    py_alarm = df.py_g.fillna(-1) > tau
    summ = {"windows": len(df), "recordings": int(df.rec.nunique()), "agree_rf": float((df.rf == df.py_rf).mean()),
            "agree_sm": float((df.sm == df.py_sm).mean()), "agree_alarm": float((df.alarm == py_alarm).mean()),
            "s_abs_diff_max": float((df.s - df.py_s).abs().max()), "feat_rel_diff_max": float(df.feat_rel_max.max()),
            "feat_rel_diff_p99": float(df.feat_rel_max.quantile(0.99)),
            "test_windows": len(tst), "f1_rf_test": f1(tst.label, tst.rf), "f1_sm_test": f1(tst.label, tst.sm),
            "us_median": float(df.us.median()), "us_max": float(df.us.max()),
            "cls_us_median": float(df.c_cls.median() / CPU * 1e6), "gate_us_median": float(df.c_gate.median() / CPU * 1e6),
            "feat_ms_median": float(df.c_feat.median() / CPU * 1e3)}
    pd.DataFrame([summ]).to_csv(run.tab_dir / f"{run.tag}_equiv_summary.csv", index=False)
    print(summ, flush=True)


# --------------------------------------------------------------------------- main
parts = ["lifecycle", "trace", "analog", "equiv"] if run.args.part == "all" else [run.args.part]
for p in parts:
    print(f"[{run.tag}] part {p}", flush=True)
    with open(HW / f"e19_{p}.log", "w", encoding="utf-8") as log, Node(run.args.port, log=log) as node:
        log.write(node.hello + "\n")
        {"lifecycle": do_lifecycle, "trace": do_trace, "analog": do_analog, "equiv": do_equiv}[p](node)
run.done()
