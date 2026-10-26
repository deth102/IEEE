"""E22 — the anchored on-chip update against other few-shot updates, same protocol as E14 (b).

Support: k labelled windows per class from block 1 of the target recordings (k = 1, 5; three draws); query: blocks
2..5. Every method starts from the same source model and sees the same support windows.
  softmax head (all feasible on the MCU):
    anchored update (KESTREL R)  full-batch gradient descent with the L2-SP anchor (cfg adaptation.update)
    unanchored update            the same without the anchor (lambda = 0)
    bias-only update             only b is updated (cf. TinyTL), same step and epochs
    nearest class mean           prototypes of the standardized support windows (cf. prototypical networks)
  CNN baselines (--cnn): last-layer fine-tuning of EdgeLite and MobileLite trained on the source load
    (Adam, 200 full-batch steps on the support set), single-source, 3 folds
"""
import copy

import numpy as np
import pandas as pd

from fdedge.adaptation import pick_fewshot
from fdedge.data import FeatureStore, Split, block_bounds, blocked_split, loads_of
from fdedge.evaluation import summarize
from fdedge.experiment import Run
from fdedge.metrics import scores
from fdedge.models import make_model

run = Run("e22_adaptation_baselines", __doc__, extra_args=[(("--cnn",), {"action": "store_true"})])
cfg, ds = run.cfg, run.dataset
C = len(cfg[ds]["classes"])
N = cfg.signal["window"]
K = cfg.protocol["n_folds"]
EV = int(N * cfg.signal["eval_hop_ratio"])
loads = loads_of(cfg, ds)
UPD = cfg.adaptation["update"]                    # lr, epochs, l2_anchor of the on-chip update


def target_blocks(st, tgt):
    pool_p, q_p = [], []
    for ri, r in enumerate(st.recs):
        if r.load != tgt:
            continue
        b = block_bounds(r.length, K)
        for lst, rng in ((pool_p, [(b[0], b[1])]), (q_p, [(b[1], b[K])])):
            X, S = st.select(r, rng, EV)
            lst.append((X, np.full(len(S), r.cls), np.full(len(S), ri), S))
    mk = lambda ps: Split(*(np.concatenate([p[i] for p in ps]) for i in range(4)))  # noqa: E731
    return mk(pool_p), mk(q_p)


def bias_only(sm, X, y, lr=0.1, epochs=200):
    m = copy.deepcopy(sm)
    Z = m._z(X)
    Y = np.eye(C)[y]
    for _ in range(epochs):
        logits = Z @ m.W.T + m.b
        P = np.exp(logits - logits.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        m.b = m.b - lr * ((P - Y) / len(y)).sum(0)
    return m


class NearestMean:
    def __init__(self, sm, X, y):
        self.sm = sm
        Z = sm._z(X)
        self.mu = np.stack([Z[y == c].mean(0) for c in range(C)])

    def predict(self, X):
        Z = self.sm._z(X)
        return np.argmin(((Z[:, None, :] - self.mu[None]) ** 2).sum(-1), axis=1)


rows = []
if not run.args.cnn:
    st = FeatureStore(cfg, ds, N)
    for multi in (False, True):
        sources = [[L for L in loads if L != t] for t in loads] if multi else [[L] for L in loads]
        for src in sources:
            for f in run.folds:
                tr, va, _ = blocked_split(st, f, loads=src)
                sm = make_model("softmax", C, cfg, seed=f).fit(tr.X, tr.y, va.X, va.y)
                for tgt in [L for L in loads if L not in src]:
                    pool, Q = target_blocks(st, tgt)
                    base = {"setting": "LOLO" if multi else "single-source", "source": "+".join(src), "target": tgt,
                            "fold": f}
                    rows.append({**base, "method": "none", "k": 0, **scores(Q.y, sm.predict(Q.X), C)})
                    for k in (1, 5):
                        for s in range(3):
                            sup, _ = pick_fewshot(pool, k, C, seed=1000 * f + s)
                            Xs, ys = pool.X[sup], pool.y[sup]
                            ms = {"anchored update (KESTREL R)": copy.deepcopy(sm).finetune(Xs, ys, **UPD),
                                  "unanchored update": copy.deepcopy(sm).finetune(Xs, ys, UPD["lr"], UPD["epochs"], 0.0),
                                  "bias-only update": bias_only(sm, Xs, ys, UPD["lr"], UPD["epochs"]),
                                  "nearest class mean": NearestMean(sm, Xs, ys)}
                            for name, m in ms.items():
                                rows.append({**base, "method": name, "k": k, "seed": s, **scores(Q.y, m.predict(Q.X), C)})
            print(f"  {'LOLO' if multi else 'single'} {src} done", flush=True)
    df = pd.DataFrame(rows)
    run.table(df, "raw", latex=False)
    s = summarize(df, ["setting", "method", "k"])
    run.table(s, "summary")
    print(s.to_string(index=False))
else:
    import torch
    from fdedge.cnn import predict_cnn, train_cnn
    stc = FeatureStore(cfg, ds, N, kind="cnn")
    cc = cfg["cnn"]
    for f in run.folds[:3]:
        for src in loads:
            tr, va, _ = blocked_split(stc, f, loads=[src])
            for name in ("edgelite", "mobilelite"):
                model = train_cnn(name, C, tr.X, tr.y, va.X, va.y, epochs=cc["epochs"], batch=cc["batch_size"],
                                  lr=cc["lr"], patience=cc["patience"], seed=f)
                for tgt in [L for L in loads if L != src]:
                    pool, Q = target_blocks(stc, tgt)
                    base = {"setting": "single-source", "source": src, "target": tgt, "fold": f, "model": name}
                    rows.append({**base, "method": "none", "k": 0, **scores(Q.y, predict_cnn(model, Q.X), C)})
                    for s in range(3):
                        sup, _ = pick_fewshot(pool, 5, C, seed=1000 * f + s)
                        m = copy.deepcopy(model)
                        for p in m.parameters():
                            p.requires_grad_(False)
                        for p in m.fc.parameters() if hasattr(m, "fc") else m.classifier[-1].parameters():
                            p.requires_grad_(True)
                        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
                        xs = torch.tensor(pool.X[sup][:, None, :], dtype=torch.float32)
                        ys = torch.tensor(pool.y[sup], dtype=torch.long)
                        m.eval()                                 # batch-norm statistics stay those of the source
                        for _ in range(200):
                            opt.zero_grad()
                            loss = torch.nn.functional.cross_entropy(m(xs), ys)
                            loss.backward()
                            opt.step()
                        rows.append({**base, "method": "last-layer fine-tuning", "k": 5, "seed": s,
                                     **scores(Q.y, predict_cnn(m, Q.X), C)})
                print(f"  fold {f} {src} {name} done", flush=True)
                pd.DataFrame(rows).to_csv(run.tab_dir / f"{run.tag}_cnn_raw.csv", index=False)
    df = pd.DataFrame(rows)
    run.table(df, "cnn_raw", latex=False)
    s = summarize(df, ["model", "method", "k"])
    run.table(s, "cnn_summary")
    print(s.to_string(index=False))
run.done()
