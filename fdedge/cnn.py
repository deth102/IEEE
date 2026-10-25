"""1-D CNN baselines (from the earlier project, D:\\baodoan\\code\\models.py) and their cost.

Input: log-magnitude spectrum of a window (N/2 bins, max-normalised), as in the
original code. Training: Adam + cross-entropy, early stopping on validation loss.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .resources import _c


class Baseline1DCNN(nn.Module):
    def __init__(self, num_classes: int, seq_len: int = 1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, 64, padding=32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 32, padding=16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 16, padding=8), nn.ReLU(), nn.MaxPool1d(2))
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, seq_len)).view(1, -1).shape[1]
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class EdgeLite1DCNN(nn.Module):
    def __init__(self, num_classes: int, seq_len: int = 1024):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2))
        self.block2 = nn.Sequential(nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.block3 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).squeeze(-1))


class DSConv1d(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, padding=k // 2, groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class MobileLite1DCNN(nn.Module):
    def __init__(self, num_classes: int, seq_len: int = 1024):
        super().__init__()
        self.block1 = nn.Sequential(DSConv1d(1, 8, 7), nn.MaxPool1d(2))
        self.block2 = nn.Sequential(DSConv1d(8, 16, 5), nn.MaxPool1d(2))
        self.block3 = DSConv1d(16, 32, 3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).squeeze(-1))


CNN_MODELS = {"baseline": Baseline1DCNN, "edgelite": EdgeLite1DCNN, "mobilelite": MobileLite1DCNN}
CNN_LABEL = {"baseline": "CNN-Baseline", "edgelite": "CNN-EdgeLite", "mobilelite": "CNN-MobileLite"}


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn(name, n_classes, Xtr, ytr, Xva, yva, epochs=30, batch=64, lr=1e-3, patience=8, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = CNN_MODELS[name](n_classes, Xtr.shape[1]).to(dev)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(Xtr[:, None, :], dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva[:, None, :], dtype=torch.float32, device=dev)
    yv = torch.tensor(yva, dtype=torch.long, device=dev)
    best, best_state, wait = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx].to(dev)), yt[idx].to(dev))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(sum(lossf(model(Xv[i:i + 1024]), yv[i:i + 1024]).item() * len(yv[i:i + 1024])
                           for i in range(0, len(yv), 1024)) / len(yv))
        if vl < best - 1e-5:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_cnn(model, X, batch=2048):
    dev = device()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch, None, :], dtype=torch.float32, device=dev)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def cnn_cost(name, n_classes, seq_len):
    """MACs, parameters and peak activation (elements) of one inference."""
    model = CNN_MODELS[name](n_classes, seq_len).eval()
    macs, acts = [0], []

    def hook(mod, inp, out):
        x = inp[0]
        if isinstance(mod, nn.Conv1d):
            macs[0] += out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0]
        elif isinstance(mod, nn.Linear):
            macs[0] += mod.in_features * mod.out_features
        if isinstance(mod, (nn.Conv1d, nn.Linear, nn.MaxPool1d, nn.AdaptiveAvgPool1d)):
            acts.append(x.numel() + out.numel())

    hs = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    with torch.no_grad():
        model(torch.zeros(1, 1, seq_len))
    for h in hs:
        h.remove()
    params = sum(p.numel() for p in model.parameters())
    return {"macs": int(macs[0]), "params": int(params), "peak_act": int(max(acts))}


def cnn_ops(macs: int):
    """Plain-C float inference: 2 flop and ~2 memory accesses per MAC."""
    return _c(flop=2 * macs, mem=2 * macs, int=macs // 4)
