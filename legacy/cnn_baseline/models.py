# ============================================================
# FILE: models.py
# ============================================================
"""
Module định nghĩa 3 kiến trúc 1D-CNN:
  - Baseline1DCNN
  - EdgeLite1DCNN
  - MobileLite1DCNN
"""
import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────
# BASELINE 1D-CNN
# ────────────────────────────────────────────────────────────
class Baseline1DCNN(nn.Module):
    """
    Desktop-class 1D-CNN, large capacity.
    3 Conv-Pool blocks + 2 FC layers.
    """
    def __init__(self, num_classes: int, seq_len: int = 1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=64, padding=32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=32, padding=16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=16, padding=8), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # auto-compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len)
            flat  = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ────────────────────────────────────────────────────────────
# EDGE-LITE 1D-CNN
# ────────────────────────────────────────────────────────────
class EdgeLite1DCNN(nn.Module):
    """
    Lightweight 1D-CNN cho MCU-class edge devices.
    3 Conv-BN-Pool blocks + GAP + 1 FC.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ────────────────────────────────────────────────────────────
# DEPTHWISE SEPARABLE BLOCK (helper)
# ────────────────────────────────────────────────────────────
class DSConv1d(nn.Module):
    """Depthwise Separable Conv1D block: DW → PW → BN → ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.dw  = nn.Conv1d(in_ch, in_ch, kernel_size,
                             padding=kernel_size//2, groups=in_ch)
        self.pw  = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.bn  = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# ────────────────────────────────────────────────────────────
# MOBILE-LITE 1D-CNN
# ────────────────────────────────────────────────────────────
class MobileLite1DCNN(nn.Module):
    """
    Ultra-compact 1D-CNN dùng depthwise separable convolutions.
    Cho thiết bị MCU có tài nguyên cực hạn chế.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(DSConv1d(1,  8,  7), nn.MaxPool1d(2))
        self.block2 = nn.Sequential(DSConv1d(8,  16, 5), nn.MaxPool1d(2))
        self.block3 = DSConv1d(16, 32, 3)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ────────────────────────────────────────────────────────────
# FACTORY
# ────────────────────────────────────────────────────────────
def get_model(name: str, num_classes: int, seq_len: int = 1024) -> nn.Module:
    """
    name: 'baseline' | 'edgelite' | 'mobilelite'
    """
    name = name.lower()
    if name == 'baseline':
        return Baseline1DCNN(num_classes, seq_len)
    elif name == 'edgelite':
        return EdgeLite1DCNN(num_classes)
    elif name == 'mobilelite':
        return MobileLite1DCNN(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. "
                         f"Choose from baseline/edgelite/mobilelite")
