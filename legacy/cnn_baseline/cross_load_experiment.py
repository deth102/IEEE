# cross_load_experiment.py
"""
Cross-load generalization experiment:
  - Train trên từng load (0Nm, 2Nm, 4Nm)
  - Test trên TẤT CẢ load (kể cả seen và unseen)
  - Vẽ 3D bar chart giống hình mẫu
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import accuracy_score

from config  import (ALL_LOADS, NUM_CLASSES, RANDOM_SEED,
                     USE_FFT, FFT_LEN, SEGMENT_LEN)
from dataset import (get_train_val_test, load_load_condition)
from models  import get_model
from trainer import train_model, make_loader

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────────────────────────────────────────
# 1. PREDICT helper
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_all(model, X: np.ndarray,
                batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64),
                         batch_size=batch_size, shuffle=False)
    preds = []
    for X_b, _ in loader:
        logits = model(X_b.to(DEVICE))
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ────────────────────────────────────────────────────────────
# 2. CHẠY CROSS-LOAD CHO 1 MÔ HÌNH
# ────────────────────────────────────────────────────────────
def run_cross_load(model_name: str) -> np.ndarray:
    """
    Trả về ma trận accuracy (len(ALL_LOADS), len(ALL_LOADS))
    acc_matrix[i, j] = accuracy khi train ở ALL_LOADS[i], test ở ALL_LOADS[j]
    """
    n     = len(ALL_LOADS)
    acc_matrix = np.zeros((n, n))
    feat_dim   = FFT_LEN if USE_FFT else SEGMENT_LEN

    for i, train_load in enumerate(ALL_LOADS):
        print(f"\n{'═'*55}")
        print(f"  Training on: {train_load}  |  Model: {model_name}")
        print(f"{'═'*55}")

        # Load train/val/test từ train_load
        data = get_train_val_test(load=train_load, use_fft=USE_FFT)

        # Build và train model
        model = get_model(model_name, NUM_CLASSES, feat_dim)
        train_model(model, data, DEVICE,
                    save_path=f"best_{model_name}_{train_load}.pt")

        # Test trên TẤT CẢ load
        for j, test_load in enumerate(ALL_LOADS):
            X_test, y_test = load_load_condition(test_load,
                                                  use_fft=USE_FFT)
            y_pred = predict_all(model, X_test)
            acc    = accuracy_score(y_test, y_pred)
            acc_matrix[i, j] = acc
            tag = "(SEEN)" if test_load == train_load else "(UNSEEN)"
            print(f"  Test on {test_load} {tag}: {acc*100:.2f}%")

    return acc_matrix


# ────────────────────────────────────────────────────────────
# 3. VẼ 3D BAR CHART
# ────────────────────────────────────────────────────────────
def plot_3d_bar(acc_matrix: np.ndarray,
                model_name: str,
                save_path: str = None):
    """
    acc_matrix: (n_train, n_test) numpy array
    """
    loads = ALL_LOADS
    n     = len(loads)

    fig   = plt.figure(figsize=(9, 6))
    ax    = fig.add_subplot(111, projection='3d')

    # Grid positions
    xpos  = np.arange(n)   # training set axis
    ypos  = np.arange(n)   # test set axis
    xm, ym = np.meshgrid(xpos, ypos, indexing='ij')
    xm    = xm.ravel()
    ym    = ym.ravel()
    zpos  = np.zeros_like(xm)
    dz    = acc_matrix.ravel()

    # Bar size
    dx = dy = 0.5

    # Color map based on accuracy value
    cmap   = cm.viridis
    norm   = plt.Normalize(vmin=dz.min() - 0.05,
                           vmax=dz.max() + 0.01)
    colors = cmap(norm(dz))

    bars = ax.bar3d(xm - dx/2, ym - dy/2, zpos,
                    dx, dy, dz,
                    color=colors, zsort='average',
                    edgecolor='white', linewidth=0.4)

    # Annotate values on top of bars
    for xi, yi, zi in zip(xm, ym, dz):
        ax.text(xi, yi, zi + 0.005,
                f'{zi:.4f}',
                ha='center', va='bottom',
                fontsize=7.5, fontweight='bold')

    # Axes labels
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"Train_{l}" for l in loads],
                        fontsize=8, rotation=10)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"Test_{l}" for l in loads],
                        fontsize=8)
    ax.set_xlabel("Training Set", labelpad=12, fontsize=10)
    ax.set_ylabel("Test Set",     labelpad=12, fontsize=10)
    ax.set_zlabel("Accuracy",     labelpad=8,  fontsize=10)
    ax.set_zlim(max(0, dz.min() - 0.1), 1.02)

    ax.set_title(f"Cross-Load Generalization — {model_name}",
                 fontsize=11, pad=15)

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(dz)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55,
                        aspect=12, pad=0.1)
    cbar.set_label("Accuracy", fontsize=9)

    ax.view_init(elev=22, azim=-50)

    plt.tight_layout()
    out = save_path or f"crossload_{model_name}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  [SAVED] {out}")
    return out


# ────────────────────────────────────────────────────────────
# 4. MAIN: chạy cho CẢ 3 mô hình
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_NAMES = ['edgelite', 'mobilelite', 'baseline']

    all_matrices = {}

    for mname in MODEL_NAMES:
        print(f"\n\n{'#'*60}")
        print(f"  CROSS-LOAD EXPERIMENT: {mname.upper()}")
        print(f"{'#'*60}")

        acc_mat = run_cross_load(mname)
        all_matrices[mname] = acc_mat

        # In bảng ma trận
        print(f"\n  Accuracy matrix (row=train, col=test):")
        header = "          " + "  ".join(f"{l:>8}" for l in ALL_LOADS)
        print(header)
        for i, tl in enumerate(ALL_LOADS):
            row = f"  {tl:>6}  " + "  ".join(
                f"{acc_mat[i,j]*100:>7.2f}%" for j in range(len(ALL_LOADS)))
            print(row)

        # Vẽ hình
        plot_3d_bar(acc_mat, mname.capitalize(),
                    save_path=f"crossload_{mname}.png")

    # ── In bảng tổng hợp so sánh 3 mô hình ──────────────────
    print(f"\n\n{'═'*65}")
    print(f"  CROSS-LOAD SUMMARY (unseen load accuracy)")
    print(f"{'═'*65}")
    print(f"{'Train→Test':<16} " +
          "  ".join(f"{m:>12}" for m in MODEL_NAMES))
    print(f"{'─'*65}")
    n = len(ALL_LOADS)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # chỉ in unseen
            key = f"{ALL_LOADS[i]}→{ALL_LOADS[j]}"
            row = f"  {key:<14} "
            for mname in MODEL_NAMES:
                v = all_matrices[mname][i, j]
                row += f"  {v*100:>10.2f}%"
            print(row)
    print(f"{'═'*65}")
