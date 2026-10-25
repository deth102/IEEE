# ============================================================
# FILE: main.py  (updated)
# ============================================================
"""
Main pipeline:
  1. Train/Val/Test trên 0Nm (seen load)   → accuracy, F1, confusion matrix
  2. Noise robustness (AWGN, SNR 15/10/5/0 dB)
  3. Variable load robustness (test trên 2Nm, 4Nm)
  4. Cross-load generalization matrix + 3D bar chart
     (train 0Nm, 2Nm, 4Nm → test tất cả)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, classification_report)

from config  import (SEEN_LOAD, UNSEEN_LOADS, NUM_CLASSES,
                     SEGMENT_LEN, FFT_LEN, USE_FFT,
                     ALL_LOADS, RANDOM_SEED, IDX2CLASS)
from dataset import (get_train_val_test, get_unseen_loads,
                     load_load_condition, add_awgn)
from models  import get_model
from trainer import train_model, make_loader
from cross_load_experiment import run_cross_load, plot_3d_bar

# ── Reproducibility ──────────────────────────────────────────
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_DIM   = FFT_LEN if USE_FFT else SEGMENT_LEN
MODEL_NAMES = ['baseline', 'edgelite', 'mobilelite']

print(f"  Device   : {DEVICE}")
print(f"  Use FFT  : {USE_FFT}  |  Feature dim: {FEAT_DIM}")
print(f"  Seen load: {SEEN_LOAD}  |  Classes: {NUM_CLASSES}")


# ════════════════════════════════════════════════════════════
# HELPER: predict full set
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64),
                         batch_size=batch_size, shuffle=False)
    preds = []
    for X_b, _ in loader:
        logits = model(X_b.to(DEVICE))
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ════════════════════════════════════════════════════════════
# HELPER: full classification report
# ════════════════════════════════════════════════════════════
def clf_metrics(model, X, y, tag=""):
    y_pred = predict(model, X)
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='macro', zero_division=0)
    rec  = recall_score   (y, y_pred, average='macro', zero_division=0)
    f1   = f1_score       (y, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y, y_pred)
    labels = [IDX2CLASS[i] for i in range(NUM_CLASSES)]

    print(f"\n{'─'*55}")
    print(f"  {tag}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-score  : {f1*100:.2f}%")

   
    report = classification_report(y, y_pred,
                                   target_names=labels,
                                   zero_division=0)
    print(f"\n{report}")

    return dict(accuracy=acc, precision=prec,
                recall=rec, f1=f1, cm=cm)


# ════════════════════════════════════════════════════════════
# HELPER: noise robustness
# ════════════════════════════════════════════════════════════
def noise_test(model, X_test, y_test,
               snr_levels=(15, 10, 5, 0)) -> dict:
    res = {'clean': accuracy_score(y_test, predict(model, X_test))}
    print(f"\n  [Noise Robustness]")
    print(f"  Clean      : {res['clean']*100:.2f}%")
    for snr in snr_levels:
        X_n = add_awgn(X_test, snr_db=snr)
        acc = accuracy_score(y_test, predict(model, X_n))
        res[f'SNR_{snr}dB'] = acc
        print(f"  SNR {snr:>3}dB  : {acc*100:.2f}%")
    return res


# ════════════════════════════════════════════════════════════
# HELPER: variable load robustness
# ════════════════════════════════════════════════════════════
def varload_test(model, unseen_data: dict) -> dict:
    res = {}
    print(f"\n  [Variable Load Robustness]")
    for load_name, (X_u, y_u) in unseen_data.items():
        acc = accuracy_score(y_u, predict(model, X_u))
        res[load_name] = acc
        print(f"  Unseen {load_name}: {acc*100:.2f}%")
    return res


# ════════════════════════════════════════════════════════════
# HELPER: plot confusion matrix
# ════════════════════════════════════════════════════════════
def plot_confusion(cm_arr, model_name, save_path):
    labels = [IDX2CLASS[i] for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_arr, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    thresh = cm_arr.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_arr[i,j]}",
                    ha='center', va='center', fontsize=7,
                    color='white' if cm_arr[i,j] > thresh else 'black')
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label",      fontsize=10)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {save_path}")


# ════════════════════════════════════════════════════════════
# HELPER: plot training history
# ════════════════════════════════════════════════════════════
def plot_history(history: dict, model_name: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} — Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs,
                 [a*100 for a in history['train_acc']],
                 label='Train Acc')
    axes[1].plot(epochs,
                 [a*100 for a in history['val_acc']],
                 label='Val Acc')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} — Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Training History — {model_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {save_path}")


# ════════════════════════════════════════════════════════════
# PHASE 1: Standard train/val/test trên SEEN LOAD (0Nm)
# ════════════════════════════════════════════════════════════
print(f"\n\n{'#'*60}")
print(f"  PHASE 1: Standard Evaluation  (Train/Val/Test = {SEEN_LOAD})")
print(f"{'#'*60}")

data        = get_train_val_test(load=SEEN_LOAD, use_fft=USE_FFT)

phase1_results = {}

for mname in MODEL_NAMES:
    print(f"\n{'═'*55}")
    print(f"  MODEL: {mname.upper()}")
    print(f"{'═'*55}")

    model = get_model(mname, NUM_CLASSES, FEAT_DIM)

    # ── Train
    history = train_model(
        model, data, DEVICE,
        save_path=f"best_{mname}_{SEEN_LOAD}.pt"
    )

    # ── Plot history
    plot_history(history, mname.capitalize(),
                 f"history_{mname}.png")

    # ── Classify (seen load test set)
    metrics = clf_metrics(
        model, data['X_test'], data['y_test'],
        tag=f"{mname.upper()} | Test ({SEEN_LOAD})"
    )

    # ── Confusion matrix
    plot_confusion(metrics['cm'], mname.capitalize(),
                   f"confusion_{mname}.png")

    # ── Noise robustness
    noise_res = noise_test(
        model, data['X_test'], data['y_test'],
        snr_levels=(15, 10, 5, 0)
    )


    phase1_results[mname] = dict(
        history=history,
        metrics=metrics,
        noise=noise_res,
    )


# ════════════════════════════════════════════════════════════
# PHASE 2: Cross-load generalization (3D bar chart)
# ════════════════════════════════════════════════════════════
print(f"\n\n{'#'*60}")
print(f"  PHASE 2: Cross-Load Generalization")
print(f"  Train on each of {ALL_LOADS}, test on all")
print(f"{'#'*60}")

cross_results = {}

for mname in MODEL_NAMES:
    print(f"\n{'═'*55}")
    print(f"  Cross-load: {mname.upper()}")
    print(f"{'═'*55}")

    acc_matrix = run_cross_load(mname)
    cross_results[mname] = acc_matrix

    # In ma trận
    print(f"\n  Accuracy matrix (row=train load, col=test load):")
    header = f"  {'':>6}  " + "  ".join(f"{l:>8}" for l in ALL_LOADS)
    print(header)
    for i, tl in enumerate(ALL_LOADS):
        row = f"  {tl:>6}  " + "  ".join(
            f"{acc_matrix[i,j]*100:>7.2f}%" for j in range(len(ALL_LOADS)))
        print(row)

    # Vẽ 3D bar chart
    plot_3d_bar(acc_matrix, mname.capitalize(),
                save_path=f"crossload_{mname}.png")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print(f"\n\n{'═'*75}")
print(f"  FINAL SUMMARY — Phase 1 (Seen Load: {SEEN_LOAD})")
print(f"{'═'*75}")
print(f"{'Model':<14} {'Acc':>7} {'F1':>7} {'SNR10':>7} "
      f"{'SNR5':>7} {'SNR0':>7} "
      + "  ".join(f"{l:>7}" for l in UNSEEN_LOADS))
print(f"{'─'*75}")
for mname in MODEL_NAMES:
    r  = phase1_results[mname]
    m  = r['metrics']
    nr = r['noise']
    vl = r['varload']
    row = (f"{mname:<14} "
           f"{m['accuracy']*100:>6.2f}% "
           f"{m['f1']*100:>6.2f}% "
           f"{nr.get('SNR_10dB',0)*100:>6.2f}% "
           f"{nr.get('SNR_5dB',0)*100:>6.2f}% "
           f"{nr.get('SNR_0dB',0)*100:>6.2f}% ")
    for l in UNSEEN_LOADS:
        row += f"{vl.get(l,0)*100:>6.2f}%  "
    print(row)

print(f"\n{'═'*75}")
print(f"  FINAL SUMMARY — Phase 2 (Cross-Load Unseen Accuracy)")
print(f"{'═'*75}")
print(f"{'Train→Test':<16} " +
      "  ".join(f"{m:>12}" for m in MODEL_NAMES))
print(f"{'─'*75}")
for i, tr in enumerate(ALL_LOADS):
    for j, te in enumerate(ALL_LOADS):
        if i == j:
            continue
        key = f"{tr}→{te}"
        row = f"  {key:<14} "
        for mname in MODEL_NAMES:
            v = cross_results[mname][i, j]
            row += f"  {v*100:>10.2f}%"
        print(row)
print(f"{'═'*75}")

print(f"\n\n[DONE] All outputs saved:")
for mname in MODEL_NAMES:
    print(f"  best_{mname}_{SEEN_LOAD}.pt")
    print(f"  history_{mname}.png")
    print(f"  confusion_{mname}.png")
    print(f"  crossload_{mname}.png")
