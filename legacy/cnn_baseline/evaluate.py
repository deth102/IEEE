# ============================================================
# FILE: evaluate.py
# ============================================================
"""
Module đánh giá chi tiết: accuracy, F1, confusion matrix,
robustness (noise + variable load).
"""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, classification_report)
from dataset import add_awgn
from trainer import make_loader, evaluate
from config import IDX2CLASS
import torch.nn as nn


# ────────────────────────────────────────────────────────────
# 1. DỰ ĐOÁN TOÀN BỘ TẬP
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, X: np.ndarray, device,
            batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64),
                         batch_size=batch_size, shuffle=False)
    preds = []
    for X_b, _ in loader:
        logits = model(X_b.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ────────────────────────────────────────────────────────────
# 2. ĐÁNH GIÁ HIỆU NĂNG PHÂN LOẠI
# ────────────────────────────────────────────────────────────
def classification_metrics(model, X: np.ndarray, y: np.ndarray,
                            device, tag: str = "") -> dict:
    """
    Tính accuracy, macro precision/recall/F1, confusion matrix.
    """
    y_pred = predict(model, X, device)
    acc    = accuracy_score(y, y_pred)
    prec   = precision_score(y, y_pred, average='macro', zero_division=0)
    rec    = recall_score(y, y_pred, average='macro', zero_division=0)
    f1     = f1_score(y, y_pred, average='macro', zero_division=0)
    cm     = confusion_matrix(y, y_pred)

    print(f"\n{'─'*55}")
    print(f"  {tag}  |  {model.__class__.__name__}")
    print(f"{'─'*55}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%")
    print(f"  Recall     : {rec*100:.2f}%")
    print(f"  F1-score   : {f1*100:.2f}%")
    print(f"\n  Per-class report:")
    labels = [IDX2CLASS[i] for i in range(len(IDX2CLASS))]
    print(classification_report(y, y_pred, target_names=labels,
                                 zero_division=0))
    return dict(accuracy=acc, precision=prec, recall=rec,
                f1=f1, confusion_matrix=cm)


# ────────────────────────────────────────────────────────────
# 3. ROBUSTNESS – NOISE (AWGN)
# ────────────────────────────────────────────────────────────
def noise_robustness(model, X_test: np.ndarray, y_test: np.ndarray,
                     device,
                     snr_levels=(15, 10, 5, 0)) -> dict:
    """
    Đánh giá accuracy ở nhiều mức SNR.
    """
    results = {'clean': accuracy_score(y_test, predict(model, X_test, device))}
    print(f"\n[Noise Robustness] {model.__class__.__name__}")
    print(f"  Clean       : {results['clean']*100:.2f}%")

    for snr in snr_levels:
        X_noisy = add_awgn(X_test, snr_db=snr)
        acc = accuracy_score(y_test, predict(model, X_noisy, device))
        results[f'SNR_{snr}dB'] = acc
        print(f"  SNR {snr:>3} dB  : {acc*100:.2f}%")

    return results


# ────────────────────────────────────────────────────────────
# 4. ROBUSTNESS – VARIABLE LOAD (generalization)
# ────────────────────────────────────────────────────────────
def variable_load_robustness(model, unseen_data: dict,
                              device) -> dict:
    """
    unseen_data: { "2Nm": (X, y), "4Nm": (X, y) }

    Trả về accuracy trên từng mức tải chưa thấy khi train.
    """
    results = {}
    print(f"\n[Variable Load Robustness] {model.__class__.__name__}")
    for load_name, (X_u, y_u) in unseen_data.items():
        acc = accuracy_score(y_u, predict(model, X_u, device))
        results[load_name] = acc
        print(f"  Unseen load {load_name}: {acc*100:.2f}%")
    return results
