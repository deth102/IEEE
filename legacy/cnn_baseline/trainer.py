# ============================================================
# FILE: trainer.py
# ============================================================
"""
Module huấn luyện và đánh giá mô hình.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, PATIENCE


# ────────────────────────────────────────────────────────────
# 1. TẠO DATALOADER TỪ NUMPY ARRAYS
# ────────────────────────────────────────────────────────────
def make_loader(X: np.ndarray, y: np.ndarray,
                batch_size: int = BATCH_SIZE,
                shuffle: bool = True) -> DataLoader:
    """
    X: (N, L), y: (N,)
    → DataLoader, mỗi item: (1, L) tensor
    """
    X_t = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ────────────────────────────────────────────────────────────
# 2. HUẤN LUYỆN MỘT EPOCH
# ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        correct    += (logits.argmax(1) == y_b).sum().item()
        total      += len(y_b)
    return total_loss / total, correct / total


# ────────────────────────────────────────────────────────────
# 3. EVALUATE
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        total_loss += loss.item() * len(y_b)
        correct    += (logits.argmax(1) == y_b).sum().item()
        total      += len(y_b)
    return total_loss / total, correct / total


# ────────────────────────────────────────────────────────────
# 4. TRAINING LOOP (với early stopping)
# ────────────────────────────────────────────────────────────
def train_model(model, data: dict, device,
                save_path: str = None) -> dict:
    """
    data: dict với X_train, y_train, X_val, y_val

    Returns dict: history (train_loss, val_loss, train_acc, val_acc)
    """
    train_loader = make_loader(data['X_train'], data['y_train'])
    val_loader   = make_loader(data['X_val'],   data['y_val'],   shuffle=False)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    history = dict(train_loss=[], val_loss=[],
                   train_acc=[], val_acc=[])
    best_val_loss = float('inf')
    patience_cnt  = 0
    best_state    = None

    print(f"\n{'─'*55}")
    print(f"  Training: {model.__class__.__name__}  |  "
          f"Device: {device}  |  Epochs: {NUM_EPOCHS}")
    print(f"{'─'*55}")
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>10} "
          f"{'ValLoss':>10} {'ValAcc':>10}")
    print(f"{'─'*55}")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(
            model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        scheduler.step(vl_loss)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc*100:>9.2f}%"
              f"  {vl_loss:>10.4f}  {vl_acc*100:>9.2f}%")

        # Early stopping
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            if save_path:
                torch.save(best_state, save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  [Early Stop] No improvement for "
                      f"{PATIENCE} epochs.")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    return history
