# dataset.py  (thêm hàm fft_transform và cập nhật load_load_condition)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    from config import (DATA_ROOT, SEGMENT_LEN, OVERLAP, NORMALIZE,
                        TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
                        RANDOM_SEED, CLASSES, CLASS2IDX, USE_FFT)
except ImportError:
    DATA_ROOT   = Path("data/vibration_x")
    SEGMENT_LEN = 2048
    OVERLAP     = 0.5
    NORMALIZE   = True
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15
    RANDOM_SEED = 42
    USE_FFT     = True
    CLASSES = [
        "Normal", "BPFI_03", "BPFI_10",
        "BPFO_03", "BPFO_10",
        "Misalign_01", "Misalign_03",
        "Unbalance_0583mg", "Unbalance_1169mg",
    ]
    CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


def load_csv(filepath: Path) -> np.ndarray:
    df = pd.read_csv(filepath, header=None)
    return df.iloc[:, 0].values.astype(np.float32)


def segment_signal(signal: np.ndarray,
                   seg_len: int = SEGMENT_LEN,
                   overlap: float = OVERLAP) -> np.ndarray:
    step   = int(seg_len * (1 - overlap))
    starts = range(0, len(signal) - seg_len + 1, step)
    return np.stack([signal[s:s + seg_len] for s in starts])


def normalize_segments(segments: np.ndarray) -> np.ndarray:
    mean = segments.mean(axis=1, keepdims=True)
    std  = segments.std(axis=1,  keepdims=True) + 1e-8
    return (segments - mean) / std


def fft_transform(segments: np.ndarray) -> np.ndarray:
    N       = segments.shape[1]
    fft_mag = np.abs(np.fft.rfft(segments, axis=1))
    fft_mag = fft_mag[:, 1: N // 2 + 1]

    # ← THÊM: log scale (dB-like), tốt hơn cho fault pattern
    fft_mag = np.log1p(fft_mag)   # log(1 + |X(f)|)

    # Normalize về [0,1]
    max_val = fft_mag.max(axis=1, keepdims=True) + 1e-8
    fft_mag = fft_mag / max_val
    return fft_mag.astype(np.float32)



def load_load_condition(load: str,
                        data_root: Path = DATA_ROOT,
                        use_fft: bool   = USE_FFT) -> tuple:
    X_list, y_list = [], []
    for label in CLASSES:
        fname = f"{load}_{label}_x.csv"
        fpath = data_root / fname
        if not fpath.exists():
            print(f"  [WARNING] Not found: {fpath}")
            continue
        raw  = load_csv(fpath)
        segs = segment_signal(raw)

        if use_fft:
            segs = fft_transform(segs)          # freq domain
        else:
            if NORMALIZE:
                segs = normalize_segments(segs)  # time domain

        X_list.append(segs)
        y_list.append(np.full(len(segs), CLASS2IDX[label],
                               dtype=np.int64))
        print(f"  Loaded  {fname}: {len(segs)} segments  "
              f"shape={segs.shape[1]}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def get_train_val_test(load: str = "0Nm",
                       use_fft: bool = USE_FFT) -> dict:
    print(f"\n[DATASET] Loading seen-load: {load}  "
          f"| FFT={use_fft}")
    X, y = load_load_condition(load, use_fft=use_fft)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_RATIO,
        stratify=y, random_state=RANDOM_SEED)

    val_adj = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_adj,
        stratify=y_tv, random_state=RANDOM_SEED)

    print(f"  Train:{len(X_train)} | Val:{len(X_val)} "
          f"| Test:{len(X_test)} | Feature dim:{X.shape[1]}")
    return dict(X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                X_test=X_test,   y_test=y_test)


def get_unseen_loads(unseen_loads: list,
                     use_fft: bool = USE_FFT) -> dict:
    result = {}
    for load in unseen_loads:
        print(f"\n[DATASET] Loading unseen-load: {load}  "
              f"| FFT={use_fft}")
        X, y = load_load_condition(load, use_fft=use_fft)
        result[load] = (X, y)
        print(f"  Total: {len(X)} segments")
    return result


def add_awgn(X: np.ndarray, snr_db: float) -> np.ndarray:
    X_noisy = np.empty_like(X)
    for i in range(len(X)):
        sig         = X[i]
        power       = np.mean(sig ** 2) + 1e-10
        noise_power = power / (10 ** (snr_db / 10.0))
        noise       = np.random.normal(0, np.sqrt(noise_power),
                                        sig.shape)
        X_noisy[i]  = sig + noise
    return X_noisy.astype(np.float32)
