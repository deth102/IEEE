# config.py
from pathlib import Path

DATA_ROOT    = Path("data/vibration_x")
SEGMENT_LEN  = 2048
OVERLAP      = 0.5
NORMALIZE    = True
USE_FFT      = True          # True = dùng freq domain, False = time domain
FFT_LEN      = SEGMENT_LEN // 2   # 512 bins sau khi rfft

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42
# Thêm vào config.py
SEEN_LOAD    = "0Nm"
UNSEEN_LOADS = ["2Nm", "4Nm"]
ALL_LOADS    = ["0Nm", "2Nm", "4Nm"]
FFT_LEN      = SEGMENT_LEN // 2   # 512

# Cross-load experiment
# Mỗi entry: train trên 1 load, test trên các load còn lại

CLASSES = [
    "Normal",
    "BPFI_03", "BPFI_10",
    "BPFO_03", "BPFO_10",
    "Misalign_01", "Misalign_03",
    "Unbalance_0583mg", "Unbalance_1169mg",
]
NUM_CLASSES  = len(CLASSES)
CLASS2IDX    = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS    = {i: c for c, i in CLASS2IDX.items()}

BATCH_SIZE   = 64
NUM_EPOCHS   = 30
LEARNING_RATE = 1e-4
PATIENCE     = 15
