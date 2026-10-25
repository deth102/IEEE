
import pandas as pd
import numpy as np
TRAIN_FILE = r"D:\lt\datn\logistic_regression_for_embedding\features_train.csv"
VAL_FILE   = r"D:\lt\datn\logistic_regression_for_embedding\features_val.csv"
TEST_FILE  = r"D:\lt\datn\logistic_regression_for_embedding\features_test.csv"
LABEL_COL = "label"

LABEL_MAP = {
    "Normal": 0,
    "BPFI": 1,
    "BPFO": 2,
    "Misalign": 3,
    "Unbalance": 4
}

# Load data
df_train = pd.read_csv(TRAIN_FILE)
df_val   = pd.read_csv(VAL_FILE)
df_test  = pd.read_csv(TEST_FILE)

# Features
X_train = df_train.drop(columns=[LABEL_COL]).values
X_val   = df_val.drop(columns=[LABEL_COL]).values
X_test  = df_test.drop(columns=[LABEL_COL]).values
# Labels
y_train = df_train[LABEL_COL].map(LABEL_MAP).values
y_val   = df_val[LABEL_COL].map(LABEL_MAP).values
y_test  = df_test[LABEL_COL].map(LABEL_MAP).values

# Safety check (rất quan trọng)
assert not np.any(pd.isnull(y_train)), "Unknown label in TRAIN set"
assert not np.any(pd.isnull(y_val)),   "Unknown label in VAL set"