import pandas as pd
from sklearn.model_selection import train_test_split

# ====== CẤU HÌNH ======
INPUT_FILE = r"D:\lt\datn\logistic_regression_for_embedding\features_5labels.csv"
OUTPUT_DIR = r"D:\lt\datn\logistic_regression_for_embedding"

LABEL_COL = "label"
RANDOM_STATE = 42
# ======================

# Load feature file
df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

# --- Split train (70%) và temp (30%) ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

# --- Split temp → val (15%) + test (15%) ---
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

# --- Gộp lại ---
df_train = pd.concat([X_train, y_train], axis=1)
df_val   = pd.concat([X_val, y_val], axis=1)
df_test  = pd.concat([X_test, y_test], axis=1)

# --- Lưu ra file ---
df_train.to_csv(f"{OUTPUT_DIR}/features_train.csv", index=False)
df_val.to_csv(f"{OUTPUT_DIR}/features_val.csv", index=False)
df_test.to_csv(f"{OUTPUT_DIR}/features_test.csv", index=False)

print("DONE!")
print(f"Train size: {len(df_train)}")
print(f"Val size:   {len(df_val)}")
print(f"Test size:  {len(df_test)}")
