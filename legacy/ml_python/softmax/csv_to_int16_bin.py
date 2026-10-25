import numpy as np
import os

# ================= CONFIG =================
input_dir = r"D:\lt\datn\logistic regression for embedding\Vibration_Data"
bin_dir   = os.path.join(input_dir, "bin")
scale_dir = os.path.join(input_dir, "scale")

# tạo folder output nếu chưa có
os.makedirs(bin_dir, exist_ok=True)
os.makedirs(scale_dir, exist_ok=True)

# =========================================
csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]

print("So file CSV:", len(csv_files))

for csv_name in csv_files:
    csv_path = os.path.join(input_dir, csv_name)

    print(f"\nDang xu ly: {csv_name}")

    # ---- load CSV (1 cot, khong header) ----
    data = np.loadtxt(csv_path, delimiter=",")

    # ---- normalize ----
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        print("  ⚠ File co toan 0, bo qua")
        continue

    data_norm = data / max_abs

    # ---- float -> int16 (Q15) ----
    data_int16 = np.int16(
        np.clip(data_norm * 32767, -32768, 32767)
    )

    # ---- save binary ----
    base_name = os.path.splitext(csv_name)[0]
    bin_path = os.path.join(bin_dir, base_name + ".bin")
    data_int16.tofile(bin_path)

    # ---- save scale factor ----
    scale_path = os.path.join(scale_dir, base_name + "_scale.txt")
    with open(scale_path, "w") as f:
        f.write(str(max_abs))

    print(f"  So mau      : {len(data_int16)}")
    print(f"  Scale factor: {max_abs}")
    print(f"  Da ghi      : {bin_path}")

print("\nHoan tat chuyen doi tat ca file CSV.")
