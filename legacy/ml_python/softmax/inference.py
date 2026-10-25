from preprocessing import hilbert_signal, normalize, windowing
from feature_extraction import frequency_domain_features, time_frequency_domain_features, time_domain_features
import pandas as pd
import numpy as np
import pickle
OUTPUT_DIR = r"D:\lt\datn\logistic_regression_for_embedding"

# vibration_data= pd.read_csv("Vibration data/0Nm_Normal.csv") 
# data = vibration_data.iloc[:, 0].values # Lấy cột đầu tiên (index = 0)


fs= 25600  #tần số lấy mẫu
lowcut = 10  #tần số cắt thấp
highcut = 4000 #tần số cắt cao
window_size = 2**13  #số mẫu mỗi đoạn
step_size = int((2**13)/4)  #bước nhảy

LABEL_MAP = {
    0:"Normal",
    1:"BPFI",
    2:"BPFO",
    3:"Misalign",
    4:"Unbalance"
}

data=pd.read_csv("Vibration_Data/2Nm_Unbalance_1169mg.csv").iloc[:, 0].values

#Tiền xử lý
envelope_data = hilbert_signal(data)['envelope']
normalized_data = normalize(envelope_data)
normalized_data = normalized_data - np.mean(normalized_data)
windows = windowing(data, window_size, step_size)

#Tính đặc trưng 
time_feats = [time_domain_features(window) for window in windows]
freq_feats = [frequency_domain_features(window,fs) for window in windows]
# time_freq_feats = [time_frequency_domain_features(window,fs,nperseg=256, noverlap=128) for window in windows]
    
time_cols = ["Mean_time", "RMS_time", "MAX", "MIN", "PTP", "CREST_FACTOR"]
freq_cols = ["Mean_freq",
             "RMS_freq", 
             "Var", 
             "Skewness",
             "Kurtosis",
             "Central", 
             "STD", 
             "Spectral_centroid",
             "Spectral_spread",
             "Spectral_entropy"]

# time_freq_cols = ["Mean_energy",
#                   "RMS_freq_TF",
#                   "Var_energy",
#                   "Skewness_energy",
#                   "Kurtosis_energy",
#                   "TF_centroid", 
#                   "TF_spread", 
#                   "Mean_time_energy",
#                   "Spectral_flux", 
#                   "TF_entropy"]

#Tạo DataFrame có tên cột
df_time = pd.DataFrame(time_feats, columns=time_cols)
df_freq = pd.DataFrame(freq_feats, columns=freq_cols)
# df_time_freq = pd.DataFrame(time_freq_feats, columns=time_freq_cols)
print("Time  :", time_feats[0][:5])
print("Freq:", freq_feats[0][:5])
# Gộp lại
features_inference = pd.concat([df_time, df_freq], axis=1)  
features_inference.to_csv(f"{OUTPUT_DIR}/features_inference.csv", index=False)
with open("softmax_logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# ===== PREDICT =====
X_test = features_inference.values              # (n_windows, n_features)
y_pred_window = model.predict(X_test)             # (n_windows,)
y_proba_window = model.predict_proba(X_test)       # (n_windows, n_classes)

# ===== MAP INDEX -> LABEL NAME =====
LABEL_MAP = {
    0: "Normal",
    1: "BPFI",
    2: "BPFO",
    3: "Misalign",
    4: "Unbalance"
}

pred_labels_window = [LABEL_MAP[i] for i in y_pred_window]
mean_proba = y_proba_window.mean(axis=0)   # (n_classes,)

pred_class_idx = np.argmax(mean_proba)
final_label = LABEL_MAP[pred_class_idx]
print("Mean probability over all windows:")
for i, p in enumerate(mean_proba):
    print(f"{LABEL_MAP[i]:10s}: {p:.4f}")

print("\nFinal predicted label (mean softmax):", final_label)


# 5. XUẤT FILE features.h
# =======================
# def export_to_arduino_header(df, filename, var_name):
#     matrix = df.astype(np.float32).values
#     rows, cols = matrix.shape

#     with open(filename, "w", encoding="utf-8") as f:
#         f.write("#ifndef FEATURES_H\n")
#         f.write("#define FEATURES_H\n\n")

#         f.write("#include <avr/pgmspace.h>\n\n")

#         f.write(f"#define NUM_SAMPLES {rows}\n")
#         f.write(f"#define NUM_FEATURES {cols}\n\n")

#         f.write(
#             f"const float {var_name}[NUM_SAMPLES][NUM_FEATURES] PROGMEM = {{\n"
#         )

#         for row in matrix:
#             row_str = ", ".join(f"{x:.6f}f" for x in row)
#             f.write(f"  {{{row_str}}},\n")

#         f.write("};\n\n")
#         f.write("#endif\n")

#     print(f"Saved Arduino header: {filename}")

# export_to_arduino_header(features_inference, "features.h", "features")