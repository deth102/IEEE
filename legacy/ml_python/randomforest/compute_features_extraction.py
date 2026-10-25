from preprocessing import   hilbert_signal, normalize, windowing
from feature_extraction import frequency_domain_features, time_frequency_domain_features, time_domain_features
import pandas as pd
import numpy as np

fs= 25600  #tần số lấy mẫu
window_size = 8192  #số mẫu mỗi đoạn
step_size = int((2048))  #bước nhảy
files = {"Vibration_Data/0Nm_BPFI_03.csv": "BPFI",
        "Vibration_Data/2Nm_BPFI_03.csv": "BPFI",
        "Vibration_Data/4Nm_BPFI_03.csv": "BPFI",
        "Vibration_Data/0Nm_BPFI_10.csv": "BPFI",
        "Vibration_Data/2Nm_BPFI_10.csv": "BPFI",
        "Vibration_Data/4Nm_BPFI_10.csv": "BPFI",
        "Vibration_Data/0Nm_BPFO_03.csv": "BPFO",
        "Vibration_Data/2Nm_BPFO_03.csv": "BPFO",
        "Vibration_Data/4Nm_BPFO_03.csv": "BPFO",
        "Vibration_Data/0Nm_BPFO_10.csv": "BPFO",
        "Vibration_Data/2Nm_BPFO_10.csv": "BPFO",
        "Vibration_Data/4Nm_BPFO_10.csv": "BPFO",
        "Vibration_Data/0Nm_Misalign_01.csv": "Misalign",
        "Vibration_Data/2Nm_Misalign_01.csv": "Misalign",
        "Vibration_Data/4Nm_Misalign_01.csv": "Misalign",
        "Vibration_Data/0Nm_Misalign_03.csv": "Misalign",
        "Vibration_Data/2Nm_Misalign_03.csv": "Misalign",
        "Vibration_Data/4Nm_Misalign_03.csv": "Misalign",
        "Vibration_Data/0Nm_Unbalance_0583mg.csv": "Unbalance",
        "Vibration_Data/2Nm_Unbalance_0583mg.csv": "Unbalance",
        "Vibration_Data/4Nm_Unbalance_0583mg.csv": "Unbalance",
        "Vibration_Data/0Nm_Unbalance_1169mg.csv": "Unbalance",
        "Vibration_Data/2Nm_Unbalance_1169mg.csv": "Unbalance",
        "Vibration_Data/4Nm_Unbalance_1169mg.csv": "Unbalance",
        "Vibration_Data/0Nm_Normal.csv": "Normal",
        "Vibration_Data/2Nm_Normal.csv": "Normal",
        "Vibration_Data/4Nm_Normal.csv": "Normal"
        }
all_features = []


for file, label in files.items():
    df = pd.read_csv(file, header= None, skiprows=1)
    data = df.iloc[:, 0].values # Lấy cột đầu tiên (index = 0)

#Tiền xử lý
    # envelope_data = hilbert_signal(data)['envelope']
    normalized_data = normalize(data)
    normalized_data = normalized_data - np.mean(normalized_data)
    windows = windowing(data, window_size, step_size)

#Tính đặc trưng 
    time_feats = [time_domain_features(window) for window in windows]
    freq_feats = [frequency_domain_features(window,fs) for window in windows]
    # time_freq_feats = [time_frequency_domain_features(window,fs,nperseg=256, noverlap=128) for window in windows]
    
    time_cols = ["Mean_time", "RMS_time", "MAX", "MIN", "PTP", "CREST_FACTOR"]
    freq_cols = ["Mean_freq", "RMS_freq", "Var", "Skewness", "Kurtosis", "Central", "STD", "Spectral_centroid", "Spectral_spread", "Spectral_entropy"]
    # time_freq_cols = ["Mean_energy", "RMS_freq_TF", "Var_energy", "Skewness_energy", "Kurtosis_energy", "TF_centroid", "TF_spread", "Mean_time_energy", "Spectral_flux", "TF_entropy"]

#Tạo DataFrame có tên cột
    df_time = pd.DataFrame(time_feats, columns=time_cols)
    df_freq = pd.DataFrame(freq_feats, columns=freq_cols)
    # df_time_freq = pd.DataFrame(time_freq_feats, columns=time_freq_cols)
# print("TF  :", time_feats[0][:5])
# print("Freq:", freq_feats[0][:5])


# Gộp lại
    features = pd.concat([df_time, df_freq], axis=1)    
    features["label"]=label
    all_features.append(features)
all_features_df = pd.concat(all_features, ignore_index=True)
all_features_df.to_csv("features_5labels.csv", index=False)