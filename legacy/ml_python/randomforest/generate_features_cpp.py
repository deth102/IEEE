import os
import numpy as np
import pandas as pd

from preprocessing import bandpass_filter, hilbert_signal, normalize, windowing
from feature_extraction import (
    time_domain_features,
    frequency_domain_features
)

# ================== CONFIG ==================
DATA_DIR = r"D:\lt\datn\logistic regression for embedding\Vibration_Data"
OUTPUT_DIR = r"D:\lt\datn\logistic regression for embedding"

fs = 25600
lowcut = 10
highcut = 4000
window_size = 2**15
step_size = window_size // 2

NUM_FEATURES = 16
# ============================================


def write_header_file(filename, num_samples):
    content = f"""#ifndef FEATURES_H
#define FEATURES_H

#include <Arduino.h>

#define NUM_SAMPLES {num_samples}
#define NUM_FEATURES {NUM_FEATURES}

extern const float features[NUM_SAMPLES][NUM_FEATURES] PROGMEM;

#endif
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def write_cpp_file(filename, features):
    with open(filename, "w", encoding="utf-8") as f:
        f.write('#include "features.h"\n\n')
        f.write("const float features[NUM_SAMPLES][NUM_FEATURES] PROGMEM = {\n")

        for row in features:
            row_str = ", ".join([f"{v:.6f}f" for v in row])
            f.write(f"  {{{row_str}}},\n")

        f.write("};\n")


# ================== MAIN LOOP ==================
for file in os.listdir(DATA_DIR):
    if not file.lower().endswith(".csv"):
        continue

    print(f"Processing: {file}")

    csv_path = os.path.join(DATA_DIR, file)
    base_name = os.path.splitext(file)[0]

    # -------- Load data --------
    data = pd.read_csv(csv_path).iloc[:, 0].values

    # -------- Preprocessing --------
    filtered = bandpass_filter(data, lowcut, highcut, fs)
    envelope = hilbert_signal(filtered)["envelope"]
    normalized = normalize(envelope)
    normalized = normalized - np.mean(normalized)

    windows = windowing(normalized, window_size, step_size)

    # -------- Feature extraction --------
    time_feats = np.array([time_domain_features(w) for w in windows])
    freq_feats = np.array([frequency_domain_features(w, fs) for w in windows])

    features = np.hstack([time_feats, freq_feats])

    num_samples = features.shape[0]

    # -------- Output files --------
    h_file = os.path.join(OUTPUT_DIR, f"features_{base_name}.h")
    cpp_file = os.path.join(OUTPUT_DIR, f"features_{base_name}.cpp")

    write_header_file(h_file, num_samples)
    write_cpp_file(cpp_file, features)

    print(f"  -> Generated {h_file}")
    print(f"  -> Generated {cpp_file}")

print("Done.")
