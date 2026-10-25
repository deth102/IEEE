import numpy as np
# from scipy.signal import butter, filtfilt #bandpass filter
from scipy.signal import hilbert
# def bandpass_filter(data, lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     y = filtfilt(b, a, data)
#     return y

# def normalize_zscore(x):
#     mu = np.mean(x)
#     sigma = np.std(x)
#     if sigma == 0:
#         return x - mu
#     return (x - mu) / sigma

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
    #return (data) / (np.max(data) - np.min(data))
    #return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

def windowing(data, window_size, step_size):
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[start:start + window_size])
    return np.array(windows)
def hilbert_signal(data):
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)
    return {
        "analytic_signal": analytic_signal,
        "envelope": envelope,
    }
