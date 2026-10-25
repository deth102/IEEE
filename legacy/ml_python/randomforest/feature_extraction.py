import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
def time_domain_features(window):
    Mean_time = np.mean(window)
    RMS_time = np.sqrt(np.mean(window**2))
    MAX = np.max(window)
    MIN = np.min(window)
    PTP = MAX - MIN
    CREST_FACTOR = MAX / RMS_time
    return [Mean_time, RMS_time, MAX, MIN, PTP, CREST_FACTOR]

# def frequency_domain_features(window, fs):
#     # Biến đổi FFT
#     N = len(window)
#     Y = np.abs(fft(window))[:N//2]
#     f = fftfreq(N, 1/fs)[:N//2]
#     Y_norm = Y / np.sum(Y)
 
#     # Các đặc trưng F1 - F10
#     Mean_freq = np.mean(Y)# Mean frequency
#     Var = np.var(Y) # Variance
#     Skewness = skew(Y)# Skewness
#     Kurtosis = kurtosis(Y)# Kurtosis
#     Central = np.sum(f * Y) / np.sum(Y)# Central frequency
#     STD = np.sqrt(np.sum(((f - Central)**2) * Y) / np.sum(Y)) #STD frequency
#     RMS_freq = np.sqrt(np.sum(f**2 * Y) / np.sum(Y))# RMS frequency
#     Spectral_centroid = np.sum(np.arange(len(Y)) * Y) / np.sum(Y)#Spectral centroid
#     Spectral_spread = np.sqrt(np.sum(((np.arange(len(Y)) - Spectral_centroid)**2) * Y) / np.sum(Y)) #Spectral spread
#     Spectral_entropy = -np.sum(Y_norm * np.log2(Y_norm + 1e-12)) #Spectral entropy

#     return [Mean_freq, RMS_freq, Var, Skewness, Kurtosis, Central, STD,  Spectral_centroid, Spectral_spread, Spectral_entropy]


def frequency_domain_features(window, fs):
    """
    window : 1-D numpy array (time-domain signal)
    fs     : sampling frequency (Hz)

    return : feature vector (numpy array)
    """
    window = np.asarray(window, dtype=np.float32)
    N = len(window)
    K = N // 2

    feat = np.zeros(16, dtype=np.float32)

    # ===== FFT magnitude =====
    x = window.copy()

    # Hamming window
    x *= np.hamming(N)

    # FFT
    X = np.fft.fft(x)
    Y = np.abs(X[:K])

    # ===== Sum spectrum =====
    sumY = np.sum(Y) + 1e-12

    # ===== Mean =====
    mean = sumY / K

    # ===== Variance =====
    var = np.sum((Y - mean) ** 2) / K

    # ===== Skewness & Kurtosis =====
    stdY = np.sqrt(var) + 1e-12
    z = (Y - mean) / stdY
    skew = np.sum(z ** 3) / K
    kurt = np.sum(z ** 4) / K - 3.0

    # ===== Central frequency =====
    freqs = fs * np.arange(K) / N
    central = np.sum(freqs * Y) / sumY

    # ===== Frequency STD =====
    std_freq = np.sqrt(np.sum((freqs - central) ** 2 * Y) / sumY)

    # ===== RMS frequency =====
    rms = np.sqrt(np.sum(freqs ** 2 * Y) / sumY)

    # ===== Spectral centroid (bin-based) =====
    sc = np.sum(np.arange(K) * Y) / sumY

    # ===== Spectral spread =====
    ss = np.sqrt(np.sum((np.arange(K) - sc) ** 2 * Y) / sumY)

    # ===== Spectral entropy =====
    p = Y / sumY
    entropy = -np.sum(p * np.log2(p + 1e-12))

    # ===== OUTPUT – GIỮ NGUYÊN INDEX =====
    return [
    mean,
    rms,
    var,
    skew,
    kurt,
    central,
    std_freq,
    sc,
    ss,
    entropy
    ]




import numpy as np
from scipy.signal import stft
from scipy.stats import skew, kurtosis

def time_frequency_domain_features(window, fs, nperseg=128, noverlap=64):
    """
    Trích xuất đặc trưng miền Time-Frequency dựa trên STFT
    """

    # Biến đổi STFT
    f, t, Zxx = stft(window, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Zxx)              # Spectrogram (magnitude)
    S_power = S ** 2             # Power spectrogram

    # Chuẩn hóa
    S_norm = S_power / (np.sum(S_power) + 1e-12)

    # Các đặc trưng TF1 - TF10

    # 1. Mean energy
    Mean_energy = np.mean(S_power)

    # 2. Variance of energy
    Var_energy = np.var(S_power)

    # 3. Skewness of energy
    Skewness_energy = skew(S_power.flatten())

    # 4. Kurtosis of energy
    Kurtosis_energy = kurtosis(S_power.flatten())

    # 5. Time-Frequency centroid
    F, T = np.meshgrid(f, t, indexing='ij')
    TF_centroid = np.sum(F * S_power) / (np.sum(S_power) + 1e-12)

    # 6. Time-Frequency spread
    TF_spread = np.sqrt(
        np.sum(((F - TF_centroid) ** 2) * S_power) / (np.sum(S_power) + 1e-12)
    )

    # 7. RMS frequency (time–frequency)
    RMS_freq = np.sqrt(
        np.sum((F ** 2) * S_power) / (np.sum(S_power) + 1e-12)
    )

    # 8. Mean temporal energy
    Mean_time_energy = np.mean(np.sum(S_power, axis=0))

    # 9. Spectral flux (biến thiên phổ theo thời gian)
    Spectral_flux = np.mean(
        np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    )

    # 10. Time-Frequency entropy
    TF_entropy = -np.sum(S_norm * np.log2(S_norm + 1e-12))

    return [
        Mean_energy,
        RMS_freq,
        Var_energy,
        Skewness_energy,
        Kurtosis_energy,
        TF_centroid,
        TF_spread,
        Mean_time_energy,
        Spectral_flux,
        TF_entropy
    ]
