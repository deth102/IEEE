"""Hand-crafted vibration features.

The definitions mirror the embedded implementation in
``firmware/lib/fdcore/src/fd_features.c`` one-to-one (same order, same
constants), so a model trained here can be exported to the MCU unchanged.

Feature vector (16 values):
  0 mean_t        time-domain mean
  1 rms_t         time-domain RMS
  2 max_t         maximum
  3 min_t         minimum
  4 ptp_t         peak-to-peak (max - min)
  5 crest_t       crest factor max / RMS
  6 mean_f        mean of the magnitude spectrum Y[k], k = 0..N/2-1
  7 rmsf_f        RMS frequency  sqrt(sum f^2 Y / sum Y)
  8 var_f         variance of Y
  9 skew_f        skewness of Y
 10 kurt_f        excess kurtosis of Y
 11 fc_f          central (mean) frequency  sum f Y / sum Y
 12 fstd_f        frequency standard deviation around fc
 13 sc_f          spectral centroid in bins (= fc * N / fs)
 14 ss_f          spectral spread in bins   (= fstd * N / fs)
 15 ent_f         spectral entropy (bits)

Shaft-order features (proposed extension, computed from the same spectrum):
 16 a1x_o       peak |Y| inside the shaft-speed search band [f_lo, f_hi]  (1X)
 17 a2x_o       max |Y| over bins 2*k1-1 .. 2*k1+1                         (2X)
 18 a1xr_o      a1x / (rms_t + 1e-9)   (load-normalised 1X level)

Bearing-order (envelope) features, enabled per dataset (``envelope`` block of the config), one triple per
band in ``bands_hz`` (indices 19, 20, 21 for the first band):
 19 bpfo_e      log ratio of the squared-envelope spectrum at 1x and 2x the outer-race order to its mean level
 20 bdf_e       same at the ball-defect order (2 x ball spin)
 21 bpfi_e      same at the inner-race order
The squared envelope is computed as on the MCU: biquad band-pass, square, block mean over ``decim`` samples,
Hann window, real FFT of N/decim points; each order is searched over the machine's speed range.

Y is |FFT(x * hamming)| of the window (DC..N/2-1, Nyquist excluded).
Note: features 13/14 are exact rescalings of 11/12 and 4 = 2 - 3; the
redundancy is kept for compatibility with the conference/thesis model and is
analysed in the feature-ablation experiment.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "mean_t", "rms_t", "max_t", "min_t", "ptp_t", "crest_t",
    "mean_f", "rmsf_f", "var_f", "skew_f", "kurt_f",
    "fc_f", "fstd_f", "sc_f", "ss_f", "ent_f",
    "a1x_o", "a2x_o", "a1xr_o",
]
N_FEATURES = len(FEATURE_NAMES)
TIME_IDX = list(range(0, 6))
FREQ_IDX = list(range(6, 16))
ORDER_IDX = [16, 17, 18]
# Non-redundant subset (drops ptp, sc, ss which are affine functions of others)
NONREDUNDANT_IDX = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15]

FEATURE_SETS = {
    "F16": TIME_IDX + FREQ_IDX,                      # conference / thesis feature set
    "F19": TIME_IDX + FREQ_IDX + ORDER_IDX,          # proposed: + shaft-order features
    "T6": TIME_IDX,
    "S10": FREQ_IDX,
    "O3": ORDER_IDX,
    "F13nr": NONREDUNDANT_IDX,
    "F16nr": NONREDUNDANT_IDX + ORDER_IDX,           # non-redundant + order
}
ENV_NAMES = ["bpfo_e", "bdf_e", "bpfi_e"]
ENV_IDX = [19, 20, 21]                               # bearing orders of the first (selected) envelope band
FEATURE_SETS.update({
    "E3": ENV_IDX,                                   # bearing-order (envelope) features alone
    "F16E": TIME_IDX + FREQ_IDX + ENV_IDX,           # baseline + bearing orders
    "F22": TIME_IDX + FREQ_IDX + ORDER_IDX + ENV_IDX,  # shaft orders + bearing orders
})


def env_band_idx(j: int) -> list[int]:
    """Columns of the bearing-order triple of envelope band j."""
    return [N_FEATURES + 3 * j + i for i in range(3)]


def band_bins(n: int, fs: float, band_hz) -> tuple[int, int]:
    """Inclusive bin range of the shaft-speed search band."""
    lo = int(np.floor(band_hz[0] * n / fs))
    hi = int(np.floor(band_hz[1] * n / fs))
    return max(lo, 1), min(hi, n // 2 - 2)

_EPS_T = 1e-9
_EPS_F = 1e-12


def hamming(n: int) -> np.ndarray:
    """Symmetric Hamming window (same as numpy.hamming / ArduinoFFT)."""
    i = np.arange(n, dtype=np.float64)
    return (0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n - 1))).astype(np.float32)


def magnitude_spectrum(x: np.ndarray) -> np.ndarray:
    """|FFT| of Hamming-windowed rows, bins 0..N/2-1, float32. x: (n, N)."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    spec = np.fft.rfft(x * hamming(n), axis=-1)[..., : n // 2]
    return np.abs(spec).astype(np.float32)


def time_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1)
    rms = np.sqrt((x * x).mean(axis=1))
    mx = x.max(axis=1)
    mn = x.min(axis=1)
    return np.stack([mean, rms, mx, mn, mx - mn, mx / (rms + _EPS_T)], axis=1)


def freq_features_from_spectrum(Y: np.ndarray, fs: float, n: int) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    K = Y.shape[1]
    k = np.arange(K, dtype=np.float64)
    f = fs * k / n
    sumY = Y.sum(axis=1) + _EPS_F
    mean = sumY / K
    d = Y - mean[:, None]
    var = (d * d).sum(axis=1) / K
    stdY = np.sqrt(var) + _EPS_F
    z = d / stdY[:, None]
    skew = (z ** 3).sum(axis=1) / K
    kurt = (z ** 4).sum(axis=1) / K - 3.0
    fc = (f * Y).sum(axis=1) / sumY
    fstd = np.sqrt((((f[None, :] - fc[:, None]) ** 2) * Y).sum(axis=1) / sumY)
    rmsf = np.sqrt(((f * f) * Y).sum(axis=1) / sumY)
    sc = (k * Y).sum(axis=1) / sumY
    ss = np.sqrt((((k[None, :] - sc[:, None]) ** 2) * Y).sum(axis=1) / sumY)
    p = Y / sumY[:, None]
    ent = -(p * np.log2(p + _EPS_F)).sum(axis=1)
    return np.stack([mean, rmsf, var, skew, kurt, fc, fstd, sc, ss, ent], axis=1).astype(np.float32)


def order_features(Y: np.ndarray, rms_t: np.ndarray, lo: int, hi: int) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32)
    K = Y.shape[1]
    k1 = lo + np.argmax(Y[:, lo:hi + 1], axis=1)          # first maximum, as in C
    rows = np.arange(len(Y))
    a1 = Y[rows, k1]
    a2 = np.maximum.reduce([Y[rows, np.clip(2 * k1 + d, 0, K - 1)] for d in (-1, 0, 1)])
    return np.stack([a1, a2, a1 / (rms_t + _EPS_T)], axis=1).astype(np.float32)


def envelope_spectrum(x: np.ndarray, fs: float, band_hz, decim: int) -> tuple[np.ndarray, float]:
    """Squared-envelope spectrum of each row: 4th-order Butterworth band-pass (two biquads, zero initial
    state), square, block mean over ``decim`` samples, mean removal, periodic Hann window, real FFT.
    Returns |E[k]|, k = 0..M/2-1 (M = N/decim), and the envelope sampling rate fs/decim."""
    from scipy.signal import butter, sosfilt
    sos = butter(2, band_hz, btype="bandpass", fs=fs, output="sos")
    y = sosfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)
    e = (y * y).reshape(y.shape[0], -1, decim).mean(axis=2)
    e = e - e.mean(axis=1, keepdims=True)
    m = e.shape[1]
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(m) / m)
    return np.abs(np.fft.rfft(e * hann, axis=1))[:, : m // 2], fs / decim


def envelope_features(E: np.ndarray, fs_env: float, orders, speed_hz, harmonics: int = 2) -> np.ndarray:
    """log( sum over harmonics of the peak inside h*order*[speed range] / (harmonics * mean level) )."""
    k2 = E.shape[1]
    res = fs_env / (2 * k2)                                   # Hz per envelope bin
    ref = E[:, 1:].mean(axis=1) + _EPS_F
    out = []
    for c in orders:
        acc = np.zeros(len(E))
        for h in range(1, harmonics + 1):
            lo = max(1, int(np.floor(h * c * speed_hz[0] / res)))
            hi = min(k2 - 1, int(np.ceil(h * c * speed_hz[1] / res)))
            acc += E[:, lo:hi + 1].max(axis=1)
        out.append(np.log(acc / (harmonics * ref) + _EPS_F))
    return np.stack(out, axis=1).astype(np.float32)


def extract_features(x: np.ndarray, fs: float, band_hz=(30.0, 70.0), chunk: int = 512, env=None) -> np.ndarray:
    """Compute the 19-D feature vector for each row of ``x`` (n_windows, N); with ``env`` (envelope config),
    append one bearing-order triple per envelope band."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    n = x.shape[1]
    lo, hi = band_bins(n, fs, band_hz)
    bands = list(env["bands_hz"]) if env else []
    out = np.empty((x.shape[0], N_FEATURES + 3 * len(bands)), dtype=np.float32)
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]
        Y = magnitude_spectrum(xb)
        out[s:s + chunk, :6] = time_features(xb)
        out[s:s + chunk, 6:16] = freq_features_from_spectrum(Y, fs, n)
        out[s:s + chunk, 16:19] = order_features(Y, out[s:s + chunk, 1], lo, hi)
        for j, band in enumerate(bands):
            E, fs_env = envelope_spectrum(xb, fs, band, int(env["decim"]))
            out[s:s + chunk, env_band_idx(j)] = envelope_features(E, fs_env, env["orders"], env["speed_hz"],
                                                                  int(env.get("harmonics", 2)))
    return out


def cnn_input(x: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Log-magnitude spectrum normalised to [0, 1] per window (CNN baseline input)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    n = x.shape[1]
    out = np.empty((x.shape[0], n // 2), dtype=np.float32)
    for s in range(0, x.shape[0], chunk):
        Y = np.log1p(np.abs(np.fft.rfft(x[s:s + chunk], axis=1))[:, 1: n // 2 + 1])
        out[s:s + chunk] = (Y / (Y.max(axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    return out
