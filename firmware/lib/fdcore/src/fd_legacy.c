/* Re-implementation of the graduation-thesis pipeline (ArduinoFFT-style complex FFT,
 * windowing with cos() per call, separate feature loops with per-bin divisions).
 * It produces the same features as the optimized pipeline and exists only to measure
 * the cost of the original implementation on the same hardware. */
#include <float.h>
#include <math.h>
#include "fd_core.h"

static void legacy_window(float *re)
{
    const float two_pi = 6.28318531f;
    for (int i = 0; i < FD_N / 2; i++) {
        const float ratio = (float)i / (float)(FD_N - 1);
        const float w = 0.54f - 0.46f * cosf(two_pi * ratio);
        re[i] *= w;
        re[FD_N - 1 - i] *= w;
    }
}

static void legacy_fft(float *re, float *im)
{
    int power = 0;
    while ((1 << power) < FD_N) power++;
    for (int i = 0, j = 0; i < FD_N - 1; i++) {        /* bit reversal */
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
        int k = FD_N >> 1;
        while (k <= j) { j -= k; k >>= 1; }
        j += k;
    }
    float c1 = -1.0f, c2 = 0.0f;
    int l2 = 1;
    for (int l = 0; l < power; l++) {
        const int l1 = l2;
        l2 <<= 1;
        float u1 = 1.0f, u2 = 0.0f;
        for (int j = 0; j < l1; j++) {
            for (int i = j; i < FD_N; i += l2) {
                const int i1 = i + l1;
                const float t1 = u1 * re[i1] - u2 * im[i1];
                const float t2 = u1 * im[i1] + u2 * re[i1];
                re[i1] = re[i] - t1;
                im[i1] = im[i] - t2;
                re[i] += t1;
                im[i] += t2;
            }
            const float z = u1 * c1 - u2 * c2;
            u2 = u1 * c2 + u2 * c1;
            u1 = z;
        }
        c2 = -sqrtf((1.0f - c1) / 2.0f);                /* forward transform */
        c1 = sqrtf((1.0f + c1) / 2.0f);
    }
}

static void legacy_freq_features(const float *Y, float *feat)
{
    const int K = FD_K;
    const float fs = (float)FD_FS;
    float sumY = 0.0f;
    for (int i = 0; i < K; i++) sumY += Y[i];
    sumY += 1e-12f;
    const float mean = sumY / K;
    float var = 0.0f;
    for (int i = 0; i < K; i++) { const float d = Y[i] - mean; var += d * d; }
    var /= K;
    const float stdY = sqrtf(var) + 1e-12f;
    float skew = 0.0f, kurt = 0.0f;
    for (int i = 0; i < K; i++) {
        const float z = (Y[i] - mean) / stdY;
        skew += z * z * z;
        kurt += z * z * z * z;
    }
    skew /= K;
    kurt = kurt / K - 3.0f;
    float central = 0.0f;
    for (int i = 0; i < K; i++) { const float f = fs * i / FD_N; central += f * Y[i]; }
    central /= sumY;
    float std_freq = 0.0f;
    for (int i = 0; i < K; i++) { const float f = fs * i / FD_N; const float d = f - central; std_freq += d * d * Y[i]; }
    std_freq = sqrtf(std_freq / sumY);
    float rms = 0.0f;
    for (int i = 0; i < K; i++) { const float f = fs * i / FD_N; rms += f * f * Y[i]; }
    rms = sqrtf(rms / sumY);
    float sc = 0.0f;
    for (int i = 0; i < K; i++) sc += i * Y[i];
    sc /= sumY;
    float ss = 0.0f;
    for (int i = 0; i < K; i++) { const float d = i - sc; ss += d * d * Y[i]; }
    ss = sqrtf(ss / sumY);
    float entropy = 0.0f;
    for (int i = 0; i < K; i++) { const float p = Y[i] / sumY; entropy -= p * log2f(p + 1e-12f); }
    feat[FD_F_MEAN_F] = mean;
    feat[FD_F_RMSF_F] = rms;
    feat[FD_F_VAR_F] = var;
    feat[FD_F_SKEW_F] = skew;
    feat[FD_F_KURT_F] = kurt;
    feat[FD_F_FC_F] = central;
    feat[FD_F_FSTD_F] = std_freq;
    feat[FD_F_SC_F] = sc;
    feat[FD_F_SS_F] = ss;
    feat[FD_F_ENT_F] = entropy;
}

#define FD_T0() uint32_t _t = fd_cycles()
#define FD_TSTAMP(st)                                  \
    do {                                                \
        if (profile) {                                  \
            uint32_t _n = fd_cycles();                  \
            profile->cycles[st] = _n - _t;              \
            _t = _n;                                    \
        }                                               \
    } while (0)

void fd_extract_features_legacy(const int16_t *in, float scale, float *re, float *im, float *mag,
                                float *feat, fd_profile_t *profile)
{
    FD_T0();                                                         /* includes int16 -> float, as the optimized path */
    for (int i = 0; i < FD_N; i++) re[i] = (float)in[i] * scale;   /* "window" buffer of the thesis */
    fd_time_features_f32(re, feat);
    FD_TSTAMP(FD_ST_TIME);
    for (int i = 0; i < FD_N; i++) im[i] = 0.0f;
    legacy_window(re);
    FD_TSTAMP(FD_ST_WINDOW);
    legacy_fft(re, im);
    FD_TSTAMP(FD_ST_FFT);
    for (int i = 0; i < FD_N; i++) re[i] = sqrtf(re[i] * re[i] + im[i] * im[i]);   /* complexToMagnitude */
    for (int i = 0; i < FD_K; i++) mag[i] = re[i];
    FD_TSTAMP(FD_ST_MAG);
    legacy_freq_features(mag, feat);
    FD_TSTAMP(FD_ST_FREQ);
    fd_order_features(mag, feat);
    FD_TSTAMP(FD_ST_ORDER);
}
