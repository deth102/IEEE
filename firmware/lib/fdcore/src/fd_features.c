/* Time, spectral and shaft-order features (optimized pipeline). */
#include <float.h>
#include <math.h>
#include "fd_core.h"

#define FD_BLK 64   /* block size of the two-level summation (limits float32 round-off) */

#ifndef FD_LOG2
#define FD_LOG2 log2f
#endif

static void fd_time_from_sums(float sum, float sumsq, float mx, float mn, float *feat)
{
    const float mean = sum / (float)FD_N;
    const float rms = sqrtf(sumsq / (float)FD_N);
    feat[FD_F_MEAN_T] = mean;
    feat[FD_F_RMS_T] = rms;
    feat[FD_F_MAX_T] = mx;
    feat[FD_F_MIN_T] = mn;
    feat[FD_F_PTP_T] = mx - mn;
    feat[FD_F_CREST_T] = mx / (rms + 1e-9f);
}

void fd_time_features_i16(const int16_t *in, float scale, float *work, float *feat)
{
    float sum = 0.0f, sumsq = 0.0f, mx = -FLT_MAX, mn = FLT_MAX;
    for (int b = 0; b < FD_N; b += FD_BLK) {
        float s = 0.0f, s2 = 0.0f;
        for (int i = b; i < b + FD_BLK; i++) {
            const float v = (float)in[i] * scale;
            work[i] = v;
            s += v;
            s2 += v * v;
            if (v > mx) mx = v;
            if (v < mn) mn = v;
        }
        sum += s;
        sumsq += s2;
    }
    fd_time_from_sums(sum, sumsq, mx, mn, feat);
}

void fd_time_features_f32(const float *x, float *feat)
{
    float sum = 0.0f, sumsq = 0.0f, mx = -FLT_MAX, mn = FLT_MAX;
    for (int b = 0; b < FD_N; b += FD_BLK) {
        float s = 0.0f, s2 = 0.0f;
        for (int i = b; i < b + FD_BLK; i++) {
            const float v = x[i];
            s += v;
            s2 += v * v;
            if (v > mx) mx = v;
            if (v < mn) mn = v;
        }
        sum += s;
        sumsq += s2;
    }
    fd_time_from_sums(sum, sumsq, mx, mn, feat);
}

void fd_apply_window(float *work)
{
    for (int i = 0; i < FD_N / 2; i++) {
        const float w = fd_hamming_half[i];     /* symmetric window: w[N-1-i] = w[i] */
        work[i] *= w;
        work[FD_N - 1 - i] *= w;
    }
}

void fd_freq_features(const float *Y, float *feat)
{
    const float df = (float)FD_FS / (float)FD_N;

    /* pass 1: sum Y, sum k Y, sum k^2 Y */
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    for (int b = 0; b < FD_K; b += FD_BLK) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
        float kf = (float)b;
        for (int k = b; k < b + FD_BLK; k++) {
            const float y = Y[k];
            p0 += y;
            p1 += kf * y;
            p2 += kf * kf * y;
            kf += 1.0f;
        }
        s0 += p0;
        s1 += p1;
        s2 += p2;
    }
    const float sumY = s0 + 1e-12f;
    const float mean = sumY / (float)FD_K;
    const float inv_sum = 1.0f / sumY;
    const float sc = s1 * inv_sum;

    /* pass 2: variance of Y and spectral spread around the centroid */
    float v = 0.0f, sp = 0.0f;
    for (int b = 0; b < FD_K; b += FD_BLK) {
        float pv = 0.0f, ps = 0.0f;
        float kf = (float)b;
        for (int k = b; k < b + FD_BLK; k++) {
            const float y = Y[k];
            const float d = y - mean;
            const float e = kf - sc;
            pv += d * d;
            ps += e * e * y;
            kf += 1.0f;
        }
        v += pv;
        sp += ps;
    }
    const float var = v / (float)FD_K;
    const float stdY = sqrtf(var) + 1e-12f;
    const float inv_std = 1.0f / stdY;

    /* pass 3: skewness, kurtosis, entropy */
    float m3 = 0.0f, m4 = 0.0f, ent = 0.0f;
    for (int b = 0; b < FD_K; b += FD_BLK) {
        float p3 = 0.0f, p4 = 0.0f, pe = 0.0f;
        for (int k = b; k < b + FD_BLK; k++) {
            const float y = Y[k];
            const float z = (y - mean) * inv_std;
            const float z2 = z * z;
            p3 += z2 * z;
            p4 += z2 * z2;
            const float p = y * inv_sum;
            pe -= p * FD_LOG2(p + 1e-12f);
        }
        m3 += p3;
        m4 += p4;
        ent += pe;
    }

    const float ss = sqrtf(sp * inv_sum);
    feat[FD_F_MEAN_F] = mean;
    feat[FD_F_RMSF_F] = df * sqrtf(s2 * inv_sum);
    feat[FD_F_VAR_F] = var;
    feat[FD_F_SKEW_F] = m3 / (float)FD_K;
    feat[FD_F_KURT_F] = m4 / (float)FD_K - 3.0f;
    feat[FD_F_FC_F] = df * sc;
    feat[FD_F_FSTD_F] = df * ss;
    feat[FD_F_SC_F] = sc;
    feat[FD_F_SS_F] = ss;
    feat[FD_F_ENT_F] = ent;
}

void fd_order_features(const float *Y, float *feat)
{
    int k1 = FD_BAND_LO;
    float a1 = Y[FD_BAND_LO];
    for (int k = FD_BAND_LO + 1; k <= FD_BAND_HI; k++) {
        if (Y[k] > a1) {
            a1 = Y[k];
            k1 = k;
        }
    }
    int lo = 2 * k1 - 1, hi = 2 * k1 + 1;
    if (lo < 0) lo = 0;
    if (hi > FD_K - 1) hi = FD_K - 1;
    float a2 = Y[lo];
    for (int k = lo + 1; k <= hi; k++)
        if (Y[k] > a2) a2 = Y[k];
    feat[FD_F_A1X_O] = a1;
    feat[FD_F_A2X_O] = a2;
    feat[FD_F_A1XR_O] = a1 / (feat[FD_F_RMS_T] + 1e-9f);
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

void fd_extract_features(const int16_t *in, float scale, float *work, float *feat, fd_profile_t *profile)
{
    FD_T0();
    fd_time_features_i16(in, scale, work, feat);
    FD_TSTAMP(FD_ST_TIME);
    fd_apply_window(work);
    FD_TSTAMP(FD_ST_WINDOW);
    fd_rfft_mag(work);                  /* FFT + split + magnitude fused */
    FD_TSTAMP(FD_ST_FFT);
    if (profile) profile->cycles[FD_ST_MAG] = 0;
    fd_freq_features(work, feat);
    FD_TSTAMP(FD_ST_FREQ);
    fd_order_features(work, feat);
    FD_TSTAMP(FD_ST_ORDER);
#if FD_ENV_BANDS > 0
    for (int j = 0; j < FD_ENV_BANDS; j++)          /* the upper half of work is free after the spectrum */
        fd_envelope_features(in, scale, j, work + FD_K, feat);
    FD_TSTAMP(FD_ST_ENV);
#endif
}
