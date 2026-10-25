/* Bearing-order features from the squared envelope (block K for bearing faults); definitions as in
 * fdedge/features.py (envelope_spectrum, envelope_features). Built only when the model config sets FD_ENV_BANDS. */
#include <math.h>
#include "fd_core.h"

#if FD_ENV_BANDS > 0
#define FD_ENV_M (FD_N / FD_ENV_DECIM)

void fd_envelope_features(const int16_t *in, float scale, int band, float *e, float *feat)
{
    /* 4th-order Butterworth band-pass as two biquads (transposed direct form II, zero initial state),
     * squared and averaged over blocks of FD_ENV_DECIM samples */
    const float *c = &fd_env_sos[band * 10];         /* per section: b0 b1 b2 a1 a2 */
    float u1 = 0.0f, u2 = 0.0f, v1 = 0.0f, v2 = 0.0f;
    const float inv_d = 1.0f / (float)FD_ENV_DECIM;
    float mean = 0.0f;
    for (int m = 0; m < FD_ENV_M; m++) {
        float acc = 0.0f;
        const int16_t *x = &in[m * FD_ENV_DECIM];
        for (int i = 0; i < FD_ENV_DECIM; i++) {
            const float xi = (float)x[i] * scale;
            const float y1 = c[0] * xi + u1;
            u1 = c[1] * xi - c[3] * y1 + u2;
            u2 = c[2] * xi - c[4] * y1;
            const float y2 = c[5] * y1 + v1;
            v1 = c[6] * y1 - c[8] * y2 + v2;
            v2 = c[7] * y1 - c[9] * y2;
            acc += y2 * y2;
        }
        e[m] = acc * inv_d;
        mean += e[m];
    }
    mean /= (float)FD_ENV_M;
    /* mean removal and periodic Hann window, w[i] = 0.5 - 0.5 cos(2 pi i / M), from the FFT cosine table */
    const int stride = FD_N / FD_ENV_M;
    for (int i = 0; i < FD_ENV_M; i++) {
        const int k = (i <= FD_ENV_M / 2) ? i : FD_ENV_M - i;
        const float cs = (k < FD_ENV_M / 2) ? fd_tw_cos[k * stride] : -1.0f;
        e[i] = (e[i] - mean) * (0.5f - 0.5f * cs);
    }
    fd_rfft_mag_n(e, FD_ENV_M);                      /* |E[k]|, k = 0..M/2-1 */
    float ref = 0.0f;
    for (int k = 1; k < FD_ENV_M / 2; k++) ref += e[k];
    ref = ref / (float)(FD_ENV_M / 2 - 1) + 1e-12f;
    const int16_t *bins = &fd_env_bins[band * 12];  /* per order: (lo, hi) of harmonics 1 and 2 */
    for (int o = 0; o < 3; o++) {
        float acc = 0.0f;
        for (int h = 0; h < 2; h++) {
            const int lo = bins[o * 4 + 2 * h], hi = bins[o * 4 + 2 * h + 1];
            float mx = e[lo];
            for (int k = lo + 1; k <= hi; k++)
                if (e[k] > mx) mx = e[k];
            acc += mx;
        }
        feat[FD_F_ENV0 + 3 * band + o] = logf(acc / (2.0f * ref) + 1e-12f);
    }
}
#endif
