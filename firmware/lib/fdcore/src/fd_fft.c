/* In-place real FFT magnitude via an N/2-point complex radix-2 FFT. */
#include <math.h>
#include "fd_core.h"

#define FD_M (FD_N / 2)

/* Complex radix-2 DIT FFT of size FD_M on interleaved data z[2*i], z[2*i+1].
 * Twiddles W_M^t = W_N^(2t) are read from the length-N/2 table of W_N^k. */
static void fd_cfft_m(float *z)
{
    /* bit reversal */
    for (int i = 1, j = 0; i < FD_M; i++) {
        int bit = FD_M >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = z[2 * i], ti = z[2 * i + 1];
            z[2 * i] = z[2 * j];
            z[2 * i + 1] = z[2 * j + 1];
            z[2 * j] = tr;
            z[2 * j + 1] = ti;
        }
    }
    for (int len = 2; len <= FD_M; len <<= 1) {
        const int half = len >> 1;
        const int tstep = FD_N / len;              /* W_M^(j*M/len) = W_N^(j*N/len) */
        for (int j = 0; j < half; j++) {
            const float wr = fd_tw_cos[j * tstep];
            const float wi = -fd_tw_sin[j * tstep];
            for (int i = j; i < FD_M; i += len) {
                float *a = &z[2 * i];
                float *b = &z[2 * (i + half)];
                const float xr = b[0] * wr - b[1] * wi;
                const float xi = b[0] * wi + b[1] * wr;
                b[0] = a[0] - xr;
                b[1] = a[1] - xi;
                a[0] += xr;
                a[1] += xi;
            }
        }
    }
}

void fd_rfft_mag(float *work)
{
    /* The real sequence x[0..N-1] is read as M complex values z[m] = x[2m] + j x[2m+1]. */
    fd_cfft_m(work);

    /* Split step: X[k] = Fe[k] + W_N^k Fo[k]; magnitudes of bins k and M-k are
     * written into the even slots of Z[k] and Z[M-k], then compacted. */
    work[0] = fabsf(work[0] + work[1]);
    for (int k = 1; k <= FD_M / 2; k++) {
        const int m = FD_M - k;
        const float ar = work[2 * k], ai = work[2 * k + 1];
        const float br = work[2 * m], bi = work[2 * m + 1];
        const float fer = 0.5f * (ar + br), fei = 0.5f * (ai - bi);
        const float forr = 0.5f * (ai + bi), foi = -0.5f * (ar - br);
        const float c = fd_tw_cos[k], s = fd_tw_sin[k];
        const float tr = c * forr + s * foi;          /* W^k * Fo, W^k = c - j s */
        const float ti = c * foi - s * forr;
        const float xr = fer + tr, xi = fei + ti;
        work[2 * k] = sqrtf(xr * xr + xi * xi);
        if (m != k) {
            const float yr = fer - c * forr - s * foi;  /* X[M-k] = conj(Fe) + W^(M-k) conj(Fo) */
            const float yi = -fei - s * forr + c * foi;
            work[2 * m] = sqrtf(yr * yr + yi * yi);
        }
    }
    for (int i = 1; i < FD_M; i++) work[i] = work[2 * i];
}

/* Same algorithm for n real points (n a power of two dividing FD_N, e.g. the squared envelope): the twiddles
 * W_n^k = W_N^(k N / n) come from the same tables, read with a stride. */
void fd_rfft_mag_n(float *work, int n)
{
    const int M = n / 2, stride = FD_N / n;
    for (int i = 1, j = 0; i < M; i++) {
        int bit = M >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = work[2 * i], ti = work[2 * i + 1];
            work[2 * i] = work[2 * j];
            work[2 * i + 1] = work[2 * j + 1];
            work[2 * j] = tr;
            work[2 * j + 1] = ti;
        }
    }
    for (int len = 2; len <= M; len <<= 1) {
        const int half = len >> 1;
        const int tstep = FD_N / len;                  /* W_len^j = W_N^(j N / len) */
        for (int j = 0; j < half; j++) {
            const float wr = fd_tw_cos[j * tstep];
            const float wi = -fd_tw_sin[j * tstep];
            for (int i = j; i < M; i += len) {
                float *a = &work[2 * i];
                float *b = &work[2 * (i + half)];
                const float xr = b[0] * wr - b[1] * wi;
                const float xi = b[0] * wi + b[1] * wr;
                b[0] = a[0] - xr;
                b[1] = a[1] - xi;
                a[0] += xr;
                a[1] += xi;
            }
        }
    }
    work[0] = fabsf(work[0] + work[1]);
    for (int k = 1; k <= M / 2; k++) {
        const int m = M - k;
        const float ar = work[2 * k], ai = work[2 * k + 1];
        const float br = work[2 * m], bi = work[2 * m + 1];
        const float fer = 0.5f * (ar + br), fei = 0.5f * (ai - bi);
        const float forr = 0.5f * (ai + bi), foi = -0.5f * (ar - br);
        const float c = fd_tw_cos[k * stride], s = fd_tw_sin[k * stride];
        const float tr = c * forr + s * foi;
        const float ti = c * foi - s * forr;
        const float xr = fer + tr, xi = fei + ti;
        work[2 * k] = sqrtf(xr * xr + xi * xi);
        if (m != k) {
            const float yr = fer - c * forr - s * foi;
            const float yi = -fei - s * forr + c * foi;
            work[2 * m] = sqrtf(yr * yr + yi * yi);
        }
    }
    for (int i = 1; i < M; i++) work[i] = work[2 * i];
}
