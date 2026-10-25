/* Self-check gate (block S): within-class Mahalanobis distance of the standardized features to the nearest prototype
 * of the predicted class, plus head disagreement; trailing mean; re-referencing with labelled support windows.
 * Definitions as in fdedge/gate.py. */
#include <math.h>
#include <string.h>
#include "fd_core.h"

#define D FD_SM_D

void fd_gate_reset(fd_gate_t *g)
{
    g->head = 0;
    g->count = 0;
}

float fd_gate_score(const fd_gate_t *g, const float *z, int cls, int disagree)
{
    float best = INFINITY, r[D];
    for (int k = 0; k < g->n_protos; k++) {
        if (g->pcls[k] != cls) continue;
        const float *p = &g->protos[k * D];
        for (int j = 0; j < D; j++) r[j] = z[j] - p[j];
        float q = 0.0f;
        for (int i = 0; i < D; i++) {
            const float *w = &g->Pw[i * D];
            float acc = 0.0f;
            for (int j = 0; j < D; j++) acc += w[j] * r[j];
            q += r[i] * acc;
        }
        if (q < best) best = q;
    }
    if (!(best < INFINITY)) best = 0.0f;
    return log1pf(best / (float)D) + (disagree ? 1.0f : 0.0f);
}

float fd_gate_push(fd_gate_t *g, float s)
{
    g->ring[g->head] = s;
    g->head = (g->head + 1) % g->w;
    if (g->count < g->w) g->count++;
    if (g->count < g->w) return -1.0f;
    float acc = 0.0f;                     /* recomputed every window: no drift of a running sum */
    for (int k = 0; k < g->w; k++) acc += g->ring[k];
    return acc / (float)g->w;
}

/* Cholesky factorization and inversion of the D x D matrix A (double, in place: A <- inv(A)). */
static int spd_inverse(double *A)
{
    static double L[D * D], M[D * D];
    memset(L, 0, sizeof(L));
    for (int j = 0; j < D; j++) {
        double s = A[j * D + j];
        for (int k = 0; k < j; k++) s -= L[j * D + k] * L[j * D + k];
        if (s <= 0.0) return -1;
        L[j * D + j] = sqrt(s);
        for (int i = j + 1; i < D; i++) {
            double t = A[i * D + j];
            for (int k = 0; k < j; k++) t -= L[i * D + k] * L[j * D + k];
            L[i * D + j] = t / L[j * D + j];
        }
    }
    memset(M, 0, sizeof(M));                  /* inverse of L, then inv(A) = inv(L)' inv(L) */
    for (int i = 0; i < D; i++) {
        M[i * D + i] = 1.0 / L[i * D + i];
        for (int j = 0; j < i; j++) {
            double s = 0.0;
            for (int k = j; k < i; k++) s -= L[i * D + k] * M[k * D + j];
            M[i * D + j] = s / L[i * D + i];
        }
    }
    for (int i = 0; i < D; i++)
        for (int j = 0; j <= i; j++) {
            double s = 0.0;
            for (int k = i; k < D; k++) s += M[k * D + i] * M[k * D + j];
            A[i * D + j] = A[j * D + i] = s;
        }
    return 0;
}

int fd_gate_rereference(fd_gate_t *g, const float *z, const uint8_t *y, int n, float weight)
{
    static double S[D * D];
    const double w = weight;
    for (int i = 0; i < D * D; i++) S[i] = g->Sw[i];
    for (int c = 0; c < FD_N_CLASSES; c++) {
        double sum[D] = {0}, mu_new[D];
        int m = 0;
        for (int k = 0; k < n; k++)
            if (y[k] == c) {
                for (int j = 0; j < D; j++) sum[j] += z[k * D + j];
                m++;
            }
        if (!m) continue;
        const double nc = g->n_c[c], n_new = nc + w * m;
        for (int j = 0; j < D; j++) mu_new[j] = (nc * g->mu[c * D + j] + w * sum[j]) / n_new;
        /* S += n_c mu mu' + w sum_s z z' - n_new mu_new mu_new' */
        for (int i = 0; i < D; i++)
            for (int j = 0; j <= i; j++) {
                double zz = 0.0;
                for (int k = 0; k < n; k++)
                    if (y[k] == c) zz += (double)z[k * D + i] * z[k * D + j];
                const double v = nc * (double)g->mu[c * D + i] * g->mu[c * D + j] + w * zz - n_new * mu_new[i] * mu_new[j];
                S[i * D + j] += v;
                if (j != i) S[j * D + i] += v;
            }
        for (int j = 0; j < D; j++) g->mu[c * D + j] = (float)mu_new[j];
        g->n_c[c] = (float)n_new;
    }
    g->n += (float)(w * n);
    static double A[D * D];
    for (int i = 0; i < D * D; i++) {
        g->Sw[i] = (float)S[i];
        A[i] = S[i] / ((double)g->n - 1.0);
    }
    for (int i = 0; i < D; i++) A[i * D + i] += g->ridge;
    if (spd_inverse(A) != 0) return -1;
    for (int i = 0; i < D * D; i++) g->Pw[i] = (float)A[i];
    for (int k = 0; k < n && g->n_protos < FD_GATE_PMAX; k++) {     /* support windows become prototypes */
        memcpy(&g->protos[g->n_protos * D], &z[k * D], D * sizeof(float));
        g->pcls[g->n_protos++] = y[k];
    }
    fd_gate_reset(g);
    return 0;
}
