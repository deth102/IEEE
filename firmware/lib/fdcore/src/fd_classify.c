/* Random forest, softmax regression (incl. on-device fine-tuning), Welford statistics, voting. */
#include <math.h>
#include <string.h>
#include "fd_core.h"

/* ------------------------------------------------------------------ RF */

static int argmax_u8(const uint8_t *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

static const fd_rf_node_t *rf_nodes = fd_rf_nodes;
static const uint16_t *rf_roots = fd_rf_roots;
static int rf_n_trees = FD_RF_N_TREES;

void fd_rf_set_forest(const fd_rf_node_t *nodes, const uint16_t *roots, int n_trees)
{
    rf_nodes = nodes ? nodes : fd_rf_nodes;
    rf_roots = nodes ? roots : fd_rf_roots;
    rf_n_trees = nodes ? n_trees : FD_RF_N_TREES;
}

int fd_rf_predict(const float *feat, uint8_t *votes)
{
    uint8_t local[FD_N_CLASSES];
    uint8_t *v = votes ? votes : local;
    memset(v, 0, FD_N_CLASSES);
    for (int t = 0; t < rf_n_trees; t++) {
        int i = rf_roots[t];
        while (rf_nodes[i].feat >= 0) {
            const fd_rf_node_t *n = &rf_nodes[i];
            i = (feat[n->feat] <= n->thr) ? i + 1 : n->right;
        }
        v[-rf_nodes[i].feat - 1]++;
    }
    return argmax_u8(v, FD_N_CLASSES);
}

int fd_rf_predict_q16(const float *feat, uint8_t *votes)
{
    int16_t q[FD_N_FEATURES];
    for (int j = 0; j < FD_N_FEATURES; j++) {
        float u = floorf((feat[j] - fd_rf_q_off[j]) * fd_rf_q_inv_step[j]) - 32768.0f;
        if (u < -32768.0f) u = -32768.0f;
        if (u > 32767.0f) u = 32767.0f;
        q[j] = (int16_t)u;
    }
    uint8_t local[FD_N_CLASSES];
    uint8_t *v = votes ? votes : local;
    memset(v, 0, FD_N_CLASSES);
    for (int t = 0; t < FD_RF_N_TREES; t++) {
        int i = fd_rf_roots[t];
        while (fd_rf_nodes_q16[i].feat >= 0) {
            const fd_rf_node_q16_t *n = &fd_rf_nodes_q16[i];
            i = (q[n->feat] <= n->thr) ? i + 1 : n->right;
        }
        v[-fd_rf_nodes_q16[i].feat - 1]++;
    }
    return argmax_u8(v, FD_N_CLASSES);
}

/* ------------------------------------------------------------- softmax */

void fd_softmax_load_default(fd_softmax_params_t *p)
{
    memcpy(p->W, fd_sm_W, sizeof(p->W));
    memcpy(p->b, fd_sm_b, sizeof(p->b));
}

static const uint8_t *sm_idx = fd_sm_idx;
static const float *sm_mu = fd_sm_mu, *sm_inv_sd = fd_sm_inv_sd;

void fd_softmax_set_standardizer(const uint8_t *idx, const float *mu, const float *inv_sd)
{
    sm_idx = idx ? idx : fd_sm_idx;
    sm_mu = idx ? mu : fd_sm_mu;
    sm_inv_sd = idx ? inv_sd : fd_sm_inv_sd;
}

void fd_softmax_standardize(const float *feat, float *z)
{
    for (int j = 0; j < FD_SM_D; j++) z[j] = (feat[sm_idx[j]] - sm_mu[j]) * sm_inv_sd[j];
}

void fd_softmax_logits(const fd_softmax_params_t *p, const float *z, float *logits)
{
    for (int c = 0; c < FD_N_CLASSES; c++) {
        const float *w = &p->W[c * FD_SM_D];
        float acc = p->b[c];
        for (int j = 0; j < FD_SM_D; j++) acc += w[j] * z[j];
        logits[c] = acc;
    }
}

static void softmax_inplace(float *v)
{
    float mx = v[0];
    for (int c = 1; c < FD_N_CLASSES; c++)
        if (v[c] > mx) mx = v[c];
    float s = 0.0f;
    for (int c = 0; c < FD_N_CLASSES; c++) { v[c] = expf(v[c] - mx); s += v[c]; }
    const float inv = 1.0f / s;
    for (int c = 0; c < FD_N_CLASSES; c++) v[c] *= inv;
}

int fd_softmax_predict(const fd_softmax_params_t *p, const float *feat, float *proba)
{
    float z[FD_SM_D], logits[FD_N_CLASSES];
    fd_softmax_standardize(feat, z);
    fd_softmax_logits(p, z, logits);
    int best = 0;
    for (int c = 1; c < FD_N_CLASSES; c++)
        if (logits[c] > logits[best]) best = c;
    if (proba) {
        memcpy(proba, logits, sizeof(logits));
        softmax_inplace(proba);
    }
    return best;
}

void fd_softmax_finetune(fd_softmax_params_t *p, const fd_softmax_params_t *p0, const float *z,
                         const uint8_t *y, int n, float lr, int epochs, float l2_anchor)
{
    float gW[FD_N_CLASSES * FD_SM_D], gb[FD_N_CLASSES], pr[FD_N_CLASSES];
    const float inv_n = 1.0f / (float)n;
    for (int e = 0; e < epochs; e++) {
        memset(gW, 0, sizeof(gW));
        memset(gb, 0, sizeof(gb));
        for (int s = 0; s < n; s++) {
            const float *zs = &z[s * FD_SM_D];
            fd_softmax_logits(p, zs, pr);
            softmax_inplace(pr);
            for (int c = 0; c < FD_N_CLASSES; c++) {
                const float g = (pr[c] - (y[s] == c ? 1.0f : 0.0f)) * inv_n;
                gb[c] += g;
                float *gw = &gW[c * FD_SM_D];
                for (int j = 0; j < FD_SM_D; j++) gw[j] += g * zs[j];
            }
        }
        for (int i = 0; i < FD_N_CLASSES * FD_SM_D; i++)
            p->W[i] -= lr * (gW[i] + l2_anchor * (p->W[i] - p0->W[i]));
        for (int c = 0; c < FD_N_CLASSES; c++)
            p->b[c] -= lr * (gb[c] + l2_anchor * (p->b[c] - p0->b[c]));
    }
}

/* ------------------------------------------------------------ Welford */

void fd_welford_reset(fd_welford_t *w) { memset(w, 0, sizeof(*w)); }

void fd_welford_update(fd_welford_t *w, const float *feat)
{
    w->n++;
    const float inv = 1.0f / (float)w->n;
    for (int j = 0; j < FD_N_FEATURES; j++) {
        const float d = feat[j] - w->mean[j];
        w->mean[j] += d * inv;
        w->m2[j] += d * (feat[j] - w->mean[j]);
    }
}

float fd_welford_std(const fd_welford_t *w, int j)
{
    return w->n ? sqrtf(w->m2[j] / (float)w->n) + 1e-12f : 0.0f;
}

/* -------------------------------------------------------------- voting */

void fd_vote_reset(fd_vote_t *v) { memset(v, 0, sizeof(*v)); }

int fd_vote_add(fd_vote_t *v, int cls)
{
    v->count[cls]++;
    v->n++;
    int best = 0;
    for (int c = 1; c < FD_N_CLASSES; c++)
        if (v->count[c] > v->count[best]) best = c;
    return best;
}
