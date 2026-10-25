/* Host-side golden test of fdcore against the Python reference.
 * Build:  python -m tools.build_host <model-tag>   (see 03_code/tools/build_host.py)
 * Output: one JSON object on stdout. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "fd_core.h"
#include "fd_testvec.h"

uint32_t fd_cycles(void) { return (uint32_t)clock(); }

static float work[FD_N], re[FD_N], im[FD_N], mag[FD_K];

static double now_us(void) { return 1e6 * (double)clock() / (double)CLOCKS_PER_SEC; }

int main(void)
{
    float feat[FD_N_FEATURES], feat_legacy[FD_N_FEATURES];
    double max_rel[FD_N_FEATURES] = {0}, max_rel_legacy[FD_N_FEATURES] = {0};
    double ref_scale[FD_N_FEATURES] = {0};
    int ok_rf = 0, ok_rfq = 0, ok_sm = 0, ok_rf_label = 0, ok_sm_label = 0;
    fd_softmax_params_t sm;
    fd_softmax_load_default(&sm);

    for (int t = 0; t < FD_TV_COUNT; t++)
        for (int j = 0; j < FD_N_FEATURES; j++)
            if (fabs(fd_tv_features[t][j]) > ref_scale[j]) ref_scale[j] = fabs(fd_tv_features[t][j]);

    for (int t = 0; t < FD_TV_COUNT; t++) {
        fd_extract_features(fd_tv_windows[t], FD_SCALE, work, feat, NULL);
        fd_extract_features_legacy(fd_tv_windows[t], FD_SCALE, re, im, mag, feat_legacy, NULL);
        for (int j = 0; j < FD_N_FEATURES; j++) {
            const double ref = fd_tv_features[t][j];
            const double den = fmax(fabs(ref), 1e-3 * ref_scale[j] + 1e-12);
            const double e1 = fabs(feat[j] - ref) / den, e2 = fabs(feat_legacy[j] - ref) / den;
            if (e1 > max_rel[j]) max_rel[j] = e1;
            if (e2 > max_rel_legacy[j]) max_rel_legacy[j] = e2;
        }
        const int prf = fd_rf_predict(feat, NULL);
        const int prq = fd_rf_predict_q16(feat, NULL);
        const int psm = fd_softmax_predict(&sm, feat, NULL);
        ok_rf += prf == fd_tv_rf[t];
        ok_rfq += prq == fd_tv_rf_q16[t];
        ok_sm += psm == fd_tv_softmax[t];
        ok_rf_label += prf == fd_tv_label[t];
        ok_sm_label += psm == fd_tv_label[t];
    }

    /* host timing (informative only) */
    const int reps = 20;
    double t0 = now_us();
    for (int r = 0; r < reps; r++) fd_extract_features(fd_tv_windows[r % FD_TV_COUNT], FD_SCALE, work, feat, NULL);
    const double t_opt = (now_us() - t0) / reps;
    t0 = now_us();
    for (int r = 0; r < reps; r++)
        fd_extract_features_legacy(fd_tv_windows[r % FD_TV_COUNT], FD_SCALE, re, im, mag, feat_legacy, NULL);
    const double t_leg = (now_us() - t0) / reps;

    /* on-device fine-tuning smoke test: adapt on the test vectors themselves */
    static float z[FD_TV_COUNT * FD_SM_D];
    static uint8_t y[FD_TV_COUNT];
    int n_lab = 0;
    for (int t = 0; t < FD_TV_COUNT; t++) {
        if (fd_tv_label[t] < 0) continue;
        fd_extract_features(fd_tv_windows[t], FD_SCALE, work, feat, NULL);
        fd_softmax_standardize(feat, &z[n_lab * FD_SM_D]);
        y[n_lab++] = (uint8_t)fd_tv_label[t];
    }
    fd_softmax_params_t p0 = sm;
    t0 = now_us();
    if (n_lab) fd_softmax_finetune(&sm, &p0, z, y, n_lab, 0.1f, 200, 1e-3f);
    const double t_ft = now_us() - t0;
    int ok_ft = 0;
    for (int s = 0; s < n_lab; s++) {
        float logits[FD_N_CLASSES];
        fd_softmax_logits(&sm, &z[s * FD_SM_D], logits);
        int best = 0;
        for (int c = 1; c < FD_N_CLASSES; c++) if (logits[c] > logits[best]) best = c;
        ok_ft += best == y[s];
    }

    printf("{\"n\": %d, \"rf_agree\": %d, \"rf_q16_agree\": %d, \"softmax_agree\": %d,", FD_TV_COUNT, ok_rf, ok_rfq, ok_sm);
    printf(" \"rf_label_acc\": %d, \"softmax_label_acc\": %d, \"finetune_fit\": %d, \"n_labelled\": %d,", ok_rf_label,
           ok_sm_label, ok_ft, n_lab);
    printf(" \"host_us_optimized\": %.1f, \"host_us_legacy\": %.1f, \"host_us_finetune\": %.1f,", t_opt, t_leg, t_ft);
    printf(" \"max_rel_err\": [");
    for (int j = 0; j < FD_N_FEATURES; j++) printf("%s%.3g", j ? ", " : "", max_rel[j]);
    printf("], \"max_rel_err_legacy\": [");
    for (int j = 0; j < FD_N_FEATURES; j++) printf("%s%.3g", j ? ", " : "", max_rel_legacy[j]);
    printf("]}\n");
    return 0;
}
