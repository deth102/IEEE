/*
 * bench_mcu — cycle-accurate benchmark of the fdcore pipeline on any Arduino-capable MCU.
 *
 * Uses the golden test vectors compiled into flash (firmware/models/active/fd_testvec.h),
 * so no SD card or sensor is needed. Prints machine-readable lines:
 *   INFO,<key>,<value>
 *   STAGE,<variant>,<stage>,<min_cycles>,<mean_cycles>
 *   CLS,<classifier>,<min_cycles>,<mean_cycles>
 *   CHECK,<name>,<value>
 * Parse them with:  python -m tools.parse_bench <logfile>
 */
#include <Arduino.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "fd_core.h"
#include "fd_testvec.h"

#ifndef F_CPU
#define F_CPU 133000000UL /* RP2040 (Arduino-mbed core does not define F_CPU) */
#endif

/* Serial.printf is not available on every Arduino core (e.g. Arduino-mbed RP2040) */
static void out(const char *fmt, ...)
{
    char buf[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    Serial.print(buf);
}

#ifndef FD_BENCH_REPS
#define FD_BENCH_REPS 10
#endif
#ifndef FD_BENCH_LEGACY
#define FD_BENCH_LEGACY 1
#endif

/* ------------------------------------------------------------ cycle counter */
#if defined(ARDUINO_ARCH_ESP32)
extern "C" uint32_t fd_cycles(void) { return (uint32_t)ESP.getCycleCount(); }
static void cycles_init(void) {}
#define FD_PLATFORM "esp32"
#elif defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__)
/* Cortex-M3/M4/M7: DWT cycle counter */
#define DEMCR_REG (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL_REG (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT_REG (*(volatile uint32_t *)0xE0001004u)
#define DWT_LAR_REG (*(volatile uint32_t *)0xE0001FB0u)
extern "C" uint32_t fd_cycles(void) { return DWT_CYCCNT_REG; }
static void cycles_init(void)
{
    DEMCR_REG |= (1u << 24);   /* TRCENA */
    DWT_LAR_REG = 0xC5ACCE55u; /* unlock (Cortex-M7) */
    DWT_CYCCNT_REG = 0;
    DWT_CTRL_REG |= 1u;
}
#define FD_PLATFORM "cortex-m"
#else
/* Cortex-M0+ (RP2040) and others: microseconds scaled to cycles */
extern "C" uint32_t fd_cycles(void) { return (uint32_t)(micros() * (uint64_t)(F_CPU / 1000000UL)); }
static void cycles_init(void) {}
#define FD_PLATFORM "generic"
#endif

static float work[FD_N];
static int16_t inbuf[FD_N];   /* acquisition buffer in RAM, as filled by ADC/DMA in the edge node */

static const int16_t *to_ram(int t)
{
    memcpy(inbuf, fd_tv_windows[t], sizeof(inbuf));
    return inbuf;
}
#if FD_BENCH_LEGACY
/* legacy buffers are heap-allocated: ESP32 static DRAM (dram0_0_seg) cannot hold 112 KiB of .bss */
static float *leg_re, *leg_im, *leg_mag;
#endif
static const char *const STAGE_NAMES[FD_ST_COUNT] = {"time_feat", "window", "fft", "magnitude", "freq_feat", "order_feat", "classifier"};

static void report_profile(const char *variant, uint32_t mn[FD_ST_COUNT], uint64_t sum[FD_ST_COUNT], int n)
{
    for (int s = 0; s < FD_ST_CLASSIFY; s++) {
        out("STAGE,%s,%s,%lu,%lu\n", variant, STAGE_NAMES[s], (unsigned long)mn[s], (unsigned long)(sum[s] / n));
    }
}

static void run_bench(void)
{
    float feat[FD_N_FEATURES];
    fd_profile_t prof;
    out("INFO,platform,%s\nINFO,f_cpu_hz,%lu\nINFO,N,%d\nINFO,fs,%.1f\nINFO,test_vectors,%d\nINFO,reps,%d\n",
                  FD_PLATFORM, (unsigned long)F_CPU, FD_N, (double)FD_FS, FD_TV_COUNT, FD_BENCH_REPS);

    /* ---- correctness ---- */
    double max_rel = 0.0;
    int agree_rf = 0, agree_rfq = 0, agree_sm = 0;
    static fd_softmax_params_t sm;
    fd_softmax_load_default(&sm);
    for (int t = 0; t < FD_TV_COUNT; t++) {
        fd_extract_features(fd_tv_windows[t], FD_SCALE, work, feat, NULL);
        for (int j = 0; j < FD_N_FEATURES; j++) {
            const double ref = fd_tv_features[t][j];
            const double e = fabs(feat[j] - ref) / (fabs(ref) + 1e-6);
            if (j != FD_F_MEAN_T && e > max_rel) max_rel = e;
        }
        agree_rf += fd_rf_predict(feat, NULL) == fd_tv_rf[t];
        agree_rfq += fd_rf_predict_q16(feat, NULL) == fd_tv_rf_q16[t];
        agree_sm += fd_softmax_predict(&sm, feat, NULL) == fd_tv_softmax[t];
    }
    out("CHECK,feature_max_rel_err,%.3g\nCHECK,rf_agree,%d/%d\nCHECK,rf_q16_agree,%d/%d\nCHECK,softmax_agree,%d/%d\n",
                  max_rel, agree_rf, FD_TV_COUNT, agree_rfq, FD_TV_COUNT, agree_sm, FD_TV_COUNT);

    /* ---- optimized pipeline ---- */
    uint32_t mn[FD_ST_COUNT];
    uint64_t sum[FD_ST_COUNT];
    for (int s = 0; s < FD_ST_COUNT; s++) { mn[s] = UINT32_MAX; sum[s] = 0; }
    int n = 0;
    uint32_t tot_min = UINT32_MAX;
    for (int r = 0; r < FD_BENCH_REPS; r++) {
        for (int t = 0; t < FD_TV_COUNT; t++, n++) {
            const int16_t *in = to_ram(t);
            const uint32_t t0 = fd_cycles();
            fd_extract_features(in, FD_SCALE, work, feat, &prof);
            const uint32_t dt = fd_cycles() - t0;
            if (dt < tot_min) tot_min = dt;
            for (int s = 0; s < FD_ST_CLASSIFY; s++) {
                if (prof.cycles[s] < mn[s]) mn[s] = prof.cycles[s];
                sum[s] += prof.cycles[s];
            }
        }
    }
    report_profile("optimized", mn, sum, n);
    out("STAGE,optimized,total_features,%lu,%lu\n", (unsigned long)tot_min, 0UL);

#if FD_BENCH_LEGACY
    leg_re = (float *)malloc(sizeof(float) * FD_N);
    leg_im = (float *)malloc(sizeof(float) * FD_N);
    leg_mag = (float *)malloc(sizeof(float) * FD_K);
    if (!leg_re || !leg_im || !leg_mag) {
        Serial.println("INFO,legacy,skipped (out of memory)");
    } else {
    for (int s = 0; s < FD_ST_COUNT; s++) { mn[s] = UINT32_MAX; sum[s] = 0; }
    n = 0;
    for (int r = 0; r < (FD_BENCH_REPS + 1) / 2; r++) {
        for (int t = 0; t < FD_TV_COUNT; t++, n++) {
            fd_extract_features_legacy(to_ram(t), FD_SCALE, leg_re, leg_im, leg_mag, feat, &prof);
            for (int s = 0; s < FD_ST_CLASSIFY; s++) {
                if (prof.cycles[s] < mn[s]) mn[s] = prof.cycles[s];
                sum[s] += prof.cycles[s];
            }
        }
    }
    report_profile("legacy", mn, sum, n);
    }
    free(leg_re);
    free(leg_im);
    free(leg_mag);
#endif

    /* ---- classifiers ---- */
    fd_extract_features(fd_tv_windows[0], FD_SCALE, work, feat, NULL);
    const int reps = 200;
    uint32_t c_rf = UINT32_MAX, c_rfq = UINT32_MAX, c_sm = UINT32_MAX;
    uint64_t s_rf = 0, s_rfq = 0, s_sm = 0;
    volatile int sink = 0;
    for (int r = 0; r < reps; r++) {
        const float *f = feat;
        uint32_t t0 = fd_cycles();
        sink += fd_rf_predict(f, NULL);
        uint32_t d = fd_cycles() - t0;
        c_rf = d < c_rf ? d : c_rf; s_rf += d;
        t0 = fd_cycles();
        sink += fd_rf_predict_q16(f, NULL);
        d = fd_cycles() - t0;
        c_rfq = d < c_rfq ? d : c_rfq; s_rfq += d;
        t0 = fd_cycles();
        sink += fd_softmax_predict(&sm, f, NULL);
        d = fd_cycles() - t0;
        c_sm = d < c_sm ? d : c_sm; s_sm += d;
    }
    out("CLS,rf,%lu,%lu\nCLS,rf_q16,%lu,%lu\nCLS,softmax,%lu,%lu\n", (unsigned long)c_rf,
                  (unsigned long)(s_rf / reps), (unsigned long)c_rfq, (unsigned long)(s_rfq / reps),
                  (unsigned long)c_sm, (unsigned long)(s_sm / reps));

    /* ---- on-device fine-tuning (support set = test vectors, 200 epochs) ---- */
    static float z[FD_TV_COUNT * FD_SM_D];
    static uint8_t y[FD_TV_COUNT];
    int nl = 0;
    for (int t = 0; t < FD_TV_COUNT; t++) {
        if (fd_tv_label[t] < 0) continue;
        fd_extract_features(fd_tv_windows[t], FD_SCALE, work, feat, NULL);
        fd_softmax_standardize(feat, &z[nl * FD_SM_D]);
        y[nl++] = (uint8_t)fd_tv_label[t];
    }
    static fd_softmax_params_t p0;
    p0 = sm;
    const uint32_t t0 = fd_cycles();
    if (nl) fd_softmax_finetune(&sm, &p0, z, y, nl, 0.1f, 200, 1e-3f);
    const uint32_t d = fd_cycles() - t0;
    out("CLS,finetune_%dx200,%lu,%lu\n", nl, (unsigned long)d, (unsigned long)d);
    out("INFO,sink,%d\nINFO,done,1\n", (int)sink);
}

void setup()
{
    Serial.begin(115200);
    delay(1500);
    cycles_init();
    run_bench();
}

void loop()
{
    delay(1000);
}
