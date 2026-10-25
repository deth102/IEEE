/*
 * hil_esp32 — hardware-in-the-loop validation of fdcore on a bare ESP32 DevKit.
 *
 * Wiring: one jumper GPIO25 (DAC1) -> GPIO34 (ADC1_CH6). Nothing else.
 * Signal set: fold-0 test-block windows of all 27 KAIST recordings (include/hil_data.h).
 *
 * The firmware runs four phases automatically after reset and prints machine-readable lines:
 *   CAL,<dac_code>,<adc_mean>,<adc_std>      DC transfer of the DAC->ADC loop
 *   FIT,<slope>,<intercept>                  linear calibration used for reconstruction
 *   RATE,<samples>,<us>,<hz>                 measured I2S-ADC sample rate
 *   ORDER,<rough_plain>,<rough_swapped>,<swap>  sample-order check with a test tone
 *   WIN,<D|A>,<rec>,<win>,<label>,<rf>,<sm>,<py_rf>,<py_sm>,<proc_us>,<c_time>,<c_win>,<c_fft>,<c_freq>,<c_order>,<c_cls>
 *   SUM,<phase>,<key>,<value>
 *   INFO,done,1
 * D = digital replay: a hardware timer pushes the stored samples at fs into the double buffer.
 * A = analog loop: the timer writes the samples to the DAC; the ADC samples the wire via I2S/DMA.
 * In both phases the processing task runs on core 1 and must finish a window before the next
 * one is complete (deadline N/fs); misses are counted as overruns.
 */
#include <Arduino.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include "driver/adc.h"
#include "driver/dac.h"
#include "driver/i2s.h"
#include "esp_heap_caps.h"
#include "soc/rtc_io_reg.h"
#include "soc/sens_reg.h"
#include "fd_core.h"
#include "hil_data.h"

#ifndef HIL_MODEL_AUG
#define HIL_MODEL_AUG 0
#endif
#if HIL_MODEL_AUG
#define PY_RF hil_py_aug_rf
#define PY_SM hil_py_aug_sm
#define MODEL_NAME "aug"
#else
#define PY_RF hil_py_clean_rf
#define PY_SM hil_py_clean_sm
#define MODEL_NAME "clean"
#endif

#define I2S_PORT I2S_NUM_0
#define ADC_CH ADC1_CHANNEL_6  /* GPIO34 */
#define DMA_LEN 512
#define DAC_SPAN 80.0f         /* peak DAC excursion (codes) around mid-scale 128 */
#define SKIP_SAMPLES 1024      /* analog: discard the start of each playback (settling) */

extern "C" uint32_t fd_cycles(void) { return (uint32_t)ESP.getCycleCount(); }

static void out(const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    Serial.print(buf);
}

/* ------------------------------------------------------------ state */
static int16_t *acq[2];
static float work[FD_N];
static volatile uint32_t acq_fill = 0, overruns = 0, n_windows = 0;
static volatile uint8_t acq_idx = 0;
static volatile bool busy = false;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static TaskHandle_t proc_task = nullptr, ctrl_task = nullptr;
static hw_timer_t *timer = nullptr;

enum { MODE_IDLE = 0, MODE_DIGITAL, MODE_ANALOG };
static volatile int mode = MODE_IDLE;
static volatile int cur_rec = 0;
static volatile uint32_t src_pos = 0, src_len = 0;
static const int16_t *volatile src = nullptr;
static volatile float dac_gain = 1.0f;
static volatile int32_t dac_gain_q16 = 65536;   /* integer copy for the ISR (no FPU use in ISRs) */
static float fit_a = 1.0f, fit_b = 0.0f;  /* adc = a * dac_code + b */
static bool swap_pairs = false;
static fd_softmax_params_t sm;

/* analog-phase feature store (for on-device adaptation) and raw-window dumps (heap) */
#define MAX_STORE 160
static float (*fstore)[FD_N_FEATURES] = nullptr;
static uint8_t fstore_rec[MAX_STORE], fstore_win[MAX_STORE];
static volatile int n_store = 0;
static int16_t *dumpbuf[FD_N_CLASSES];
static int first_rec_of_class[FD_N_CLASSES];

/* The I2S-ADC clock divider runs ~0.39 % slow at a nominal 25.6 kHz (measured on this board);
 * request a proportionally higher rate so that the analog loop is sampled at fs. */
#ifndef HIL_ADC_RATE_CORR
#define HIL_ADC_RATE_CORR (25600.0f / 25500.6f)
#endif

/* ------------------------------------------------ double buffer push */
static inline void IRAM_ATTR push_sample(int16_t v)
{
    acq[acq_idx][acq_fill++] = v;
    if (acq_fill >= FD_N) {
        acq_fill = 0;
        if (busy) {
            overruns++;
        } else {
            busy = true;
            n_windows++;
            BaseType_t woken = pdFALSE;
            xTaskNotifyFromISR(proc_task, acq_idx, eSetValueWithOverwrite, &woken);
            acq_idx ^= 1;
            if (woken) portYIELD_FROM_ISR();
        }
    }
}

static void reset_acq(void)
{
    portENTER_CRITICAL(&mux);
    acq_fill = 0;
    acq_idx = 0;
    overruns = 0;
    n_windows = 0;
    portEXIT_CRITICAL(&mux);
}

/* --------------------------------------------------------- DAC write */
static inline void IRAM_ATTR dac_write_fast(uint8_t v)
{
    SET_PERI_REG_BITS(RTC_IO_PAD_DAC1_REG, RTC_IO_PDAC1_DAC, v, RTC_IO_PDAC1_DAC_S);
}

/* -------------------------------------------------- sample timer ISR */
static void IRAM_ATTR on_sample(void)
{
    if (src_pos >= src_len) return;
    const int16_t s = src[src_pos++];
    if (mode == MODE_DIGITAL) {
        portENTER_CRITICAL_ISR(&mux);
        push_sample(s);
        portEXIT_CRITICAL_ISR(&mux);
    } else if (mode == MODE_ANALOG) {
        int32_t c = 128 + (int32_t)(((int64_t)s * dac_gain_q16 + 32768) >> 16);
        if (c < 0) c = 0;
        if (c > 255) c = 255;
        dac_write_fast((uint8_t)c);
    }
}

static void timer_start(void)
{
    /* 80 MHz / 25 = 3.2 MHz; 125 ticks -> exactly 25.6 kHz */
    const uint32_t ticks = (uint32_t)lroundf(3.2e6f / FD_FS);
    timer = timerBegin(0, 25, true);
    timerAttachInterrupt(timer, &on_sample, true);
    timerAlarmWrite(timer, ticks, true);
    timerAlarmEnable(timer);
}

static void timer_stop(void)
{
    if (!timer) return;
    timerAlarmDisable(timer);
    timerDetachInterrupt(timer);
    timerEnd(timer);
    timer = nullptr;
}

/* ------------------------------------------------------------ I2S ADC */
static uint16_t dma_buf[DMA_LEN];

static void adc_start(void)
{
    i2s_config_t c = {};
    c.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN);
    c.sample_rate = (uint32_t)lroundf(FD_FS * HIL_ADC_RATE_CORR);
    c.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    c.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    c.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    c.intr_alloc_flags = 0;
    c.dma_buf_count = 8;
    c.dma_buf_len = DMA_LEN;
    c.use_apll = false;
    i2s_driver_install(I2S_PORT, &c, 0, NULL);
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC_CH, ADC_ATTEN_DB_11);
    i2s_set_adc_mode(ADC_UNIT_1, ADC_CH);
    i2s_adc_enable(I2S_PORT);
}

/* read n samples (12-bit codes) into dst, optionally undoing the pairwise swap of the I2S-ADC */
static void adc_read(uint16_t *dst, int n)
{
    int got = 0;
    while (got < n) {
        size_t bytes = 0;
        const int want = (n - got) < DMA_LEN ? (n - got) : DMA_LEN;
        i2s_read(I2S_PORT, dma_buf, want * 2, &bytes, portMAX_DELAY);
        const int k = bytes / 2;
        for (int i = 0; i < k; i++) {
            int j = swap_pairs ? (i ^ 1) : i;
            if (j >= k) j = i;
            dst[got + i] = dma_buf[j] & 0x0FFF;
        }
        got += k;
    }
}

/* -------------------------------------------------- processing task */
static void proc_fn(void *)
{
    float feat[FD_N_FEATURES];
    fd_profile_t prof;
    for (;;) {
        uint32_t idx;
        xTaskNotifyWait(0, 0xFFFFFFFFu, &idx, portMAX_DELAY);
        const int rec = cur_rec;
        const int win = (int)n_windows - 1;
        const char m = (mode == MODE_ANALOG) ? 'A' : 'D';
        const uint32_t t0 = micros();
        fd_extract_features(acq[idx & 1], HIL_SCALE, work, feat, &prof);
        const uint32_t c0 = fd_cycles();
        const int prf = fd_rf_predict(feat, NULL);
        const int psm = fd_softmax_predict(&sm, feat, NULL);
        if (m == 'A' && fstore && n_store < MAX_STORE) {
            memcpy(fstore[n_store], feat, sizeof(feat));
            fstore_rec[n_store] = (uint8_t)rec;
            fstore_win[n_store] = (uint8_t)win;
            n_store++;
            const int lab = hil_label[rec];
            if (win == 1 && rec == first_rec_of_class[lab] && dumpbuf[lab])
                memcpy(dumpbuf[lab], acq[idx & 1], FD_N * sizeof(int16_t));
        }
        prof.cycles[FD_ST_CLASSIFY] = fd_cycles() - c0;
        const uint32_t dt = micros() - t0;
        busy = false;
        int py_rf = -1, py_sm = -1;
        if (m == 'D' && win < HIL_WIN_PER_REC) {
            py_rf = PY_RF[rec * HIL_WIN_PER_REC + win];
            py_sm = PY_SM[rec * HIL_WIN_PER_REC + win];
        }
        out("WIN,%c,%d,%d,%d,%d,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n", m, rec, win, hil_label[rec], prf, psm, py_rf,
            py_sm, (unsigned long)dt, (unsigned long)prof.cycles[FD_ST_TIME], (unsigned long)prof.cycles[FD_ST_WINDOW],
            (unsigned long)prof.cycles[FD_ST_FFT], (unsigned long)prof.cycles[FD_ST_FREQ],
            (unsigned long)prof.cycles[FD_ST_ORDER], (unsigned long)prof.cycles[FD_ST_CLASSIFY]);
    }
}

static void wait_idle(uint32_t expect_windows, uint32_t timeout_ms)
{
    const uint32_t t0 = millis();
    while ((n_windows < expect_windows || busy) && millis() - t0 < timeout_ms) vTaskDelay(pdMS_TO_TICKS(5));
    vTaskDelay(pdMS_TO_TICKS(30));   /* let the log line of the last window go out */
}

/* ------------------------------------------------------ phases */
static uint16_t cal_buf[2048];

static void phase_calibration(void)
{
    dac_output_enable(DAC_CHANNEL_1);
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int np = 0;
    for (int code = 20; code <= 240; code += 10) {
        dac_write_fast((uint8_t)code);
        vTaskDelay(pdMS_TO_TICKS(20));
        adc_read(cal_buf, 2048);            /* flush */
        adc_read(cal_buf, 2048);
        double m = 0, v = 0;
        for (int i = 0; i < 2048; i++) m += cal_buf[i];
        m /= 2048;
        for (int i = 0; i < 2048; i++) v += (cal_buf[i] - m) * (cal_buf[i] - m);
        out("CAL,%d,%.2f,%.2f\n", code, m, sqrt(v / 2047));
        if (code == 20 || code == 240) {
            out("RAW");
            for (int i = 0; i < 8; i++) out(",%04x", dma_buf[i]);
            out("\n");
        }
        if (code >= 40 && code <= 220) {    /* fit on the central, most linear part */
            sx += code; sy += m; sxx += (double)code * code; sxy += code * m; np++;
        }
    }
    fit_a = (float)((np * sxy - sx * sy) / (np * sxx - sx * sx));
    fit_b = (float)((sy - fit_a * sx) / np);
    out("FIT,%.5f,%.3f\n", fit_a, fit_b);
    dac_write_fast(128);
}

static void phase_rate_and_order(void)
{
    /* sample rate */
    adc_read(cal_buf, 2048);
    const uint32_t t0 = micros();
    const int n = 25600;
    for (int i = 0; i < n; i += 2048) adc_read(cal_buf, (n - i) < 2048 ? (n - i) : 2048);
    const uint32_t dt = micros() - t0;
    out("RATE,%d,%lu,%.1f\n", n, (unsigned long)dt, n * 1e6 / dt);

    /* order check: DAC tone at fs/64 = 400 Hz; the correct order gives the smoother sequence */
    int16_t *tone = dumpbuf[0];   /* borrow a dump buffer (not used before the analog phase) */
    for (int i = 0; i < FD_N; i++) tone[i] = (int16_t)(1000.0f * sinf(2.0f * (float)M_PI * i / 64.0f));
    src = tone; src_pos = 0; src_len = FD_N; dac_gain = DAC_SPAN / 1000.0f;
    dac_gain_q16 = (int32_t)lroundf(dac_gain * 65536.0f);
    mode = MODE_ANALOG;
    timer_start();
    vTaskDelay(pdMS_TO_TICKS(60));
    swap_pairs = false;
    adc_read(cal_buf, 2048);
    timer_stop();
    mode = MODE_IDLE;
    double r_plain = 0, r_swap = 0;
    for (int i = 2; i < 2046; i++) {
        const double d1 = (double)cal_buf[i + 1] - 2.0 * cal_buf[i] + cal_buf[i - 1];
        const int a = i ^ 1, b = (i + 1) ^ 1, c = (i - 1) ^ 1;
        const double d2 = (double)cal_buf[b] - 2.0 * cal_buf[a] + cal_buf[c];
        r_plain += d1 * d1;
        r_swap += d2 * d2;
    }
    swap_pairs = r_swap < 0.5 * r_plain;
    out("ORDER,%.0f,%.0f,%d\n", r_plain, r_swap, swap_pairs ? 1 : 0);
    dac_write_fast(128);
    src = nullptr;
}

static void phase_digital(void)
{
    for (int r = 0; r < HIL_N_REC; r++) {
        reset_acq();
        cur_rec = r;
        src = hil_sig[r]; src_pos = 0; src_len = HIL_WIN_PER_REC * FD_N;
        mode = MODE_DIGITAL;
        timer_start();
        while (src_pos < src_len) vTaskDelay(pdMS_TO_TICKS(10));
        timer_stop();
        wait_idle(HIL_WIN_PER_REC, 2000);
        out("SUM,D,overruns_rec%d,%lu\n", r, (unsigned long)overruns);
        mode = MODE_IDLE;
    }
}

static void phase_analog(void)
{
    static uint16_t chunk[DMA_LEN];
    const float inv_a = 1.0f / fit_a;
    for (int r = 0; r < HIL_N_REC; r++) {
        reset_acq();
        cur_rec = r;
        dac_gain = DAC_SPAN / (float)(hil_peak[r] > 0 ? hil_peak[r] : 1);
        dac_gain_q16 = (int32_t)lroundf(dac_gain * 65536.0f);
        const float to_q = 1.0f / dac_gain;   /* DAC code -> int16 units of the recording */
        src = hil_sig[r]; src_pos = 0; src_len = HIL_WIN_PER_REC * FD_N;
        mode = MODE_ANALOG;
        adc_read(chunk, DMA_LEN);             /* drop stale DMA content */
        timer_start();
        uint32_t seen = 0;
        while (src_pos < src_len) {
            adc_read(chunk, DMA_LEN);
            for (int i = 0; i < DMA_LEN; i++) {
                if (++seen <= SKIP_SAMPLES) continue;
                const float code = ((float)chunk[i] - fit_b) * inv_a - 128.0f;
                float q = code * to_q;
                if (q > 32767.0f) q = 32767.0f;
                if (q < -32768.0f) q = -32768.0f;
                portENTER_CRITICAL(&mux);
                push_sample((int16_t)lroundf(q));
                portEXIT_CRITICAL(&mux);
            }
        }
        timer_stop();
        dac_write_fast(128);
        wait_idle(HIL_WIN_PER_REC - 1, 2000);
        out("SUM,A,overruns_rec%d,%lu\n", r, (unsigned long)overruns);
        mode = MODE_IDLE;
    }
}

/* On-device few-shot adaptation to the analog front end: the first complete window of every
 * recording (labelled) forms the support set, the softmax head is fine-tuned on the ESP32, and
 * the remaining windows are re-classified from the stored features. */
static void phase_adapt(void)
{
    float *z = (float *)malloc(HIL_N_REC * FD_SM_D * sizeof(float));
    uint8_t *y = (uint8_t *)malloc(HIL_N_REC);
    bool *is_sup = (bool *)calloc(MAX_STORE, sizeof(bool));
    int ns = 0;
    for (int i = 0; i < n_store; i++) {
        if (fstore_win[i] != 0 || ns >= HIL_N_REC) continue;
        fd_softmax_standardize(fstore[i], &z[ns * FD_SM_D]);
        y[ns++] = hil_label[fstore_rec[i]];
        is_sup[i] = true;
    }
    fd_softmax_params_t p0 = sm;
    const uint32_t c0 = fd_cycles();
    fd_softmax_finetune(&sm, &p0, z, y, ns, 0.1f, 200, 1e-3f);
    const uint32_t dc = fd_cycles() - c0;
    out("ADAPT,%d,%lu,%.3f\n", ns, (unsigned long)dc, dc / 240000.0);
    for (int i = 0; i < n_store; i++) {
        const int before = fd_softmax_predict(&p0, fstore[i], NULL);
        const int after = fd_softmax_predict(&sm, fstore[i], NULL);
        out("ADAPTWIN,%d,%d,%d,%d,%d,%d\n", fstore_rec[i], fstore_win[i], hil_label[fstore_rec[i]], is_sup[i] ? 1 : 0,
            before, after);
    }
    for (int i = 0; i < n_store; i++) {
        out("FEAT,%d,%d,%d", fstore_rec[i], fstore_win[i], hil_label[fstore_rec[i]]);
        for (int j = 0; j < FD_N_FEATURES; j++) out(",%.7g", fstore[i][j]);
        out("\n");
    }
    sm = p0;
    free(z);
    free(y);
    free(is_sup);
}

static void dump_raw(void)
{
    for (int c = 0; c < FD_N_CLASSES; c++) {
        if (!dumpbuf[c] || first_rec_of_class[c] < 0) continue;
        out("RAWA,%d,%d\n", c, first_rec_of_class[c]);
        for (int i = 0; i < FD_N; i += 64) {
            out("R");
            for (int k = 0; k < 64; k++) out(",%d", dumpbuf[c][i + k]);
            out("\n");
        }
    }
}

static void ctrl_fn(void *)
{
    /* runs on core 0: the sample-timer ISR and the ADC reader live here, processing on core 1 */
    out("INFO,phase,calibration\n");
    phase_calibration();
    phase_rate_and_order();
    out("INFO,phase,digital\n");
    phase_digital();
    out("INFO,phase,analog\n");
    phase_analog();
    out("INFO,phase,adapt\n");
    phase_adapt();
    dump_raw();
    out("INFO,free_heap,%lu\nINFO,min_free_heap,%lu\n", (unsigned long)ESP.getFreeHeap(), (unsigned long)ESP.getMinFreeHeap());
    out("INFO,done,1\n");
    vTaskDelete(NULL);
}

void setup()
{
    Serial.begin(115200);
    delay(800);
    out("INFO,hil_esp32,model=%s,N=%d,fs=%.0f,recordings=%d,windows_per_rec=%d,cpu_mhz=%lu\n", MODEL_NAME, FD_N,
        (double)FD_FS, HIL_N_REC, HIL_WIN_PER_REC, (unsigned long)getCpuFrequencyMhz());
    acq[0] = (int16_t *)heap_caps_malloc(FD_N * sizeof(int16_t), MALLOC_CAP_8BIT);
    acq[1] = (int16_t *)heap_caps_malloc(FD_N * sizeof(int16_t), MALLOC_CAP_8BIT);
    fd_softmax_load_default(&sm);
    fstore = (float (*)[FD_N_FEATURES])malloc(MAX_STORE * FD_N_FEATURES * sizeof(float));
    for (int c = 0; c < FD_N_CLASSES; c++) {
        first_rec_of_class[c] = -1;
        dumpbuf[c] = (int16_t *)malloc(FD_N * sizeof(int16_t));
    }
    for (int r = HIL_N_REC - 1; r >= 0; r--) first_rec_of_class[hil_label[r]] = r;
    /* wiring pre-check with the one-shot ADC driver, before I2S takes ADC1 */
    dac_output_enable(DAC_CHANNEL_1);
    for (int code = 0; code <= 255; code += 51) {
        dac_output_voltage(DAC_CHANNEL_1, (uint8_t)code);
        delay(20);
        uint32_t acc = 0;
        for (int i = 0; i < 64; i++) acc += analogRead(34);
        out("PRECHECK,%d,%lu,%lu\n", code, (unsigned long)(acc / 64), (unsigned long)analogReadMilliVolts(34));
    }
    dac_output_voltage(DAC_CHANNEL_1, 128);
    adc_start();
    xTaskCreatePinnedToCore(proc_fn, "proc", 8192, NULL, 5, &proc_task, 1);
    xTaskCreatePinnedToCore(ctrl_fn, "ctrl", 8192, NULL, 6, &ctrl_task, 0);
}

void loop() { vTaskDelay(pdMS_TO_TICKS(1000)); }
