/*
 * node_esp32 — the complete KESTREL node on a bare ESP32 DevKit, driven by a host that plays the gateway.
 *
 * Wiring: one jumper GPIO25 (DAC1) -> GPIO34 (ADC1_CH6) for the analog loop. Nothing else.
 *
 * The host uploads the models (forest, softmax head, gate statistics) and streams signals over USB serial:
 *   batch  : one window per frame, processed at once (equivalence checks and long campaigns)
 *   stream : samples go into a ring buffer and a 25.6-kHz timer moves them into the acquisition double buffer
 *            (real-time replay: each window must be finished before the next one is complete)
 *   analog : the timer writes the samples to the DAC and the I2S-ADC samples the jumper (as hil_esp32)
 * For every window the node computes the features, both heads, and the self-check gate (s_t, g_t, alarm).
 * Windows flagged as support are stored; the on-chip update retunes the softmax head and re-references the gate.
 * A forest refreshed by the gateway and the new threshold tau' can be uploaded at run time.
 *
 * Host -> node frame: 0xA5, cmd, uint32 length (LE), payload, uint8 sum of the payload bytes.
 * Node -> host: one text line per event:
 *   HELLO,...                         after reset
 *   W,<mode>,<rec>,<win>,<label>,<flags>,<rf>,<sm>,<s>,<g>,<alarm>,<us>,<c_feat>,<c_cls>,<c_gate>,<c_env>[,features]
 *   K                                  one credit per stream chunk taken into the ring buffer
 *   ADAPT,<n>,<cycles_update>,<cycles_reref>,<ok>,<prototypes>   followed by SMW,... (W, b) and GATEP,... (Pw)
 *   CAL,<dac>,<adc_mean>,<adc_std> / FIT,<a>,<b> / ORDER,...   analog calibration
 *   SEND,<mode>,<windows>,<overruns>,<underruns>               end of a stream or analog recording
 *   OK,<cmd>,... / ERR,<text>
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

#ifndef NODE_BAUD
#define NODE_BAUD 2000000
#endif
#define RX_BUF 24576            /* one batch window (stop-and-wait) or the stream chunks in flight; 40 KB breaks reception */
#define MAX_TREES 32
#define MAX_NODES 512             /* 10 trees x at most 43 nodes (22 leaves) */
#define MAX_SUP 64
#define RING_N 8192             /* int16 samples (0.32 s at 25.6 kHz) */
#define CHUNK_MAX 4096
#define META_N 32
#define I2S_PORT I2S_NUM_0
#define ADC_CH ADC1_CHANNEL_6   /* GPIO34 */
#define DMA_LEN 512
#define SKIP_SAMPLES 1024       /* analog: settling at the start of every recording */
#ifndef ADC_RATE_CORR
#define ADC_RATE_CORR (25600.0f / 25500.6f)   /* I2S-ADC clock runs ~0.39 % slow at a nominal 25.6 kHz */
#endif

enum { F_RESET = 1, F_SUPPORT = 2, F_FEATURES = 4 };
enum { MODE_IDLE = 0, MODE_STREAM, MODE_ANALOG };

extern "C" uint32_t fd_cycles(void) { return (uint32_t)ESP.getCycleCount(); }

/* The CH340 USB bridge drops bytes when the node sends continuously at 921600 baud; lines therefore leave in
 * 32-byte packets with a pause after each, about 24 KB/s. A mutex keeps the lines of different tasks whole. */
static SemaphoreHandle_t tx_lock;

static void tx(const char *s, int n)
{
    xSemaphoreTake(tx_lock, portMAX_DELAY);
    for (int off = 0; off < n; off += 32) {
        const int k = (n - off) < 32 ? (n - off) : 32;
        Serial.write((const uint8_t *)s + off, k);
        Serial.flush();
        delayMicroseconds(700);                   /* also after the last packet: lines follow each other */
    }
    xSemaphoreGive(tx_lock);
}

static void out(const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    tx(buf, n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1);
}

/* ------------------------------------------------------------------ model in RAM */
static fd_rf_node_t *rf_nodes;                        /* heap: static DRAM holds the FFT tables */
static uint16_t rf_roots[MAX_TREES];
static uint8_t sm_idx[FD_SM_D];
static float sm_mu[FD_SM_D], sm_inv_sd[FD_SM_D];
static fd_softmax_params_t sm, sm_src;
static fd_gate_t gate;
static float sig_scale = FD_SCALE;
static bool model_ok = false;

/* support set for the on-chip update */
static float (*sup_z)[FD_SM_D];
static uint8_t sup_y[MAX_SUP];
static int n_sup = 0;

/* ------------------------------------------------------------------ acquisition */
struct Meta {
    uint16_t rec, win;
    int8_t label;
    uint8_t flags;
};
static int16_t *acq[2];
static float *work;
static int16_t *batch_buf;
static Meta batch_meta;
static volatile uint32_t acq_fill = 0, overruns = 0, underruns = 0, n_windows = 0;
static volatile uint8_t acq_idx = 0;
static volatile bool busy = false;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static TaskHandle_t proc_task = nullptr, done_waiter = nullptr;
static volatile int mode = MODE_IDLE;
static hw_timer_t *timer = nullptr;

static int16_t *ring;                                 /* SPSC: reader task produces, timer ISR consumes */
static volatile uint32_t ring_head = 0, ring_tail = 0;
static volatile bool ring_eos = false;                /* host has sent the last chunk */
static Meta meta_q[META_N];                           /* stream mode: one entry per window, in order */
static volatile uint32_t meta_head = 0, meta_tail = 0;
static Meta ana_meta;                                 /* analog mode: recording being played */
static volatile uint16_t ana_win = 0;
static volatile uint32_t ana_limit = 0xFFFFFFFFu;     /* samples to keep: the captured windows lie inside the played signal */
static volatile int32_t dac_gain_q16 = 65536;
static float dac_gain = 1.0f, fit_a = 15.25f, fit_b = -105.0f;
static bool swap_pairs = false;

static inline uint32_t ring_count(void) { return ring_head - ring_tail; }

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

static inline void IRAM_ATTR dac_write_fast(uint8_t v)
{
    SET_PERI_REG_BITS(RTC_IO_PAD_DAC1_REG, RTC_IO_PDAC1_DAC, v, RTC_IO_PDAC1_DAC_S);
}

static void IRAM_ATTR on_sample(void)
{
    if (ring_head == ring_tail) {                     /* nothing to play */
        if (!ring_eos) underruns++;
        return;
    }
    const int16_t s = ring[ring_tail & (RING_N - 1)];
    ring_tail++;
    if (mode == MODE_STREAM) {
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
    const uint32_t ticks = (uint32_t)lroundf(3.2e6f / FD_FS);     /* 80 MHz / 25 = 3.2 MHz */
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

static void reset_acq(void)
{
    portENTER_CRITICAL(&mux);
    acq_fill = 0;
    acq_idx = 0;
    overruns = 0;
    underruns = 0;
    n_windows = 0;
    portEXIT_CRITICAL(&mux);
}

/* ------------------------------------------------------------------ I2S ADC (analog loop) */
static uint16_t dma_buf[DMA_LEN];
static bool adc_running = false;

static void adc_start(void)
{
    if (adc_running) return;
    /* one-shot conversions first, as in hil_esp32's wiring pre-check: without them the I2S-ADC returns zeros */
    dac_output_enable(DAC_CHANNEL_1);
    for (int code = 0; code <= 255; code += 51) {
        dac_output_voltage(DAC_CHANNEL_1, (uint8_t)code);
        delay(5);
        for (int i = 0; i < 16; i++) analogRead(34);
        analogReadMilliVolts(34);
    }
    dac_output_voltage(DAC_CHANNEL_1, 128);
    i2s_config_t c = {};
    c.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN);
    c.sample_rate = (uint32_t)lroundf(FD_FS * ADC_RATE_CORR);
    c.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    c.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    c.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    c.dma_buf_count = 8;
    c.dma_buf_len = DMA_LEN;
    i2s_driver_install(I2S_PORT, &c, 0, NULL);
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC_CH, ADC_ATTEN_DB_11);
    i2s_set_adc_mode(ADC_UNIT_1, ADC_CH);
    i2s_adc_enable(I2S_PORT);
    adc_running = true;
}

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

/* ------------------------------------------------------------------ processing (core 1) */
static void process(const int16_t *x, const Meta &m, char mode_c)
{
    float feat[FD_N_FEATURES], z[FD_SM_D], logits[FD_N_CLASSES];
    fd_profile_t prof;
    memset(&prof, 0, sizeof(prof));
    if (m.flags & F_RESET) fd_gate_reset(&gate);
    const uint32_t t0 = micros();
    const uint32_t c0 = fd_cycles();
    fd_extract_features(x, sig_scale, work, feat, &prof);
    const uint32_t c1 = fd_cycles();
    const int prf = fd_rf_predict(feat, NULL);
    fd_softmax_standardize(feat, z);
    fd_softmax_logits(&sm, z, logits);
    int psm = 0;
    for (int c = 1; c < FD_N_CLASSES; c++)
        if (logits[c] > logits[psm]) psm = c;
    const uint32_t c2 = fd_cycles();
    const float s = fd_gate_score(&gate, z, psm, prf != psm);
    const float g = fd_gate_push(&gate, s);
    const int alarm = (g >= 0.0f && g > gate.tau) ? 1 : 0;
    const uint32_t c3 = fd_cycles();
    const uint32_t dt = micros() - t0;
    if ((m.flags & F_SUPPORT) && n_sup < MAX_SUP && m.label >= 0) {
        memcpy(sup_z[n_sup], z, sizeof(z));
        sup_y[n_sup++] = (uint8_t)m.label;
    }
    static char line[768];
    int n = snprintf(line, sizeof(line), "W,%c,%u,%u,%d,%u,%d,%d,%.7g,%.7g,%d,%lu,%lu,%lu,%lu,%lu", mode_c, m.rec, m.win,
                     m.label, m.flags, prf, psm, (double)s, (double)g, alarm, (unsigned long)dt,
                     (unsigned long)(c1 - c0), (unsigned long)(c2 - c1), (unsigned long)(c3 - c2),
                     (unsigned long)prof.cycles[FD_ST_ENV]);
    if (m.flags & F_FEATURES)
        for (int j = 0; j < FD_N_FEATURES && n < (int)sizeof(line) - 20; j++) n += snprintf(line + n, sizeof(line) - n, ",%.9g", (double)feat[j]);
    line[n++] = '\n';
    tx(line, n);
}

static void proc_fn(void *)
{
    for (;;) {
        uint32_t idx;
        xTaskNotifyWait(0, 0xFFFFFFFFu, &idx, portMAX_DELAY);
        if (idx == 2) {                               /* batch window */
            process(batch_buf, batch_meta, 'B');
            if (done_waiter) xTaskNotifyGive(done_waiter);
            continue;
        }
        Meta m;
        char mc;
        if (mode == MODE_ANALOG) {
            m = ana_meta;
            m.win = ana_win++;
            if (m.win != 0) m.flags &= (uint8_t)~(F_RESET | F_SUPPORT);   /* reset/support: first captured window */
            mc = 'A';
        } else {
            if (meta_tail != meta_head) {
                m = meta_q[meta_tail % META_N];
                meta_tail++;
            } else {
                m = {0xFFFF, 0xFFFF, -1, 0};
            }
            mc = 'S';
        }
        process(acq[idx & 1], m, mc);
        busy = false;
    }
}

/* ------------------------------------------------------------------ host commands */
static uint8_t *frame;                                /* largest payload: one window */
static const size_t FRAME_MAX = FD_N * 2 + 64;      /* one window; a model upload is smaller */

static bool read_exact(uint8_t *dst, size_t n, uint32_t timeout_ms)
{
    size_t got = 0;
    uint32_t t0 = millis();
    while (got < n) {
        int a = Serial.available();
        if (a > 0) {
            size_t k = Serial.read(dst + got, (size_t)a < n - got ? (size_t)a : n - got);
            got += k;
            t0 = millis();
        } else {
            if (millis() - t0 > timeout_ms) return false;
            vTaskDelay(1);
        }
    }
    return true;
}

template <typename T>
static const uint8_t *take(const uint8_t *p, T *dst, size_t count)
{
    memcpy(dst, p, sizeof(T) * count);
    return p + sizeof(T) * count;
}

static bool load_forest(const uint8_t *&p, const uint8_t *end)
{
    uint16_t nt, nn;
    p = take(p, &nt, 1);
    p = take(p, &nn, 1);
    if (nt > MAX_TREES || nn > MAX_NODES || p + 2 * nt + 8 * nn > end) return false;
    p = take(p, rf_roots, nt);
    for (int i = 0; i < nn; i++) {
        p = take(p, &rf_nodes[i].feat, 1);
        p = take(p, &rf_nodes[i].right, 1);
        p = take(p, &rf_nodes[i].thr, 1);
    }
    fd_rf_set_forest(rf_nodes, rf_roots, nt);
    out("OK,F,%u,%u\n", nt, nn);
    return true;
}

static void cmd_model(const uint8_t *p, size_t len)
{
    const uint8_t *end = p + len;
    uint16_t d, c;
    p = take(p, &d, 1);
    p = take(p, &c, 1);
    if (d != FD_SM_D || c != FD_N_CLASSES) {
        out("ERR,model dims %u %u (expected %d %d)\n", d, c, FD_SM_D, FD_N_CLASSES);
        return;
    }
    p = take(p, &sig_scale, 1);
    if (!load_forest(p, end)) {
        out("ERR,forest\n");
        return;
    }
    p = take(p, sm_idx, FD_SM_D);
    p = take(p, sm_mu, FD_SM_D);
    p = take(p, sm_inv_sd, FD_SM_D);
    p = take(p, sm.W, FD_N_CLASSES * FD_SM_D);
    p = take(p, sm.b, FD_N_CLASSES);
    p = take(p, gate.Pw, FD_SM_D * FD_SM_D);
    p = take(p, gate.Sw, FD_SM_D * FD_SM_D);
    p = take(p, gate.mu, FD_N_CLASSES * FD_SM_D);
    p = take(p, gate.n_c, FD_N_CLASSES);
    p = take(p, &gate.n, 1);
    uint16_t np_;
    p = take(p, &np_, 1);
    if (np_ > FD_GATE_PMAX) {
        out("ERR,prototypes %u\n", np_);
        return;
    }
    p = take(p, gate.protos, (size_t)np_ * FD_SM_D);
    p = take(p, gate.pcls, np_);
    gate.n_protos = np_;
    p = take(p, &gate.ridge, 1);
    p = take(p, &gate.tau, 1);
    uint16_t w;
    p = take(p, &w, 1);
    if (p != end || w < 1 || w > FD_GATE_WMAX) {
        out("ERR,model length %d\n", (int)(end - p));
        return;
    }
    gate.w = w;
    fd_gate_reset(&gate);
    fd_softmax_set_standardizer(sm_idx, sm_mu, sm_inv_sd);
    sm_src = sm;
    n_sup = 0;
    model_ok = true;
    out("OK,M,%d,%d,%.7g,%d\n", FD_SM_D, FD_N_CLASSES, (double)gate.tau, gate.w);
}

static void cmd_batch(const uint8_t *p, size_t len)
{
    if (len != 6 + 2 * FD_N) {
        out("ERR,window length %u\n", (unsigned)len);
        return;
    }
    p = take(p, &batch_meta.rec, 1);
    p = take(p, &batch_meta.win, 1);
    p = take(p, &batch_meta.label, 1);
    p = take(p, &batch_meta.flags, 1);
    batch_buf = (int16_t *)p;                         /* processed in place: the reader waits */
    done_waiter = xTaskGetCurrentTaskHandle();
    xTaskNotify(proc_task, 2, eSetValueWithOverwrite);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    done_waiter = nullptr;
}

static void cmd_adapt(const uint8_t *p, size_t len)
{
    float lr, l2, weight;
    uint16_t epochs;
    uint8_t reref;
    if (len != 15) {
        out("ERR,adapt length\n");
        return;
    }
    p = take(p, &lr, 1);
    p = take(p, &epochs, 1);
    p = take(p, &l2, 1);
    p = take(p, &weight, 1);
    p = take(p, &reref, 1);
    float *zbuf = &sup_z[0][0];                         /* contiguous n_sup x D */
    const uint32_t c0 = fd_cycles();
    fd_softmax_finetune(&sm, &sm_src, zbuf, sup_y, n_sup, lr, epochs, l2);
    const uint32_t c1 = fd_cycles();
    int ok = reref ? fd_gate_rereference(&gate, zbuf, sup_y, n_sup, weight) : 0;
    const uint32_t c2 = fd_cycles();
    out("ADAPT,%d,%lu,%lu,%d,%d\n", n_sup, (unsigned long)(c1 - c0), (unsigned long)(c2 - c1), ok == 0 ? 1 : 0, gate.n_protos);
    out("SMW");
    for (int i = 0; i < FD_N_CLASSES * FD_SM_D; i++) out(",%.9g", (double)sm.W[i]);
    for (int i = 0; i < FD_N_CLASSES; i++) out(",%.9g", (double)sm.b[i]);
    out("\nGATEP");
    for (int i = 0; i < FD_SM_D * FD_SM_D; i++) out(",%.9g", (double)gate.Pw[i]);
    out("\n");
}

static void wait_idle(uint32_t timeout_ms)
{
    const uint32_t t0 = millis();
    while (busy && millis() - t0 < timeout_ms) vTaskDelay(pdMS_TO_TICKS(2));
    vTaskDelay(pdMS_TO_TICKS(20));
}

/* stream / analog: chunk of int16 samples into the ring (blocks until there is room) */
static void cmd_chunk(const uint8_t *p, size_t len)
{
    const int n = len / 2;
    while (RING_N - ring_count() < (uint32_t)n) vTaskDelay(1);
    const int16_t *s = (const int16_t *)p;
    for (int i = 0; i < n; i++) ring[(ring_head + i) & (RING_N - 1)] = s[i];
    ring_head += n;
    tx("K\n", 2);
}

static void cmd_stream_begin(void)
{
    timer_stop();
    reset_acq();
    ring_head = ring_tail = 0;
    ring_eos = false;
    meta_head = meta_tail = 0;
    mode = MODE_STREAM;
    out("OK,S\n");
}

static void cmd_meta(const uint8_t *p, size_t len)
{
    Meta m;
    p = take(p, &m.rec, 1);
    p = take(p, &m.win, 1);
    p = take(p, &m.label, 1);
    p = take(p, &m.flags, 1);
    while (meta_head - meta_tail >= META_N) vTaskDelay(1);
    meta_q[meta_head % META_N] = m;
    meta_head++;
    if (!timer && mode == MODE_STREAM && ring_count() >= FD_N / 2) timer_start();
}

static void cmd_stream_end(void)
{
    ring_eos = true;
    if (!timer && ring_count()) timer_start();
    while (ring_count()) vTaskDelay(pdMS_TO_TICKS(2));
    timer_stop();
    wait_idle(2000);
    out("SEND,S,%lu,%lu,%lu\n", (unsigned long)n_windows, (unsigned long)overruns, (unsigned long)underruns);
    mode = MODE_IDLE;
}

/* analog recording: header (meta + peak), then chunks, then 'E' */
static void cmd_analog_begin(const uint8_t *p, size_t len)
{
    int16_t peak;
    float span;
    p = take(p, &ana_meta.rec, 1);
    p = take(p, &ana_meta.win, 1);
    p = take(p, &ana_meta.label, 1);
    p = take(p, &ana_meta.flags, 1);
    p = take(p, &peak, 1);
    p = take(p, &span, 1);
    adc_start();
    dac_output_enable(DAC_CHANNEL_1);
    timer_stop();
    reset_acq();
    ring_head = ring_tail = 0;
    ring_eos = false;
    ana_win = 0;
    ana_limit = 0xFFFFFFFFu;
    dac_gain = span / (float)(peak > 0 ? peak : 1);
    dac_gain_q16 = (int32_t)lroundf(dac_gain * 65536.0f);
    mode = MODE_ANALOG;
    out("OK,G,%u\n", ana_meta.rec);
}

static TaskHandle_t adc_task = nullptr;
static volatile bool adc_capture = false;

static void adc_fn(void *)
{
    static uint16_t chunk[DMA_LEN];
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);       /* one recording */
        const float inv_a = 1.0f / fit_a, to_q = 1.0f / dac_gain;
        uint32_t seen = 0, kept = 0;
        adc_read(chunk, DMA_LEN);                     /* drop stale DMA content */
        while (adc_capture) {
            adc_read(chunk, DMA_LEN);
            for (int i = 0; i < DMA_LEN; i++) {
                if (++seen <= SKIP_SAMPLES || kept >= ana_limit) continue;
                kept++;
                const float code = ((float)chunk[i] - fit_b) * inv_a - 128.0f;
                float q = code * to_q;
                if (q > 32767.0f) q = 32767.0f;
                if (q < -32768.0f) q = -32768.0f;
                portENTER_CRITICAL(&mux);
                push_sample((int16_t)lroundf(q));
                portEXIT_CRITICAL(&mux);
            }
        }
    }
}

static void analog_maybe_start(void)
{
    if (!timer && mode == MODE_ANALOG && ring_count() >= FD_N / 2) {
        adc_capture = true;
        xTaskNotifyGive(adc_task);
        timer_start();
    }
}

static void cmd_analog_end(void)
{
    ana_limit = ring_head > SKIP_SAMPLES + 64 ? ring_head - SKIP_SAMPLES - 64 : 0;   /* 64: loop latency */
    ring_eos = true;
    if (!timer && ring_count()) analog_maybe_start();
    while (ring_count()) vTaskDelay(pdMS_TO_TICKS(2));
    timer_stop();
    dac_write_fast(128);
    vTaskDelay(pdMS_TO_TICKS(40));                     /* last DMA buffers */
    adc_capture = false;
    wait_idle(2000);
    out("SEND,A,%lu,%lu,%lu\n", (unsigned long)n_windows, (unsigned long)overruns, (unsigned long)underruns);
    mode = MODE_IDLE;
}

static uint16_t cal_buf[2048];

static void cmd_calibrate(void)
{
    adc_start();
    dac_output_enable(DAC_CHANNEL_1);
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int np = 0;
    for (int code = 20; code <= 240; code += 10) {
        dac_write_fast((uint8_t)code);
        vTaskDelay(pdMS_TO_TICKS(20));
        adc_read(cal_buf, 2048);
        adc_read(cal_buf, 2048);
        double m = 0, v = 0;
        for (int i = 0; i < 2048; i++) m += cal_buf[i];
        m /= 2048;
        for (int i = 0; i < 2048; i++) v += (cal_buf[i] - m) * (cal_buf[i] - m);
        out("CAL,%d,%.2f,%.2f\n", code, m, sqrt(v / 2047));
        if (code == 20 || code == 240) out("RAW,%04x,%04x,%04x,%04x,%04x,%04x\n", dma_buf[0], dma_buf[1], dma_buf[2], dma_buf[3], dma_buf[4], dma_buf[5]);
        if (code >= 40 && code <= 220) {
            sx += code; sy += m; sxx += (double)code * code; sxy += code * m; np++;
        }
    }
    fit_a = (float)((np * sxy - sx * sy) / (np * sxx - sx * sx));
    fit_b = (float)((sy - fit_a * sx) / np);
    out("FIT,%.5f,%.3f\n", fit_a, fit_b);
    /* sample-order check of the I2S-ADC with a 400-Hz tone */
    for (int i = 0; i < FD_N; i++) ring[i] = (int16_t)(1000.0f * sinf(2.0f * (float)M_PI * i / 64.0f));
    ring_head = FD_N;
    ring_tail = 0;
    ring_eos = true;
    dac_gain = 80.0f / 1000.0f;
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
    ring_head = ring_tail = 0;
}

static void reader_fn(void *)
{
    out("READY\n");
    for (;;) {
        uint8_t h[6];
        if (!read_exact(h, 1, 60000)) continue;
        if (h[0] != 0xA5) continue;
        if (!read_exact(h + 1, 5, 2000)) { out("ERR,header\n"); continue; }
        const char cmd = (char)h[1];
        uint32_t len;
        memcpy(&len, h + 2, 4);
        if (len > FRAME_MAX) { out("ERR,length %lu\n", (unsigned long)len); continue; }
        uint8_t sum;
        if (!read_exact(frame, len, 2000) || !read_exact(&sum, 1, 2000)) { out("ERR,timeout %c\n", cmd); continue; }
        uint8_t s = 0;
        for (uint32_t i = 0; i < len; i++) s += frame[i];
        if (s != sum) { out("ERR,checksum %c\n", cmd); continue; }
        switch (cmd) {
        case 'P': out("OK,P,%d,%d,%d,%d\n", FD_N, FD_SM_D, FD_N_CLASSES, FD_N_FEATURES); break;
        case 'M': cmd_model(frame, len); break;
        case 'F': { const uint8_t *p = frame; if (!load_forest(p, frame + len)) out("ERR,forest\n"); break; }
        case 'T': memcpy(&gate.tau, frame, 4); fd_gate_reset(&gate); out("OK,T,%.7g\n", (double)gate.tau); break;
        case 'C': n_sup = 0; sm = sm_src; out("OK,C\n"); break;
        case 'B': cmd_batch(frame, len); break;
        case 'A': cmd_adapt(frame, len); break;
        case 'S': cmd_stream_begin(); break;
        case 'Q': cmd_meta(frame, len); break;
        case 'X': cmd_chunk(frame, len); if (mode == MODE_STREAM && !timer && meta_head != meta_tail && ring_count() >= FD_N / 2) timer_start();
                  if (mode == MODE_ANALOG) analog_maybe_start(); break;
        case 'E': if (mode == MODE_ANALOG) cmd_analog_end(); else cmd_stream_end(); break;
        case 'G': cmd_analog_begin(frame, len); break;
        case 'L': cmd_calibrate(); break;
        case 'H': out("OK,H,%lu,%lu\n", (unsigned long)ESP.getFreeHeap(), (unsigned long)ESP.getMinFreeHeap()); break;
        case 'V':                                     /* wiring check with the one-shot ADC (before I2S takes ADC1) */
            if (adc_running) { out("ERR,adc busy\n"); break; }
            dac_output_enable(DAC_CHANNEL_1);
            for (int code = 0; code <= 255; code += 51) {
                dac_output_voltage(DAC_CHANNEL_1, (uint8_t)code);
                delay(20);
                uint32_t acc = 0;
                for (int i = 0; i < 64; i++) acc += analogRead(34);
                out("PRECHECK,%d,%lu,%lu\n", code, (unsigned long)(acc / 64), (unsigned long)analogReadMilliVolts(34));
            }
            dac_output_voltage(DAC_CHANNEL_1, 128);
            out("OK,V\n");
            break;
        case 'Z': {                                   /* link test: lines of a known pattern */
            static char zl[1600];
            for (int n = 50; n <= 1500; n += 50) {
                int k = snprintf(zl, sizeof(zl), "Z,%d,", n);
                for (int i = 0; i < n; i++) zl[k++] = '0' + (i % 10);
                zl[k++] = '\n';
                tx(zl, k);
            }
            break;
        }
        default: out("ERR,cmd %c\n", cmd);
        }
    }
}

void setup()
{
    const size_t rxb = Serial.setRxBufferSize(RX_BUF);
    Serial.setTxBufferSize(8192);
    Serial.begin(NODE_BAUD);
    tx_lock = xSemaphoreCreateMutex();
    delay(300);
    /* task stacks first: after the large buffers the heap is fragmented (largest free block ~6 KB) */
    const int t1 = xTaskCreatePinnedToCore(proc_fn, "proc", 8192, NULL, 5, &proc_task, 1);
    const int t2 = xTaskCreatePinnedToCore(adc_fn, "adc", 4096, NULL, 6, &adc_task, 0);
    const int t3 = xTaskCreatePinnedToCore(reader_fn, "reader", 8192, NULL, 4, NULL, 0);
    /* largest buffers first: the heap is split into regions and fragments quickly */
    work = (float *)heap_caps_malloc(FD_N * sizeof(float), MALLOC_CAP_8BIT);
    frame = (uint8_t *)heap_caps_malloc(FRAME_MAX, MALLOC_CAP_8BIT);
    rf_nodes = (fd_rf_node_t *)heap_caps_malloc(MAX_NODES * sizeof(fd_rf_node_t), MALLOC_CAP_8BIT);
    acq[0] = (int16_t *)heap_caps_malloc(FD_N * sizeof(int16_t), MALLOC_CAP_8BIT);
    acq[1] = (int16_t *)heap_caps_malloc(FD_N * sizeof(int16_t), MALLOC_CAP_8BIT);
    ring = (int16_t *)heap_caps_malloc(RING_N * sizeof(int16_t), MALLOC_CAP_8BIT);
    sup_z = (float (*)[FD_SM_D])heap_caps_malloc(MAX_SUP * FD_SM_D * sizeof(float), MALLOC_CAP_8BIT);
    fd_softmax_load_default(&sm);
    sm_src = sm;
    memset(&gate, 0, sizeof(gate));
    gate.w = 10;
    gate.tau = 1e9f;
    out("HELLO,node_esp32,N=%d,fs=%.0f,D=%d,C=%d,F=%d,env=%d,baud=%d,rx=%u,tasks=%d%d%d,largest=%u,heap=%lu,alloc=%d\n", FD_N,
        (double)FD_FS, FD_SM_D, FD_N_CLASSES, FD_N_FEATURES, FD_ENV_BANDS, NODE_BAUD, (unsigned)rxb, t1, t2, t3,
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT), (unsigned long)ESP.getFreeHeap(),
        (acq[0] ? 1 : 0) | (acq[1] ? 2 : 0) | (ring ? 4 : 0) | (frame ? 8 : 0) | (work ? 16 : 0) | (rf_nodes ? 32 : 0) | (sup_z ? 64 : 0));
}

void loop() { vTaskDelay(pdMS_TO_TICKS(1000)); }
