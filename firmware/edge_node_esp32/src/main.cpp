/*
 * edge_node_esp32 — real-time vibration fault-diagnosis node (hardware-in-the-loop).
 *
 * Signal path (emulated sensor):
 *   SD card (.fdv int16 file) --SD task, core 0--> ring buffer
 *   --hardware timer ISR, one block of FD_BLOCK samples every FD_BLOCK/fs s (DMA emulation)-->
 *   double acquisition buffer (2 x FD_N int16)
 *   --task notification--> processing task (core 1): features + classifier + voting
 *   --> Serial log (machine-readable) + LCD 20x4
 *
 * The timer paces the samples exactly at fs, so the processing task sees the same
 * real-time constraints as with an ADC+DMA front end: a window must be processed before
 * the next one is complete (deadline = FD_N / fs); misses are counted as overruns.
 *
 * Serial commands (115200 Bd, newline terminated):
 *   help | ls | mem
 *   run <file> [rf|q16|sm] [V]   stream a file in real time, majority vote over V windows
 *   bench <file> [n]             unpaced per-stage cycle benchmark (optimized vs legacy)
 *   ft <file> <class> <k> [skip] add k windows of <file> labelled <class> to the support set
 *   adapt [lr] [epochs]          fine-tune the softmax head on the support set (on device)
 *   save | load | reset          persist / restore / reset the softmax parameters (NVS)
 *
 * Wiring (same as the graduation project): SD CS=5 SCK=18 MISO=19 MOSI=23, LCD I2C SDA=21 SCL=22.
 */
#include <Arduino.h>
#include <LiquidCrystal_I2C.h>
#include <Preferences.h>
#include <SD.h>
#include <SPI.h>
#include <Wire.h>
#include "esp_heap_caps.h"
#include "fd_core.h"

#define SD_CS_PIN 5
#define LCD_ADDR 0x27
#define DATA_DIR "/fd"
#define FD_BLOCK 64            /* samples per emulated DMA transfer */
#define RING_SAMPLES 16384     /* SD -> ISR ring buffer (32 KiB) */
#define SD_CHUNK 2048          /* samples per SD read */
#define MAX_SUPPORT 64         /* few-shot support set capacity (windows) */

extern "C" uint32_t fd_cycles(void) { return (uint32_t)ESP.getCycleCount(); }

/* ---------------------------------------------------------------- state */
static int16_t *acq[2];
static int16_t *ring;
static float work[FD_N];
static volatile uint32_t ring_head = 0, ring_tail = 0;   /* head: SD task, tail: ISR */
static volatile uint32_t acq_fill = 0, underruns = 0, overruns = 0, windows_ready = 0;
static volatile uint8_t acq_idx = 0;
static volatile bool busy = false, streaming = false, sd_eof = false;
static TaskHandle_t proc_task = nullptr, sd_task = nullptr;
static hw_timer_t *timer = nullptr;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static File stream_file;
static float stream_scale = FD_SCALE;

enum { CLS_RF = 0, CLS_Q16, CLS_SM };
static int cls_mode = CLS_RF;
static int vote_len = 5;
static fd_vote_t vote;
static fd_softmax_params_t sm, sm0;
static float support_z[MAX_SUPPORT * FD_SM_D];
static uint8_t support_y[MAX_SUPPORT];
static int n_support = 0;

static LiquidCrystal_I2C lcd(LCD_ADDR, 20, 4);
static bool lcd_ok = false;
static Preferences prefs;

/* stream statistics */
static uint32_t n_proc = 0, t_proc_max = 0;
static uint64_t t_proc_sum = 0;
static uint32_t cls_count[FD_N_CLASSES];

/* ------------------------------------------------------------- file I/O */
typedef struct {
    char magic[4];
    uint32_t fs;
    float scale;
    uint32_t n;
} fdv_header_t;

static bool open_fdv(const char *name, File &f, fdv_header_t &h)
{
    String path = String(DATA_DIR) + "/" + name;
    if (!path.endsWith(".fdv")) path += ".fdv";
    f = SD.open(path.c_str(), FILE_READ);
    if (!f) {
        Serial.printf("ERR,open,%s\n", path.c_str());
        return false;
    }
    if (f.read((uint8_t *)&h, sizeof(h)) != sizeof(h) || memcmp(h.magic, "FDV1", 4) != 0) {
        Serial.printf("ERR,header,%s\n", path.c_str());
        f.close();
        return false;
    }
    if ((float)h.fs != FD_FS) Serial.printf("WARN,fs_mismatch,%lu,%.1f\n", (unsigned long)h.fs, (double)FD_FS);
    return true;
}

static int read_samples(File &f, int16_t *dst, int n)
{
    return f.read((uint8_t *)dst, n * sizeof(int16_t)) / (int)sizeof(int16_t);
}

/* ---------------------------------------------------- acquisition (ISR) */
static void IRAM_ATTR on_timer()
{
    portENTER_CRITICAL_ISR(&mux);
    const uint32_t avail = ring_head - ring_tail;
    if (avail < FD_BLOCK) {
        underruns++;
        portEXIT_CRITICAL_ISR(&mux);
        return;
    }
    int16_t *dst = &acq[acq_idx][acq_fill];
    for (int i = 0; i < FD_BLOCK; i++) dst[i] = ring[(ring_tail + i) % RING_SAMPLES];
    ring_tail += FD_BLOCK;
    acq_fill += FD_BLOCK;
    if (acq_fill >= FD_N) {
        acq_fill = 0;
        if (busy) {
            overruns++;                      /* deadline miss: previous window still in progress */
        } else {
            busy = true;
            BaseType_t woken = pdFALSE;
            xTaskNotifyFromISR(proc_task, acq_idx, eSetValueWithOverwrite, &woken);
            acq_idx ^= 1;
            windows_ready++;
            portEXIT_CRITICAL_ISR(&mux);
            if (woken) portYIELD_FROM_ISR();
            return;
        }
    }
    portEXIT_CRITICAL_ISR(&mux);
}

static void timer_start(uint32_t period_us)
{
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    timer = timerBegin(1000000);
    timerAttachInterrupt(timer, &on_timer);
    timerAlarm(timer, period_us, true, 0);
#else
    timer = timerBegin(0, 80, true);          /* 80 MHz APB / 80 = 1 MHz */
    timerAttachInterrupt(timer, &on_timer, true);
    timerAlarmWrite(timer, period_us, true);
    timerAlarmEnable(timer);
#endif
}

static void timer_stop()
{
    if (!timer) return;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    timerEnd(timer);
#else
    timerAlarmDisable(timer);
    timerDetachInterrupt(timer);
    timerEnd(timer);
#endif
    timer = nullptr;
}

/* ------------------------------------------------------ SD producer task */
static void sd_task_fn(void *)
{
    static int16_t chunk[SD_CHUNK];
    for (;;) {
        if (!streaming || sd_eof) {
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }
        const uint32_t used = ring_head - ring_tail;
        if (RING_SAMPLES - used < SD_CHUNK) {
            vTaskDelay(pdMS_TO_TICKS(2));
            continue;
        }
        const int n = read_samples(stream_file, chunk, SD_CHUNK);
        for (int i = 0; i < n; i++) ring[(ring_head + i) % RING_SAMPLES] = chunk[i];
        portENTER_CRITICAL(&mux);
        ring_head += n;
        portEXIT_CRITICAL(&mux);
        if (n < SD_CHUNK) sd_eof = true;
    }
}

/* -------------------------------------------------------------- classify */
static int classify(const float *feat, uint32_t *cycles)
{
    const uint32_t t0 = fd_cycles();
    int c;
    if (cls_mode == CLS_RF) c = fd_rf_predict(feat, NULL);
    else if (cls_mode == CLS_Q16) c = fd_rf_predict_q16(feat, NULL);
    else c = fd_softmax_predict(&sm, feat, NULL);
    *cycles = fd_cycles() - t0;
    return c;
}

static void lcd_show(int pred, int voted, uint32_t proc_us)
{
    if (!lcd_ok) return;
    lcd.setCursor(0, 0);
    lcd.printf("Win %-5lu %-3s V=%-2d  ", (unsigned long)n_proc, cls_mode == CLS_SM ? "SM" : (cls_mode == CLS_Q16 ? "Q16" : "RF"), vote_len);
    lcd.setCursor(0, 1);
    lcd.printf("Now : %-13s", fd_class_names[pred]);
    lcd.setCursor(0, 2);
    lcd.printf("Vote: %-13s", fd_class_names[voted]);
    lcd.setCursor(0, 3);
    lcd.printf("%5.1f ms ovr %-5lu", proc_us / 1000.0, (unsigned long)overruns);
}

/* ------------------------------------------------------ processing task */
static void proc_task_fn(void *)
{
    float feat[FD_N_FEATURES];
    fd_profile_t prof;
    for (;;) {
        uint32_t idx;
        xTaskNotifyWait(0, 0xFFFFFFFFu, &idx, portMAX_DELAY);
        const uint32_t t0 = micros();
        fd_extract_features(acq[idx & 1], stream_scale, work, feat, &prof);
        const int pred = classify(feat, &prof.cycles[FD_ST_CLASSIFY]);
        const uint32_t dt = micros() - t0;
        busy = false;                                     /* buffer released: deadline met */
        const int voted = fd_vote_add(&vote, pred);
        n_proc++;
        cls_count[pred]++;
        t_proc_sum += dt;
        if (dt > t_proc_max) t_proc_max = dt;
        Serial.printf("WIN,%lu,%d,%d,%lu", (unsigned long)n_proc, pred, voted, (unsigned long)dt);
        for (int s = 0; s < FD_ST_COUNT; s++) Serial.printf(",%lu", (unsigned long)prof.cycles[s]);
        Serial.println();
        if (vote.n >= vote_len) {
            Serial.printf("DECISION,%s,%d/%d\n", fd_class_names[voted], vote.count[voted], vote.n);
            fd_vote_reset(&vote);
        }
        lcd_show(pred, voted, dt);
    }
}

/* ------------------------------------------------------------- commands */
static void cmd_ls()
{
    File dir = SD.open(DATA_DIR);
    if (!dir) { Serial.println("ERR,no " DATA_DIR " directory"); return; }
    for (File f = dir.openNextFile(); f; f = dir.openNextFile()) {
        Serial.printf("FILE,%s,%lu\n", f.name(), (unsigned long)f.size());
        f.close();
    }
}

static void cmd_mem()
{
    Serial.printf("MEM,free_heap,%lu\nMEM,largest_block,%lu\nMEM,min_free_heap,%lu\n", (unsigned long)ESP.getFreeHeap(),
                  (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT), (unsigned long)ESP.getMinFreeHeap());
    Serial.printf("MEM,proc_stack_free,%lu\nMEM,sd_stack_free,%lu\n", (unsigned long)uxTaskGetStackHighWaterMark(proc_task),
                  (unsigned long)uxTaskGetStackHighWaterMark(sd_task));
    Serial.printf("MEM,static_work_bytes,%u\nMEM,acq_bytes,%u\nMEM,ring_bytes,%u\n", (unsigned)sizeof(work),
                  (unsigned)(2 * FD_N * sizeof(int16_t)), (unsigned)(RING_SAMPLES * sizeof(int16_t)));
}

static void cmd_run(const char *file, const char *mode, int V)
{
    fdv_header_t h;
    if (!open_fdv(file, stream_file, h)) return;
    cls_mode = !strcmp(mode, "sm") ? CLS_SM : (!strcmp(mode, "q16") ? CLS_Q16 : CLS_RF);
    vote_len = V > 0 ? V : 5;
    stream_scale = h.scale;
    fd_vote_reset(&vote);
    ring_head = ring_tail = acq_fill = 0;
    acq_idx = 0;
    underruns = overruns = windows_ready = n_proc = t_proc_max = 0;
    t_proc_sum = 0;
    memset(cls_count, 0, sizeof(cls_count));
    busy = false;
    sd_eof = false;
    ring_head = read_samples(stream_file, ring, RING_SAMPLES / 2);   /* pre-fill */
    streaming = true;
    const uint32_t period_us = (uint32_t)lroundf(FD_BLOCK * 1e6f / FD_FS);
    Serial.printf("RUN,%s,%lu samples,period_us=%lu,deadline_ms=%.1f\n", file, (unsigned long)h.n, (unsigned long)period_us,
                  1000.0 * FD_N / FD_FS);
    timer_start(period_us);
    /* run until the file is consumed */
    while (!(sd_eof && (ring_head - ring_tail) < FD_BLOCK)) {
        if (Serial.available() && Serial.peek() == 'q') { Serial.read(); break; }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    timer_stop();
    vTaskDelay(pdMS_TO_TICKS(FD_N * 1000 / (int)FD_FS + 50));        /* let the last window finish */
    streaming = false;
    stream_file.close();
    int best = 0;
    for (int c = 1; c < FD_N_CLASSES; c++) if (cls_count[c] > cls_count[best]) best = c;
    Serial.printf("SUMMARY,windows=%lu,overruns=%lu,underruns=%lu,proc_mean_ms=%.2f,proc_max_ms=%.2f,majority=%s\n",
                  (unsigned long)n_proc, (unsigned long)overruns, (unsigned long)underruns,
                  n_proc ? t_proc_sum / 1000.0 / n_proc : 0.0, t_proc_max / 1000.0, fd_class_names[best]);
}

static void cmd_bench(const char *file, int nwin)
{
    fdv_header_t h;
    File f;
    if (!open_fdv(file, f, h)) return;
    float *re = (float *)malloc(4 * FD_N), *im = (float *)malloc(4 * FD_N), *mag = (float *)malloc(4 * FD_K);
    float feat[FD_N_FEATURES];
    fd_profile_t po, pl;
    for (int w = 0; w < nwin; w++) {
        if (read_samples(f, acq[0], FD_N) < FD_N) break;
        fd_extract_features(acq[0], h.scale, work, feat, &po);
        uint32_t cc;
        const int pred = classify(feat, &cc);
        po.cycles[FD_ST_CLASSIFY] = cc;
        Serial.printf("BENCH,optimized,%d,%d", w, pred);
        for (int s = 0; s < FD_ST_COUNT; s++) Serial.printf(",%lu", (unsigned long)po.cycles[s]);
        Serial.println();
        if (re && im && mag) {
            fd_extract_features_legacy(acq[0], h.scale, re, im, mag, feat, &pl);
            pl.cycles[FD_ST_CLASSIFY] = cc;
            Serial.printf("BENCH,legacy,%d,%d", w, pred);
            for (int s = 0; s < FD_ST_COUNT; s++) Serial.printf(",%lu", (unsigned long)pl.cycles[s]);
            Serial.println();
        }
    }
    free(re); free(im); free(mag);
    f.close();
    Serial.printf("BENCH,cpu_mhz,%lu\n", (unsigned long)getCpuFrequencyMhz());
}

static void cmd_ft(const char *file, int cls, int k, int skip)
{
    fdv_header_t h;
    File f;
    if (cls < 0 || cls >= FD_N_CLASSES) { Serial.println("ERR,class"); return; }
    if (!open_fdv(file, f, h)) return;
    f.seek(sizeof(fdv_header_t) + (uint32_t)skip * FD_N * sizeof(int16_t));
    float feat[FD_N_FEATURES];
    int added = 0;
    while (added < k && n_support < MAX_SUPPORT && read_samples(f, acq[0], FD_N) == FD_N) {
        fd_extract_features(acq[0], h.scale, work, feat, NULL);
        fd_softmax_standardize(feat, &support_z[n_support * FD_SM_D]);
        support_y[n_support++] = (uint8_t)cls;
        added++;
    }
    f.close();
    Serial.printf("FT,added=%d,class=%s,support=%d\n", added, fd_class_names[cls], n_support);
}

static void cmd_adapt(float lr, int epochs)
{
    if (!n_support) { Serial.println("ERR,empty support set"); return; }
    const uint32_t t0 = fd_cycles();
    fd_softmax_finetune(&sm, &sm0, support_z, support_y, n_support, lr, epochs, 1e-3f);
    const uint32_t dt = fd_cycles() - t0;
    int ok = 0;
    for (int s = 0; s < n_support; s++) {
        float lg[FD_N_CLASSES];
        fd_softmax_logits(&sm, &support_z[s * FD_SM_D], lg);
        int b = 0;
        for (int c = 1; c < FD_N_CLASSES; c++) if (lg[c] > lg[b]) b = c;
        ok += b == support_y[s];
    }
    Serial.printf("ADAPT,n=%d,epochs=%d,cycles=%lu,ms=%.2f,support_acc=%d/%d\n", n_support, epochs, (unsigned long)dt,
                  dt / (getCpuFrequencyMhz() * 1000.0), ok, n_support);
}

static void handle(String line)
{
    line.trim();
    if (!line.length()) return;
    char buf[160];
    line.toCharArray(buf, sizeof(buf));
    char *argv[6] = {0};
    int argc = 0;
    for (char *t = strtok(buf, " "); t && argc < 6; t = strtok(NULL, " ")) argv[argc++] = t;
    const char *c = argv[0];
    if (!strcmp(c, "help")) {
        Serial.println("cmds: ls | mem | run <file> [rf|q16|sm] [V] | bench <file> [n] | ft <file> <class> <k> [skip] | "
                       "adapt [lr] [epochs] | save | load | reset");
    } else if (!strcmp(c, "ls")) cmd_ls();
    else if (!strcmp(c, "mem")) cmd_mem();
    else if (!strcmp(c, "run") && argc >= 2) cmd_run(argv[1], argc > 2 ? argv[2] : "rf", argc > 3 ? atoi(argv[3]) : 5);
    else if (!strcmp(c, "bench") && argc >= 2) cmd_bench(argv[1], argc > 2 ? atoi(argv[2]) : 10);
    else if (!strcmp(c, "ft") && argc >= 4) cmd_ft(argv[1], atoi(argv[2]), atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 0);
    else if (!strcmp(c, "adapt")) cmd_adapt(argc > 1 ? atof(argv[1]) : 0.1f, argc > 2 ? atoi(argv[2]) : 200);
    else if (!strcmp(c, "save")) {
        prefs.putBytes("sm", &sm, sizeof(sm));
        Serial.println("OK,saved");
    } else if (!strcmp(c, "load")) {
        Serial.println(prefs.getBytes("sm", &sm, sizeof(sm)) == sizeof(sm) ? "OK,loaded" : "ERR,no saved model");
    } else if (!strcmp(c, "reset")) {
        fd_softmax_load_default(&sm);
        n_support = 0;
        prefs.remove("sm");
        Serial.println("OK,reset");
    } else Serial.printf("ERR,unknown command '%s' (help)\n", c);
}

/* ---------------------------------------------------------------- setup */
void setup()
{
    Serial.begin(115200);
    delay(500);
    Serial.printf("INFO,fdcore edge node,N=%d,fs=%.0f,classes=%d,rf_trees=%d,rf_nodes=%d\n", FD_N, (double)FD_FS,
                  FD_N_CLASSES, FD_RF_N_TREES, FD_RF_N_NODES);
    acq[0] = (int16_t *)heap_caps_malloc(FD_N * sizeof(int16_t), MALLOC_CAP_8BIT);
    acq[1] = (int16_t *)heap_caps_malloc(FD_N * sizeof(int16_t), MALLOC_CAP_8BIT);
    ring = (int16_t *)heap_caps_malloc(RING_SAMPLES * sizeof(int16_t), MALLOC_CAP_8BIT);
    if (!acq[0] || !acq[1] || !ring) Serial.println("ERR,alloc");

    Wire.begin(21, 22);
    Wire.beginTransmission(LCD_ADDR);
    lcd_ok = Wire.endTransmission() == 0;
    if (lcd_ok) {
        lcd.init();
        lcd.backlight();
        lcd.print("fdcore edge node");
    }
    if (!SD.begin(SD_CS_PIN)) Serial.println("ERR,SD init");

    prefs.begin("fdcore", false);
    fd_softmax_load_default(&sm0);
    sm = sm0;
    if (prefs.getBytes("sm", &sm, sizeof(sm)) == sizeof(sm)) Serial.println("INFO,adapted softmax loaded from NVS");

    xTaskCreatePinnedToCore(proc_task_fn, "proc", 8192, NULL, 5, &proc_task, 1);
    xTaskCreatePinnedToCore(sd_task_fn, "sd", 6144, NULL, 4, &sd_task, 0);
    cmd_mem();
    Serial.println("READY (type help)");
}

void loop()
{
    if (Serial.available()) handle(Serial.readStringUntil('\n'));
    delay(5);
}
