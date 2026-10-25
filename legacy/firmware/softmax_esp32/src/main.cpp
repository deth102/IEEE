#include <Arduino.h>
#include <Wire.h>
#include "LiquidCrystal_I2C.h"
#include <math.h>
#include <SPI.h>
#include <SD.h>
#include "esp_heap_caps.h"
#include <arduinoFFT.h>
#include <stdint.h>

/* ================= GLOBALS ================= */
float *window = nullptr;

/* ================= CONFIG ================= */
#define FS 25600
#define WINDOW_SIZE 8192 // 2^13
#define STEP_SIZE 2048   // 50% overlap
#define NUM_CLASSES 5
#define NUM_FEATURES 16

#define MAX_WINDOWS 15 
#define LCD_ADDR 0x27
#define SD_CS_PIN 5
#define DATA_DIR "/Vibration_Data"
const char *FILE_LIST[] = {
    "0Nm_Normal.csv",
    "2Nm_BPFO_10.csv",
    "2Nm_BPFI_10.csv",
    "2Nm_Misalign_01.csv",
    
    "2Nm_Unbalance_0583mg.csv"
};
const int TOTAL_FILES = 5;
int current_file_index = 0; 

const float W[NUM_CLASSES][NUM_FEATURES] = {
    {1.34298632e-02, -4.20027607e-01, -1.52225688e-02, 7.08524353e-01, -3.62402676e-01, -3.31600385e-01, -4.14010697e-01, -1.34319407e+00, 3.11273130e-01, 1.36694216e+00, -1.95960946e+00, -2.47680306e+00, 3.21048838e+00, -2.47680306e+00, 3.21048838e+00, -5.53443015e-01},
    {9.70783723e-03, 9.30586038e-01, 1.12933880e+00, -1.17027562e+00, 1.15129177e+00, 8.63716252e-01, 1.16209325e+00, 7.32156029e-01, 9.40005921e-01, -8.22675736e-02, -1.57746991e-01, 6.78447750e-01, 3.44006446e-01, 6.78447750e-01, 3.44006446e-01, 5.20547615e-01},
    {-3.52842998e-02, 1.03927961e+00, 5.75117531e-01, -2.56218077e-01, 4.16174089e-01, -1.47565524e-02, 8.62568878e-01, 1.20599044e-01, 3.82463264e-02, -8.27570596e-01, -6.25276297e-01, 4.83707527e-01, -1.19754975e+00, 4.83707527e-01, -1.19754975e+00, -3.04204731e+00},
    {1.97686457e-02, -6.19382261e-01, -7.09436333e-01, 5.91483766e-01, -6.51287029e-01, 1.18487162e-01, -1.28109179e-01, 1.67458153e-01, -6.51231231e-01, -9.78277073e-01, 4.47354703e+00, 1.20059040e+00, -3.74007199e+00, 1.20059040e+00, -3.74007199e+00, 4.45213646e+00},
    {-7.62204638e-03, -9.30455783e-01, -9.79797427e-01, 1.26485573e-01, -5.53776152e-01, -6.35846477e-01, -1.48254226e+00, 3.22980849e-01, -6.38294147e-01, 5.21173079e-01, -1.73091428e+00, 1.14057384e-01, 1.38312691e+00, 1.14057384e-01, 1.38312691e+00, -1.37719375e+00},
};

const float b[5] = {
    -1.99177451e+00, -8.95803530e-01, 1.24752715e+00, 1.24837374e+00, 3.91677144e-01};

const float mean[16] = {
    -2.22359397e-05, 4.38219157e+00, 1.88185900e+01, -1.86723302e+01, 3.74909202e+01, 3.86203993e+00, 8.64045224e+01, 5.77639602e+03, 1.25980091e+05, 1.41736559e+01, 3.27305237e+02, 4.74062901e+03, 3.28006539e+03, 1.51700128e+03, 1.04962092e+03, 1.07184185e+01};

const float feature_std[16] = {
    6.67357785e-03, 4.91879159e+00, 2.33247780e+01, 2.33332682e+01, 4.65980329e+01, 8.85770769e-01, 8.83944302e+01, 6.05075084e+02, 2.54605318e+05, 5.07282045e+00, 1.82885189e+02, 6.64087211e+02, 2.43849698e+02, 2.12507908e+02, 7.80319032e+01, 3.84686924e-01};

/* ================= SD CARD ================= */
bool open_current_csv(File &file)
{
  String filename = String(FILE_LIST[current_file_index]);
  String path = String(DATA_DIR) + "/" + filename;
  
  Serial.print("Opening [");
  Serial.print(current_file_index + 1);
  Serial.print("/");
  Serial.print(TOTAL_FILES);
  Serial.print("]: ");
  Serial.println(path);

  file = SD.open(path.c_str(), FILE_READ);
  return (bool)file;
}

/* ================= LCD ================= */
LiquidCrystal_I2C lcd(LCD_ADDR, 20, 04);

/* ================= LABEL ================= */
const char *LABELS[NUM_CLASSES] = {
    "Nm",
    "IN",
    "OT",
    "MA",
    "Ub"};

/* ================= RUNTIME ================= */
uint32_t vote_count[NUM_CLASSES] = {0};
uint32_t processed_windows = 0;
bool sd_ok = true;
bool fatal_error = false;
bool lcd_ok = false;

bool probe_i2c_addr(uint8_t addr)
{
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

/* ================= FEATURE EXTRACTION ================= */

void extract_time_features(float *x, float *feat)
{
  float sum = 0, sumsq = 0;
  float maxv = -1e9, minv = 1e9;

  for (uint32_t i = 0; i < WINDOW_SIZE; i++)
  {
    float v = x[i];
    sum += v;
    sumsq += v * v;
    if (v > maxv)
      maxv = v;
    if (v < minv)
      minv = v;
  }

  float mean = sum / WINDOW_SIZE;
  float rms = sqrt(sumsq / WINDOW_SIZE);

  feat[0] = mean;
  feat[1] = rms;
  feat[2] = maxv;
  feat[3] = minv;
  feat[4] = maxv - minv;
  feat[5] = maxv / (rms + 1e-9);
}

#include <ArduinoFFT.h>

static float fft_real[WINDOW_SIZE];
static float fft_imag[WINDOW_SIZE];

ArduinoFFT<float> FFT;

void fft_magnitude(float *x, float *mag)
{
  for (uint32_t i = 0; i < WINDOW_SIZE; i++)
  {
    fft_real[i] = x[i];
    fft_imag[i] = 0.0f;
  }

  FFT.windowing(
      fft_real,
      WINDOW_SIZE,
      FFT_WIN_TYP_HAMMING,
      FFT_FORWARD);

  FFT.compute(
      fft_real,
      fft_imag,
      WINDOW_SIZE,
      FFT_FORWARD);

  FFT.complexToMagnitude(
      fft_real,
      fft_imag,
      WINDOW_SIZE);

  for (uint32_t i = 0; i < WINDOW_SIZE / 2; i++)
  {
    mag[i] = fft_real[i];
  }
}

void extract_freq_features(float *x, float *feat)
{
  const uint32_t N = WINDOW_SIZE;
  const uint32_t K = N / 2;
  const float fs = FS;

  static float Y[WINDOW_SIZE / 2];

  /* FFT magnitude */
  fft_magnitude(x, Y);

  /* ===== Sum spectrum ===== */
  float sumY = 0.0f;
  for (uint32_t i = 0; i < K; i++)
    sumY += Y[i];
  sumY += 1e-12f;

  /* ===== Mean ===== */
  float mean = sumY / K;

  /* ===== Variance ===== */
  float var = 0.0f;
  for (uint32_t i = 0; i < K; i++)
  {
    float d = Y[i] - mean;
    var += d * d;
  }
  var /= K;

  /* ===== Skewness & Kurtosis ===== */
  float stdY = sqrtf(var) + 1e-12f;
  float skew = 0.0f;
  float kurt = 0.0f;

  for (uint32_t i = 0; i < K; i++)
  {
    float z = (Y[i] - mean) / stdY;
    skew += z * z * z;
    kurt += z * z * z * z;
  }
  skew /= K;
  kurt = kurt / K - 3.0f;

  /* ===== Central frequency ===== */
  float central = 0.0f;
  for (uint32_t i = 0; i < K; i++)
  {
    float f = fs * i / N;
    central += f * Y[i];
  }
  central /= sumY;

  /* ===== Frequency STD ===== */
  float std_freq = 0.0f;
  for (uint32_t i = 0; i < K; i++)
  {
    float f = fs * i / N;
    float d = f - central;
    std_freq += d * d * Y[i];
  }
  std_freq = sqrtf(std_freq / sumY);

  /* ===== RMS frequency ===== */
  float rms = 0.0f;
  for (uint32_t i = 0; i < K; i++)
  {
    float f = fs * i / N;
    rms += f * f * Y[i];
  }
  rms = sqrtf(rms / sumY);

  /* ===== Spectral centroid (bin-based) ===== */
  float sc = 0.0f;
  for (uint32_t i = 0; i < K; i++)
  {
    sc += i * Y[i];
  }
  sc /= sumY;

  /* ===== Spectral spread ===== */
  float ss = 0.0f;
  for (uint32_t i = 0; i < K; i++)
  {
    float d = i - sc;
    ss += d * d * Y[i];
  }
  ss = sqrtf(ss / sumY);

  /* ===== Spectral entropy ===== */
  float entropy = 0.0f;
  for (uint32_t i = 0; i < K; i++)
  {
    float p = Y[i] / sumY;
    entropy -= p * log2f(p + 1e-12f);
  }

  /* ===== OUTPUT – GIỮ NGUYÊN INDEX ===== */
  feat[6] = mean;
  feat[7] = rms;
  feat[8] = var;
  feat[9] = skew;
  feat[10] = kurt;
  feat[11] = central;
  feat[12] = std_freq;
  feat[13] = sc;
  feat[14] = ss;
  feat[15] = entropy;
}
void print_features_serial(const char *label, float *feat, int start, int count)
{
  Serial.print(label);
  Serial.print(": ");
  for (int i = 0; i < count; i++)
  {
    Serial.print(feat[start + i], 6);
    Serial.print(" ");
  }
  Serial.println();
}

/* ================= MODEL ================= */

/* Normalize giống Python */
void normalize_features(float *x)
{
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    x[i] = (x[i] - mean[i]) / feature_std[i];
  }
}

void linear_layer(float *x, float *z)
{
  for (int c = 0; c < NUM_CLASSES; c++)
  {
    float acc = b[c];
    for (int i = 0; i < NUM_FEATURES; i++)
    {
      acc += W[c][i] * x[i];
    }
    z[c] = acc;
  }
}

void softmax(float *z, float *p)
{
  float maxz = z[0];
  for (int i = 1; i < NUM_CLASSES; i++)
    if (z[i] > maxz)
      maxz = z[i];

  float sum = 0;
  for (int i = 0; i < NUM_CLASSES; i++)
  {
    p[i] = exp(z[i] - maxz);
    sum += p[i];
  }

  for (int i = 0; i < NUM_CLASSES; i++)
    p[i] /= sum;
}

uint8_t predict_class(float *feat)
{
  float z[NUM_CLASSES];
  float p[NUM_CLASSES];

  normalize_features(feat);
  linear_layer(feat, z);
  softmax(z, p);

  uint8_t best = 0;
  float best_p = p[0];
  for (int i = 1; i < NUM_CLASSES; i++)
  {
    if (p[i] > best_p)
    {
      best_p = p[i];
      best = i;
    }
  }
  return best;
}

/* ================= LCD ================= */
void update_lcd()
{
  static uint8_t offset = 0;

  if (!lcd_ok)
  {
    Serial.println("=== Status ===");
    Serial.print("Win ");
    Serial.print(processed_windows);
    Serial.print("/");
    Serial.println(MAX_WINDOWS);
    return;
  }

  lcd.clear();

  /* Line 0 */
  lcd.setCursor(0, 0);
  lcd.print("Win ");
  lcd.print(processed_windows);
  lcd.print("/");
  lcd.print(MAX_WINDOWS);

  /* Line 1–3 */
  for (int i = 0; i < 3; i++)
  {
    int cls = (i + offset) % NUM_CLASSES;
    uint32_t pct = (processed_windows == 0)
                       ? 0
                       : (vote_count[cls] * 100) / processed_windows;

    lcd.setCursor(0, i + 1);
    lcd.print(LABELS[cls]);
    lcd.print(": ");
    lcd.print(pct);
    lcd.print("%");
  }

  offset = (offset + 1) % NUM_CLASSES;
}

void print_final_result()
{
  Serial.println();
  Serial.println("====== FINAL RESULT FOR THIS FILE ======");

  uint8_t best_cls = 0;
  float best_p = 0.0f;

  for (int i = 0; i < NUM_CLASSES; i++)
  {
    float p = (float)vote_count[i] / (float)processed_windows;
    Serial.print(LABELS[i]);
    Serial.print(" : ");
    Serial.println(p, 4);

    if (p > best_p)
    {
      best_p = p;
      best_cls = i;
    }
  }

  Serial.println();
  Serial.print("Predicted: ");
  Serial.println(LABELS[best_cls]);

  if (!lcd_ok) return;

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("FINAL:");
  lcd.setCursor(0, 1);
  lcd.print(LABELS[best_cls]);
}

void lcd_show_prob_3_2_and_final()
{
  if (!lcd_ok)
    return;

  lcd.clear();

  float p[NUM_CLASSES];
  for (int i = 0; i < NUM_CLASSES; i++)
  {
    p[i] = (processed_windows == 0) ? 0 : (float)vote_count[i] / (float)processed_windows;
  }

  /* Line 0 */
  lcd.setCursor(0, 0);
  lcd.print(LABELS[0]); lcd.print(":"); lcd.print(p[0], 1);
  lcd.setCursor(7, 0);
  lcd.print(LABELS[1]); lcd.print(":"); lcd.print(p[1], 1);
  lcd.setCursor(14, 0);
  lcd.print(LABELS[2]); lcd.print(":"); lcd.print(p[2], 1);

  /* Line 1 */
  lcd.setCursor(0, 1);
  lcd.print(LABELS[3]); lcd.print(":"); lcd.print(p[3], 1);
  lcd.setCursor(7, 1);
  lcd.print(LABELS[4]); lcd.print(":"); lcd.print(p[4], 1);

  /* Best class */
  uint8_t best = 0;
  for (int i = 1; i < NUM_CLASSES; i++)
  {
    if (p[i] > p[best])
      best = i;
  }

  lcd.setCursor(0, 3);
  lcd.print("=> ");
  lcd.print(LABELS[best]);
  lcd.print(" (");
  lcd.print(processed_windows);
  lcd.print(")");
}

void lcd_show_processing(uint32_t current, uint32_t total, uint8_t last_pred)
{
  if (!lcd_ok)
  {
    Serial.print("Processing ");
    Serial.print(current);
    Serial.print("/");
    Serial.println(total);
    return;
  }

  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print("File: ");
  lcd.print(current_file_index + 1);
  lcd.print("/");
  lcd.print(TOTAL_FILES);

  lcd.setCursor(0, 1);
  lcd.print("Seg: ");
  lcd.print(current);
  lcd.print("/");
  lcd.print(total);

  lcd.setCursor(0, 2);
  lcd.print("Last: ");
  lcd.print(LABELS[last_pred]);
}

bool read_window_from_csv(File &file, float *buffer)
{
  uint32_t count = 0;

  while (file.available() && count < WINDOW_SIZE)
  {
    String line = file.readStringUntil('\n');
    buffer[count++] = line.toFloat();
  }
  return (count == WINDOW_SIZE);
}

void reset_variables() {
  processed_windows = 0;
  memset(vote_count, 0, sizeof(vote_count));
  fatal_error = false;
  
}

/* ================= SETUP ================= */

void setup()
{
  Serial.begin(115200);
  delay(1000);

  Serial.println("ESP32 started");

  Wire.begin(21, 22);

  // LCD
  lcd_ok = probe_i2c_addr(LCD_ADDR);
  if (lcd_ok)
  {
    lcd.init();
    lcd.backlight();
    lcd.clear();
    lcd.print("LCD OK");
  }
  else
  {
    Serial.println("LCD not found");
  }
  Serial.print("Free heap: ");
  Serial.println(ESP.getFreeHeap());

  Serial.print("Largest block: ");
  Serial.println(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

  window = (float *)malloc(WINDOW_SIZE * sizeof(float));

  if (window == nullptr)
  {
    Serial.println("Malloc FAIL (internal heap)");
    fatal_error = true;
    return;
  }

  if (!SD.begin(SD_CS_PIN))
  {
    Serial.println("SD init FAIL");
    if (lcd_ok)
    {
      lcd.clear();
      lcd.print("SD FAIL");
    }
    fatal_error = true;
    return;
  }

  Serial.println("SD init OK");
  if (lcd_ok)
  {
    lcd.clear();
    lcd.print("SD OK");
    delay(1000);
  }
}

void loop()
{
  static bool file_is_open = false;
  static File dataFile;
  static float features[NUM_FEATURES];

  if (!file_is_open)
  {
    reset_variables();

    if (!open_current_csv(dataFile))
    {
      Serial.println("Failed to open file! Skipping...");
      if (lcd_ok) { lcd.clear(); lcd.print("Open Fail!"); delay(2000); }
      current_file_index = (current_file_index + 1) % TOTAL_FILES;
      return; 
    }

    if (lcd_ok)
    {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Processing File:");
      lcd.setCursor(0, 1);
      String fname = String(FILE_LIST[current_file_index]);
      lcd.print(fname.substring(0, 16)); 
      delay(2000);
    }
    
    file_is_open = true;
  }

  if (processed_windows < MAX_WINDOWS)
  {
    if (!read_window_from_csv(dataFile, window))
    {
      Serial.println("End of File reached early");
      processed_windows = MAX_WINDOWS; 
    }
    else 
    {
     
      extract_time_features(window, features);
      extract_freq_features(window, features);

      uint8_t pred = predict_class(features);
      vote_count[pred]++;
      processed_windows++;

      lcd_show_processing(processed_windows, MAX_WINDOWS, pred);
      
      Serial.print("File "); Serial.print(current_file_index);
      Serial.print(" | Win "); Serial.print(processed_windows);
      Serial.print(" | Pred: "); Serial.println(LABELS[pred]);
      
      delay(50); 
      return; 
    }
  }

  if (file_is_open)
  {
    dataFile.close();
    file_is_open = false; 
    print_final_result();
    lcd_show_prob_3_2_and_final();

    Serial.println(">>> WAITING 10s FOR NEXT FILE <<<");
    for(int i=10; i>0; i--) {
        if(lcd_ok) {
            lcd.setCursor(15, 3);
            // lcd.print("N:");
            // lcd.print(i);
            if(i<10) lcd.print(" ");
        }
        delay(1000);
    }
    current_file_index++;
    if (current_file_index >= TOTAL_FILES) {
      current_file_index = 0;
    }
  }
}