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

#define MAX_WINDOWS 15 // user selectable
#define LCD_ADDR 0x27
#define SD_CS_PIN 5
#define DATA_DIR "/Vibration_Data"
// #define CSV_FILE "2Nm_BPFO_10.csv"
// #define CSV_FILE1 "2Nm_BPFI_10.csv"
// #define CSV_FILE2 "2Nm_Misalign_01.csv"
// #define CSV_FILE3 "2Nm_Normal.csv"
// #define CSV_FILE4 "2Nm_Unbalance_0583mg.csv"
const char *FILE_LIST[] = {
    "0Nm_Normal.csv",
    "2Nm_BPFO_10.csv",
    "2Nm_BPFI_10.csv",
    "2Nm_Misalign_01.csv",
    "2Nm_Unbalance_0583mg.csv"};
const int TOTAL_FILES = 5;
int current_file_index = 0;

/* ================= SD CARD ================= */
// bool open_csv_from_dir(File &file) {
//   String path = String(DATA_DIR) + "/" + String(CSV_FILE);
//   Serial.print("Opening: ");
//   Serial.println(path);

//   file = SD.open(path.c_str(), FILE_READ);
//   return (bool)file;
// }
bool open_current_csv(File &file)
{
  // Lấy tên file từ danh sách
  String filename = String(FILE_LIST[current_file_index]);
  String path = String(DATA_DIR) + "/" + filename;

  Serial.println("--------------------------------");
  Serial.print("Opening File [");
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
bool finished = false;

// Probe an I2C address; return true if device ACKs
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

/* Frequency-domain (proxy – consistent & stable) */
// void extract_freq_features(float* x, float* feat) {
//   float energy = 0;
//   float centroid = 0;

//   for (uint32_t i = 0; i < WINDOW_SIZE; i++) {
//     float e = x[i] * x[i];
//     energy += e;
//     centroid += i * e;
//   }

//   centroid /= (energy + 1e-9);

//   feat[6]  = energy / WINDOW_SIZE;
//   feat[7]  = sqrt(energy / WINDOW_SIZE);
//   feat[8]  = energy;
//   feat[9]  = 0.0f;                 // Skewness
//   feat[10] = 0.0f;                 // Kurtosis
//   feat[11] = centroid;
//   feat[12] = sqrt(centroid);
//   feat[13] = centroid;
//   feat[14] = sqrt(centroid);
//   feat[15] = log(energy + 1e-9f) / log(2.0f);
// }

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

/* ================= RANDOM FOREST STRUCT ================= */

typedef struct
{
  int feature;
  float threshold;
  int left;
  int right;
  int value; // -1 = internal node
} TreeNode;

typedef struct
{
  const TreeNode *nodes;
  int node_count;
} DecisionTree;

typedef struct
{
  const DecisionTree *trees;
  int n_trees;
  int n_classes;
} RandomForest;

/* ================= MODEL ================= */

int decision_tree_predict(const DecisionTree *tree, const float *x)
{
  int idx = 0;
  while (tree->nodes[idx].value == -1)
  {
    int f = tree->nodes[idx].feature;
    float t = tree->nodes[idx].threshold;
    idx = (x[f] <= t) ? tree->nodes[idx].left
                      : tree->nodes[idx].right;
  }
  return tree->nodes[idx].value;
}

int random_forest_predict(const RandomForest *rf, const float *x)
{
  int votes[NUM_CLASSES] = {0};

  for (int i = 0; i < rf->n_trees; i++)
  {
    int c = decision_tree_predict(&rf->trees[i], x);
    votes[c]++;
  }

  int best = 0;
  for (int c = 1; c < rf->n_classes; c++)
  {
    if (votes[c] > votes[best])
      best = c;
  }
  return best;
}

// ===== model parameters =====
/* ---- Tree 0 ---- */
static const TreeNode tree0_nodes[] = {
    {2, 10.534968f, 1, 48, -1}, {14, 1068.547760f, 2, 17, -1}, {12, 3331.640507f, 3, 10, -1}, {15, 10.788668f, 4, 9, -1}, {7, 5615.590762f, 5, 8, -1}, {0, 0.003619f, 6, 7, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 3}, {9, 19.974707f, 11, 16, -1}, {9, 16.199371f, 12, 15, -1}, {1, 2.510067f, 13, 14, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {11, 4781.045773f, 18, 47, -1}, {11, 4561.773577f, 19, 34, -1}, {12, 3392.175744f, 20, 27, -1}, {6, 42.830727f, 21, 24, -1}, {10, 480.769170f, 22, 23, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {7, 5570.850855f, 25, 26, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {10, 175.809860f, 28, 31, -1}, {10, 147.368500f, 29, 30, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {10, 481.995150f, 32, 33, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {10, 148.250990f, 35, 42, -1}, {11, 4589.964579f, 36, 39, -1}, {12, 3492.166991f, 37, 38, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {10, 136.138320f, 40, 41, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {6, 34.723320f, 43, 46, -1}, {6, 26.050829f, 44, 45, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 2}};

/* ---- Tree 1 ---- */
static const TreeNode tree1_nodes[] = {
    {6, 72.426380f, 1, 50, -1}, {14, 1071.680542f, 2, 33, -1}, {12, 3342.160887f, 3, 18, -1}, {12, 3336.642734f, 4, 9, -1}, {15, 10.759720f, 5, 8, -1}, {8, 16603.719000f, 6, 7, -1}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {7, 5724.168613f, 10, 17, -1}, {9, 16.326496f, 11, 14, -1}, {0, -0.012702f, 12, 13, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {3, -8.315884f, 15, 16, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {13, 1485.104674f, 19, 30, -1}, {9, 15.720224f, 20, 25, -1}, {1, 1.594754f, 21, 22, -1}, {-1, 0, -1, -1, 4}, {8, 7706.418500f, 23, 24, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {6, 43.817978f, 26, 29, -1}, {1, 1.661286f, 27, 28, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {5, 3.566078f, 31, 32, -1}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 4}, {10, 616.473630f, 34, 49, -1}, {13, 1452.001928f, 35, 48, -1}, {10, 385.182500f, 36, 43, -1}, {10, 145.960330f, 37, 40, -1}, {11, 4514.556470f, 38, 39, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {12, 3376.710104f, 41, 42, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {14, 1082.514772f, 44, 47, -1}, {6, 42.271650f, 45, 46, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 1}};

/* ---- Tree 2 ---- */
static const TreeNode tree2_nodes[] = {
    {1, 2.804831f, 1, 48, -1}, {12, 3344.145255f, 2, 15, -1}, {14, 1062.903055f, 3, 4, -1}, {-1, 0, -1, -1, 3}, {7, 5756.442016f, 5, 14, -1}, {9, 16.025171f, 6, 9, -1}, {8, 7003.962000f, 7, 8, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {9, 19.974707f, 10, 13, -1}, {0, 0.006212f, 11, 12, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 3}, {9, 20.243895f, 16, 47, -1}, {13, 1455.066203f, 17, 32, -1}, {12, 3389.148935f, 18, 25, -1}, {1, 2.489502f, 19, 22, -1}, {9, 16.098616f, 20, 21, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {10, 357.325130f, 23, 24, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {10, 149.362800f, 26, 29, -1}, {13, 1442.369237f, 27, 28, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {10, 476.784400f, 30, 31, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {9, 9.526072f, 33, 40, -1}, {13, 1465.369381f, 34, 37, -1}, {10, 138.142680f, 35, 36, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {10, 132.868970f, 38, 39, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {6, 34.246075f, 41, 44, -1}, {6, 26.062750f, 42, 43, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {1, 2.352284f, 45, 46, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 2}};

/* ---- Tree 3 ---- */
static const TreeNode tree3_nodes[] = {
    {2, 10.855757f, 1, 40, -1}, {14, 1046.924261f, 2, 3, -1}, {-1, 0, -1, -1, 3}, {9, 20.735657f, 4, 33, -1}, {11, 4555.008642f, 5, 20, -1}, {14, 1083.835129f, 6, 13, -1}, {1, 2.495624f, 7, 10, -1}, {15, 10.806876f, 8, 9, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {10, 336.598180f, 11, 12, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {10, 158.772050f, 14, 17, -1}, {7, 5647.960144f, 15, 16, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {9, 19.461914f, 18, 19, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {9, 9.398555f, 21, 28, -1}, {10, 130.752240f, 22, 25, -1}, {12, 3491.335016f, 23, 24, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {13, 1461.285430f, 26, 27, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {12, 3285.929758f, 29, 30, -1}, {-1, 0, -1, -1, 3}, {6, 25.813128f, 31, 32, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {7, 5664.064216f, 34, 35, -1}, {-1, 0, -1, -1, 4}, {10, 603.189800f, 36, 39, -1}, {11, 4650.821749f, 37, 38, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 3}, {15, 10.422857f, 41, 42, -1}, {-1, 0, -1, -1, 2}, {6, 234.457690f, 43, 46, -1}, {12, 3099.526481f, 44, 45, -1}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 2}, {-1, 0, -1, -1, 1}};

/* ---- Tree 4 ---- */
static const TreeNode tree4_nodes[] = {
    {8, 21733.832000f, 1, 50, -1}, {12, 3353.595499f, 2, 31, -1}, {15, 10.760626f, 3, 18, -1}, {7, 5722.206117f, 4, 17, -1}, {10, 337.771940f, 5, 10, -1}, {10, 335.801000f, 6, 9, -1}, {1, 1.638344f, 7, 8, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {4, 12.294322f, 11, 14, -1}, {0, 0.004986f, 12, 13, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {5, 3.020454f, 15, 16, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {6, 35.687164f, 19, 26, -1}, {7, 5518.234080f, 20, 21, -1}, {-1, 0, -1, -1, 3}, {10, 410.222930f, 22, 25, -1}, {10, 335.688870f, 23, 24, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {14, 1068.463668f, 27, 28, -1}, {-1, 0, -1, -1, 3}, {1, 1.689134f, 29, 30, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 3}, {11, 4776.597971f, 32, 49, -1}, {13, 1456.673325f, 33, 48, -1}, {14, 1085.836839f, 34, 41, -1}, {6, 42.741370f, 35, 38, -1}, {6, 41.337845f, 36, 37, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {1, 2.495716f, 39, 40, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {10, 147.741100f, 42, 45, -1}, {11, 4507.424736f, 43, 44, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {6, 26.579033f, 46, 47, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 1}};

/* ---- Tree 5 ---- */
static const TreeNode tree5_nodes[] = {
    {1, 5.196017f, 1, 52, -1}, {12, 3351.370110f, 2, 27, -1}, {15, 10.774474f, 3, 14, -1}, {2, 18.306869f, 4, 13, -1}, {9, 21.606995f, 5, 12, -1}, {11, 4629.227953f, 6, 9, -1}, {10, 344.665560f, 7, 8, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {0, -0.003631f, 10, 11, -1}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 2}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 2}, {14, 1065.303563f, 15, 20, -1}, {6, 35.405502f, 16, 19, -1}, {11, 4468.584945f, 17, 18, -1}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {10, 432.578770f, 21, 26, -1}, {2, 5.473491f, 22, 23, -1}, {-1, 0, -1, -1, 4}, {7, 5685.944553f, 24, 25, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 3}, {11, 4784.974655f, 28, 51, -1}, {11, 4565.772363f, 29, 44, -1}, {14, 1086.106818f, 30, 37, -1}, {6, 42.905727f, 31, 34, -1}, {9, 19.033964f, 32, 33, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {1, 2.502227f, 35, 36, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {10, 163.369700f, 38, 41, -1}, {11, 4496.732585f, 39, 40, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {9, 19.486115f, 42, 43, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {10, 613.753800f, 45, 50, -1}, {10, 163.322830f, 46, 49, -1}, {10, 152.709920f, 47, 48, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 2}};

/* ---- Tree 6 ---- */
static const TreeNode tree6_nodes[] = {
    {4, 22.924721f, 1, 36, -1}, {12, 3319.209411f, 2, 3, -1}, {-1, 0, -1, -1, 3}, {10, 621.915100f, 4, 35, -1}, {13, 1454.339269f, 5, 20, -1}, {14, 1083.977926f, 6, 13, -1}, {1, 2.494497f, 7, 10, -1}, {3, -7.063856f, 8, 9, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {10, 347.576200f, 11, 12, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {10, 157.624200f, 14, 17, -1}, {10, 126.062940f, 15, 16, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {10, 473.045720f, 18, 19, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {10, 159.218340f, 21, 28, -1}, {11, 4598.312438f, 22, 25, -1}, {14, 1107.477020f, 23, 24, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {10, 145.610180f, 26, 27, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {6, 26.091272f, 29, 32, -1}, {11, 4636.451738f, 30, 31, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {10, 445.409270f, 33, 34, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {6, 208.219330f, 37, 48, -1}, {6, 71.136140f, 38, 39, -1}, {-1, 0, -1, -1, 3}, {2, 11.357554f, 40, 41, -1}, {-1, 0, -1, -1, 3}, {2, 12.856815f, 42, 43, -1}, {-1, 0, -1, -1, 3}, {6, 98.187290f, 44, 47, -1}, {2, 13.473875f, 45, 46, -1}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 2}, {-1, 0, -1, -1, 2}, {-1, 0, -1, -1, 1}};

/* ---- Tree 7 ---- */
static const TreeNode tree7_nodes[] = {
    {3, -16.507732f, 1, 12, -1}, {15, 10.521600f, 2, 7, -1}, {13, 1606.445674f, 3, 4, -1}, {-1, 0, -1, -1, 2}, {4, 43.327959f, 5, 6, -1}, {-1, 0, -1, -1, 2}, {-1, 0, -1, -1, 1}, {8, 185256.100000f, 8, 11, -1}, {1, 5.447052f, 9, 10, -1}, {-1, 0, -1, -1, 2}, {-1, 0, -1, -1, 1}, {-1, 0, -1, -1, 1}, {12, 3364.009545f, 13, 46, -1}, {15, 10.787761f, 14, 37, -1}, {7, 5748.816545f, 15, 30, -1}, {9, 16.582924f, 16, 23, -1}, {10, 318.773160f, 17, 20, -1}, {6, 45.754433f, 18, 19, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 2}, {2, 7.789809f, 21, 22, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {6, 43.545887f, 24, 27, -1}, {4, 13.483250f, 25, 26, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {1, 2.540559f, 28, 29, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {2, 7.935360f, 31, 36, -1}, {6, 40.963010f, 32, 35, -1}, {1, 1.649367f, 33, 34, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 2}, {6, 34.233437f, 38, 45, -1}, {10, 396.795560f, 39, 44, -1}, {7, 5708.874288f, 40, 41, -1}, {-1, 0, -1, -1, 4}, {6, 33.895176f, 42, 43, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 4}};

/* ---- Tree 8 ---- */
static const TreeNode tree8_nodes[] = {
    {2, 11.807789f, 1, 46, -1}, {12, 3341.187862f, 2, 13, -1}, {15, 10.799828f, 3, 12, -1}, {11, 4696.820769f, 4, 11, -1}, {9, 16.199371f, 5, 8, -1}, {2, 7.630556f, 6, 7, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {5, 3.428749f, 9, 10, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 3}, {10, 621.417540f, 14, 45, -1}, {13, 1463.174352f, 15, 30, -1}, {14, 1085.548729f, 16, 23, -1}, {1, 2.499006f, 17, 20, -1}, {9, 16.014029f, 18, 19, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {7, 5516.813790f, 21, 22, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {10, 162.404300f, 24, 27, -1}, {10, 130.759570f, 25, 26, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {10, 487.212220f, 28, 29, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {10, 151.384540f, 31, 38, -1}, {13, 1480.789272f, 32, 35, -1}, {14, 1120.657670f, 33, 34, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {11, 4630.481234f, 36, 37, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {11, 4777.354787f, 39, 42, -1}, {10, 416.791930f, 40, 41, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {6, 34.771416f, 43, 44, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 2}};

/* ---- Tree 9 ---- */
static const TreeNode tree9_nodes[] = {
    {2, 10.731720f, 1, 48, -1}, {12, 3343.445426f, 2, 17, -1}, {15, 10.832901f, 3, 16, -1}, {11, 4776.102755f, 4, 15, -1}, {9, 16.104546f, 5, 10, -1}, {8, 7003.962000f, 6, 7, -1}, {-1, 0, -1, -1, 4}, {2, 7.676589f, 8, 9, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {10, 546.872600f, 11, 14, -1}, {12, 3342.247386f, 12, 13, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 3}, {10, 548.477800f, 18, 47, -1}, {11, 4553.047548f, 19, 34, -1}, {14, 1082.155494f, 20, 27, -1}, {6, 43.033460f, 21, 24, -1}, {15, 10.719741f, 22, 23, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {10, 337.248300f, 25, 26, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {6, 26.119642f, 28, 31, -1}, {9, 9.663543f, 29, 30, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {13, 1442.249043f, 32, 33, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 0}, {10, 153.893600f, 35, 42, -1}, {13, 1471.646873f, 36, 39, -1}, {14, 1112.571516f, 37, 38, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {11, 4629.533385f, 40, 41, -1}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {6, 34.256610f, 43, 46, -1}, {6, 25.920246f, 44, 45, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 4}, {-1, 0, -1, -1, 3}, {-1, 0, -1, -1, 1}};

/* ---- Forest ---- */
static const DecisionTree forest_trees[] = {
    {tree0_nodes, 49}, {tree1_nodes, 51}, {tree2_nodes, 49}, {tree3_nodes, 47}, {tree4_nodes, 51}, {tree5_nodes, 53}, {tree6_nodes, 49}, {tree7_nodes, 47}, {tree8_nodes, 47}, {tree9_nodes, 49}};

static const RandomForest rf_model = {
    forest_trees, 10, 5};

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

    for (int i = 0; i < 3; i++)
    {
      int cls = (i + offset) % NUM_CLASSES;
      uint32_t pct = (processed_windows == 0)
                         ? 0
                         : (vote_count[cls] * 100) / processed_windows;
      Serial.print(LABELS[cls]);
      Serial.print(": ");
      Serial.print(pct);
      Serial.println("%");
    }
    offset = (offset + 1) % NUM_CLASSES;
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
  Serial.println("Mean probability over all windows:");

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
  Serial.print("Final predicted label: ");
  Serial.print(LABELS[best_cls]);
  Serial.print(" (");
  Serial.print(processed_windows);
  Serial.println(" segments)");

  if (!lcd_ok)
    return;

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
    p[i] = (float)vote_count[i] / (float)processed_windows;
  }

  /* Line 0 */
  lcd.setCursor(0, 0);
  lcd.print(LABELS[0]);
  lcd.print(":");
  lcd.print(p[0], 1);

  lcd.setCursor(7, 0);
  lcd.print(LABELS[1]);
  lcd.print(":");
  lcd.print(p[1], 1);

  lcd.setCursor(14, 0);
  lcd.print(LABELS[2]);
  lcd.print(":");
  lcd.print(p[2], 1);

  /* Line 1 */
  lcd.setCursor(0, 1);
  lcd.print(LABELS[3]);
  lcd.print(":");
  lcd.print(p[3], 1);

  lcd.setCursor(7, 1);
  lcd.print(LABELS[4]);
  lcd.print(":");
  lcd.print(p[4], 1);

  /* Best class */
  uint8_t best = 0;
  for (int i = 1; i < NUM_CLASSES; i++)
  {
    if (p[i] > p[best])
      best = i;
  }

  /* Line 3 */
  lcd.setCursor(0, 3);
  lcd.print("=> ");
  lcd.print(LABELS[best]);
  lcd.print(" (");
  lcd.print(processed_windows);
  lcd.print(" segments)");
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
  lcd.print("Processing");

  lcd.setCursor(0, 1);
  lcd.print("Segments ");
  lcd.print(current);
  lcd.print("/");
  lcd.print(total);

  lcd.setCursor(0, 2);
  lcd.print("Last Pred: ");
  lcd.print(LABELS[last_pred]);
}

/* ================= CSV READING ================= */
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

  // Allocate window
  window = (float *)malloc(WINDOW_SIZE * sizeof(float));

  if (window == nullptr)
  {
    Serial.println("Malloc FAIL (internal heap)");
    fatal_error = true;
    return;
  }

  // SD init
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
  }
  Serial.println("First 10 samples:");
  for (int i = 0; i < 10; i++)
  {
    Serial.println(window[i], 6);
  }
}
void reset_variables()
{
  processed_windows = 0;
  memset(vote_count, 0, sizeof(vote_count));
  fatal_error = false;
}


void loop()
{
  static bool file_is_open = false;
  static File dataFile;
  static float features[NUM_FEATURES];
  if (!file_is_open)
  {
    if (!open_current_csv(dataFile))
    {
      Serial.println("FAIL to open file! Skipping to next...");
      if (lcd_ok)
      {
        lcd.clear();
        lcd.print("Open Fail!");
        delay(2000);
      }
      current_file_index = (current_file_index + 1) % TOTAL_FILES;
      return;
    }
    file_is_open = true;
    reset_variables();

    if (lcd_ok)
    {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("File:");
      lcd.print(current_file_index + 1);
      lcd.print("/");
      lcd.print(TOTAL_FILES);
      lcd.setCursor(0, 1);
      lcd.print(String(FILE_LIST[current_file_index]).substring(0, 16));
      delay(2000);
    }
  }
  if (processed_windows < MAX_WINDOWS)
  {

    if (!read_window_from_csv(dataFile, window))
    {
      Serial.println("EOF reached early (File ended)");
      processed_windows = MAX_WINDOWS;
    }
    else
    {
      extract_time_features(window, features);
      extract_freq_features(window, features);

      uint8_t pred = random_forest_predict(&rf_model, features);
      vote_count[pred]++;
      processed_windows++;

      lcd_show_processing(processed_windows, MAX_WINDOWS, pred);

      Serial.print("File ");
      Serial.print(current_file_index);
      Serial.print(" | Win ");
      Serial.print(processed_windows);
      Serial.print(" | Pred: ");
      Serial.println(LABELS[pred]);

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

    Serial.println(">>> WAITING 10 SECONDS BEFORE NEXT FILE <<<");

    for (int i = 10; i > 0; i--)
    {
      if (lcd_ok)
      {
        lcd.setCursor(14, 3);
        // lcd.print("Nx:");
        // lcd.print(i);
        if (i < 10)
          lcd.print(" ");
      }
      delay(1000);
    }
    current_file_index++;
    if (current_file_index >= TOTAL_FILES)
    {
      current_file_index = 0;
    }
  }
}
