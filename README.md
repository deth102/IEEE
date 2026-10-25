# Code for the JESTIE paper — resource-aware vibration fault diagnosis on MCUs

Everything needed to reproduce the numbers, figures and firmware of the paper.
Run all commands from `D:\IEEE\03_code`.

```
python -m pip install -r requirements.txt
python run_all.py            # every experiment, KAIST + CWRU  (~1 h CPU, CNN on GPU)
python run_all.py --quick    # smoke test
python -m pytest tests -q    # leak-free split, feature definitions, C == Python
```
Outputs go to `D:\IEEE\results\` (`figures/` PDF+PNG in IEEE column widths,
`tables/` CSV+LaTeX, `logs/`, `SUMMARY.md` with every headline number).

## Layout

| Path | Content |
|---|---|
| `configs/default.yaml` | windows, protocol, datasets, model hyper-parameters |
| `configs/platforms.yaml` | MCU parameters of the latency/energy model (**un-calibrated estimates**, see below) |
| `fdedge/data.py` | KAIST/CWRU loaders (signal cache), feature store, **blocked** and legacy random protocols |
| `fdedge/features.py` | 16 conference features + 3 shaft-order features (F19); identical to the C code |
| `fdedge/models.py` | RF (scikit-learn training, library-free C inference), softmax regression (+ fine-tuning) |
| `fdedge/adaptation.py` | HBR, DFS reference, few-shot selection, Welford statistics |
| `fdedge/resources.py` | operation counts per stage (legacy/optimized), memory model, platform estimates, calibration |
| `fdedge/export_c.py` | code generation: model headers, FFT/window tables, int16 test vectors |
| `fdedge/cnn.py` | 1-D CNN baselines (from `D:\baodoan\code`) + MAC/parameter/activation cost |
| `experiments/e01…e13` | one script per experiment (below) |
| `firmware/lib/fdcore` | portable C core: in-place real FFT, fused features, RF (8-B / 6-B int16 nodes), softmax + on-device fine-tuning, Welford, voting, legacy pipeline for comparison |
| `firmware/host_test` | gcc golden test (C vs Python) |
| `firmware/bench_mcu` | PlatformIO cycle benchmark: ESP32, ESP32-S3, STM32F446, STM32H743, RP2040 |
| `firmware/edge_node_esp32` | real-time HIL edge node: SD → timer/DMA emulation at fs → double buffer → dual-core processing → LCD/Serial, few-shot fine-tuning over Serial, NVS persistence |
| `firmware/models/<tag>` | exported models (`active/` = the one the firmware builds with) |
| `tools/` | `build_host`, `select_model`, `make_sd_binary`, `parse_bench` (+calibration), `elf_memory`, `make_summary` |
| `legacy/` | graduation-thesis code and ELF (reference only) |

## Experiments

| ID | Question | Main output |
|---|---|---|
| e01 | leaky random split vs time-blocked CV; F16 vs F19 | `e01_protocol_summary`, `f1_by_protocol`, `confusion_blocked` |
| e02 | hyper-parameter validation curves (blocked validation block) | `validation_curves`, `rf_size_tradeoff` |
| e03 | feature ablation, permutation importance, cost per group | `ablation_importance` |
| e04 | unseen load: single-source 3×3 and leave-one-load-out | `crossload_matrix` |
| e05 | adaptation: none / HBR / DFS(oracle) / few-shot (on-device FT, host retrain) | `fewshot_curve`, `finetune_cost` |
| e06 | window length N: accuracy vs RAM vs latency, legacy vs optimized | `design_space` |
| e07 | majority voting; AWGN robustness; noise-augmented training | `voting_noise` |
| e08 | node formats, int16 thresholds, int8 softmax, 16-bit ADC | `formats`, `forest_grid` |
| e09 | 1-D CNN baselines, same protocols, MACs/RAM/latency | `accuracy`, `cost` |
| e11 | per-stage op counts, memory (validated on the thesis ELF), platform estimates | `stage_breakdown`, `platform_summary` |
| e12 | export to C + host golden test for N = 2048…16384 | `golden` |
| e13 | exact RAM/flash from cross-compiled firmware | `memory` |
| e14 | review checks: unseen severity, time-separated few-shot, MLP/SVM, replay subset | `severity`, `fewshot_time` |
| e15 | why the 99.52 % of the conference paper is not reproduced (subset, window, overlap, legacy RF) | `accounting`, `summary` |
| e16 | adaptation checks: matched test set, cross-severity support, hyper-parameters from source loads only | `matched`, `cross_severity`, `hp_outer` |

`--dataset cwru` runs e01, e03, e04, e05 on the CWRU 48 kHz drive-end data (4 classes, 0–3 hp).

## Firmware workflow

```
python -m experiments.e12_export               # models + test vectors (+ host golden test)
python -m tools.select_model kaist_n8192       # choose the model the firmware compiles
cd firmware/bench_mcu && pio run -e esp32dev -t upload -t monitor > bench_esp32.log
python -m tools.parse_bench firmware/bench_mcu/bench_esp32.log --platform esp32 --calibrate
```
Copy the printed `cycles:` line into `configs/platforms.yaml` (set `calibrated: true`) and
re-run e06/e09/e11 — the latency/energy estimates then rest on measured cycles.

Edge node (ESP32 DevKit + microSD SPI CS=5 + LCD 20×4 I2C 0x27, same wiring as the thesis):
```
python -m tools.make_sd_binary --test-block-only    # results/sd_card/fd/*.fdv -> copy "fd" to the SD root
cd firmware/edge_node_esp32 && pio run -t upload -t monitor
> run 2Nm_BPFO_10 rf 5        # real-time streaming, vote over 5 windows; prints WIN/DECISION/SUMMARY lines
> bench 2Nm_Normal 10         # per-stage cycles, optimized vs legacy, on the real board
> ft 4Nm_Normal 0 5 ...       # collect 5 labelled windows per class at the new load, then
> adapt 0.1 200               # fine-tune the softmax head on the device;  save  -> NVS
```

## Hardware-in-the-loop on a bare ESP32 (one jumper GPIO25 -> GPIO34)

```
python -m tools.make_hil_data                       # signal set (fold-0 test blocks) + noise-aug model export
cd firmware/hil_esp32 && pio run -e hil_clean -t upload --upload-port COM3
python -m tools.capture_serial --port COM3 --out ../results/hw/hil_clean.log --timeout 500
(same with -e hil_aug -> hil_aug.log)
python -m tools.parse_hil ../results/hw/hil_clean.log ../results/hw/hil_aug.log
```
Phases (automatic): DAC->ADC calibration, ADC-rate check, digital replay (timer at fs), analog loop, on-device
fine-tuning on the analog support windows, raw-window dump. Benchmark: `firmware/bench_mcu` envs `esp32dev`
(tables in flash) and `esp32dev_ramtables` (tables in DRAM, `-DFD_TABLES_IN_RAM`).

## The complete node on a bare ESP32, driven by the host as gateway (e19, e20)

`firmware/node_esp32` runs the whole node: features (with the bearing-order stage in the CWRU build), both heads,
the prototype self-check gate (`fdedge/gate.py`, `fd_gate.c`), the on-chip softmax update and the re-referencing of
the gate, and a forest/threshold refresh received at run time. `tools/node_host.py` uploads models and streams
recordings over USB (921.6 kbaud; the CH340 bridge needs the paced output of the firmware).
```
cd firmware/node_esp32 && pio run -e node_kaist -t upload --upload-port COM3     # platform espressif32@6.9.0
cd ../.. && python -m experiments.e19_onchip --part trace|lifecycle|analog|equiv
python -m experiments.e20_cwru_onchip --dataset cwru --config configs/cwru_env.yaml --part export   # then node_cwru
python -m experiments.e20_cwru_onchip --dataset cwru --config configs/cwru_env.yaml --part all
python -m tools.onchip_tables                                                     # summaries for the paper
```
Further experiments of the target draft: e21 (gate vs. kNN, class-conditional Mahalanobis, energy, PCA-SPE,
isolation forest), e22 (few-shot updates incl. CNN last-layer fine-tuning, `--cnn`), e23 (design of the gate statistic).

## What is measured and what is estimated

* **Exact without hardware**: accuracy numbers; RAM of the buffer layout (validated
  byte-exactly on the thesis ELF: 114 688 B); flash/RAM of the cross-compiled firmware (e13);
  C-vs-Python equivalence (e12; max relative feature error ≈ 1e-5, 100 % prediction agreement).
* **Measured on an ESP32** (results/hw): per-stage cycles, real-time replay, DAC->ADC loop, on-device fine-tuning.
  The ESP32 entry of `platforms.yaml` is calibrated on these measurements (stage MAPE 8.3 %, other N within 3 %).
* **Estimated**: latency of the other MCUs and of the CNNs (literature cycle costs), all energy figures.

## Known facts to report honestly

* The 99.52 % of the conference paper could not be reproduced: the legacy RF code on the
  legacy feature file gives 91.7 % (random split, N = 32768); with the blocked protocol
  the conference feature set (F16) reaches macro-F1 0.88 ± 0.03.
* The thesis firmware computed features on 8192-sample windows while the model was trained
  on 32768-sample windows (inferred from the feature file), and the random split mixed
  overlapping windows between train and test.
* HBR (unsupervised healthy-baseline recalibration) is not consistently beneficial; it is
  reported as an ablation, the proposed adaptation is few-shot on-device fine-tuning.
* Order features help shaft faults (KAIST: unbalance, misalignment) but are neutral on the
  bearing-only CWRU data.
