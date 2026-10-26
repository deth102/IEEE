# KESTREL

Code, firmware, and experiment scripts for the paper
**"KESTREL: A Self-Checking, Retunable Edge Learner for Trustworthy Vibration Fault Diagnosis on Microcontrollers"**
(M.-C. Nguyen and T.-D. Nguyen, submitted to the IEEE Journal of Emerging and Selected Topics in Industrial Electronics).

KESTREL is a fault-diagnosis node that closes the monitor–alert–adapt–verify loop on a microcontroller. It has five parts:
- kinematic-order features (shaft orders, and bearing orders from a squared envelope);
- a random forest and a softmax head computed from one in-place FFT;
- a prototype self-check gate;
- an anchored on-chip update of the softmax head;
- the complete lifecycle on an ESP32.

## Data

| Dataset | Use | Source |
|---|---|---|
| KAIST | x channel of bearing housing A | Jung et al., *Data in Brief*, 2023 |
| CWRU | drive-end 48 kHz data | Case Western Reserve University bearing data center |

Paths to the data are set in `configs/default.yaml` (`paths`), relative to the parent folder of this repository.

## Setup

```
python -m pip install -r requirements.txt     # scikit-learn 1.8.0 reproduces the published numbers
python -m pytest tests -q                     # leak-free split, feature definitions, C == Python
```

## Reproducing the results

Run each experiment as `python -m experiments.<name> [--dataset kaist|cwru] [--config configs/cwru_env.yaml]`. The CWRU results of the paper use the bearing orders (`configs/cwru_env.yaml`). Tables go to `../results/tables`, figures to `../results/figures`.

| ID | Content |
|---|---|
| e01 | random split vs time-blocked cross-validation; feature sets F16, F19, F22 (Table II) |
| e03, e06, e07, e08 | feature ablation, window length, noise and voting, quantization |
| e09 | 1-D CNN baselines and their cost |
| e14, e16 | few-shot adaptation on unseen loads, label-free baselines, cross-severity support (Table V) |
| e17, e21, e23 | self-check gate, comparison with other shift detectors, design of the gate statistic (Table IV) |
| e18 | bearing-order (squared-envelope) features on CWRU |
| e22, e24, e27 | few-shot updates compared, labels at the new load (Table VI), role of the anchor |
| e26 | hyper-parameter search on validation data (`tools/e26_report.py` applies the selection rule) |
| e19, e20 | the complete node on the ESP32: equivalence, lifecycle, real-time replay, DAC–ADC loop |

**Hyper-parameters.**
- They are selected per dataset on validation data only (e26) and stored in `configs/default.yaml`.
- `dataset_overrides` holds the CWRU values; `fdedge.experiment.Run` applies them when `--dataset cwru` is given.

**Figures.**
- `python -m tools.fig_data` and `python -m tools.fig_data_target` prepare the figure data.
- `python -m tools.visio_figs` draws the figures in Microsoft Visio (Windows).

## Firmware

`firmware/lib/fdcore` is the portable C99 core:
- in-place real FFT and fused feature passes;
- bearing-order stage;
- random forest with compact nodes, and the softmax head;
- self-check gate, on-chip update, and re-referencing.

The other firmware folders:

| Folder | Content |
|---|---|
| `firmware/node_esp32` | the complete node, driven over USB by `tools/node_host.py`, which plays the gateway |
| `firmware/bench_mcu` | stage-by-stage cycle benchmark (ESP32, ESP32-S3, STM32F446, STM32H743, RP2040) |
| `firmware/host_test` | golden test: C against Python |

```
cd firmware/node_esp32
pio run -e node_kaist -t upload --upload-port COM3      # PlatformIO, platform espressif32@6.9.0
cd ../..
python -m experiments.e19_onchip --part lifecycle|trace|equiv|analog
python -m experiments.e20_cwru_onchip --dataset cwru --config configs/cwru_env.yaml --part all   # after node_cwru
```

The analog loop of e19 needs one jumper on the ESP32 board, from GPIO25 (DAC1) to GPIO34 (ADC1_CH6).

## Layout

| Path | Content |
|---|---|
| `configs/` | protocol, datasets, features, hyper-parameters |
| `fdedge/` | data and blocked protocol, features, models, gate, adaptation, resource model, C export, CNN baselines |
| `experiments/` | one script per experiment |
| `firmware/` | C library, node firmware, benchmarks, exported models |
| `tools/` | figure data and drawing, host tools for the board, summaries |
| `legacy/` | code of the earlier pipeline, kept for the comparison of Supplementary Note S1 |
