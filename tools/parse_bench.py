"""Parse serial logs of bench_mcu / edge_node_esp32 and calibrate the latency model.

usage:
  python -m tools.parse_bench <log.txt> --platform esp32 [--calibrate]

* STAGE lines (bench_mcu) and BENCH lines (edge node) -> per-stage cycles table
* --calibrate: fit effective cycles per operation class (NNLS) from the measured
  stages of the optimized and legacy pipelines and print a platforms.yaml snippet
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from fdedge.resources import (calibrate, ops_fft, ops_freq_features, ops_magnitude, ops_order_features,
                              ops_time_features, ops_window)

STAGE_OPS = {
    "time_feat": ops_time_features, "window": ops_window, "fft": ops_fft,
    "magnitude": ops_magnitude, "freq_feat": ops_freq_features,
}
EDGE_STAGES = ["time_feat", "window", "fft", "magnitude", "freq_feat", "order_feat", "classifier"]


def parse(path: Path) -> tuple[dict, pd.DataFrame]:
    info, rows = {}, []
    bench = defaultdict(list)
    for line in Path(path).read_text(errors="ignore").splitlines():
        p = line.strip().split(",")
        if p[0] == "INFO" and len(p) >= 3:
            info[p[1]] = p[2]
        elif p[0] == "STAGE" and len(p) == 5:
            rows.append({"variant": p[1], "stage": p[2], "min_cycles": int(p[3]), "mean_cycles": int(p[4])})
        elif p[0] == "CLS" and len(p) == 4:
            rows.append({"variant": "classifier", "stage": p[1], "min_cycles": int(p[2]), "mean_cycles": int(p[3])})
        elif p[0] == "BENCH" and len(p) == 4 + len(EDGE_STAGES):
            for s, v in zip(EDGE_STAGES, p[4:]):
                bench[(p[1], s)].append(int(v))
        elif p[0] == "BENCH" and p[1] == "cpu_mhz":
            info["f_cpu_hz"] = str(int(p[2]) * 1_000_000)
        elif p[0] == "CHECK":
            info[f"check_{p[1]}"] = p[2]
    for (variant, s), v in bench.items():
        rows.append({"variant": variant, "stage": s, "min_cycles": int(np.min(v)), "mean_cycles": int(np.mean(v))})
    return info, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--platform", default="esp32")
    ap.add_argument("--n", type=int, default=None, help="window length (default: from the log)")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    info, df = parse(Path(a.log))
    n = a.n or int(info.get("N", 8192))
    f = float(info.get("f_cpu_hz", 240e6))
    df["ms"] = df["min_cycles"] / f * 1e3
    print(info)
    print(df.to_string(index=False))
    out = Path(a.log).with_suffix(".csv")
    df.to_csv(out, index=False)
    print(f"-> {out}")
    if a.calibrate:
        pairs = []
        for _, r in df.iterrows():
            if r["stage"] in STAGE_OPS and r["variant"] in ("optimized", "legacy"):
                v = "legacy" if r["variant"] == "legacy" else "opt"
                if v == "opt" and r["stage"] == "magnitude":
                    continue                     # fused into the FFT stage in fdcore
                ops = STAGE_OPS[r["stage"]](n, v)
                if v == "opt" and r["stage"] == "fft":
                    ops = ops + ops_magnitude(n, v)
                if sum(ops.values()) and r["min_cycles"] > 0:
                    pairs.append((Counter(ops), float(r["min_cycles"])))
        if len(pairs) < 3:
            print("not enough stages to calibrate")
            return
        cal = calibrate(pairs)
        print(f"calibrated cycles for {a.platform} (MAPE {100 * cal['mape']:.1f} %):")
        print("    cycles: {" + ", ".join(f"{k}: {v:.2f}" for k, v in cal["cycles"].items()) + "}")
        print("    calibrated: true")


if __name__ == "__main__":
    main()
