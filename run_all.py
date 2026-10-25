"""Run every experiment of the study in order.

usage (from D:\\IEEE\\03_code):
  python run_all.py                 full study (~1 h CPU + CNN on GPU)
  python run_all.py --quick         smoke test (2 folds / seeds)
  python run_all.py --only e01,e04  selected experiments
  python run_all.py --skip e09,e13  skip CNN training / firmware builds
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = [
    ("e01", "experiments.e01_protocol", ["kaist", "cwru"]),
    ("e02", "experiments.e02_hyperparams", ["kaist"]),
    ("e03", "experiments.e03_features", ["kaist", "cwru"]),
    ("e04", "experiments.e04_crossload", ["kaist", "cwru"]),
    ("e05", "experiments.e05_adaptation", ["kaist", "cwru"]),
    ("e06", "experiments.e06_window_sweep", ["kaist"]),
    ("e07", "experiments.e07_voting_noise", ["kaist"]),
    ("e08", "experiments.e08_compression", ["kaist"]),
    ("e09", "experiments.e09_cnn", ["kaist"]),
    ("e11", "experiments.e11_resources", ["kaist"]),
    ("e12", "experiments.e12_export", ["kaist"]),
    ("e13", "experiments.e13_firmware", ["kaist"]),
    ("sum", "tools.make_summary", [None]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    skip = set(filter(None, a.skip.split(",")))
    failed = []
    for key, module, datasets in STEPS:
        if (only and key not in only) or key in skip:
            continue
        for ds in datasets:
            cmd = [sys.executable, "-m", module]
            if ds:
                cmd += ["--dataset", ds]
            if a.quick and key != "sum":
                cmd.append("--quick")
            print(f"\n=== {key} {ds or ''} ===", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd)
            print(f"=== {key} {ds or ''}: {'OK' if r.returncode == 0 else 'FAILED'} ({time.time() - t0:.0f} s)")
            if r.returncode:
                failed.append(f"{key}:{ds}")
    print("\nFAILED: " + ", ".join(failed) if failed else "\nall steps OK")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
