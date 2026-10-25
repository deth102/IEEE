"""Build and run the host golden test of fdcore for an exported model.

usage: python -m tools.build_host <model-tag> [--no-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
FW = CODE / "firmware"


def build_and_run(tag: str, run: bool = True) -> dict | None:
    gcc = shutil.which("gcc")
    if gcc is None:
        raise RuntimeError("gcc not found on PATH")
    model_dir = FW / "models" / tag
    out = FW / "host_test" / "build" / tag
    out.mkdir(parents=True, exist_ok=True)
    exe = out / ("test_golden.exe" if sys.platform == "win32" else "test_golden")
    srcs = sorted(str(p) for p in (FW / "lib" / "fdcore" / "src").glob("*.c"))
    cmd = [gcc, "-O2", "-std=c11", "-Wall", "-Wextra", "-Wno-unused-parameter",
           f"-I{FW / 'lib' / 'fdcore' / 'src'}", f"-I{model_dir}",
           *srcs, str(FW / "host_test" / "test_golden.c"), "-o", str(exe), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"host build failed:\n{r.stderr}")
    if r.stderr.strip():
        print(r.stderr, file=sys.stderr)
    if not run:
        return None
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"host test failed:\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build_and_run(a.tag, not a.no_run), indent=2))
