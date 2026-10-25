"""Convert recordings into .fdv files for the edge node's microSD card.

.fdv layout (little endian): "FDV1" | uint32 fs | float32 scale | uint32 n_samples | int16[n]
All files share ONE scale (the dataset-wide int16 full scale of the emulated ADC),
equal to FD_SCALE of the exported model.

usage: python -m tools.make_sd_binary [--dataset kaist] [--out results/sd_card] [--test-block-only]
Copy the resulting "fd" folder to the root of the SD card.
"""
from __future__ import annotations

import argparse
import struct

import numpy as np

from fdedge.config import load_config
from fdedge.data import block_bounds, list_recordings, load_signal
from fdedge.export_c import global_scale, to_int16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kaist")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test-block-only", action="store_true",
                    help="write only the blocked test block of fold 0 (unseen by the fold-0 model)")
    a = ap.parse_args()
    cfg = load_config()
    out = (cfg.path("results_dir") / "sd_card" / "fd") if a.out is None else __import__("pathlib").Path(a.out) / "fd"
    out.mkdir(parents=True, exist_ok=True)
    recs = list_recordings(cfg, a.dataset)
    sigs = {r.name: load_signal(r, cfg) for r in recs}
    scale = global_scale(max(float(np.max(np.abs(x))) for x in sigs.values()))
    for r in recs:
        x = np.asarray(sigs[r.name], dtype=np.float32)
        if a.test_block_only:
            b = block_bounds(len(x), cfg.protocol["n_folds"])
            x = x[b[0]:b[1]]
        q = to_int16(x, scale)
        with open(out / f"{r.name}.fdv", "wb") as f:
            f.write(b"FDV1" + struct.pack("<IfI", int(r.fs), scale, len(q)))
            f.write(q.astype("<i2").tobytes())
    print(f"{len(recs)} files -> {out}  (scale = {scale:.9g}, must equal FD_SCALE of the model)")


if __name__ == "__main__":
    main()
