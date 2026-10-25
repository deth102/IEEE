"""Copy firmware/models/<tag>/ into firmware/models/active/ (the include path of the firmware projects).

usage: python -m tools.select_model <tag>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "firmware" / "models"


def select(tag: str) -> Path:
    src = MODELS / tag
    if not (src / "fd_model_config.h").exists():
        raise FileNotFoundError(f"model '{tag}' not exported (run experiments.e12_export)")
    dst = MODELS / "active"
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.iterdir():
        f.unlink()
    for f in src.iterdir():
        shutil.copy2(f, dst / f.name)
    (dst / "TAG").write_text(tag, encoding="utf-8")
    return dst


if __name__ == "__main__":
    print(select(sys.argv[1]))
