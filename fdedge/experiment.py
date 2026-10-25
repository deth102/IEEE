"""Shared scaffolding for the experiment scripts (CLI, output folders, tables)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, results_dir
from .plotting import setup


class Run:
    def __init__(self, name: str, description: str = "", extra_args=None):
        ap = argparse.ArgumentParser(description=description)
        ap.add_argument("--config", default=None)
        ap.add_argument("--quick", action="store_true", help="fewer folds/seeds for a smoke test")
        ap.add_argument("--dataset", default="kaist", choices=["kaist", "cwru"])
        ap.add_argument("--plot-only", action="store_true", help="redraw figures from saved tables")
        for args, kw in (extra_args or []):
            ap.add_argument(*args, **kw)
        self.args = ap.parse_args()
        self.cfg = load_config(self.args.config)
        self.name = name
        self.quick = self.args.quick
        self.dataset = self.args.dataset
        self.plot_only = self.args.plot_only
        tag = name if self.dataset == "kaist" else f"{name}_{self.dataset}"
        if self.cfg.get("tag_suffix"):                   # variant configs write their own result files
            tag = f"{tag}_{self.cfg['tag_suffix']}"
        self.tag = tag
        sub = ("quick",) if self.quick else ()           # smoke tests never overwrite the full results
        self.fig_dir = results_dir(self.cfg, *sub, "figures")
        self.tab_dir = results_dir(self.cfg, *sub, "tables")
        self.log_dir = results_dir(self.cfg, *sub, "logs")
        self.t0 = time.time()
        setup()
        print(f"[{tag}] start (quick={self.quick})")

    @property
    def folds(self) -> list[int]:
        k = self.cfg.protocol["n_folds"]
        return list(range(2 if self.quick else k))

    @property
    def seeds(self) -> list[int]:
        return [0, 1] if self.quick else [0, 1, 2, 3, 4]

    def table(self, df: pd.DataFrame, name: str, latex: bool = True, float_fmt="%.3f"):
        path = self.tab_dir / f"{self.tag}_{name}.csv"
        df.to_csv(path, index=False)
        if latex:
            with open(self.tab_dir / f"{self.tag}_{name}.tex", "w", encoding="utf-8") as f:
                f.write(df.to_latex(index=False, float_format=lambda v: float_fmt % v, escape=True))
        print(f"[{self.tag}] table -> {path.name}")
        return path

    def load_table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.tab_dir / f"{self.tag}_{name}.csv")

    def json(self, obj, name: str):
        path = self.log_dir / f"{self.tag}_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_np_default)
        return path

    def fig_name(self, name: str) -> str:
        return f"{self.tag}_{name}"

    def done(self):
        print(f"[{self.tag}] done in {time.time() - self.t0:.1f} s")


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"
