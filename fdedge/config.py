"""Configuration loading and repository paths."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CODE_DIR = Path(__file__).resolve().parents[1]      # D:\IEEE\03_code
REPO_DIR = CODE_DIR.parent                           # D:\IEEE
DEFAULT_CONFIG = CODE_DIR / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access and resolved paths."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else REPO_DIR / p


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, **overrides) -> Config:
    """Load a YAML config; a variant config may name its base with ``extends: default.yaml``."""
    path = Path(path or DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "extends" in cfg:
        base = path.parent / cfg.pop("extends")
        cfg = _merge(dict(load_config(base)), cfg)
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = cfg
        keys = dotted.split("__")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return Config(cfg)


def results_dir(cfg: Config, *parts: str) -> Path:
    d = cfg.path("results_dir").joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
