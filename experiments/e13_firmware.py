"""E13 — Exact memory footprint from cross-compilation (no hardware needed).

Builds firmware/bench_mcu for several MCUs (PlatformIO downloads missing
toolchains on first use) and the ESP32 edge node, then reads the ELF:
  * PlatformIO RAM / flash summary (whole image incl. framework)
  * fdcore share: symbols of the library (code + constant tables + model),
    static work buffer, excluding the benchmark test vectors
Also builds the ESP32 benchmark for several window lengths.
"""
import re
import subprocess
from pathlib import Path

import pandas as pd

from fdedge.config import CODE_DIR
from fdedge.experiment import Run
from tools.elf_memory import find_tool, section_sizes, summarize_sections
from tools.select_model import select

run = Run("e13_firmware", __doc__, extra_args=[(("--envs",), dict(default=None, help="comma separated env list"))])
cfg, ds = run.cfg, run.dataset
PIO = str(Path.home() / ".platformio" / "penv" / "Scripts" / "pio.exe")
FW = CODE_DIR / "firmware"
ENVS = {"esp32dev": "xtensa-esp32-elf", "esp32s3": "xtensa-esp32s3-elf", "nucleo_f446re": "arm-none-eabi",
        "nucleo_h743zi": "arm-none-eabi", "pico": "arm-none-eabi"}
if run.args.envs:
    ENVS = {k: v for k, v in ENVS.items() if k in run.args.envs.split(",")}
elif run.quick:
    ENVS = {"esp32dev": ENVS["esp32dev"]}
SFX = "_quick" if run.quick else ""
N0 = cfg.signal["window"]

RX_RAM = re.compile(r"RAM:\s+\[.*?\]\s+[\d.]+% \(used (\d+) bytes from (\d+) bytes\)")
RX_FLASH = re.compile(r"Flash:\s+\[.*?\]\s+[\d.]+% \(used (\d+) bytes from (\d+) bytes\)")


def build(project: Path, env: str) -> dict:
    r = subprocess.run([PIO, "run", "-e", env], cwd=project, capture_output=True, text=True, timeout=3600)
    out = r.stdout + r.stderr
    res = {"ok": r.returncode == 0}
    m1, m2 = RX_RAM.search(out), RX_FLASH.search(out)
    if m1:
        res.update(ram_used=int(m1.group(1)), ram_total=int(m1.group(2)))
    if m2:
        res.update(flash_used=int(m2.group(1)), flash_total=int(m2.group(2)))
    if not res["ok"]:
        res["error"] = "\n".join(l for l in out.splitlines() if "rror" in l)[-600:]
    return res


def elf_totals(elf: Path, prefix: str) -> dict:
    """Flash / static RAM read from the ELF sections (independent of the PlatformIO summary)."""
    if not elf.exists():
        return {}
    t = summarize_sections(section_sizes(elf, prefix), "xtensa" if prefix.startswith("xtensa") else "arm")
    return {"elf_flash": t["flash"], "elf_static_ram": t["static_dram"]}


def fd_symbols(elf: Path, prefix: str) -> dict:
    nm = find_tool(prefix, "nm")
    if nm is None or not elf.exists():
        return {}
    out = subprocess.run([nm, "-S", "-C", str(elf)], capture_output=True, text=True).stdout
    code = const = ram = tv = 0
    for line in out.splitlines():
        p = line.split(maxsplit=3)
        if len(p) != 4:
            continue
        size, typ, name = int(p[1], 16), p[2], p[3]
        if name.startswith("fd_tv_"):
            tv += size
        elif name.startswith(("fd_", "fd_cfft", "legacy_", "softmax_inplace", "argmax_u8")) or "fd_" in name.split("(")[0]:
            if typ in "tTwW":
                code += size
            elif typ in "rRdD":
                const += size
            elif typ in "bB":
                ram += size
        elif name in ("work",) or name.startswith("work"):
            ram += size
    return {"fd_code": code, "fd_const": const, "fd_static_ram": ram, "testvec": tv}


rows = []
select(f"{ds}_n{N0}{SFX}")
for env, prefix in ENVS.items():
    print(f"building bench_mcu [{env}] ...", flush=True)
    b = build(FW / "bench_mcu", env)
    elf = FW / "bench_mcu" / ".pio" / "build" / env / "firmware.elf"
    sym = {**elf_totals(elf, prefix), **fd_symbols(elf, prefix)} if b["ok"] else {}
    rows.append({"project": "bench_mcu", "env": env, "N": N0, **b, **sym})
    print(rows[-1], flush=True)
print("building edge_node_esp32 ...", flush=True)
b = build(FW / "edge_node_esp32", "esp32dev")
elf = FW / "edge_node_esp32" / ".pio" / "build" / "esp32dev" / "firmware.elf"
rows.append({"project": "edge_node_esp32", "env": "esp32dev", "N": N0, **b,
             **elf_totals(elf, "xtensa-esp32-elf"), **fd_symbols(elf, "xtensa-esp32-elf")})
print(rows[-1], flush=True)
if not run.quick and "esp32dev" in ENVS:
    for N in (2048, 4096, 16384):
        try:
            select(f"{ds}_n{N}")
        except FileNotFoundError:
            continue
        b = build(FW / "bench_mcu", "esp32dev")
        elf = FW / "bench_mcu" / ".pio" / "build" / "esp32dev" / "firmware.elf"
        rows.append({"project": "bench_mcu", "env": "esp32dev", "N": N, **b,
                     **elf_totals(elf, "xtensa-esp32-elf"), **fd_symbols(elf, "xtensa-esp32-elf")})
        print(rows[-1], flush=True)
    select(f"{ds}_n{N0}")
if run.quick:
    select(f"{ds}_n{N0}")                                  # leave the deployed model active
df = pd.DataFrame(rows)
run.table(df.drop(columns=[c for c in ["error"] if c in df]), "memory", float_fmt="%.0f")
if "error" in df:
    run.json(df[["env", "error"]].dropna().to_dict("records"), "build_errors")
print(df.to_string(index=False))
run.done()
