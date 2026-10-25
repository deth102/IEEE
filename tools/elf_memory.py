"""Read static memory usage and symbol sizes from an ELF with the GNU binutils of a toolchain."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

PIO_PACKAGES = Path.home() / ".platformio" / "packages"


def find_tool(prefix: str, tool: str) -> str | None:
    exe = f"{prefix}-{tool}"
    hit = shutil.which(exe)
    if hit:
        return hit
    for p in PIO_PACKAGES.glob("toolchain-*/bin"):
        for cand in (p / f"{exe}.exe", p / exe):
            if cand.exists():
                return str(cand)
    return None


def section_sizes(elf: Path, prefix: str) -> dict[str, int]:
    size = find_tool(prefix, "size")
    if size is None:
        raise FileNotFoundError(f"{prefix}-size not found")
    out = subprocess.run([size, "-A", str(elf)], capture_output=True, text=True, check=True).stdout
    res = {}
    for line in out.splitlines():
        m = re.match(r"^(\.\S+)\s+(\d+)\s+(\d+)", line)
        if m:
            res[m.group(1)] = int(m.group(2))
    return res


def symbol_sizes(elf: Path, prefix: str, pattern: str) -> dict[str, int]:
    nm = find_tool(prefix, "nm")
    out = subprocess.run([nm, "-S", "-C", "--size-sort", str(elf)], capture_output=True, text=True, check=True).stdout
    rx = re.compile(pattern)
    res = {}
    for line in out.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4 and rx.search(parts[3]):
            res[parts[3]] = int(parts[1], 16)
    return res


def summarize_sections(sec: dict[str, int], family: str) -> dict[str, int]:
    """Flash / static RAM totals for the ESP32 (Xtensa) or ARM (GNU ld) section naming."""
    if family == "xtensa":
        ram = sum(v for k, v in sec.items() if k.startswith((".dram0.data", ".dram0.bss", ".iram0.text", ".iram0.vectors")))
        dram = sum(v for k, v in sec.items() if k.startswith((".dram0.data", ".dram0.bss")))
        flash = sum(v for k, v in sec.items() if k.startswith((".flash.text", ".flash.rodata", ".flash.appdesc",
                                                                 ".iram0.text", ".iram0.vectors", ".dram0.data")))
        return {"static_dram": dram, "iram_dram": ram, "flash": flash}
    # ARM (GNU ld). Some cores keep one section per function (-ffunction-sections without
    # merging, e.g. Arduino-mbed RP2040), so sum by prefix. Reserved .heap/.stack areas are excluded.
    def total(prefixes):
        return sum(v for k, v in sec.items() if k.startswith(prefixes))
    data = total((".data", ".tdata", ".ram_vector", ".ramfunc"))
    bss = total((".bss", ".tbss", ".noinit", ".uninitialized"))
    code = total((".text", ".rodata", ".isr_vector", ".boot", ".ARM.exidx", ".ARM.extab", ".init", ".fini",
                  ".preinit_array", ".init_array", ".fini_array", ".binary_info", ".flash"))
    return {"static_dram": data + bss, "iram_dram": data + bss, "flash": code + data}
