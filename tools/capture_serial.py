"""Reset a board through the USB-serial control lines and log its output until a marker line.

usage: python -m tools.capture_serial --port COM3 --out results/hw/bench_esp32.log [--until INFO,done] [--timeout 120]
Requires pyserial (bundled with PlatformIO: ~/.platformio/penv/Scripts/python.exe).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def capture(port: str, out: Path, until: str = "INFO,done", timeout: float = 120.0, baud: int = 115200,
            reset: bool = True, send: list[str] | None = None) -> list[str]:
    import serial
    lines: list[str] = []
    with serial.Serial(port, baud, timeout=1) as s:
        if reset:                                   # EN low -> high (ESP32 DevKit auto-reset circuit)
            s.dtr = False
            s.rts = True
            time.sleep(0.2)
            s.rts = False
        t0 = time.time()
        sent = False
        while time.time() - t0 < timeout:
            raw = s.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").rstrip()
            lines.append(line)
            print(line, flush=True)
            if send and not sent and line.startswith("READY"):
                for cmd in send:
                    s.write((cmd + "\n").encode())
                sent = True
            if until and line.startswith(until):
                break
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--until", default="INFO,done")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--send", nargs="*", default=None, help="commands sent after the READY line")
    a = ap.parse_args()
    capture(a.port, Path(a.out), a.until, a.timeout, send=a.send)
