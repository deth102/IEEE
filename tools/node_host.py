"""Host side of firmware/node_esp32: the gateway that uploads models and streams signals over USB serial.

    from tools.node_host import Node
    with Node("COM3") as node:
        node.upload_model(rf, sm, gate, scale)
        rows = node.batch(windows_i16, metas)            # one dict per window (features with flags & F_FEATURES)
"""
from __future__ import annotations

import struct
import time

import numpy as np

from fdedge.gate import ProtoGate  # noqa: F401  (re-exported for the experiments)

F_RESET, F_SUPPORT, F_FEATURES = 1, 2, 4
W_COLS = ["mode", "rec", "win", "label", "flags", "rf", "sm", "s", "g", "alarm", "us", "c_feat", "c_cls", "c_gate", "c_env"]


def parse_w(line: str) -> dict:
    p = line.split(",")
    d = {"mode": p[1]}
    for k, v in zip(W_COLS[1:], p[2:16]):
        d[k] = float(v) if k in ("s", "g") else int(v)
    if len(p) > 16:
        d["feat"] = np.array([float(v) for v in p[16:]], dtype=np.float32)
    return d



def forest_blob(rf) -> bytes:
    trees = rf.flat_trees()
    roots, nodes = [], []
    for t in trees:
        base = len(nodes)
        roots.append(base)
        for i, f in enumerate(t["feature"]):
            if f >= 0:
                nodes.append((int(f), base + int(t["right"][i]), float(t["threshold"][i])))
            else:
                nodes.append((-int(t["leaf"][i]) - 1, -1, 0.0))
    b = struct.pack("<HH", len(roots), len(nodes)) + struct.pack(f"<{len(roots)}H", *roots)
    b += b"".join(struct.pack("<hhf", f, r, thr) for f, r, thr in nodes)
    return b


class Node:
    def __init__(self, port: str, baud: int = 921_600, log=None, reset: bool = True):
        import serial
        self.s = serial.Serial(port, baud, timeout=0.05)
        if hasattr(self.s, "set_buffer_size"):           # Windows: the default 4-KiB driver buffer drops bytes
            self.s.set_buffer_size(rx_size=1 << 20, tx_size=1 << 20)
        self.log = log                       # open text file: every line from the node is appended
        self.buf = b""
        self.hello = None
        if reset:
            self.s.dtr = False
            self.s.rts = True
            time.sleep(0.15)
            self.s.rts = False
            self.hello = self.expect("HELLO", timeout=8)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.s.close()

    # ---------------------------------------------------------------- framing
    def send(self, cmd: str, payload: bytes = b""):
        self.s.write(b"\xa5" + cmd.encode() + struct.pack("<I", len(payload)) + payload +
                     bytes([sum(payload) & 0xFF]))

    def readline(self, timeout: float = 10.0) -> str:
        t0 = time.time()
        while b"\n" not in self.buf:
            chunk = self.s.read(max(1, self.s.in_waiting))   # returns as soon as bytes are there
            if chunk:
                self.buf += chunk
                t0 = time.time()
            elif time.time() - t0 > timeout:
                raise TimeoutError(f"no line from the node within {timeout} s")
        line, self.buf = self.buf.split(b"\n", 1)
        line = line.decode("utf-8", "replace").rstrip("\r")
        if self.log:
            self.log.write(line + "\n")
        return line

    def expect(self, prefix: str, timeout: float = 10.0, on_other=None) -> str:
        while True:
            line = self.readline(timeout)
            if line.startswith(prefix):
                return line
            if prefix == "HELLO" and prefix in line:      # the boot ROM talks at 115200 before the first line
                return line[line.index(prefix):]
            if line.startswith("ERR"):
                raise RuntimeError(line)
            if on_other:
                on_other(line)

    # ---------------------------------------------------------------- commands
    def ping(self):
        self.send("P")
        return self.expect("OK,P")

    def heap(self):
        self.send("H")
        return self.expect("OK,H")

    def upload_model(self, rf, sm, gate: ProtoGate, scale: float, tau: float, w: int):
        d, C = len(sm.mu), sm.W.shape[0]
        idx = sm.features if sm.features is not None else list(range(d))
        b = struct.pack("<HHf", d, C, float(scale)) + forest_blob(rf)
        f32 = lambda a: np.asarray(a, dtype="<f4").tobytes()  # noqa: E731
        b += bytes(int(i) for i in idx) + f32(sm.mu) + f32(1.0 / sm.sd) + f32(sm.W) + f32(sm.b)
        b += f32(gate.Pw) + f32(gate.Sw) + f32(gate.mu) + f32(gate.n_c) + f32([gate.n])
        b += struct.pack("<H", len(gate.protos)) + f32(gate.protos) + bytes(int(c) for c in gate.pcls)
        b += struct.pack("<ff", gate.ridge, float(tau)) + struct.pack("<H", w)
        self.send("M", b)
        self.expect("OK,F")
        return self.expect("OK,M")

    def upload_forest(self, rf):
        self.send("F", forest_blob(rf))
        return self.expect("OK,F")

    def set_tau(self, tau: float):
        self.send("T", struct.pack("<f", float(tau)))
        return self.expect("OK,T")

    def clear_support(self):
        self.send("C")
        return self.expect("OK,C")

    def adapt(self, lr=0.1, epochs=200, l2=1e-3, weight=50.0, reref=True):
        self.send("A", struct.pack("<fHffB", lr, epochs, l2, weight, 1 if reref else 0))
        a = self.expect("ADAPT", timeout=60).split(",")
        smw = np.array([float(v) for v in self.expect("SMW").split(",")[1:]])
        gp = np.array([float(v) for v in self.expect("GATEP").split(",")[1:]])
        return {"n": int(a[1]), "cycles_update": int(a[2]), "cycles_reref": int(a[3]), "ok": int(a[4]), "prototypes": int(a[5]),
                "smw": smw, "P": gp}

    @staticmethod
    def _meta(rec, win, label, flags) -> bytes:
        return struct.pack("<HHbB", int(rec), int(win), int(label), int(flags))

    def batch(self, windows: np.ndarray, metas, in_flight: int = 1, progress=None) -> list[dict]:
        """windows: (n, N) int16; metas: (rec, win, label, flags) per window. Returns the W lines as dicts."""
        out, sent = [], 0
        n = len(windows)
        while len(out) < n:
            while sent < n and sent - len(out) < in_flight:
                self.send("B", self._meta(*metas[sent]) + np.asarray(windows[sent], dtype="<i2").tobytes())
                sent += 1
            out.append(parse_w(self.expect("W,B", timeout=20)))
            if progress and len(out) % 200 == 0:
                progress(len(out), n)
        return out

    def _stream_chunks(self, samples: np.ndarray, chunk: int, max_inflight: int, before_chunk=None):
        inflight = 0
        rows = []

        def other(line):
            if line.startswith("W,"):
                rows.append(parse_w(line))

        for k in range(0, len(samples), chunk):
            if before_chunk:
                before_chunk(k)
            while inflight >= max_inflight:
                self.expect("K", timeout=20, on_other=other)
                inflight -= 1
            self.send("X", np.asarray(samples[k:k + chunk], dtype="<i2").tobytes())
            inflight += 1
        while inflight:
            self.expect("K", timeout=20, on_other=other)
            inflight -= 1
        return rows, other

    def stream(self, windows: np.ndarray, metas, chunk: int = 2048, max_inflight: int = 4):
        """Real-time replay: contiguous windows, the node's timer moves them into the acquisition buffer at fs."""
        self.send("S")
        self.expect("OK,S")
        N = windows.shape[1]
        flat = np.asarray(windows, dtype=np.int16).ravel()

        def meta_at(k):
            if k % N == 0:
                self.send("Q", self._meta(*metas[k // N]))

        rows, other = self._stream_chunks(flat, chunk, max_inflight, before_chunk=meta_at)
        self.send("E")
        end = self.expect("SEND,S", timeout=30, on_other=other)
        p = end.split(",")
        return rows, {"windows": int(p[2]), "overruns": int(p[3]), "underruns": int(p[4])}

    def calibrate(self):
        self.send("L")
        lines = []
        while True:
            line = self.readline(20)
            lines.append(line)
            if line.startswith("ORDER"):
                return lines

    def analog(self, samples: np.ndarray, rec, label, flags, peak: int, span: float = 80.0,
               chunk: int = 2048, max_inflight: int = 4):
        """Play one recording through the DAC -> ADC jumper; returns the W lines of the captured windows."""
        self.send("G", self._meta(rec, 0, label, flags) + struct.pack("<hf", int(peak), float(span)))
        self.expect("OK,G")
        rows, other = self._stream_chunks(np.asarray(samples, dtype=np.int16), chunk, max_inflight)
        self.send("E")
        end = self.expect("SEND,A", timeout=30, on_other=other)
        p = end.split(",")
        return rows, {"windows": int(p[2]), "overruns": int(p[3]), "underruns": int(p[4])}
