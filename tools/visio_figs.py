"""Draw the KESTREL manuscript figures in Microsoft Visio (Arial 8 pt, final print size).

usage: python -m tools.visio_figs [--export] [kestrel protocol staircase ... staircase_target crossload_target ...]

Visio (Windows) is driven through COM; an open Visio window is reused, otherwise Visio is started.
Data come from results/tables (run python -m tools.fig_data first) and results/hw/hil_clean.log. Every figure
has its own sheet of 05_manuscript_JESTIE/figures/visio/figures.vsdx, drawn as a group named after its PDF
(fig01_kestrel, ...) with an invisible frame of the printed size; the authors' photos live on the sheet "Photos".
For LaTeX, every group is copied onto a temporary page of exactly that size and exported to figures/<name>.pdf,
so the manuscript includes the PDFs unscaled and every label is printed at 8 pt. The *_target variants hold the
values of the target draft and go to target/figures/, together with copies of the shared figures.

After editing a figure by hand in Visio:  python -m tools.visio_figs --export [names]   (PDFs only)
"""
from __future__ import annotations

import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageFont

from fdedge.config import load_config
from fdedge.plotting import INK, INK2, MUTED, GRID, SERIES

# no grey in the figures (user, 2026-10-26): secondary text and frames in black, grid in a light blue tint
INK2, MUTED, GRID = INK, INK, "#d6e3f1"

FS = 8.0                        # font size of every label [pt]
PT = 1 / 72                     # inch per point
COL_W, PAGE_W = 3.46, 7.13      # IEEEtran column and text width [in] (249.5 pt, 514 pt)
TICK = 2.5 * PT
MARKERS = ["o", "s", "^", "D", "v", "p"]
DASHES = ["-", "--", "-.", ":", "--", "-."]
_LINE_PATTERN = {"-": "1", "--": "2", ":": "3", "-.": "4"}
_FONT_FILES = {"Times New Roman": ("times.ttf", "timesbd.ttf"), "Arial": ("arial.ttf", "arialbd.ttf")}
_FONT_NAME = "Arial"
_FONT = _FONT_B = None


def use_font(name: str):
    """Font of the figure being drawn (labels and the text metrics used for layout)."""
    global _FONT_NAME, _FONT, _FONT_B
    reg, bold = _FONT_FILES[name]
    _FONT_NAME = name
    _FONT = ImageFont.truetype(rf"C:\Windows\Fonts\{reg}", 100)
    _FONT_B = ImageFont.truetype(rf"C:\Windows\Fonts\{bold}", 100)


use_font("Arial")
RF_C, SM_C = "#7ea6d8", "#eea283"           # muted fills of the two heads
RF_L, SM_L = "#3f74b5", "#d5683f"           # line versions
P_BLUE, P_LAV, P_PINK, P_GREEN, P_SAND, P_GREY = "#dde6f0", "#e4e2f3", "#f9dcdc", "#d9eed6", "#f6ecd0", "#f6dde8"   # P_GREY: pastel rose
TAG_BLUE, TAG_ORANGE, TAG_GREEN, TAG_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
LIGHT = "#eef3f7"

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "05_manuscript_JESTIE" / "figures"
TARGET_FIG_DIR = FIG_DIR      # since 2026-10-26 the target draft lives in 05_manuscript_JESTIE itself


def rgb(hexcol: str) -> str:
    h = hexcol.lstrip("#")
    return "RGB({},{},{})".format(*(int(h[i:i + 2], 16) for i in (0, 2, 4)))


def shade(hexcol: str, f: float) -> str:
    h = hexcol.lstrip("#")
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(int(h[i:i + 2], 16) * f))) for i in (0, 2, 4))


def tint(hexcol: str, a: float) -> str:
    """Blend a colour towards white by a (0..1)."""
    h = hexcol.lstrip("#")
    return "#%02x%02x%02x" % tuple(round(int(h[i:i + 2], 16) * (1 - a) + 255 * a) for i in (0, 2, 4))


def viridis(v: float, lo=0.3, hi=1.0) -> str:
    from matplotlib import cm
    r, g, b, _ = cm.viridis(min(max((v - lo) / (hi - lo), 0.0), 1.0))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


# --------------------------------------------------------------------------- rich text
def rich(s: str):
    """Mini markup: _{sub}, ^{sup}, *italic*, ~{bold}. Returns plain text and runs (start, end, kind)."""
    out, runs, i, italic_from = [], [], 0, None
    while i < len(s):
        if s[i] in "_^~" and i + 1 < len(s) and s[i + 1] == "{":
            j = s.index("}", i)
            start = len("".join(out))
            out.append(s[i + 2:j])
            runs.append((start, start + j - i - 2, {"_": "sub", "^": "sup", "~": "bold"}[s[i]]))
            i = j + 1
        elif s[i] == "*":
            pos = len("".join(out))
            if italic_from is None:
                italic_from = pos
            else:
                runs.append((italic_from, pos, "italic"))
                italic_from = None
            i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out), runs


def text_width(s: str, size: float = FS, bold: bool = False) -> float:
    """Width in inches of one line of (marked-up) text; sub/superscripts counted at 70 %."""
    plain, runs = rich(s)
    f = _FONT_B if bold else _FONT
    w = f.getlength(plain)
    for a, b, kind in runs:
        if kind in ("sub", "sup"):
            w -= 0.3 * f.getlength(plain[a:b])
        elif kind == "bold" and not bold:
            w += _FONT_B.getlength(plain[a:b]) - _FONT.getlength(plain[a:b])
    return w / 100 * size * PT


def block_width(s: str) -> float:
    return max(text_width(l) for l in s.split("\n"))


# --------------------------------------------------------------------------- canvas
VSDX = FIG_DIR / "visio" / "figures.vsdx"
SHEET = "Figures"
SIZE = {                      # printed size of every figure [in]
    "fig01_kestrel": (PAGE_W, 2.45), "fig01_kestrel_target": (PAGE_W, 2.45), "fig02_protocol": (COL_W, 2.12), "fig03_staircase": (PAGE_W, 2.35), "fig03_staircase_target": (PAGE_W, 2.35),
    "fig04_ablation": (PAGE_W, 2.02), "fig05_gate": (PAGE_W, 2.40), "fig05_gate_target": (PAGE_W, 2.40),
    "fig06_crossload3d": (PAGE_W, 2.80),
    "fig06_crossload3d_target": (PAGE_W, 2.80),
    "fig07_cost": (COL_W, 2.40), "fig08_hil_spectra": (COL_W, 1.88), "fig09_overview": (PAGE_W, 5.97),
    "fig09_overview_target": (PAGE_W, 5.97), "fig10_tradeoff": (COL_W, 1.95),
    "fig06_heatmap": (COL_W, 1.62), "fig06_heatmap_target": (COL_W, 1.62), "fig11_bench": (COL_W, 1.15)}
ROWS = [["fig01_kestrel"], ["fig02_protocol", "fig07_cost"], ["fig03_staircase"], ["fig04_ablation"],
        ["fig05_gate"], ["fig06_crossload3d"], ["fig08_hil_spectra"], ["fig09_overview"], ["fig10_tradeoff"], ["fig03_staircase_target"]]
FIG_NO = {n: f"Fig. {int(n[3:5])}" for n in SIZE}
MARGIN, LABEL_H, ROW_GAP, COL_GAP = 0.4, 0.3, 0.25, 0.12


def sheet_layout():
    """Size of the common page and the lower-left corner of every figure on it (rows as in the paper)."""
    width = PAGE_W + 2 * MARGIN
    height = 2 * MARGIN + sum(LABEL_H + max(SIZE[n][1] for n in row) for row in ROWS) + ROW_GAP * (len(ROWS) - 1)
    pos, y = {}, height - MARGIN
    for row in ROWS:
        rh = max(SIZE[n][1] for n in row)
        y -= LABEL_H + rh
        x = MARGIN
        for n in row:
            pos[n] = (x, y + rh - SIZE[n][1])
            x += SIZE[n][0] + COL_GAP
        y -= ROW_GAP
    return width, height, pos


def find_shape(page, name):
    try:
        return page.Shapes.ItemU(name)
    except Exception:
        return None


PHOTOS = "Photos"


def get_page(doc, name):
    return next((doc.Pages.Item(i) for i in range(1, doc.Pages.Count + 1) if doc.Pages.Item(i).Name == name), None)


def open_pages(doc):
    """One sheet per figure. Migrates an old single 'Figures' sheet: the authors' photos go to the 'Photos' sheet."""
    for i in range(doc.Pages.Count, 0, -1):
        if doc.Pages.Item(i).Name == "_export":
            doc.Pages.Item(i).Delete(0)
    photos = get_page(doc, PHOTOS)
    old = get_page(doc, SHEET)
    if old is not None:
        if photos is None:
            photos = doc.Pages.Add()
            photos.Name = PHOTOS
        x = 1.0
        for i in range(old.Shapes.Count, 0, -1):
            shp = old.Shapes.Item(i)
            if shp.Name.startswith("photo_") and find_shape(photos, shp.Name) is None:
                w = shp.CellsU("Width").ResultIU
                cp = photos.Drop(shp, x + w / 2, 3.0)
                cp.Name = cp.NameU = shp.Name
                x += w + 0.5
        old.Delete(0)
    return photos


def fig_page(doc, name):
    """Sheet of one figure, emptied for a redraw; page size = printed size."""
    page = get_page(doc, name)
    if page is None:
        page = doc.Pages.Add()
        page.Name = name
    for i in range(page.Shapes.Count, 0, -1):
        page.Shapes.Item(i).Delete()
    w, h = SIZE[name]
    page.AutoSize = False                             # keep the printed size; no automatic page growth
    page.PageSheet.CellsU("PageWidth").FormulaU = f"{w} in"
    page.PageSheet.CellsU("PageHeight").FormulaU = f"{h} in"
    return page


def order_pages(doc):
    names = sorted(SIZE) + [PHOTOS]
    for k, n in enumerate([n for n in names if get_page(doc, n) is not None], start=1):
        get_page(doc, n).Index = k


def open_drawing(app):
    """The drawing that holds the figures (created on first use)."""
    VSDX.parent.mkdir(parents=True, exist_ok=True)
    for d in app.Documents:
        if Path(d.FullName).resolve() == VSDX.resolve():
            return d
    return app.Documents.Open(str(VSDX)) if VSDX.exists() else app.Documents.Add("")


class Canvas:
    """One figure, drawn in local coordinates [in] at its place on the common page."""

    def __init__(self, page, name: str, font: str = "Arial"):
        use_font(font)
        self.page, self.name = page, name
        self.w, self.h = SIZE[name]
        self.ox, self.oy = 0.0, 0.0                  # each figure has its own sheet
        self.shapes = []

    def _add(self, shp):
        self.shapes.append(shp)
        return shp

    def _rect(self, x1, y1, x2, y2):
        return self._add(self.page.DrawRectangle(x1 + self.ox, y1 + self.oy, x2 + self.ox, y2 + self.oy))

    # -- primitives
    def _stroke(self, shp, color=INK, lw=0.75, dash="-", arrow=False):
        c = shp.CellsU
        c("LineColor").FormulaU = rgb(color)
        c("LineWeight").FormulaU = f"{lw} pt"
        c("LinePattern").FormulaU = _LINE_PATTERN[dash]
        if arrow:
            c("EndArrow").FormulaU = "13"
            c("EndArrowSize").FormulaU = str(int(arrow))      # True -> 1; 2, 3 for bold arrows

    def _fill(self, shp, fill):
        if fill is None:
            shp.CellsU("FillPattern").FormulaU = "0"
        else:
            shp.CellsU("FillForegnd").FormulaU = rgb(fill)
            shp.CellsU("FillPattern").FormulaU = "1"

    def line(self, x1, y1, x2, y2, color=INK, lw=0.75, dash="-", arrow=False):
        shp = self._add(self.page.DrawLine(x1 + self.ox, y1 + self.oy, x2 + self.ox, y2 + self.oy))
        self._stroke(shp, color, lw, dash, arrow)
        return shp

    def polyline(self, pts, color=INK, lw=0.75, dash="-", arrow=False, fill=None):
        flat = tuple(float(v) for x, y in pts for v in (x + self.ox, y + self.oy))
        shp = self._add(self.page.DrawPolyline(flat, 0))
        self._stroke(shp, color, lw, dash, arrow)
        self._fill(shp, fill)
        return shp

    def polygon(self, pts, fill, edge=None, lw=0.4):
        shp = self.polyline(list(pts) + [pts[0]], color=edge or fill, lw=lw, fill=fill)
        if edge is None:
            shp.CellsU("LinePattern").FormulaU = "0"
        return shp

    def rect(self, x, y, w, h, fill=None, edge=None, lw=0.75, dash="-", rounding=0.0, grad=None, to=None,
             shadow=False):
        """grad: 'v' (fill at the top, `to` at the bottom), 'h' (fill left), 'r' (radial)."""
        shp = self._rect(x, y, x + w, y + h)
        self._fill(shp, fill)
        if grad and fill is not None:
            shp.CellsU("FillBkgnd").FormulaU = rgb(to or tint(fill, 0.6))
            shp.CellsU("FillPattern").FormulaU = {"v": "28", "h": "25", "r": "40"}[grad]
        if edge is None:
            shp.CellsU("LinePattern").FormulaU = "0"
        else:
            self._stroke(shp, edge, lw, dash)
        if rounding:
            shp.CellsU("Rounding").FormulaU = f"{rounding} in"
        if shadow:
            self.shadow(shp)
        return shp

    @staticmethod
    def shadow(shp):
        c = shp.CellsU
        c("ShdwPattern").FormulaU = "1"
        c("ShdwForegnd").FormulaU = rgb("#9fa4aa")
        c("ShapeShdwType").FormulaU = "1"
        c("ShdwForegndTrans").FormulaU = "70%"
        c("ShapeShdwOffsetX").FormulaU = "0.014 in"
        c("ShapeShdwOffsetY").FormulaU = "-0.014 in"
        c("ShapeShdwBlur").FormulaU = "1 pt"

    def circle(self, x, y, r, face="#ffffff", edge=INK, lw=0.6):
        shp = self._add(self.page.DrawOval(x - r + self.ox, y - r + self.oy, x + r + self.ox, y + r + self.oy))
        self._fill(shp, face)
        if edge is None:
            shp.CellsU("LinePattern").FormulaU = "0"
        else:
            self._stroke(shp, edge, lw)
        return shp

    def marker(self, x, y, kind="o", ms=4.5, face="#ffffff", edge=INK, lw=0.6):
        r = ms * PT / 2
        if kind == "o":
            return self.circle(x, y, r, face, edge, lw)
        if kind == "s":
            pts = [(x - .85 * r, y - .85 * r), (x + .85 * r, y - .85 * r), (x + .85 * r, y + .85 * r), (x - .85 * r, y + .85 * r)]
        elif kind == "^":
            pts = [(x - r, y - .8 * r), (x + r, y - .8 * r), (x, y + r)]
        elif kind == "v":
            pts = [(x - r, y + .8 * r), (x + r, y + .8 * r), (x, y - r)]
        elif kind == "D":
            pts = [(x, y - 1.1 * r), (x + .9 * r, y), (x, y + 1.1 * r), (x - .9 * r, y)]
        else:  # pentagon
            pts = [(x + r * math.cos(math.pi / 2 + k * 2 * math.pi / 5), y + r * math.sin(math.pi / 2 + k * 2 * math.pi / 5))
                   for k in range(5)]
        return self.polyline(pts + [pts[0]], color=edge, lw=lw, fill=face)

    def text(self, x, y, s, ha="center", va="center", rot=0, color=INK, bold=False, size=FS):  # noqa: C901
        """Place text; for rot=90 the anchors ha/va refer to the page frame of the rotated text."""
        plain, runs = rich(s)
        lines = s.split("\n")
        w = max(text_width(l, size, bold) for l in lines) + 0.08
        h = len(lines) * size * 1.2 * PT
        if rot == 0:
            x0 = {"left": x, "center": x - w / 2, "right": x - w}[ha]
            y0 = {"bottom": y, "center": y - h / 2, "top": y - h}[va]
            shp = self._rect(x0, y0, x0 + w, y0 + h)
            align = {"left": 0, "center": 1, "right": 2}[ha]
        else:
            cx = {"left": x + h / 2, "center": x, "right": x - h / 2}[ha]
            cy = {"bottom": y + w / 2, "center": y, "top": y - w / 2}[va]
            shp = self._rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            shp.CellsU("Angle").FormulaU = f"{rot} deg"
            align = {"bottom": 0, "center": 1, "top": 2}[va]
        shp.Text = plain
        c = shp.CellsU
        c("LinePattern").FormulaU = "0"
        c("FillPattern").FormulaU = "0"
        for m in ("LeftMargin", "RightMargin", "TopMargin", "BottomMargin"):
            c(m).FormulaU = "0 pt"
        c("VerticalAlign").FormulaU = "1"
        c("Para.HorzAlign").FormulaU = str(align)
        c("Char.Font").FormulaU = f'FONT("{_FONT_NAME}")'
        c("Char.Size").FormulaU = f"{size} pt"
        c("Char.Color").FormulaU = rgb(color)
        c("Char.Style").FormulaU = "1" if bold else "0"
        for a, b, kind in runs:
            ch = shp.Characters
            ch.Begin, ch.End = a, b
            if kind == "sub":
                ch.CharProps(4, 2)
            elif kind == "sup":
                ch.CharProps(4, 1)
            elif kind == "italic":
                ch.CharProps(2, 2 | (1 if bold else 0))
            elif kind == "bold":
                ch.CharProps(2, 1)
        return shp

    def text_angle(self, x, y, s, angle, anchor="center", color=INK, size=FS):
        """Text rotated by `angle` degrees; anchor left/right/center refers to the start/end/middle of the line."""
        w = text_width(s, size) + 0.06
        h = size * 1.2 * PT
        c, sn = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        k = {"left": 0.5, "center": 0.0, "right": -0.5}[anchor]
        cx, cy = x + k * w * c, y + k * w * sn
        shp = self.text(cx, cy, s, color=color, size=size)
        shp.CellsU("Angle").FormulaU = f"{angle} deg"
        return shp

    def box(self, x, y, w, h, label, fill="#ffffff", edge="#333333", lw=0.75, dash="-", rounding=0.04, shadow=False):
        self.rect(x, y, w, h, fill=fill, edge=edge, lw=lw, dash=dash, rounding=rounding)
        self.text(x + w / 2, y + h / 2, label)

    def tag(self, x, y, label, color=None):
        """KESTREL block letter in a small white disc centred at (x, y)."""
        self.circle(x, y, 0.07, face="#ffffff", edge="#333333", lw=0.6)
        self.text(x, y, label, color=INK, bold=True)

    def finish(self):
        """Group the figure with an invisible frame of its printed size and label it on the page."""
        frame = self._rect(0, 0, self.w, self.h)
        frame.CellsU("LinePattern").FormulaU = "0"
        frame.CellsU("FillPattern").FormulaU = "0"
        frame.Name = "frame_" + self.name
        sel = self.page.CreateSelection(0)
        for shp in self.shapes:
            sel.Select(shp, 2)
        grp = sel.Group()
        grp.Name = self.name
        self.shapes = []


def export_pdfs(doc, names):
    """Copy each figure group onto a temporary page of its frame size and export that page to PDF."""
    for name in names:
        page = get_page(doc, name)
        grp = find_shape(page, name) if page is not None else None
        if grp is None:
            print(f"  {name}: not on the page, skipped")
            continue
        frame = next(s for s in grp.Shapes if s.Name.startswith("frame_"))
        w, h = frame.CellsU("Width").ResultIU, frame.CellsU("Height").ResultIU
        tmp = doc.Pages.Add()
        tmp.Name = "_export"
        tmp.AutoSize = False
        tmp.PageSheet.CellsU("PageWidth").ResultIU = w
        tmp.PageSheet.CellsU("PageHeight").ResultIU = h
        cp = tmp.Drop(grp, w / 2, h / 2)
        fr = next(s for s in cp.Shapes if s.Name.startswith("frame_"))
        x0, y0 = fr.XYToPage(0.0, 0.0)
        for cell, d in (("PinX", x0), ("PinY", y0)):
            c = cp.CellsU(cell)
            c.ResultIU = c.ResultIU - d
        cp.Ungroup()        # Visio renders a group twice in PDF exports (text and fills doubled)
        pdf = (TARGET_FIG_DIR if name.endswith("_target") else FIG_DIR) / f"{name}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        try:
            pdf.unlink(missing_ok=True)
        except PermissionError:
            tmp.Delete(0)
            print(f"  {pdf.name}: open in another program, not exported")
            continue
        # without structure tags: Visio otherwise strokes a frame around the page
        doc.ExportAsFixedFormat(1, str(pdf), 1, 1, tmp.Index, tmp.Index, False, True, True, False)
        tmp.Delete(0)
        print(f"  {pdf.name}  ({w:.2f} x {h:.2f} in)")
        # the target draft shares every figure without a target variant
        if TARGET_FIG_DIR != FIG_DIR and TARGET_FIG_DIR.exists() and not name.endswith("_target") \
                and f"{name}_target" not in SIZE:
            shutil.copy2(pdf, TARGET_FIG_DIR / pdf.name)


def legend(cv, x, y, items, ncol=1, colw=None, rowh=0.15):
    """items: dicts with label, color and either patch=True or dash/marker. (x, y): top-left corner."""
    for k, it in enumerate(items):
        cx, cy = x + (k % ncol) * (colw or 0), y - (k // ncol) * rowh - rowh / 2
        if it.get("patch"):
            cv.rect(cx + 0.06, cy - 0.045, 0.14, 0.09, fill=it["color"], edge=it.get("edge", "#333333"), lw=0.5)
        elif it.get("circle"):
            cv.circle(cx + 0.13, cy, 0.05, face=it.get("face", it["color"]), edge="#333333", lw=0.6)
        else:
            cv.line(cx, cy, cx + 0.26, cy, color=it["color"], lw=1.0, dash=it.get("dash", "-"))
            if it.get("marker"):
                cv.marker(cx + 0.13, cy, it["marker"], face=it.get("face", it["color"]),
                          edge=it.get("edge", it["color"]))
        cv.text(cx + 0.32, cy, it["label"], ha="left")


def legend_width(items, ncol=1):
    w = [0.32 + text_width(it["label"]) + 0.14 for it in items]
    return max(w) * ncol


# --------------------------------------------------------------------------- axes
class Axes:
    def __init__(self, cv, x0, y0, w, h, xlim, ylim, xlog=None, ylog=None):
        self.cv, self.x0, self.y0, self.w, self.h = cv, x0, y0, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog
        self.ylab_w = 0.0
        self.xlab_h = FS * 1.2 * PT

    @staticmethod
    def _map(v, lim, log, a0, length):
        a, b = lim
        if log:
            v, a, b = (math.log(t, log) for t in (v, a, b))
        return a0 + (v - a) / (b - a) * length

    def X(self, v):
        return self._map(v, self.xlim, self.xlog, self.x0, self.w)

    def Y(self, v):
        return self._map(v, self.ylim, self.ylog, self.y0, self.h)

    def grid(self, yvals=(), xvals=()):
        for v in yvals:
            self.cv.line(self.x0, self.Y(v), self.x0 + self.w, self.Y(v), color=GRID, lw=0.4)
        for v in xvals:
            self.cv.line(self.X(v), self.y0, self.X(v), self.y0 + self.h, color=GRID, lw=0.4)

    def spines(self):
        self.cv.line(self.x0, self.y0, self.x0 + self.w, self.y0, color=INK2, lw=0.6)
        self.cv.line(self.x0, self.y0, self.x0, self.y0 + self.h, color=INK2, lw=0.6)

    def xticks(self, vals, labels=None, rot=0, at=None):
        labels = labels or [f"{v:g}" for v in vals]
        for v, lab in zip(vals, labels):
            x = self.X(v) if at is None else at(v)
            self.cv.line(x, self.y0, x, self.y0 - TICK, color=MUTED, lw=0.6)
            if lab != "":
                self.cv.text(x, self.y0 - TICK - 1.5 * PT, lab, va="top", rot=rot)
        if rot:
            self.xlab_h = max(text_width(l) for l in labels)
        else:
            self.xlab_h = max(len(l.split("\n")) for l in labels) * FS * 1.2 * PT

    def yticks(self, vals, labels=None):
        labels = labels or [f"{v:g}" for v in vals]
        for v, lab in zip(vals, labels):
            y = self.Y(v)
            self.cv.line(self.x0, y, self.x0 - TICK, y, color=MUTED, lw=0.6)
            if lab != "":
                self.cv.text(self.x0 - TICK - 2 * PT, y, lab, ha="right")
        self.ylab_w = max([text_width(l) for l in labels if l != ""], default=0.0)

    def xlabel(self, s):
        self.cv.text(self.x0 + self.w / 2, self.y0 - TICK - 3 * PT - self.xlab_h, s, va="top")

    def ylabel(self, s):
        self.cv.text(self.x0 - TICK - 5 * PT - self.ylab_w, self.y0 + self.h / 2, s, ha="right", rot=90)

    def title(self, s):
        self.cv.text(self.x0 + self.w / 2, self.y0 + self.h + 2 * PT, s, va="bottom")

    def hline(self, y, color=MUTED, lw=0.75, dash="-", x1=None, x2=None):
        a = self.x0 if x1 is None else self.X(x1)
        b = self.x0 + self.w if x2 is None else self.X(x2)
        self.cv.line(a, self.Y(y), b, self.Y(y), color=color, lw=lw, dash=dash)

    def plot(self, xs, ys, color, lw=1.0, dash="-", marker=None, face=None, edge=None, ms=4.5):
        pts = [(self.X(x), self.Y(y)) for x, y in zip(xs, ys)]
        if lw > 0 and len(pts) > 1:
            self.cv.polyline(pts, color=color, lw=lw, dash=dash)
        if marker:
            for p in pts:
                self.cv.marker(*p, marker, ms=ms, face=face or color, edge=edge or color)

    def errorbars(self, xs, ys, es, color=INK2, lw=0.6, cap=2.5 * PT, horizontal_offset=0.0):
        for x, y, e in zip(xs, ys, es):
            if not np.isfinite(e) or e <= 0:
                continue
            px = self.X(x) + horizontal_offset
            lo, hi = self.Y(max(y - e, self.ylim[0])), self.Y(min(y + e, self.ylim[1]))
            self.cv.line(px, lo, px, hi, color=color, lw=lw)
            for yy, clipped in ((lo, y - e < self.ylim[0]), (hi, y + e > self.ylim[1])):
                if not clipped:
                    self.cv.line(px - cap / 2, yy, px + cap / 2, yy, color=color, lw=lw)

    def bar(self, x, width, top, color, bottom=None):
        bottom = self.ylim[0] if bottom is None else bottom
        y1, y2 = sorted((self.Y(bottom), self.Y(top)))
        x1 = self.X(x - width / 2)
        self.cv.rect(x1, y1, self.X(x + width / 2) - x1, max(y2 - y1, 0.002), fill=color, edge="#333333", lw=0.4)

    def frame(self):
        self.cv.line(self.x0, self.y0 + self.h, self.x0 + self.w, self.y0 + self.h, color=MUTED, lw=0.6)
        self.cv.line(self.x0 + self.w, self.y0, self.x0 + self.w, self.y0 + self.h, color=MUTED, lw=0.6)


# --------------------------------------------------------------------------- data
def tables(cfg):
    d = cfg.path("results_dir") / "tables"
    names = ["e03_features_summary", "e03_features_permutation_importance", "fig_staircase", "fig_gate_trace",
             "fig_gate_meta", "fig_gate_roc", "fig_cost", "fig_crossload3d"]
    out = {n: pd.read_csv(d / f"{n}.csv") for n in names}
    for n in ("fig_gate_trace_target", "fig_gate_meta_target", "fig_gate_roc_target"):   # target draft only
        if (d / f"{n}.csv").exists():
            out[n] = pd.read_csv(d / f"{n}.csv")
    return out


FEATURE_LABELS = {
    "mean_t": "time mean", "rms_t": "RMS", "max_t": "max", "min_t": "min", "ptp_t": "peak–peak",
    "crest_t": "crest factor", "mean_f": "spec. mean", "rmsf_f": "*f*_{RMS}", "var_f": "spec. var.",
    "skew_f": "skewness", "kurt_f": "kurtosis", "fc_f": "*f*_{c}", "fstd_f": "*σ*_{f}", "sc_f": "centroid",
    "ss_f": "spread", "ent_f": "entropy", "a1x_o": "*A*_{1X}", "a2x_o": "*A*_{2X}", "a1xr_o": "*A*_{1X}/RMS"}   # one line: sub + sup break when rotated


def arrow_path(cv, pts, color=INK, lw=0.9, dash="-", size=1):
    return cv.polyline(pts, color=color, lw=lw, dash=dash, arrow=size)


# --------------------------------------------------------------------------- small icons
def icon_wave(cv, x, y, w, h, color="#1f4e8c"):
    xs = np.linspace(0, 1, 60)
    ys = np.sin(2 * np.pi * 3.2 * xs) * (0.55 + 0.45 * np.sin(2 * np.pi * 0.7 * xs + 0.6))
    cv.polyline([(x + a * w, y + h / 2 + b * h / 2) for a, b in zip(xs, ys)], color=color, lw=0.6)


def icon_spec(cv, x, y, w, h, color="#1f4e8c"):
    hs = [0.25, 0.9, 0.35, 0.6, 0.2, 0.45, 0.15]
    for k, f in enumerate(hs):
        xx = x + (k + 0.5) * w / len(hs)
        cv.line(xx, y, xx, y + f * h, color=color, lw=1.1)
    cv.line(x, y, x + w, y, color=color, lw=0.5)


def icon_warn(cv, x, y, s=0.16, color="#f3d27a"):
    cv.polygon([(x - s / 2, y - s * 0.42), (x + s / 2, y - s * 0.42), (x, y + s * 0.5)], color, edge="#333333", lw=0.5)
    cv.text(x, y - s * 0.08, "!", color=INK, bold=True, size=7)


def zone(cv, ax, x0, x1, fill, label=None, color=INK2, y_label=None):
    if fill:
        cv.rect(ax.X(x0), ax.y0, ax.X(x1) - ax.X(x0), ax.h, fill=fill)
    if label:
        cv.text(ax.X((x0 + x1) / 2), ax.Y(y_label) if y_label is not None else ax.y0 + ax.h - 0.10, label,
                color=color, bold=True)


# --------------------------------------------------------------------------- Fig. 1: KESTREL
GATE_T2 = "distance to the training data\n+ head disagreement"      # what the gate measures, in words
GATE_PROTO = "distance to class prototypes\n+ head disagreement"


def fig_kestrel(page, cfg, T, name="fig01_kestrel", gate=GATE_T2):
    cv = Canvas(page, name)
    W, H = cv.w, cv.h
    pad = 0.08
    for y, h, lab in ((1.56, 0.86, "Per-window path on the MCU (every 0.32 s)"), (0.05, 0.85, "Lifecycle (L)")):
        cv.rect(0.03, y, W - 0.06, h, fill="#eef4fb", edge="#5b7fa6", lw=0.6, dash="--", rounding=0.08)
        cv.text(0.13, y + h - 0.06, lab, ha="left", va="top", color=INK, bold=True)
    # ---- row A
    rowA = [("Vibration signal\n(ADC + DMA)", P_BLUE, "wave"),
            ("Spectrum\n(in-place FFT)", P_LAV, "spec"),
            ("Spectral statistics\n+ kinematic orders", P_GREEN, None)]
    heads = [("Random forest", P_PINK), ("Softmax head", P_PINK)]
    vote = ("Diagnosis", P_BLUE)
    icon_w = 0.26
    wA = [block_width(s) + 2 * pad + (icon_w if ic else 0) for s, _, ic in rowA]
    wA[2] += 0.12                                   # two corner discs on this block
    wh = max(block_width(s) for s, _ in heads) + 2 * pad
    wv = block_width(vote[0]) + 2 * pad
    bh, hh = 0.50, 0.22
    yA = 1.60
    gap = (W - 0.26 - sum(wA) - wh - wv) / 4
    xs, x = [], 0.13
    for (s, fill, ic), w in zip(rowA, wA):
        cv.rect(x, yA, w, bh, fill=fill, edge="#333333", lw=0.75, rounding=0.05)
        tx = x + (icon_w if ic else 0)
        cv.text(tx + (w - (icon_w if ic else 0)) / 2, yA + bh / 2, s)
        if ic == "wave":
            icon_wave(cv, x + 0.06, yA + 0.13, icon_w - 0.08, bh - 0.26, RF_L)
        elif ic == "spec":
            icon_spec(cv, x + 0.06, yA + 0.12, icon_w - 0.08, bh - 0.24, RF_L)
        xs.append((x, w))
        x += w + gap
    xh = x
    y_rf, y_sm = yA + bh - hh, yA
    for (s, fill), yy in zip(heads, (y_rf, y_sm)):
        cv.rect(xh, yy, wh, hh, fill=fill, edge="#333333", lw=0.75, rounding=0.05)
        cv.text(xh + wh / 2, yy + hh / 2, s)
    xv = xh + wh + gap
    cv.rect(xv, yA + 0.07, wv, bh - 0.14, fill=vote[1], edge="#333333", lw=0.75, rounding=0.05)
    cv.text(xv + wv / 2, yA + bh / 2, vote[0])
    for (xa, wa), (xb, _) in zip(xs, xs[1:]):
        cv.line(xa + wa, yA + bh / 2, xb, yA + bh / 2, arrow=True, lw=0.9, color=INK)
    kx, kw = xs[2]
    mid = kx + kw + gap / 2
    arrow_path(cv, [(kx + kw, yA + bh / 2), (mid, yA + bh / 2), (mid, y_rf + hh / 2), (xh, y_rf + hh / 2)], color=INK, lw=0.9)
    arrow_path(cv, [(mid, yA + bh / 2), (mid, y_sm + hh / 2), (xh, y_sm + hh / 2)], color=INK, lw=0.9)
    vmid = xh + wh + gap / 2
    arrow_path(cv, [(xh + wh, y_rf + hh / 2), (vmid, y_rf + hh / 2), (vmid, yA + bh / 2), (xv, yA + bh / 2)], color=INK, lw=0.9)
    cv.line(xh + wh, y_sm + hh / 2, vmid, y_sm + hh / 2, lw=0.9, color=INK)
    cv.line(vmid, y_sm + hh / 2, vmid, yA + bh / 2, lw=0.9, color=INK)
    # ---- row C: lifecycle (computed first, it fixes the route of the retune arrow)
    states = [("Monitor", P_BLUE), ("Alert:\nask for labels", P_SAND),
              ("Collect a few\nlabeled windows", P_BLUE), ("Retune on chip (R),\nre-reference gate", P_GREEN),
              ("Verify", P_BLUE)]
    ws = [block_width(s) + 2 * pad for s, _ in states]
    sh, ys = 0.36, 0.30
    g3 = (W - 0.26 - sum(ws)) / 4
    xs3, x = [], 0.13
    for (s, fill), w in zip(states, ws):
        cv.rect(x, ys, w, sh, fill=fill, edge="#333333", lw=0.75, rounding=0.09)
        cv.text(x + w / 2, ys + sh / 2, s)
        xs3.append((x, w))
        x += w + g3
    for (xa, wa), (xb, _) in zip(xs3, xs3[1:]):
        cv.line(xa + wa, ys + sh / 2, xb, ys + sh / 2, arrow=True, lw=0.9, color=INK)
    (m0, mw), (vx, vw) = xs3[0], xs3[4]
    yr = ys - 0.17
    arrow_path(cv, [(vx + vw / 2, ys), (vx + vw / 2, yr), (m0 + mw / 2, yr), (m0 + mw / 2, ys)], color=INK, lw=0.9)
    lab = cv.text((vx + vw / 2 + m0 + mw / 2) / 2, yr, " resume monitoring ", color=INK2)
    lab.CellsU("FillForegnd").FormulaU = rgb("#eef4fb")
    lab.CellsU("FillPattern").FormulaU = "1"
    # ---- row B: self-check gate, placed so that the retune arrow passes to its right
    gtxt = "~{Self-check gate}\n" + gate
    gw, gh = block_width(gtxt) + 2 * pad, 0.48
    ax_, aw_ = xs3[3]
    rx = min(ax_ + aw_, xh + wh) - 0.16
    gx, gy = rx - 0.16 - gw, 0.98
    cv.rect(gx, gy, gw, gh, fill=P_SAND, edge="#333333", lw=0.75, rounding=0.05)
    cv.text(gx + gw / 2, gy + gh / 2, gtxt)
    cv.line(kx + kw / 2, yA, kx + kw / 2, gy + gh, arrow=True, lw=0.9, color=INK)
    sx = xh + 0.16
    cv.line(sx, y_sm, sx, gy + gh, arrow=True, lw=0.9, dash="--", color=INK)
    a1x, a1w = xs3[1]
    acx = a1x + a1w / 2
    if acx < gx:
        arrow_path(cv, [(gx, gy + gh / 2), (acx, gy + gh / 2), (acx, ys + sh)], color=INK, lw=0.9)
        cv.text((gx + acx) / 2, gy + gh / 2 + 1.5 * PT, "alarm", va="bottom", color=INK)
    else:
        cv.line(acx, gy, acx, ys + sh, arrow=True, lw=0.9, color=INK)
        cv.text(acx + 0.05, (gy + ys + sh) / 2, "alarm", ha="left", color=INK)
    arrow_path(cv, [(rx, ys + sh), (rx, y_sm)], color=INK, dash="--", lw=0.9)
    cv.text(rx + 0.05, gy + gh / 2, "retuned head", ha="left", color=INK)
    # ---- block letters
    for (x0, w0), lab_ in zip(xs, ["", "T", "K"]):
        if lab_:
            cv.tag(x0 + 0.01, yA + bh - 0.01, lab_, TAG_BLUE)
    cv.tag(xh + 0.01, y_rf + hh - 0.01, "E", TAG_BLUE)       # one disc per letter: E marks the heads only
    cv.tag(xh + 0.01, y_sm + 0.01, "R", TAG_GREEN)
    cv.tag(gx + gw - 0.01, gy + gh - 0.01, "S", TAG_ORANGE)   # right corner: the feature arrow meets the left
    cv.tag(m0 + mw - 0.01, ys + sh - 0.01, "L", TAG_GREY)
    cv.finish()


# --------------------------------------------------------------------------- Fig. 2: protocols
def fig_protocol(page, cfg, T):
    cv = Canvas(page, "fig02_protocol")
    W, H = cv.w, cv.h
    C_TR, C_VA, C_TE, C_SU, C_NO = "#c9dbf0", "#f3e3a6", "#f6cdbd", "#cfe8cb", "#ffffff"
    lab_w = max(text_width(s) for s in ["(a) Blocked, fold *b*", "(b) Unseen load", "(c) Severity A", "(d) Gate"]) + 0.10
    x0, bw, bh = lab_w, (W - lab_w - 0.06) / 5, 0.17
    cv.rect(0.0, 0.04, W, H - 0.40, fill="#eef4fb", edge="#5b7fa6", lw=0.6, dash="--", rounding=0.06)
    legend(cv, 0.05, H - 0.01, [dict(label="train", color=C_TR, patch=True), dict(label="validation", color=C_VA, patch=True),
                                dict(label="test", color=C_TE, patch=True)], ncol=3, colw=1.08)
    legend(cv, 0.05, H - 0.16, [dict(label="support", color=C_SU, patch=True),
                                dict(label="not used", color=C_NO, patch=True)], ncol=3, colw=1.08)

    def seg(x, y, w, c, txt=None):
        cv.rect(x, y, w, bh, fill=c, rounding=0.02, edge="#333333", lw=0.5)
        if txt:
            cv.text(x + w / 2, y + bh / 2, txt, color=INK)

    def row(y, label, cols):
        cv.text(x0 - 0.07, y + bh / 2, label, ha="right")
        for i, c in enumerate(cols):
            seg(x0 + i * bw + 0.01, y, bw - 0.02, c)

    y = H - 0.66
    row(y, "(a) Blocked, fold *b*", [C_TR, C_TR, C_TE, C_VA, C_TR])
    cv.text(x0 + 2.5 * bw, y + bh + 1 * PT, "*b*", va="bottom", color=INK2)
    cv.text(x0 + 3.5 * bw, y + bh + 1 * PT, "*b*+1", va="bottom", color=INK2)
    y -= 0.31
    row(y, "(b) Unseen load", [C_SU, C_TE, C_TE, C_TE, C_TE])
    y -= 0.31
    row(y, "(c) Severity A", [C_SU, C_NO, C_NO, C_NO, C_NO])
    y -= 0.21
    row(y, "Severity B", [C_NO, C_TE, C_TE, C_TE, C_TE])
    y -= 0.31
    cv.text(x0 - 0.07, y + bh / 2, "(d) Gate", ha="right")
    seg(x0 + 0.01, y, 2 * bw - 0.02, C_TR, "source load")
    seg(x0 + 2 * bw + 0.01, y, 3 * bw - 0.02, C_TE, "new condition")
    yb = y - 0.11
    cv.line(x0, yb, W - 0.06, yb, color=INK2, lw=0.75, arrow=True)
    cv.text((x0 + W) / 2, yb - 1 * PT, "time within each recording", va="top", color=INK2)
    cv.finish()


# --------------------------------------------------------------------------- Fig. 3: staircase
def fig_staircase(page, cfg, T, name="fig03_staircase", targets=None, auroc="98%"):
    """targets: {(row, column): value} replacing the data, for the target draft only."""
    cv = Canvas(page, name)
    W, H = cv.w, cv.h
    s = T["fig_staircase"].copy()
    for (k_, col_), v_ in (targets or {}).items():
        s.loc[k_, col_] = v_
    n = len(s)
    ax = Axes(cv, 0.52, 0.50, W - 0.58, H - 0.50 - 0.25, (-0.55, n - 0.45), (0.0, 1.42))
    zone(cv, ax, -0.55, 1.5, None, "Leak-free accuracy", INK, 1.33)
    zone(cv, ax, 1.5, 4.5, "#fbf3df", "Load change: detect (S) and retune (R)", INK, 1.33)
    zone(cv, ax, 4.5, n - 0.45, None, "On the ESP32", INK, 1.33)
    for xb in (1.5, 4.5):                                  # the three groups of tests, clearly apart
        cv.line(ax.X(xb), ax.y0, ax.X(xb), ax.y0 + ax.h, color=MUTED, lw=0.6)
    ax.grid(yvals=[0.2, 0.4, 0.6, 0.8, 1.0])
    bw = 0.38
    for i, (col, c) in enumerate((("rf", RF_C), ("sm", SM_C))):
        for k, v in enumerate(s[col]):
            x = k + (i - 0.5) * bw
            ax.bar(x, bw * 0.88, v, c)
            cv.text(ax.X(x), ax.Y(v) + 1.5 * PT, f"{100 * v:.1f}", va="bottom")
    # the gate fires on the unseen load, the retunable head recovers it; gains above the values, outside the bars
    cv.text(ax.X(2.0), ax.Y(0.90), f"S alarm\n(AUROC {auroc})", color=INK)
    for i, (col, c) in enumerate((("rf", RF_C), ("sm", SM_C))):
        gain = (round(100 * s[col][3], 1) - round(100 * s[col][2], 1)) / 100
        cv.text(ax.X(3 + (i - 0.5) * bw), ax.Y(s[col][3]) + 2.5 * PT + FS * 1.2 * PT, f"+{100 * gain:.1f}",
                va="bottom", color=INK, bold=True)
    ax.spines()
    ax.frame()
    ax.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    ax.xticks(range(n), ["Random\nsplit", "Time-\nblocked", "Unseen\nload", "Unseen load\n+ R (*q* = 5)",
                         "Other\nseverity + R", "ESP32\nreplay", "DAC–ADC\n+ on-chip R"])
    ax.ylabel("Macro-F1 (%)")
    legend(cv, 0.55, H - 0.01, [dict(label="RF head", color=RF_C, patch=True), dict(label="Softmax head", color=SM_C, patch=True)],
           ncol=2, colw=0.85)
    cv.finish()


# --------------------------------------------------------------------------- Fig. 4: ablation
def fig_ablation(page, cfg, T):
    cv = Canvas(page, "fig04_ablation")
    W, H = cv.w, cv.h
    s, imp = T["e03_features_summary"], T["e03_features_permutation_importance"]
    sets = ["T6", "S10", "O3", "F16", "F13nr", "F16nr", "F19"]
    set_labels = ["T6", "S10", "O3", "*F*_{16}", "F13nr", "F16nr", "*F*_{19}"]
    models = ["RF", "Softmax"]
    HL = "#fbf1cc"
    y0, top = 0.76, H - 0.27
    legend(cv, 0.62, H - 0.01, [dict(label="RF", color=RF_C, patch=True), dict(label="Softmax", color=SM_C, patch=True),
                                dict(label="contains shaft-order features", color=HL, patch=True)],
           ncol=3, colw=0.86)
    ax = Axes(cv, 0.62, y0, 1.95, top - y0, (-0.6, len(sets) - 0.4), (0.5, 1.12))
    for k in (2, 5, 6):
        cv.rect(ax.X(k - 0.48), ax.y0, ax.X(k + 0.48) - ax.X(k - 0.48), ax.h, fill=HL)
    ax.grid(yvals=[0.6, 0.7, 0.8, 0.9, 1.0])
    for i, m in enumerate(models):
        g = s[s.model == m].set_index("features").reindex(sets)
        xs = np.arange(len(sets)) + (i - 0.5) * 0.38
        for x, v in zip(xs, g.f1_mean):
            ax.bar(x, 0.35, v, (RF_C, SM_C)[i])
        ax.errorbars(xs, g.f1_mean, g.f1_std)
    yb = 1.03
    cv.polyline([(ax.X(3), ax.Y(0.99)), (ax.X(3), ax.Y(yb)), (ax.X(6), ax.Y(yb)), (ax.X(6), ax.Y(1.005))],
                color=INK, lw=0.7)
    cv.text(ax.X(4.5), ax.Y(yb) + 1 * PT, "+11.6 pp (RF)", va="bottom", color=INK, bold=True)
    ax.spines()
    ax.frame()
    ax.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["50", "60", "70", "80", "90", "100"])
    ax.xticks(range(len(sets)), set_labels, rot=90)
    ax.ylabel("Blocked macro-F1 (%)")
    order = imp[imp.model == "RF"].sort_values("mean", ascending=False)["feature"].tolist()
    ax = Axes(cv, 3.45, y0, W - 3.45 - 0.05, top - y0, (-0.6, len(order) - 0.4), (-0.05, 0.20))
    for k, f in enumerate(order):
        if f in ("a1x_o", "a2x_o", "a1xr_o"):
            cv.rect(ax.X(k - 0.48), ax.y0, ax.X(k + 0.48) - ax.X(k - 0.48), ax.h, fill=HL)
    ax.grid(yvals=[0.05, 0.10, 0.15, 0.20])
    ax.hline(0, color=MUTED, lw=0.6)
    for i, m in enumerate(models):
        g = imp[imp.model == m].set_index("feature").reindex(order)
        xs = np.arange(len(order)) + (i - 0.5) * 0.4
        for x, v in zip(xs, g["mean"]):
            ax.bar(x, 0.37, v, (RF_C, SM_C)[i], bottom=0.0)
        ax.errorbars(xs, g["mean"], g["std"])
    ax.spines()
    ax.frame()
    ax.yticks([-0.05, 0, 0.05, 0.10, 0.15, 0.20], ["−5", "0", "5", "10", "15", "20"])
    ax.xticks(range(len(order)), [FEATURE_LABELS[f] for f in order], rot=90)
    ax.ylabel("Macro-F1 drop (pp)\nwhen permuted")
    cv.finish()


# --------------------------------------------------------------------------- Fig. 5: gate
def fig_gate(page, cfg, T, name="fig05_gate", sfx="", title_a="(a) The lifecycle on a load change"):
    cv = Canvas(page, name)
    W, H = cv.w, cv.h
    tr, meta, rc = T[f"fig_gate_trace{sfx}"], T[f"fig_gate_meta{sfx}"].iloc[0], T[f"fig_gate_roc{sfx}"]
    y0, top = 0.47, H - 0.28
    LINE = "#1f3b5b"
    gmax = max(3.0, float(np.ceil((tr.g.max() + 0.55) * 2) / 2))   # room for the zone labels above the trace
    ax = Axes(cv, 0.52, y0, 3.55, top - y0, (0, 130), (0.0, gmax))
    ylab = gmax - 0.20
    zone(cv, ax, 0, meta.t_change, P_BLUE, "Monitor (0 N·m)", INK, ylab)
    zone(cv, ax, meta.t_change, meta.t_adapt0, P_SAND, "Alert", INK, ylab)
    zone(cv, ax, meta.t_adapt0, meta.t_adapt1, P_GREY)
    zone(cv, ax, meta.t_adapt1, 130, P_GREEN, "Verify → Monitor", INK, ylab)
    # the adaptation band is narrower than a line of text: horizontal label beside it, arrow into the band
    ya = (meta.tau_after + ylab) / 2
    cv.text(ax.X(meta.t_adapt1 + 4.0), ax.Y(ya), "Adapt (R)", ha="left", color=INK)
    cv.line(ax.X(meta.t_adapt1 + 3.6), ax.Y(ya), ax.X((meta.t_adapt0 + meta.t_adapt1) / 2), ax.Y(ya),
            color=INK, lw=0.75, arrow=True)
    ax.grid(yvals=list(np.arange(0.5, gmax, 0.5)))
    ax.hline(meta.tau, color=INK, dash="--", x1=0, x2=meta.t_adapt0)
    ax.hline(meta.tau_after, color=INK, dash="--", x1=meta.t_adapt1 + 2.0, x2=130)   # clear of the Adapt label
    cv.text(ax.X(1.5), ax.Y(meta.tau) + 1 * PT, "*τ*", ha="left", va="bottom", color=INK)
    cv.text(ax.X(129), ax.Y(meta.tau_after) + 1 * PT, "*τ*′", ha="right", va="bottom", color=INK)
    for ph in ("before", "after"):
        g = tr[tr.phase == ph]
        ax.plot(g.t, g.g, LINE, lw=1.1)
    cv.line(ax.X(meta.t_change), ax.y0, ax.X(meta.t_change), ax.Y(gmax - 0.38), color=MUTED, lw=0.75, dash="--")
    y_load = meta.tau - 0.25 if meta.tau - 0.25 > 0.6 else meta.tau + 0.35       # clear of the tau line
    cv.text(ax.X(meta.t_change) - 0.05, ax.Y(y_load), "load 0 → 2 N·m", ha="right", color=INK2)
    cv.text(ax.X(meta.t_alarm) + 0.05, ax.Y(meta.tau) - 0.03, "alarm", ha="left", va="top", color=INK)
    ax.spines()
    ax.frame()
    yt = list(np.arange(0, gmax + 1e-9, 0.5))
    ax.yticks(yt, ["0" if v == 0 else f"{v:.1f}" for v in yt])
    ax.xticks([0, 20, 40, 60, 80, 100, 120])
    ax.xlabel("Time (s), healthy machine")
    ax.ylabel("Gate statistic *g*_{t}")
    ax.title(title_a)
    # (b) square ROC in neutral colours (blue/orange mean RF/softmax elsewhere); the axes are padded so that the
    # perfect curve stays visible inside the frame, and the key sits outside the axes, to the right
    side = top - y0
    ax = Axes(cv, 0.52 + 3.55 + 0.62, y0, side, side, (-0.03, 1.0), (0.0, 1.03))
    cases = list(dict.fromkeys(rc.case))
    ax.grid(yvals=[0.25, 0.5, 0.75, 1.0], xvals=[0.25, 0.5, 0.75, 1.0])
    cv.line(ax.X(0), ax.Y(0), ax.X(1), ax.Y(1), color=MUTED, lw=0.5, dash="--")
    COLS, SHORT = ("#1a1a1a", "#b71c4a"), ("Load", "DAC–ADC")     # black, crimson (crimson = DAC-ADC as in Fig. 9)
    curves = []
    for i, case in enumerate(cases):
        for score, dash in (("maxp", ":"), ("gate", "-")):
            g = rc[(rc.case == case) & (rc.score == score)]
            lw_ = (1.4 if i == 0 else 2.0) if score == "gate" else 1.0
            curves.append((i, score, g, COLS[i], lw_, dash))
    for i, score, g, col, lw_, dash in sorted(curves, key=lambda c: (c[1] == "gate", c[0])):   # gates on top
        ax.plot(g.fpr, g.tpr, col, lw=lw_, dash=dash)
    ax.spines()
    ax.frame()
    ax.yticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax.xticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax.xlabel("False-alarm rate (%)")
    ax.ylabel("Detection rate (%)")
    cv.text(ax.x0 + ax.w / 2, top + 2 * PT, "(b) ROC", va="bottom")
    rowh, smp = 0.15, 0.20
    kx = ax.x0 + ax.w + 0.10
    ky = ax.y0 + ax.h / 2 + 3 * rowh
    for i in range(2):
        cv.text(kx, ky - (3 * i + 0.5) * rowh, SHORT[i], ha="left", color=INK)
        for j, (score, dash, lab) in enumerate((("gate", "-", "gate"), ("maxp", ":", "max *p*"))):
            yy = ky - (3 * i + j + 1.5) * rowh
            lw_ = (1.4 if i == 0 else 2.0) if score == "gate" else 1.0
            cv.line(kx, yy, kx + smp, yy, color=COLS[i], lw=lw_, dash=dash)
            cv.text(kx + smp + 0.04, yy, lab, ha="left", color=INK)
    cv.finish()


# --------------------------------------------------------------------------- Fig. 6: 3-D cross-load
def fig_crossload(page, cfg, T, name="fig06_crossload3d", targets=None):
    """3-D bars in the style of Duan et al. (MSSP 2025, Fig. 14): height-graded viridis faces, gold tops with the
    value on top, diagonal view with two back walls and a floor grid, axis titles along the axes.
    targets: {(panel, source, target): value} replacing the data, for the target draft only."""
    import itertools
    cv = Canvas(page, name)
    W, H = cv.w, cv.h
    d = T["fig_crossload3d"]
    LAB = {"0Nm": "0 N·m", "2Nm": "2 N·m", "4Nm": "4 N·m"}
    ei, ej, hz, foot = (0.36, -0.23), (0.38, 0.20), 1.04, 0.64   # projection [in per unit], bar footprint
    ZLO, ZHI = 0.0, 1.0                                            # axis range
    CLO, CHI = 0.2, 1.0                                            # colour range of the faces and the colour bar
    TOP, TOP_EDGE, EDGE = "#d9b40a", "#8c7400", "#3a3a3a"
    NS = 26                                                        # gradient slices over the full height
    lo, hi = -0.5, 2.5
    ang_i = math.degrees(math.atan2(ei[1], ei[0]))
    ang_j = math.degrees(math.atan2(ej[1], ej[0]))
    cells = {p: {(r.source, r.target): r.f1 for r in d[d.panel == p].itertuples()} for p in ("before", "after")}
    for (p, src, tgt), v in (targets or {}).items():
        cells[p][(src, tgt)] = v

    def P(ox, oy, i, j, z):
        return (ox + i * ei[0] + j * ej[0], oy + i * ei[1] + j * ej[1] + (z - ZLO) / (ZHI - ZLO) * hz)

    def hull(pts):
        pts = sorted(set(pts))
        cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
        lower, upper = [], []
        for q in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
                lower.pop()
            lower.append(q)
        for q in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
                upper.pop()
            upper.append(q)
        return lower[:-1] + upper[:-1]

    def inside(pt, poly):
        return all((b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]) >= 0
                   for a, b in zip(poly, poly[1:] + poly[:1]))

    def silhouette(i, j, v):
        return hull([P(0, 0, a, b, z) for a in (i - foot / 2, i + foot / 2) for b in (j - foot / 2, j + foot / 2)
                     for z in (ZLO, v)])

    order = sorted(itertools.product(range(3), range(3)), key=lambda q: (-(q[1] - q[0]), q[0]))   # far to near

    def hidden(train, test):
        """Number of top labels covered by nearer bars for this axis order (both panels)."""
        n = 0
        for panel in ("before", "after"):
            for k, (i, j) in enumerate(order):
                v = cells[panel][(train[i], test[j])]
                cx, cy = P(0, 0, i, j, v)
                w = text_width(f"{100 * v:.1f}") / 2
                probes = [(cx - w, cy), (cx + w, cy), (cx, cy + 0.05)]
                for (i2, j2) in order[k + 1:]:
                    sil = silhouette(i2, j2, cells[panel][(train[i2], test[j2])])
                    n += sum(inside(q, sil) for q in probes)
        return n

    loads = ["0Nm", "2Nm", "4Nm"]
    perms = [tuple(loads), tuple(reversed(loads))]        # monotonic axis orders only
    train_o, test_o = min(((a, b) for a in perms for b in perms),
                          key=lambda ab: (hidden(*ab), ab[0] != tuple(loads), ab[1] != tuple(loads)))

    def panel(ox, oy, key, title, show_z):
        # walls and floor
        for z in np.arange(ZLO, ZHI + 1e-9, 0.2):
            cv.polyline([P(ox, oy, lo, lo, z), P(ox, oy, lo, hi, z), P(ox, oy, hi, hi, z)], color=GRID, lw=0.4)
        for t in (0.5, 1.5):
            cv.line(*P(ox, oy, lo, t, ZLO), *P(ox, oy, lo, t, ZHI), color=GRID, lw=0.4)
            cv.line(*P(ox, oy, t, hi, ZLO), *P(ox, oy, t, hi, ZHI), color=GRID, lw=0.4)
            cv.line(*P(ox, oy, t, lo, ZLO), *P(ox, oy, t, hi, ZLO), color=GRID, lw=0.4)
            cv.line(*P(ox, oy, lo, t, ZLO), *P(ox, oy, hi, t, ZLO), color=GRID, lw=0.4)
        cv.polyline([P(ox, oy, lo, lo, ZHI), P(ox, oy, lo, hi, ZHI), P(ox, oy, hi, hi, ZHI)], color=MUTED, lw=0.5)
        cv.polyline([P(ox, oy, lo, lo, ZLO), P(ox, oy, hi, lo, ZLO), P(ox, oy, hi, hi, ZLO), P(ox, oy, lo, hi, ZLO),
                     P(ox, oy, lo, lo, ZLO)], color=INK2, lw=0.6)
        for a, b in ((lo, lo), (lo, hi), (hi, hi)):
            cv.line(*P(ox, oy, a, b, ZLO), *P(ox, oy, a, b, ZHI), color=MUTED if (a, b) != (lo, lo) else INK2, lw=0.5)
        # z axis on the left edge
        for z in np.arange(ZLO, ZHI + 1e-9, 0.2):
            x_, y_ = P(ox, oy, lo, lo, z)
            cv.line(x_, y_, x_ - TICK, y_, color=INK2, lw=0.6)
            cv.text(x_ - TICK - 1.5 * PT, y_, f"{100 * z:.0f}", ha="right")
        if show_z:
            x_, y_ = P(ox, oy, lo, lo, (ZLO + ZHI) / 2)
            cv.text(x_ - 0.32, y_, "Macro-F1 (%)", rot=90)
        # bars, far to near: graded front and side faces, gold top, value on top
        for (i, j) in order:
            v = cells[key][(train_o[i], test_o[j])]
            a0, a1, b0, b1 = i - foot / 2, i + foot / 2, j - foot / 2, j + foot / 2
            n = max(2, int(math.ceil((v - ZLO) / (ZHI - ZLO) * NS)))
            zs = np.linspace(ZLO, v, n + 1)
            for z0, z1 in zip(zs[:-1], zs[1:]):
                c = viridis((z0 + z1) / 2, CLO, CHI)
                z1 = min(v, z1 + 0.6 * (z1 - z0))                 # slices overlap: no anti-aliasing seams
                cv.polygon([P(ox, oy, a0, b0, z0), P(ox, oy, a1, b0, z0), P(ox, oy, a1, b0, z1), P(ox, oy, a0, b0, z1)], c)
                cv.polygon([P(ox, oy, a1, b0, z0), P(ox, oy, a1, b1, z0), P(ox, oy, a1, b1, z1), P(ox, oy, a1, b0, z1)],
                           shade(c, 0.78))
            cv.polyline([P(ox, oy, a0, b0, v), P(ox, oy, a0, b0, ZLO), P(ox, oy, a1, b0, ZLO), P(ox, oy, a1, b1, ZLO),
                         P(ox, oy, a1, b1, v)], color=EDGE, lw=0.4)
            cv.line(*P(ox, oy, a1, b0, ZLO), *P(ox, oy, a1, b0, v), color=EDGE, lw=0.4)
            cv.polygon([P(ox, oy, a0, b0, v), P(ox, oy, a1, b0, v), P(ox, oy, a1, b1, v), P(ox, oy, a0, b1, v)], TOP,
                       edge=TOP_EDGE, lw=0.5)
        for (i, j) in order:                                   # values last, so that none is hidden
            v = cells[key][(train_o[i], test_o[j])]
            cx, cy = P(ox, oy, i, j, v)
            cv.text(cx, cy, f"{100 * v:.1f}")
        # tick labels parallel to the other axis, axis titles along their axis
        ui = (ei[0] / math.hypot(*ei), ei[1] / math.hypot(*ei))
        uj = (ej[0] / math.hypot(*ej), ej[1] / math.hypot(*ej))
        for i in range(3):
            x_, y_ = P(ox, oy, i, lo, ZLO)
            cv.line(x_, y_, x_ - 0.05 * uj[0], y_ - 0.05 * uj[1], color=INK2, lw=0.6)
            cv.text_angle(x_ - 0.08 * uj[0], y_ - 0.08 * uj[1], LAB[train_o[i]], ang_j, "right")
        for j in range(3):
            x_, y_ = P(ox, oy, hi, j, ZLO)
            cv.line(x_, y_, x_ + 0.05 * ui[0], y_ + 0.05 * ui[1], color=INK2, lw=0.6)
            cv.text_angle(x_ + 0.08 * ui[0], y_ + 0.08 * ui[1], LAB[test_o[j]], ang_i, "left")
        x_, y_ = P(ox, oy, 1, lo, ZLO)
        cv.text_angle(x_ - 0.62 * uj[0], y_ - 0.62 * uj[1], "Training load", ang_i, "center")
        x_, y_ = P(ox, oy, hi, 1, ZLO)
        cv.text_angle(x_ + 0.62 * ui[0], y_ + 0.62 * ui[1], "Test load", ang_j, "center")
        tx, ty = P(ox, oy, 1, 1, ZHI)
        cv.text(tx, H - 0.10, title)

    panel(1.02, 0.93, "before", "(a) Before adaptation", True)
    panel(4.20, 0.93, "after", "(b) After on-device adaptation, *q* = 5", False)
    # colour bar
    cx, cy, ch = W - 0.40, 0.85, 1.50
    for k in range(48):
        v = CLO + (CHI - CLO) * (k + 0.5) / 48
        cv.rect(cx, cy + ch * k / 48, 0.12, ch / 48 + 0.002, fill=viridis(v, CLO, CHI))
    cv.rect(cx, cy, 0.12, ch, edge=INK2, lw=0.5)
    for v in (CLO, (CLO + CHI) / 2, CHI):
        yy = cy + ch * (v - CLO) / (CHI - CLO)
        cv.line(cx + 0.12, yy, cx + 0.12 + TICK, yy, color=INK2, lw=0.6)
        cv.text(cx + 0.12 + TICK + 1.5 * PT, yy, f"{100 * v:.0f}", ha="left")
    cv.finish()


# --------------------------------------------------------------------------- Fig. 6: cross-load heatmaps
def fig_heatmap(page, cfg, T, name="fig06_heatmap", targets=None):
    """Softmax macro-F1 for every training/test load pair before and after the on-device update: one-hue ramp
    (light = low), the diagonal (no shift) in grey. targets as in fig_crossload, for the target draft only."""
    cv = Canvas(page, name)
    d = T["fig_crossload3d"]
    cells = {p: {(r.source, r.target): r.f1 for r in d[d.panel == p].itertuples()} for p in ("before", "after")}
    for (p, src, tgt), v in (targets or {}).items():
        cells[p][(src, tgt)] = v
    loads = ["0Nm", "2Nm", "4Nm"]
    LAB = {"0Nm": "0 N·m", "2Nm": "2 N·m", "4Nm": "4 N·m"}
    CLO, CHI = 0.3, 1.0
    RAMP = ((0.0, "#f4f7fb"), (0.5, "#a9c3e3"), (1.0, "#2f63a6"))

    def color(v):
        t = min(max((v - CLO) / (CHI - CLO), 0.0), 1.0)
        for (t0, c0), (t1, c1) in zip(RAMP, RAMP[1:]):
            if t <= t1:
                return _blend(c0, c1, (t - t0) / (t1 - t0))

    def dark(c):
        r, g, b = (int(c.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45

    cs, y0 = 0.35, 0.40
    x0s = (0.58, 0.58 + 3 * cs + 0.28)
    for x0, key, title in zip(x0s, ("before", "after"), ("(a) Before adaptation", "(b) After adaptation, *q* = 5")):
        for i, src in enumerate(loads):                 # rows: training load, top to bottom
            for j, tgt in enumerate(loads):
                v = cells[key][(src, tgt)]
                x, y = x0 + j * cs, y0 + (2 - i) * cs
                fill = P_GREY if src == tgt else color(v)
                cv.rect(x, y, cs, cs, fill=fill, edge="#ffffff", lw=1.0)
                cv.text(x + cs / 2, y + cs / 2, f"{100 * v:.1f}", color="#ffffff" if dark(fill) else INK)
        cv.rect(x0, y0, 3 * cs, 3 * cs, edge=INK2, lw=0.6)
        for j, tgt in enumerate(loads):
            cv.text(x0 + (j + 0.5) * cs, y0 - 2 * PT, LAB[tgt], va="top")
        cv.text(x0 + 1.5 * cs, y0 - 3 * PT - FS * 1.2 * PT, "Test load", va="top")
        cv.text(x0 + 1.5 * cs, y0 + 3 * cs + 2 * PT, title, va="bottom")
    for i, src in enumerate(loads):
        cv.text(x0s[0] - 3 * PT, y0 + (2.5 - i) * cs, LAB[src], ha="right")
    cv.text(x0s[0] - 6 * PT - text_width("0 N·m"), y0 + 1.5 * cs, "Training load", ha="right", rot=90)
    bx, bw, bh = x0s[1] + 3 * cs + 0.09, 0.09, 3 * cs   # colour bar
    for k in range(48):
        v = CLO + (CHI - CLO) * (k + 0.5) / 48
        cv.rect(bx, y0 + bh * k / 48, bw, bh / 48 + 0.002, fill=color(v))
    cv.rect(bx, y0, bw, bh, edge=INK2, lw=0.5)
    for v in (0.4, 0.6, 0.8, 1.0):
        yy = y0 + bh * (v - CLO) / (CHI - CLO)
        cv.line(bx + bw, yy, bx + bw + TICK, yy, color=INK2, lw=0.6)
        cv.text(bx + bw + TICK + 1.5 * PT, yy, f"{100 * v:.0f}", ha="left")
    cv.finish()


# --------------------------------------------------------------------------- Fig. 11: hardware-in-the-loop bench
def fig_bench(page, cfg, T):
    """The ESP32 bench in a few words: host as gateway over USB, acquisition core, KESTREL core, DAC-ADC loop."""
    cv = Canvas(page, "fig11_bench")
    H = cv.h
    DARK = "#333333"
    hx0, hx1, hy0, hy1 = 0.04, 0.76, 0.42, 0.90                    # host
    cv.rect(hx0, hy0, hx1 - hx0, hy1 - hy0, fill=P_GREY, edge=DARK, lw=0.75, rounding=0.05)
    cv.text((hx0 + hx1) / 2, (hy0 + hy1) / 2, "~{Host PC}\n(gateway)")
    gx0, gx1, gy0, gy1 = 1.14, 3.42, 0.04, H - 0.03                 # node
    cv.rect(gx0, gy0, gx1 - gx0, gy1 - gy0, fill=LIGHT, edge=MUTED, lw=0.75, dash="--", rounding=0.06)
    cv.text(gx0 + 0.10, gy1 - 0.05, "~{ESP32}", ha="left", va="top")
    c0, c1, cy0, cy1 = (1.26, 2.12), (2.42, 3.32), 0.56, 0.92
    cv.box(c0[0], cy0, c0[1] - c0[0], cy1 - cy0, "Core 0\nacquisition", fill=P_GREEN)
    cv.box(c1[0], cy0, c1[1] - c1[0], cy1 - cy0, "Core 1\nKESTREL node", fill=P_PINK)
    cv.line(c0[1], (cy0 + cy1) / 2, c1[0], (cy0 + cy1) / 2, color=INK, lw=0.9, arrow=True)
    dac, adc, dy0, dy1 = (1.26, 1.60), (1.78, 2.12), 0.13, 0.37    # analog loop below core 0
    cv.box(dac[0], dy0, dac[1] - dac[0], dy1 - dy0, "DAC", fill=P_SAND)
    cv.box(adc[0], dy0, adc[1] - adc[0], dy1 - dy0, "ADC", fill=P_SAND)
    xd, xa = (dac[0] + dac[1]) / 2, (adc[0] + adc[1]) / 2
    cv.line(xd, cy0, xd, dy1, color=INK, lw=0.9, arrow=True)
    cv.line(xa, dy1, xa, cy0, color=INK, lw=0.9, arrow=True)
    cv.line(dac[1], (dy0 + dy1) / 2, adc[0], (dy0 + dy1) / 2, color=DARK, lw=1.8)    # jumper wire
    for y, (a, b) in ((0.72, (hx1 + 0.02, gx0 - 0.02)), (0.60, (gx0 - 0.02, hx1 + 0.02))):
        cv.line(a, y, b, y, color=INK, lw=0.9, arrow=True)
    cv.text((hx1 + gx0) / 2, 0.72 + 2 * PT, "USB", va="bottom")
    cv.finish()


# --------------------------------------------------------------------------- Fig. 7: cost
def fig_cost(page, cfg, T):
    cv = Canvas(page, "fig07_cost")
    W, H = cv.w, cv.h
    d = T["fig_cost"]
    ax = Axes(cv, 0.60, 0.47, W - 0.68, H - 0.47 - 0.08, (5, 2e5), (0.57, 1.03), xlog=10)
    cv.rect(ax.X(5), ax.y0, ax.X(320) - ax.X(5), ax.h, fill="#e6f2e4")
    ax.grid(yvals=[0.6, 0.7, 0.8, 0.9, 1.0], xvals=[10, 100, 1000, 10000, 100000])
    xv = ax.X(320)
    cv.line(xv, ax.y0, xv, ax.y0 + ax.h, color=INK2, dash="--", lw=0.75)
    cv.text(ax.X(6), ax.Y(0.80), "real-time\nzone\n(< 320 ms)", ha="left", color=INK)
    # measured vs estimated in crimson/charcoal (blue/orange stay reserved for the RF and softmax heads)
    MEAS, EST = "#b71c4a", "#00897b"
    labels = {"CNN 34.4 M": ("CNN, 34.4 M\n(1569 KiB)", "below"), "EdgeLite": ("EdgeLite\n(416 KiB)", "left"),
              "MobileLite": ("MobileLite\n(224 KiB)", "above"),
              "KESTREL, tables in RAM": ("112 KiB", "under"), "KESTREL, tables in flash": ("64 KiB", "beside")}
    kes = []
    for r in d.itertuples():
        rad = 0.0050 * math.sqrt(r.ram_kib)
        x, y = ax.X(r.latency_ms), ax.Y(r.f1)
        cv.circle(x, y, rad, face=MEAS if r.measured else EST, edge="#333333", lw=0.6)
        txt, ha = labels[r.label]
        if ha == "above":
            cv.text(x, y + rad + 0.03, txt, va="bottom")
        elif ha == "below":
            w_ = max(text_width(l) for l in txt.split("\n")) / 2 + 0.04
            cv.text(min(x, ax.x0 + ax.w - w_), y - rad - 0.04, txt, va="top")
        elif ha == "under":                                  # each KESTREL point labeled on its own
            cv.text(x, y - rad - 0.05, txt, va="top")
            kes.append((x, y - rad - 0.05 - FS * 1.2 * PT))
        elif ha == "beside":
            cv.text(x + rad + 0.04, y + 0.02, txt, ha="left")
            kes.append((x, None))
        else:
            cv.text(x + rad + 0.05, y, txt, ha="left")
    if kes:
        cv.text(sum(k[0] for k in kes) / len(kes), min(k[1] for k in kes if k[1] is not None) - 0.01, "KESTREL",
                va="top", color=INK)
    ax.spines()
    ax.frame()
    ax.yticks([0.6, 0.7, 0.8, 0.9, 1.0], ["60", "70", "80", "90", "100"])
    ax.xticks([10, 100, 1000, 10000, 100000], ["10", "10^{2}", "10^{3}", "10^{4}", "10^{5}"])
    ax.xlabel("Latency per decision on ESP32 (ms)")
    ax.ylabel("Time-blocked macro-F1 (%)")
    legend(cv, ax.X(1500), ax.Y(0.78), [dict(label="measured", color=MEAS, circle=True),
                                        dict(label="estimated", color=EST, circle=True)])
    cv.text(ax.X(1500) + 0.05, ax.Y(0.78) - 0.37, "bubble area ∝ RAM", ha="left", color=INK2)
    cv.finish()


# --------------------------------------------------------------------------- Fig. 8: HIL spectra
def fig_spectra(page, cfg, T):
    from tools.parse_hil import parse, spectra
    cv = Canvas(page, "fig08_hil_spectra")
    W, H = cv.w, cv.h
    _, _, info = parse(cfg.path("results_dir") / "hw" / "hil_clean.log")
    f, psd = spectra(cfg, {"clean": info["raw"]})
    names = {0: "Normal", 1: "Inner race (BPFI)"}
    STORED, AFTER, DIFF = "#1a1a1a", "#b71c4a", "#f4d3dd"    # not blue/orange: those mean RF/softmax
    items = [dict(label="stored", color=STORED, dash="-"), dict(label="after DAC–ADC", color=AFTER, dash="-"),
             dict(label="difference", color=DIFF, patch=True)]
    for it, x_ in zip(items, (0.12, 0.90, 2.22)):
        legend(cv, x_, H - 0.01, [it])
    y0, top, x0, gap = 0.47, H - 0.40, 0.62, 0.12
    pw = (W - x0 - gap - 0.06) / 2
    for p, c in enumerate(sorted(psd)[:2]):
        ax = Axes(cv, x0 + p * (pw + gap), y0, pw, top - y0, (0, f[-1] / 1e3), (-45, 3))
        ref = psd[c][0].max()
        idx = np.arange(0, len(f), 6)
        db0 = np.clip(10 * np.log10(psd[c][0] / ref + 1e-12), -45, 3)[idx]
        db1 = np.clip(10 * np.log10(psd[c][1] / ref + 1e-12), -45, 3)[idx]
        fx = f[idx] / 1e3
        upper = np.maximum(db0, db1)
        cv.polygon([(ax.X(a), ax.Y(b)) for a, b in zip(fx, upper)] + [(ax.X(a), ax.Y(b)) for a, b in zip(fx[::-1], db0[::-1])],
                   DIFF)
        ax.grid(yvals=[-40, -30, -20, -10, 0])
        ax.plot(fx, db0, STORED, lw=0.8)
        ax.plot(fx, db1, AFTER, lw=0.8)
        ax.spines()
        ax.frame()
        ax.yticks([-40, -30, -20, -10, 0], ["−40", "−30", "−20", "−10", "0"] if p == 0 else [""] * 5)
        ax.xticks([0, 5, 10])
        ax.title(names.get(c, str(c)))
        if p == 0:
            ax.ylabel("PSD (dB re peak)")
    cv.text(x0 + pw + gap / 2, y0 - TICK - 3 * PT - FS * 1.2 * PT, "Frequency (kHz)", va="top")
    cv.finish()


# --------------------------------------------------------------------------- Fig. 9 (draft): overview
def _blend(c0, c1, a):
    h0, h1 = c0.lstrip("#"), c1.lstrip("#")
    return "#%02x%02x%02x" % tuple(round(int(h0[i:i + 2], 16) * (1 - a) + int(h1[i:i + 2], 16) * a) for i in (0, 2, 4))


def _envelope(x, n):
    """Min/max envelope of a long signal as n alternating points (dense look of a raw vibration trace)."""
    x = np.asarray(x[: len(x) // n * n], dtype=np.float64).reshape(n, -1)
    lo, hi = x.min(1), x.max(1)
    return np.ravel(np.column_stack([lo, hi]))


def _signal(cfg, name):
    return np.load(cfg.path("cache_dir") / "signals" / "kaist" / f"{name}.npy", mmap_mode="r")


def _photo(cv, name, x, y, h):
    """Copy of the authors' photo shape `name` (kept beside the sheet) scaled to height h at (x, y)."""
    src = get_page(cv.page.Document, PHOTOS).Shapes.ItemU(name)
    w0, h0 = src.CellsU("Width").ResultIU, src.CellsU("Height").ResultIU
    w = h * w0 / h0
    shp = cv.page.Drop(src, w / 2, h / 2)
    c = shp.CellsU
    c("Width").ResultIU, c("Height").ResultIU = w, h
    c("PinX").ResultIU, c("PinY").ResultIU = cv.ox + x + w / 2, cv.oy + y + h / 2
    shp.Name = f"copy_{name}"
    cv.shapes.append(shp)
    cv.rect(x, y, w, h, edge=INK2, lw=0.6)
    return w


GEN_DIR = FIG_DIR / "visio" / "gen"


def _mpl_png(name, w, h, draw):
    """Render `draw(fig)` with matplotlib at w x h inches (600 dpi, Arial) and return the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6}):
        fig = plt.figure(figsize=(w, h))
        draw(fig)
        path = GEN_DIR / f"{name}.png"
        fig.savefig(path, dpi=600, transparent=True)
        plt.close(fig)
    return path


def _image(cv, path, x, y, w, h):
    shp = cv.page.Import(str(path))
    c = shp.CellsU
    c("Width").ResultIU, c("Height").ResultIU = w, h
    c("PinX").ResultIU, c("PinY").ResultIU = cv.ox + x + w / 2, cv.oy + y + h / 2
    cv.shapes.append(shp)
    return shp


def _fat_arrow(cv, p0, p1, w=0.03, hl=0.11, hw=0.10, color="#333333"):
    """Solid arrow: shaft and head are one polygon, so no stroke runs under the head."""
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    dx, dy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -dy, dx
    bx, by = x1 - dx * hl, y1 - dy * hl
    cv.polygon([(x0 + nx * w / 2, y0 + ny * w / 2), (bx + nx * w / 2, by + ny * w / 2), (bx + nx * hw / 2, by + ny * hw / 2),
                (x1, y1), (bx - nx * hw / 2, by - ny * hw / 2), (bx - nx * w / 2, by - ny * w / 2),
                (x0 - nx * w / 2, y0 - ny * w / 2)], color)


def _fat_path(cv, pts, w=0.03, hl=0.11, hw=0.10, color="#333333"):
    """Orthogonal path: solid bars for the first segments (extended to close the corners), solid arrow last."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        L = math.hypot(x1 - x0, y1 - y0)
        dx, dy = (x1 - x0) / L, (y1 - y0) / L
        nx, ny = -dy, dx
        ax_, ay_, bx_, by_ = x0 - dx * w / 2, y0 - dy * w / 2, x1 + dx * w / 2, y1 + dy * w / 2
        cv.polygon([(ax_ + nx * w / 2, ay_ + ny * w / 2), (bx_ + nx * w / 2, by_ + ny * w / 2),
                    (bx_ - nx * w / 2, by_ - ny * w / 2), (ax_ - nx * w / 2, ay_ - ny * w / 2)], color)
    _fat_arrow(cv, pts[-2], pts[-1], w, hl, hw, color)


def _draw_band(sig, n=320):
    """Two-tone signal band: min/max envelope (light) and +-2 sigma (dark) per bin."""
    def draw(fig):
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        x = np.asarray(sig[: len(sig) // n * n], dtype=np.float64).reshape(n, -1)
        m = x.mean()
        s = np.percentile(np.abs(x - m), 99.7) or 1.0
        lo, hi, sd = (x.min(1) - m) / s, (x.max(1) - m) / s, 2 * x.std(1) / s
        k = np.arange(n)
        ax.fill_between(k, np.clip(lo, -1.1, 0), np.clip(hi, 0, 1.1), color="#a9c8ea", lw=0)
        ax.fill_between(k, -sd, sd, color="#1f5fa8", lw=0)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(-1.15, 1.15)
    return draw


def _draw_snippets(sig, ws, bw, n=420, pad=0.03):
    """One raw snippet per time block (from the middle of each block), aligned with the block columns."""
    def draw(fig):
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        L = len(sig)
        parts = []
        for k in range(5):
            mid = int((k + 0.5) * L / 5)
            parts.append(np.asarray(sig[mid - n // 2: mid + n // 2], dtype=np.float64))
        m = np.mean([p_.mean() for p_ in parts])
        s = max(np.abs(p_ - m).max() for p_ in parts) or 1.0
        for k, p_ in enumerate(parts):
            x0, x1 = k * bw + 0.04 - pad, (k + 1) * bw - 0.04 - pad
            ax.plot(np.linspace(x0, x1, len(p_)), 0.92 * (p_ - m) / s, color="#1f5fa8", lw=0.5)
        ax.set_xlim(0, ws - 2 * pad)
        ax.set_ylim(-1.0, 1.0)
    return draw


def _draw_wave(sig, n=1400):
    def draw(fig):
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        x = np.asarray(sig[20000:20000 + n], dtype=np.float64)
        x = (x - x.mean()) / (np.abs(x - x.mean()).max() + 1e-12)
        ax.axhline(0, color="#a9c8ea", lw=0.4)
        ax.plot(np.arange(n), x, color="#1f5fa8", lw=0.6)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(-1.1, 1.1)
    return draw


MINI_AX = (0.24, 0.72)      # bottom and height of the two mini charts of Fig. 1, so that their baselines align


def _draw_gate(tr, meta):
    def draw(fig):
        ax = fig.add_axes([0.08, MINI_AX[0], 0.90, MINI_AX[1]])
        ax.axvspan(meta.t_change, meta.t_adapt0, color="#f6e3c3", lw=0)
        ax.axvspan(meta.t_adapt0, meta.t_adapt1, color="#f6dde8", lw=0)
        ax.plot([0, meta.t_adapt0], [meta.tau] * 2, color="#1a1a1a", lw=0.7, ls=(0, (4, 2)))
        ax.plot([meta.t_adapt1, 130], [meta.tau_after] * 2, color="#1a1a1a", lw=0.7, ls=(0, (4, 2)))
        b_ = tr[tr.phase == "before"]
        a_ = tr[tr.phase == "after"]
        ax.plot(b_.t, b_.g, color="#1f3b5b", lw=1.0)     # as in Fig. 6; blue/orange mean RF/softmax
        ax.plot(a_.t, a_.g, color="#1f3b5b", lw=1.0)
        ax.text(2, meta.tau + 0.08, r"$\tau$", fontsize=7, color="#333333")
        ax.text(128, meta.tau_after + 0.08, r"$\tau'$", fontsize=7, color="#333333", ha="right")
        ax.annotate("load change", xy=(meta.t_change, 0.05), xytext=(meta.t_change, -0.22), fontsize=6,
                    ha="center", va="top", color="#1a1a1a", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#1a1a1a", mutation_scale=5))
        ax.set_xlim(0, 130)
        ax.set_ylim(0, 1.06 * max(2.85, float(tr.g.max())))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    return draw


def _curve_arrow(cv, p0, p1, w=0.03, hl=0.11, hw=0.11, color="#333333", n=120):
    """Solid S-curve arrow (cubic Bezier with horizontal tangents): shaft and head form one polygon."""
    P0, P1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    k = 0.5 * (P1[0] - P0[0])
    C0, C1 = P0 + [k, 0.0], P1 - [k, 0.0]
    ts = np.linspace(0.0, 1.0, n)[:, None]
    P = (1 - ts) ** 3 * P0 + 3 * (1 - ts) ** 2 * ts * C0 + 3 * (1 - ts) * ts ** 2 * C1 + ts ** 3 * P1
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    j = int(np.searchsorted(s, s[-1] - hl))
    f = (s[-1] - hl - s[j - 1]) / (s[j] - s[j - 1])
    Pb = P[j - 1] + f * (P[j] - P[j - 1])
    shaft = np.vstack([P[:j], Pb])
    d = np.gradient(shaft, axis=0)
    d /= np.hypot(d[:, 0], d[:, 1])[:, None]
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    left, right = shaft + nrm * w / 2, shaft - nrm * w / 2
    u = (P1 - Pb) / np.hypot(*(P1 - Pb))
    nh = np.array([-u[1], u[0]])
    pts = [tuple(q) for q in left] + [tuple(Pb + nh * hw / 2), tuple(P1), tuple(Pb - nh * hw / 2)] + \
          [tuple(q) for q in right[::-1]]
    cv.polygon(pts, color)


def _draw_retune(vals):
    """Grouped bars before (sand) and after (head colour) retuning, values on top."""
    def draw(fig):
        ax = fig.add_axes([0.04, MINI_AX[0], 0.92, MINI_AX[1]])
        for k, (b_, a_, c) in enumerate(vals):
            ax.bar(k - 0.19, b_, 0.36, color="#e9d8a6", edgecolor="#333333", lw=0.4)
            ax.bar(k + 0.19, a_, 0.36, color=c, edgecolor="#333333", lw=0.4)
            ax.text(k - 0.19, b_ + 2, f"{b_:.0f}", ha="center", va="bottom", fontsize=6)
            ax.text(k + 0.19, a_ + 2, f"{a_:.0f}", ha="center", va="bottom", fontsize=6)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 112)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["RF", "SM"])
        ax.tick_params(axis="x", length=0, labelsize=7, pad=2)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    return draw


def fig_overview(page, cfg, T, name="fig09_overview", retune=((62.8, 96.4), (51.1, 95.1)), sfx="",
                 gate_box="Self-check gate (S): *T*^{2} + head disagreement"):
    """Workflow in the style of Duan et al. (MSSP 2025, Fig. 15): data, training, deployment.
    retune: macro-F1 (%) before and after retuning, RF and softmax (single-source KAIST, q = 5)."""
    import json
    cv = Canvas(page, name)
    W, H = cv.w, cv.h
    ph, gap = 1.83, 0.22
    gh = ph - 0.37                                          # height of the content boxes inside a panel
    tops = [H - 0.02 - k * (ph + gap) for k in range(3)]
    fills = ["#ebeaf6", "#e4eef8", "#e6f2e4"]
    titles = ["Data preprocessing", "Train the model", "Deploy the model"]
    RED, DARK = "#a50f15", "#333333"
    for top, fill, title in zip(tops, fills, titles):
        cv.rect(0.03, top - ph, W - 0.06, ph, fill=fill, edge=DARK, lw=1.5, rounding=0.12)
        cv.text(W / 2, top - 0.14, f"~{{{title}}}", color=RED, size=10)
    for top in tops[:2]:                                    # down arrows between panels
        cx, y0 = W / 2, top - ph
        _fat_arrow(cv, (cx, y0 - 0.005), (cx, y0 - gap + 0.01), w=0.22, hl=0.12, hw=0.42, color=DARK)
    C_TR, C_VA, C_TE = "#c9dbf0", "#f3e3a6", "#f6cdbd"
    BIG, SMALL = dict(w=0.04, hl=0.13, hw=0.13), dict(w=0.02, hl=0.07, hw=0.075)

    # ---- panel 1: KAIST testbed photo -> time-blocked recordings
    top, bot = tops[0], tops[0] - ph
    photo_h = ph - 0.40
    pw = _photo(cv, "photo_kaist", 0.12, bot + 0.07, photo_h)
    names = [("0Nm_Normal", "Normal"), ("0Nm_Misalign_03", "Misalignment"), ("0Nm_BPFI_10", "Inner race")]
    lab_w = max(text_width(n) for _, n in names) + 0.10
    xs0 = 0.12 + pw + 0.03                                   # arrows start on the photo edge
    sx0, sx1 = xs0 + 0.64 + lab_w, W - 0.15
    sh, sgap = 0.36, 0.10
    y_top = top - 0.49
    rows = [y_top - (r + 1) * sh - r * sgap for r in range(3)]
    blocks = [C_TR, C_TR, C_TE, C_VA, C_TR]
    bw = (sx1 - sx0) / 5
    for k, c in enumerate(blocks):                          # time blocks across all recordings
        cv.rect(sx0 + k * bw + 0.01, rows[-1] - 0.03, bw - 0.02, y_top - rows[-1] + 0.03, fill=c)
    for k, word in ((0, "Training"), (2, "Test"), (3, "Validation")):
        tw = text_width(word, size=7, bold=True) + 0.10
        xt = sx0 + (k + 0.5) * bw - tw / 2
        cv.rect(xt, y_top + 0.03, tw, 0.17, fill=shade(blocks[k], 0.93), edge="#333333", lw=0.75, rounding=0.05)
        cv.text(xt + tw / 2, y_top + 0.115, f"~{{{word}}}", size=7)
    ymid = rows[1] + sh / 2
    for (fn, lab), y in zip(names, rows):
        cv.rect(sx0, y, sx1 - sx0, sh, edge=DARK, lw=0.6, rounding=0.03)
        png = _mpl_png(f"strip_{fn}", sx1 - sx0 - 0.06, sh - 0.06, _draw_snippets(_signal(cfg, fn), sx1 - sx0, bw))
        _image(cv, png, sx0 + 0.03, y + 0.03, sx1 - sx0 - 0.06, sh - 0.06)
        cv.text(sx0 - 0.06, y + sh / 2, lab, ha="right")
        _curve_arrow(cv, (xs0, ymid), (sx0 - lab_w - 0.02, y + sh / 2), w=0.03, hl=0.11, hw=0.11, color=DARK)

    # ---- panel 2: per-window pipeline -> results on the host
    top, bot = tops[1], tops[1] - ph
    gx1 = 4.60
    cv.rect(0.15, bot + 0.07, gx1 - 0.15, gh, fill="#eef4fb", edge="#5b7fa6", lw=0.6, dash="--", rounding=0.06)
    steps = [("Time-blocked\nwindows", P_BLUE), ("FFT + kinematic\norders (K)", P_LAV),
             ("RF + softmax\nheads (E)", P_PINK), ("C code +\ngolden tests", P_GREEN)]
    ws = [block_width(s) + 0.18 for s, _ in steps]
    bh, y = 0.48, bot + 0.88
    g = (gx1 - 0.15 - 0.30 - sum(ws)) / 3
    x, xs = 0.30, []
    for (s, fill), w in zip(steps, ws):
        cv.box(x, y, w, bh, s, fill=fill)
        xs.append((x, w))
        x += w + g
    for (xa, wa), (xb, _) in zip(xs, xs[1:]):
        _fat_arrow(cv, (xa + wa + 0.01, y + bh / 2), (xb - 0.01, y + bh / 2), **SMALL)
    gtxt = gate_box
    gw = block_width(gtxt) + 0.24
    gx = max(0.30, min(xs[1][0] + xs[1][1] / 2 - 0.25, gx1 - 0.12 - gw))   # inside the dashed region
    cv.box(gx, bot + 0.22, gw, 0.34, gtxt, fill=P_SAND)
    for k in (1, 2):
        xk = xs[k][0] + xs[k][1] / 2
        _fat_arrow(cv, (xk, y - 0.01), (xk, bot + 0.57), **SMALL)
    rx0, rx1 = 4.85, W - 0.15
    ym = bot + 0.07 + gh / 2
    cv.rect(rx0, bot + 0.07, rx1 - rx0, gh, fill="#ffffff", edge=DARK, lw=0.8, rounding=0.06)
    _fat_arrow(cv, (gx1 + 0.01, ym), (rx0 - 0.02, ym), **BIG)
    s = T["e03_features_summary"].set_index(["features", "model"]).f1_mean
    ax = Axes(cv, 5.22, bot + 0.36, 0.78, 0.86, (-0.6, 1.6), (0.8, 1.0))
    for k, fs in enumerate(["F16", "F19"]):
        for i, (m, c) in enumerate((("RF", RF_C), ("Softmax", SM_C))):
            ax.bar(k + (i - 0.5) * 0.38, 0.34, s[(fs, m)], c)
    ax.spines()
    ax.yticks([0.8, 0.9, 1.0], ["80", "90", "100"])
    ax.xticks([0, 1], ["*F*_{16}", "*F*_{19}"])
    ax.title("macro-F1 (%)")
    cm = np.array(json.load(open(cfg.path("results_dir") / "logs" / "e01_protocol_confusion.json"))["rf_F19"])
    cs, cx0, cy0 = 0.16, 6.15, bot + 0.36
    for i in range(5):
        for j in range(5):
            cv.rect(cx0 + j * cs, cy0 + (4 - i) * cs, cs, cs, fill=_blend("#ffffff", "#184f95", min(cm[i, j], 1.0) ** 0.6),
                    edge="#c9d6e6", lw=0.4)
    for k, l in enumerate("NIOMU"):
        cv.text(cx0 - 0.03, cy0 + (4 - k + 0.5) * cs, l, ha="right", size=7)
        cv.text(cx0 + (k + 0.5) * cs, cy0 - 1 * PT, l, va="top", size=7)
    cv.text(cx0 + 2.5 * cs, cy0 + 5 * cs + 2 * PT, "confusion (RF)", va="bottom")

    # ---- panel 3: signal -> ESP32 photo -> on-chip lifecycle -> results
    top, bot = tops[2], tops[2] - ph
    sb0, sb1, sy0, sy1 = 0.15, 1.28, bot + 0.50, bot + 1.10
    cv.rect(sb0, sy0, sb1 - sb0, sy1 - sy0, fill="#ffffff", edge=DARK, lw=0.6, rounding=0.03)
    png = _mpl_png("stream_2Nm_Normal", sb1 - sb0 - 0.08, sy1 - sy0 - 0.08, _draw_wave(_signal(cfg, "2Nm_Normal")))
    _image(cv, png, sb0 + 0.04, sy0 + 0.04, sb1 - sb0 - 0.08, sy1 - sy0 - 0.08)
    cv.text((sb0 + sb1) / 2, sy0 - 1 * PT, "25.6 kHz stream", va="top")
    ex, ey, eh = 1.50, bot + 0.12, 1.14
    ew = _photo(cv, "photo_esp32", ex, ey, eh)
    cv.text(ex + ew / 2, ey + eh + 1 * PT, "ESP32, 240 MHz, 320 KiB", va="bottom")
    _fat_arrow(cv, (sb1 + 0.02, (sy0 + sy1) / 2), (ex - 0.02, (sy0 + sy1) / 2), **BIG)
    nx0, nx1, ny0, ny1 = 3.40, 4.94, bot + 0.10, bot + 1.48
    cv.rect(nx0, ny0, nx1 - nx0, ny1 - ny0, fill="#ffffff", edge=DARK, lw=0.8, rounding=0.06)
    cv.line(ex + ew, ey + eh, nx0, ny1, color=MUTED, lw=0.6, dash="--")
    cv.line(ex + ew, ey, nx0, ny0, color=MUTED, lw=0.6, dash="--")
    chips = [("Monitor", P_BLUE), ("Alert", P_SAND), ("Adapt (R)", P_GREEN), ("Verify", P_BLUE)]
    cwid, chh = 0.56, 0.26
    cpos = [(nx0 + 0.10, ny1 - 0.55), (nx1 - 0.10 - cwid, ny1 - 0.55), (nx1 - 0.10 - cwid, ny1 - 1.03),
            (nx0 + 0.10, ny1 - 1.03)]
    for (lab, fill), (x, y) in zip(chips, cpos):
        cv.box(x, y, cwid, chh, lab, fill=fill, rounding=0.05)
    LC = dict(w=0.018, hl=0.065, hw=0.07)
    _fat_arrow(cv, (cpos[0][0] + cwid + 0.01, cpos[0][1] + chh / 2), (cpos[1][0] - 0.01, cpos[1][1] + chh / 2), **LC)
    _fat_arrow(cv, (cpos[1][0] + cwid / 2, cpos[1][1] - 0.01), (cpos[2][0] + cwid / 2, cpos[2][1] + chh + 0.01), **LC)
    _fat_arrow(cv, (cpos[2][0] - 0.01, cpos[2][1] + chh / 2), (cpos[3][0] + cwid + 0.01, cpos[3][1] + chh / 2), **LC)
    _fat_arrow(cv, (cpos[3][0] + cwid / 2, cpos[3][1] + chh + 0.01), (cpos[0][0] + cwid / 2, cpos[0][1] - 0.01), **LC)
    cv.text((nx0 + nx1) / 2, ny0 + 0.10, "20.9 ms per window · 64 KiB RAM", size=7)
    cv.text((nx0 + nx1) / 2, ny1 - 0.08, "~{on-chip lifecycle}", va="top")
    rx0, rx1 = 5.14, W - 0.15
    cv.rect(rx0, bot + 0.07, rx1 - rx0, gh, fill="#ffffff", edge=DARK, lw=0.8, rounding=0.06)
    _fat_arrow(cv, (nx1 + 0.01, (ny0 + ny1) / 2), (rx0 - 0.02, (ny0 + ny1) / 2), w=0.035, hl=0.10, hw=0.12)
    tr, meta = T[f"fig_gate_trace{sfx}"], T[f"fig_gate_meta{sfx}"].iloc[0]
    gx0, gy0, gw_, gh_ = 5.20, bot + 0.14, 0.92, 0.98
    png = _mpl_png("gate_trace" + name[len("fig09_overview"):], gw_, gh_, _draw_gate(tr, meta))
    _image(cv, png, gx0, gy0, gw_, gh_)
    rtx0, rtw = 6.18, 0.76
    (rb, ra), (sb, sa) = retune
    png = _mpl_png("retune_bars" + name[len("fig09_overview"):], rtw, gh_,
                   _draw_retune(((rb, ra, RF_C), (sb, sa, SM_C))))
    _image(cv, png, rtx0, gy0, rtw, gh_)
    for (x_, w_), (t1, t2) in (((gx0, gw_), ("~{Self-check gate (S)}", "statistic *g*_{t}")),
                               ((rtx0, rtw), ("~{Retuning (R)}", "before → after (%)"))):
        cv.text(x_ + w_ / 2, gy0 + gh_ + 0.24, t1)
        cv.text(x_ + w_ / 2, gy0 + gh_ + 0.10, t2, size=7, color=INK2)
    cv.finish()


# --------------------------------------------------------------------------- Fig. 10 (draft): trade-off triangle
def _icon_target(cv, x, y):
    for r, face in ((0.13, "#ffffff"), (0.088, SM_C), (0.046, "#ffffff")):
        cv.circle(x, y, r, face=face, edge=INK, lw=0.5)
    cv.circle(x, y, 0.016, face=INK, edge=None)
    cv.line(x + 0.15, y + 0.15, x + 0.025, y + 0.025, color=INK, lw=0.8, arrow=True)


def _icon_chip(cv, x, y):
    a = 0.085
    for k in (-1, 0, 1):
        o = k * 0.05
        for (x1, y1, x2, y2) in ((x - a, y + o, x - a - 0.04, y + o), (x + a, y + o, x + a + 0.04, y + o),
                                 (x + o, y - a, x + o, y - a - 0.04), (x + o, y + a, x + o, y + a + 0.04)):
            cv.line(x1, y1, x2, y2, color=INK, lw=0.7)
    cv.rect(x - a, y - a, 2 * a, 2 * a, fill="#c8ccd2", edge=INK, lw=0.5, rounding=0.015)
    cv.rect(x - 0.035, y - 0.035, 0.07, 0.07, fill="#ffffff", edge=INK, lw=0.4)


def _icon_loop(cv, x, y):
    ang = np.radians(np.linspace(60, 380, 28))
    cv.polyline([(x + 0.095 * math.cos(t), y + 0.095 * math.sin(t)) for t in ang], color=INK, lw=1.0, arrow=True)
    cv.circle(x, y, 0.03, face=RF_C, edge=INK, lw=0.4)


def fig_tradeoff(page, cfg, T):
    """Trade-off triangle (one column): vertex names and knob names only; costs are given in the text."""
    cv = Canvas(page, "fig10_tradeoff")
    W, H = cv.w, cv.h
    s, y0 = 1.55, 0.30
    cx = 1.73
    E, AU, A = (cx - s / 2, y0), (cx + s / 2, y0), (cx, y0 + s * math.sqrt(3) / 2)
    cv.polygon([E, AU, A], "#f4f4f4", edge=INK2, lw=1.0)

    def at(p, q, t_):
        return (p[0] + t_ * (q[0] - p[0]), p[1] + t_ * (q[1] - p[1]))

    def knob(p_, n):
        cv.circle(*p_, 0.072, face="#ffffff", edge=INK, lw=0.6)
        cv.text(*p_, str(n), bold=True)

    k = 1
    for t_, lab in ((0.29, "Window length *N*"), (0.50, "Quantization"), (0.71, "Bearing orders")):
        p_ = at(E, A, t_)
        knob(p_, k)
        cv.text(p_[0] - 0.12, p_[1], lab, ha="right")
        k += 1
    for t_, lab in ((0.36, "Retuned head"), (0.64, "Labels per class *q*")):
        p_ = at(AU, A, t_)
        knob(p_, k)
        cv.text(p_[0] + 0.12, p_[1], lab, ha="left")
        k += 1
    p_ = at(E, AU, 0.5)
    knob(p_, k)
    cv.text(p_[0], y0 + 0.25, "Self-check +\non-chip update")

    for (x, y), fill, icon in ((A, P_PINK, _icon_target), (E, P_BLUE, _icon_chip), (AU, P_GREEN, _icon_loop)):
        cv.circle(x, y, 0.20, face=fill, edge=INK, lw=0.75)
        icon(cv, x, y)
    cv.text(A[0] + 0.25, A[1], "~{Accuracy}", ha="left")
    cv.text(E[0] - 0.25, E[1], "~{Efficiency}", ha="right")
    cv.text(AU[0] + 0.25, AU[1], "~{Autonomy}", ha="left")

    gx, gy = cx, y0 + 0.74
    star = [(gx + (0.072 if i % 2 == 0 else 0.03) * math.cos(math.pi / 2 + i * math.pi / 5),
             gy + (0.072 if i % 2 == 0 else 0.03) * math.sin(math.pi / 2 + i * math.pi / 5)) for i in range(10)]
    cv.polygon(star, "#f3d27a", edge=INK, lw=0.6)
    cv.text(gx, gy - 0.09, "~{KESTREL}", va="top")
    cv.finish()


# --------------------------------------------------------------------------- target draft (target/main_target.tex)
# Values still to be obtained; every one is listed in target/target_key.md.
def fig_staircase_target(page, cfg, T):
    fig_staircase(page, cfg, T, name="fig03_staircase_target", auroc="99.2%",
                  targets={(3, "sm"): 0.962, (4, "rf"): 0.935, (4, "sm"): 0.928,
                           (5, "rf"): 0.991, (5, "sm"): 0.983,      # measured: every fold-0 test window on the board (E19)
                           (6, "rf"): 0.743, (6, "sm"): 0.763})     # measured: DAC-ADC campaign, one window per recording


def fig_crossload_target(page, cfg, T):
    fig_crossload(page, cfg, T, name="fig06_crossload3d_target", targets={("after", "2Nm", "4Nm"): 0.907})


def fig_heatmap_target(page, cfg, T):
    fig_heatmap(page, cfg, T, name="fig06_heatmap_target", targets={("after", "2Nm", "4Nm"): 0.907})


def fig_gate_target(page, cfg, T):
    fig_gate(page, cfg, T, name="fig05_gate_target", sfx="_target",
             title_a="(a) The lifecycle on a load change, measured on the ESP32 in real time")


def fig_overview_target(page, cfg, T):
    fig_overview(page, cfg, T, name="fig09_overview_target", retune=((62.8, 96.4), (51.1, 96.2)), sfx="_target",
                 gate_box="Self-check gate (S): class prototypes + head disagreement")


def fig_kestrel_target(page, cfg, T):
    fig_kestrel(page, cfg, T, name="fig01_kestrel_target", gate=GATE_PROTO)


FIGURES = {"kestrel": fig_kestrel, "protocol": fig_protocol, "staircase": fig_staircase, "ablation": fig_ablation,
           "gate": fig_gate, "crossload": fig_crossload, "cost": fig_cost, "spectra": fig_spectra,
           "overview": fig_overview, "tradeoff": fig_tradeoff,
           "staircase_target": fig_staircase_target, "crossload_target": fig_crossload_target, "gate_target": fig_gate_target,
           "kestrel_target": fig_kestrel_target,
           "overview_target": fig_overview_target, "heatmap": fig_heatmap, "heatmap_target": fig_heatmap_target,
           "bench": fig_bench}
FIG_NAME = {"kestrel": "fig01_kestrel", "protocol": "fig02_protocol", "staircase": "fig03_staircase",
            "ablation": "fig04_ablation", "gate": "fig05_gate", "crossload": "fig06_crossload3d",
            "cost": "fig07_cost", "spectra": "fig08_hil_spectra", "overview": "fig09_overview",
            "tradeoff": "fig10_tradeoff", "staircase_target": "fig03_staircase_target",
            "crossload_target": "fig06_crossload3d_target", "overview_target": "fig09_overview_target",
            "gate_target": "fig05_gate_target", "kestrel_target": "fig01_kestrel_target",
            "heatmap": "fig06_heatmap", "heatmap_target": "fig06_heatmap_target", "bench": "fig11_bench"}


def visio_app():
    import win32com.client as win32
    try:
        return win32.GetActiveObject("Visio.Application")
    except Exception:
        import subprocess
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\visio.exe")
        subprocess.Popen([winreg.QueryValue(key, None)])
        for _ in range(60):
            time.sleep(1)
            try:
                return win32.GetActiveObject("Visio.Application")
            except Exception:
                pass
        raise RuntimeError("Visio did not start")


def main(args):
    export_only = "--export" in args
    keys = [a for a in args if not a.startswith("--")] or list(FIGURES)
    app = visio_app()
    app.AlertResponse = 1
    app.ScreenUpdating = False
    doc = open_drawing(app)
    try:
        open_pages(doc)
        if not export_only:
            cfg = load_config()
            T = tables(cfg)
            for k in keys:
                print(k)
                FIGURES[k](fig_page(doc, FIG_NAME[k]), cfg, T)
        order_pages(doc)
        export_pdfs(doc, [FIG_NAME[k] for k in keys])
        if VSDX.exists():
            doc.Save()
        else:
            doc.SaveAs(str(VSDX))
        print(f"saved {VSDX}")
    finally:
        app.ScreenUpdating = True
        app.AlertResponse = 0


if __name__ == "__main__":
    main(sys.argv[1:])
