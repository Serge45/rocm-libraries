# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SVG drawing of the final LDS allocation and the rotation read order (see lds_layout.py)."""

import hashlib
import os
from xml.sax.saxutils import escape

from .lds_layout import lds_layout_model
from .segment_interleave import SEG

_FONT = "Segoe UI, Helvetica, Arial, sans-serif"
_MONO = "Consolas, Menlo, monospace"
_INK = "#1e1e1e"
# fill per (tensor, component); a single-component tensor uses component 0
_FILL = {("A", 0): "#a5d8ff", ("A", 1): "#d0bfff", ("B", 0): "#b2f2bb", ("B", 1): "#ffec99",
         ("MXSA", 0): "#fcc2d7", ("MXSB", 0): "#99e9f2", ("Metadata", 0): "#dee2e6"}
_WIDTH = 1000


class _Svg:
    def __init__(self):
        self.items = []

    def rect(self, x, y, w, h, fill="none", stroke=_INK, sw=1.2, rx=4, dash=None, opacity=None):
        extra = (f' stroke-dasharray="{dash}"' if dash else "") + \
                (f' fill-opacity="{opacity}"' if opacity is not None else "")
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>')

    def text(self, x, y, s, size=12, anchor="middle", weight="normal", color=_INK, mono=False):
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{_MONO if mono else _FONT}" '
                          f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
                          f'fill="{color}">{escape(str(s))}</text>')

    def line(self, x1, y1, x2, y2, color="#adb5bd", sw=1.0):
        self.items.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{color}" stroke-width="{sw}"/>')

    def render(self, height, title):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{height}" '
                f'viewBox="0 0 {_WIDTH} {height}">\n<title>{escape(title)}</title>\n'
                '<rect width="100%" height="100%" fill="white"/>\n'
                + "\n".join(self.items) + "\n</svg>\n")


def _is_second(phase):
    return phase != "ping" and not phase.endswith("0")


def build_svg(model):
    """Return the SVG text for one kernel's layout model (see lds_layout_model)."""
    s = _Svg()
    tile = "MT%sx%s" % model["macro_tile"] if None not in model["macro_tile"] else "kernel"
    s.text(20, 26, f"LDS layout: {tile}  MIWaveGroup {model['wave_group']}  layout={model['layout']}  "
           f"Rotate(A,B)=({model['rotate'][0]}, {model['rotate'][1]})", 16, anchor="start", weight="bold")
    s.text(20, 46, f"reserved {model['reserved']} bytes; ranges include padding; pale = second buffer "
           "(pong); dashed red = unused", 12, anchor="start", color="#555")
    nseg = max(1, (model["reserved"] + SEG - 1) // SEG)
    x0, width = 40, _WIDTH - 80
    total = nseg * SEG
    sx = lambda b: x0 + width * b / total
    y = 70

    # panel 1: to-scale allocation bar
    s.text(20, y, "1. Allocation (to scale)", 14, anchor="start", weight="bold")
    y += 12
    for k in range(nseg):
        s.rect(sx(k * SEG), y, sx(SEG) - x0, 50, fill="#f8f9fa", stroke="#adb5bd", sw=1, rx=0)
        s.text(sx(k * SEG) + (sx(SEG) - x0) / 2, y + 68, f"segment {k}", 11, color="#777")
    cursor = 0
    for start, end, name, tc, comp, phase in model["regions"]:
        if start > cursor and model["has_offsets"]:
            s.rect(sx(cursor), y + 4, sx(start) - sx(cursor), 42, fill="#fff5f5", stroke="#c92a2a",
                   sw=1, rx=3, dash="4,3")
        fill = _FILL.get((tc, comp if (tc, comp) in _FILL else 0), "#e9ecef")
        s.rect(sx(start), y + 4, max(sx(end) - sx(start), 1.0), 42, fill=fill, rx=3,
               opacity=0.55 if _is_second(phase) else 1.0)
        if sx(end) - sx(start) >= 34:
            s.text((sx(start) + sx(end)) / 2, y + 30, name.replace(" ping", "p").replace(" pong", "q")
                   .replace(" buffer", " b"), 11, weight="bold")
        cursor = max(cursor, end)
    y += 90

    # region table
    for i, (start, end, name, *_rest) in enumerate(model["regions"]):
        col, row = divmod(i, (len(model["regions"]) + 1) // 2)
        s.text(30 + col * 480, y + row * 16, f"[{start:7d}, {end:7d})  {name}  ({end - start} B)",
               11, anchor="start", mono=True)
    y += ((len(model["regions"]) + 1) // 2) * 16 + 24

    # panel 2: rotation read order
    if model["rotations"]:
        s.text(20, y, "2. Rotation read order (step s, pair p reads component s xor p)", 14,
               anchor="start", weight="bold")
        y += 14
        for entry in model["rotations"]:
            col = 0 if not _is_second(entry["phase"]) else 1
            if col == 0 and entry is not model["rotations"][0]:
                y += 190
            ox = 30 + col * 480
            s.text(ox, y + 14, f"{entry['tensor']} {entry['phase']}", 13, anchor="start", weight="bold")
            for st in entry["steps"]:
                bx = ox + 80 + st["step"] * 190
                s.text(bx + 80, y + 34, f"step {st['step']}", 11, color="#555")
                for p, cell in enumerate((st["pair0"], st["pair1"])):
                    comp, segs = cell
                    by = y + 42 + p * 52
                    s.rect(bx, by, 160, 44, fill=_FILL.get((entry["tensor"], comp), "#e9ecef"),
                           opacity=0.55 if _is_second(entry["phase"]) else 1.0)
                    s.text(bx + 80, by + 18, f"{entry['tensor']}{comp}", 13, weight="bold")
                    s.text(bx + 80, by + 35, "segment " + "/".join(str(v) for v in segs), 11)
                s.rect(bx, y + 150, 160, 22, fill="#c3fae8" if st["ok"] else "#ffc9c9")
                s.text(bx + 80, y + 165, "no conflict" if st["ok"] else "CONFLICT", 11, weight="bold")
            s.text(ox, y + 70, "pair 0", 11, anchor="start", color="#555")
            s.text(ox, y + 122, "pair 1", 11, anchor="start", color="#555")
        y += 190
    else:
        s.text(20, y, "No tensor is rotated (LDSSegmentRotate off or not applicable).", 12,
               anchor="start", color="#555")
        y += 20
    return s.render(y + 20, model["name"])


def svg_file_name(model):
    tile = "MT%sx%s" % model["macro_tile"] if None not in model["macro_tile"] else "kernel"
    wg = "x".join(str(v) for v in model["wave_group"]) if model["wave_group"] else "wg"
    digest = hashlib.sha1(model["name"].encode()).hexdigest()[:10]
    return f"{tile}_{wg}_{digest}.svg"


def write_layout_svg(state, kernel_name, directory):
    """Write the kernel's layout SVG into `directory`; return the path. Safe to call repeatedly."""
    model = lds_layout_model(state, kernel_name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, svg_file_name(model))
    with open(path, "w") as f:
        f.write(build_svg(model))
    return path
