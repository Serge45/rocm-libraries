# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Format the final matrix-staging LDS allocation without changing solution state."""

import os
import sys

from ..Components.DecouplePGR import decouplePGRBlocks
from .segment_interleave import SEG, _footprint, _tensor_footprint

# Bar letter and ANSI color per tensor (second buffer is drawn in lower case).
_BAR_STYLE = {"A": ("A", "34"), "B": ("B", "32"), "MXSA": ("S", "35"), "MXSB": ("S", "35"),
              "Metadata": ("M", "36")}
_BAR_CELL = 1024                       # bytes per bar character


def _segments(start, end):
    """64 KiB segments overlapped by [start, end)."""
    return tuple(range(start // SEG, (end - 1) // SEG + 1))


def lds_layout_model(state, kernel_name):
    """Collect the final staging regions and, per rotated tensor, the per-pair read order.

    Read-only. A region is (start, end, name, tensor, component, phase) including internal padding.
    """
    offsets = {}
    if state.get("LDSSegmentInterleave") == 1:
        offsets = state.get("LDSSegInterleaveOffsets") or {}
    paired = offsets.get("pairedBuffers", False)
    single_buffer = state.get("1LDSBuffer") == 1 or not state.get("PrefetchGlobalRead")
    count = 1 if single_buffer else max(1, state["NumLdsBlk"])
    decoupled, count_a, count_b = decouplePGRBlocks(state)
    divergent = decoupled and count_a != count_b
    regions = []
    strides = []
    for tc in ("A", "B", "MXSA", "MXSB", "Metadata"):
        direct = state.get("DirectToVgpr" + tc)
        if tc == "Metadata":
            direct = state.get("DirectToVgprSparseMetadata")
        if direct:
            continue
        if offsets and tc in ("A", "B"):
            whole = offsets.get("aBaseline" if tc == "A" else "bBaseline", False)
            size = _tensor_footprint(state, tc) if whole else _footprint(state, tc)
            components = 1 if whole else 2
        elif tc == "MXSA" and offsets.get("mxsSplitA"):
            size = offsets["mxsaHalfBytes"]
            components = 2
        else:
            size_key = "LdsNumElementsAligned" if offsets else "ldsNumBytes"
            size = state.get(size_key + tc, 0)
            components = 1
        if size <= 0:
            continue
        base = offsets.get("ldsBase" + tc, state.get("LdsOffset" + tc, 0))
        copies = count
        if paired:
            stride = offsets["bufferStride" + tc]
        elif divergent:
            side = "A" if tc in ("A", "MXSA") else "B"
            copies = count_a if side == "A" else count_b
            stride = state["LdsOffsetBlk" + side]
        else:
            stride = state.get("LdsOffsetA_Blk", 0)
        strides.append(f"{tc}={stride if copies > 1 else 0}")
        for component in range(components):
            group = tc
            if components > 1:
                group += f" group{component}" if state.get("TDMSplit") else str(component)
            for buffer in range(copies):
                phase = ("ping", "pong")[buffer] if copies == 2 else f"buffer{buffer}"
                component_stride = offsets.get("mxsaStrideBytes", 0) if tc == "MXSA" else offsets.get("writeStrideBytes", 0)
                start = int(base + component * component_stride + buffer * stride)
                regions.append((start, start + int(size), f"{group} {phase}", tc, component, phase))
    regions.sort(key=lambda r: r[:3])
    layout = "paired" if paired else "interleaved" if offsets else "ordinary"
    model = dict(name=kernel_name, layout=layout, strides=strides, regions=regions,
                 wave_group=state.get("MIWaveGroup"), tdm_split=bool(state.get("TDMSplit")),
                 rotate=(bool(state.get("LDSSegRotateA")), bool(state.get("LDSSegRotateB"))),
                 reserved=state["LdsNumBytes"], gfx1250=tuple(state.get("ISA", ())) == (12, 5, 0),
                 macro_tile=(state.get("MacroTile0"), state.get("MacroTile1")),
                 has_offsets=bool(offsets))
    model["rotations"] = _rotation_model(model)
    return model


def _rotation_model(model):
    """Per rotated tensor and buffer phase: which component each SIMD pair reads at each step.

    Step s, pair p reads component (s ^ p), the same rule the rotation emits (pair 1 starts with component 1).
    A step conflicts when the two pairs' components share a 64 KiB segment. Components that straddle a
    boundary count for every segment they touch (conservative).
    """
    out = []
    if not model["gfx1250"]:
        return out
    for tc, rotated in zip(("A", "B"), model["rotate"]):
        if not rotated:
            continue
        comps = {}
        for start, end, _, rtc, component, phase in model["regions"]:
            if rtc == tc:
                comps.setdefault(phase, {})[component] = (start, end)
        for phase, by_comp in comps.items():
            if sorted(by_comp) != [0, 1]:
                continue
            steps = []
            for step in range(2):
                cells = []
                for pair in range(2):
                    comp = step ^ pair
                    cells.append((comp, _segments(*by_comp[comp])))
                steps.append(dict(step=step, pair0=cells[0], pair1=cells[1],
                                  ok=not set(cells[0][1]) & set(cells[1][1])))
            out.append(dict(tensor=tc, phase=phase, steps=steps))
    return out


def _use_color(color):
    if color is not None:
        return color
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _bar_lines(model, color):
    """One 64-character bar per 64 KiB segment (1 character = 1 KiB)."""
    lines = ["  LDS bars (1 character = 1 KiB, one row per 64 KiB segment; "
             "upper case = first buffer, lower case = second buffer, . = unused):"]
    for segment in range((model["reserved"] + SEG - 1) // SEG):
        cells = []
        for cell in range(SEG // _BAR_CELL):
            lo = segment * SEG + cell * _BAR_CELL
            hit = next((r for r in model["regions"] if r[0] < lo + _BAR_CELL and r[1] > lo), None)
            if lo >= model["reserved"]:
                cells.append(" ")
            elif hit is None:
                cells.append(".")
            else:
                letter, code = _BAR_STYLE.get(hit[3], ("?", "37"))
                second = hit[5] != "ping" and not hit[5].endswith("0")
                letter = letter.lower() if second else letter
                cells.append(f"\033[{code}m{letter}\033[0m" if color else letter)
        lines.append(f"  seg {segment} |{''.join(cells)}|")
    return lines


def _rotation_lines(model):
    lines = []
    for entry in model["rotations"]:
        lines.append(f"  Rotation {entry['tensor']} {entry['phase']}: step, pair0 reads, pair1 reads "
                     "(component, 64 KiB segments)")
        for st in entry["steps"]:
            def cell(c):
                return f"{entry['tensor']}{c[0]} (S{'/S'.join(str(s) for s in c[1])})"
            lines.append(f"    step {st['step']}: pair0 {cell(st['pair0'])}   pair1 {cell(st['pair1'])}"
                         f"   {'ok' if st['ok'] else 'CONFLICT'}")
    return lines


def format_lds_layout(state, kernel_name, color=None):
    """Return physical byte ranges and, on gfx1250, their 64 KiB segment map.

    Ranges include internal padding. This describes matrix staging, not the
    contents of LDS at runtime; reductions and the epilogue can reuse the memory.
    On gfx1250 it also draws one bar per segment and, for rotated tensors, the read order of the
    two SIMD pairs. color=None colors the bars only when stdout is a terminal and NO_COLOR is unset.
    """
    model = lds_layout_model(state, kernel_name)
    regions = model["regions"]
    lines = [f"[LDS layout] {kernel_name}",
             f"  layout={model['layout']}, MIWaveGroup={state.get('MIWaveGroup')}, "
             f"TDMSplit={model['tdm_split']}, Rotate(A,B)=({model['rotate'][0]}, {model['rotate'][1]})",
             f"  Reserved LDS: {state['LdsNumBytes']} bytes; ranges are [start, end), including padding.",
             "  Buffer strides (bytes): " + ", ".join(model["strides"]),
             "  Matrix staging regions (ascending address):"]
    for start, end, name, *_ in regions:
        lines.append(f"    [{start:6d}, {end:6d}) {name} ({end - start} bytes)")
    if model["gfx1250"]:
        lines.append("  64 KiB segments (regions crossing a boundary appear in both):")
        for segment in range((state["LdsNumBytes"] + SEG - 1) // SEG):
            lo, hi = segment * SEG, min((segment + 1) * SEG, state["LdsNumBytes"])
            cursor = lo
            blocks = []
            for start, end, name, *_ in regions:
                if end <= lo or start >= hi:
                    continue
                if start > cursor:
                    blocks.append(f"[gap {start - cursor} B]")
                blocks.append(f"[{name}]")
                cursor = max(cursor, min(end, hi))
            if cursor < hi:
                blocks.append(f"[gap {hi - cursor} B]")
            lines.append(f"    S{segment} [{lo}, {hi}): " + "".join(blocks))
    if model["has_offsets"] and state.get("TDMSplit"):
        lines.append("  group0/1 name physical read groups; "
                     "see segment_interleave.py for matrix-slice mapping.")
    if model["gfx1250"]:
        lines += _bar_lines(model, _use_color(color))
        lines += _rotation_lines(model)
    return "\n".join(lines)
