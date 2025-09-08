# -*- coding: utf-8 -*-
"""
rigid_end_utils.py

Implements the split hook used by beams.py / columns.py to support ETABS rigid ends.
Signature matches current usage:

    parts = split_with_rigid_ends(
        kind="BEAM"|"COLUMN",
        line_name=str,
        story_index=int,
        nI=int, nJ=int,           # endpoint node tags
        pI=(x,y,z), pJ=(x,y,z),   # endpoint world coords
        LoffI=float, LoffJ=float  # LENGTHOFF at I/J ends
    )

Returns:
{
  "segments": [
    {"role": "rigid_i", "i": nI,   "j": nIm, "suffix": "::rigidI"},
    {"role": "deformable", "i": nIm, "j": nJm, "suffix": ""},
    {"role": "rigid_j", "i": nJm, "j": nJ,   "suffix": "::rigidJ"}
  ],
  "nodes":  {"nI": nI, "nIm": nIm_or_nI, "nJm": nJm_or_nJ, "nJ": nJ},
  "coords": {"nI": pI, "nIm": pIm_or_pI, "nJm": pJm_or_pJ, "nJ": pJ},
  "meta":   {...}
}

The function also **registers** any created interface nodes into the centralized
intermediate-node registry via emit_nodes.register_intermediate_node(), which is
later merged by emit_nodes.emit_nodes_json() without changing nodes.json schema.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import math
import os

try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = "out"

# Uses the registrar implemented in emit_nodes.py
from emit_nodes import register_intermediate_node, _load_json  # type: ignore


def _story_name_from_index(idx: int) -> str:
    sg = _load_json(os.path.join(OUT_DIR, "story_graph.json"))
    order = sg.get("story_order_top_to_bottom") or []
    if 0 <= idx < len(order):
        return order[idx]
    return f"StoryIndex-{idx}"


def _clamp_offset(li: float, L: float) -> float:
    if L <= 0.0:
        return 0.0
    return max(0.0, min(float(li or 0.0), float(L)))


def _intermediate_point(pI: Tuple[float, float, float],
                        pJ: Tuple[float, float, float],
                        L: float, off: float, end: str) -> Tuple[float, float, float]:
    t = 0.0 if L <= 0.0 else _clamp_offset(off, L) / L
    xi, yi, zi = pI
    xj, yj, zj = pJ
    vx, vy, vz = (xj - xi), (yj - yi), (zj - zi)
    if str(end).upper() == "I":
        return (xi + vx * t, yi + vy * t, zi + vz * t)
    else:
        return (xj - vx * t, yj - vy * t, zj - vz * t)


def split_with_rigid_ends(
    *,
    kind: str,
    line_name: str,
    story_index: int,
    nI: int,
    nJ: int,
    pI: Tuple[float, float, float],
    pJ: Tuple[float, float, float],
    LoffI: float = 0.0,
    LoffJ: float = 0.0,
) -> Dict[str, Any]:
    """
    Main hook used by beams.py / columns.py.
    """
    xi, yi, zi = pI
    xj, yj, zj = pJ
    vx, vy, vz = (xj - xi), (yj - yi), (zj - zi)
    L = math.sqrt(vx * vx + vy * vy + vz * vz)

    has_i = (LoffI or 0.0) > 0.0
    has_j = (LoffJ or 0.0) > 0.0

    # Defaults (no offsets)
    nIm: int = nI
    nJm: int = nJ
    pIm: Tuple[float, float, float] = pI
    pJm: Tuple[float, float, float] = pJ

    segments = []

    # Register I-end interface if needed
    if has_i and L > 0.0:
        pIm = _intermediate_point(pI, pJ, L, float(LoffI), "I")
        sidx_I = int(nI) % 1000
        sname_I = _story_name_from_index(sidx_I)
        nIm = register_intermediate_node(
            OUT_DIR, int(nI), int(nJ), "I",
            float(pIm[0]), float(pIm[1]), float(pIm[2]),
            sidx_I, sname_I, kind="rigid_interface"
        )
        segments.append({"role": "rigid_i", "i": int(nI), "j": int(nIm), "suffix": "::rigidI"})

    # Register J-end interface if needed
    if has_j and L > 0.0:
        pJm = _intermediate_point(pI, pJ, L, float(LoffJ), "J")
        sidx_J = int(nJ) % 1000
        sname_J = _story_name_from_index(sidx_J)
        nJm = register_intermediate_node(
            OUT_DIR, int(nI), int(nJ), "J",
            float(pJm[0]), float(pJm[1]), float(pJm[2]),
            sidx_J, sname_J, kind="rigid_interface"
        )

    # Middle deformable segment
    if has_i and has_j:
        segments.append({"role": "deformable", "i": int(nIm), "j": int(nJm), "suffix": ""})
    elif has_i and not has_j:
        segments.append({"role": "deformable", "i": int(nIm), "j": int(nJ), "suffix": ""})
    elif (not has_i) and has_j:
        segments.append({"role": "deformable", "i": int(nI), "j": int(nJm), "suffix": ""})
    else:
        segments.append({"role": "deformable", "i": int(nI), "j": int(nJ), "suffix": ""})

    # J-end rigid (after the middle)
    if has_j and L > 0.0:
        segments.append({"role": "rigid_j", "i": int(nJm), "j": int(nJ), "suffix": "::rigidJ"})

    return {
        "segments": segments,
        "nodes": {"nI": int(nI), "nIm": int(nIm), "nJm": int(nJm), "nJ": int(nJ)},
        "coords": {
            "nI": (float(pI[0]), float(pI[1]), float(pI[2])),
            "nIm": (float(pIm[0]), float(pIm[1]), float(pIm[2])),
            "nJm": (float(pJm[0]), float(pJm[1]), float(pJm[2])),
            "nJ": (float(pJ[0]), float(pJ[1]), float(pJ[2])),
        },
        "meta": {
            "kind": str(kind),
            "line_name": str(line_name),
            "story_index": int(story_index),
            "lengths": {"LoffI": float(LoffI or 0.0), "LoffJ": float(LoffJ or 0.0)},
        },
    }
