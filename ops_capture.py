# -*- coding: utf-8 -*-
"""
ops_capture.py

Minimal, opt-in monkey-patch capture layer for OpenSeesPy.

Usage (preferred via verify_domain_vs_artifacts.py):
    from ops_capture import capture_session, save_capture, get_capture

Captured signals (best-effort; wrappers used in our codebase):
- node(tag, x, y, z)
- fix(nodeTag, dof1..dof6)
- mass(nodeTag, m1..m6)
- rigidDiaphragm(perpDirn, masterTag, *slaveTags)
- geomTransf(type, tag, ...)
- element(type, tag, i, j, ...)
- elasticBeamColumn(tag, i, j, A, E, G, J, Iy, Iz, transfTag, ...)

Python: 3.11+
"""
from __future__ import annotations

import json
import os
import sys
import importlib
from contextlib import contextmanager
from typing import Any, Dict, List

from openseespy import opensees as _ops

# Store originals so we can restore them
_ORIG: Dict[str, Any] = {}
_CAP: Dict[str, Any] = {
    "nodes": [],            # {"tag": int, "x": float, "y": float, "z": float}
    "fixes": [],            # {"node": int, "ux":int,"uy":int,"uz":int,"rx":int,"ry":int,"rz":int}
    "masses": [],           # {"node": int, "m": [m1..m6]}
    "rigid_diaphragms": [], # {"perp": int, "master": int, "slaves": [int,..]}
    "geom_transf": [],      # {"ttype": str, "tag": int, "args": list}
    "elements": [],         # {"etype": str, "tag": int, "i": int, "j": int, "args": list}
}


def reset_capture() -> None:
    _CAP["nodes"].clear()
    _CAP["fixes"].clear()
    _CAP["masses"].clear()
    _CAP["rigid_diaphragms"].clear()
    _CAP["geom_transf"].clear()
    _CAP["elements"].clear()


def get_capture() -> Dict[str, Any]:
    return _CAP


def save_capture(out_dir: str = "out", filename: str = "domain_capture.json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"capture": _CAP}, f, indent=2)
    print(f"[CAPTURE] Wrote runtime domain snapshot: {path}")
    return path


# -----------------------------
# Wrappers
# -----------------------------
def _wrap_node(tag: int, x: float, y: float, z: float, *args: Any) -> None:
    _CAP["nodes"].append({"tag": int(tag), "x": float(x), "y": float(y), "z": float(z)})
    _ORIG["node"](tag, x, y, z, *args)


def _wrap_fix(tag: int, ux: int, uy: int, uz: int, rx: int, ry: int, rz: int) -> None:
    _CAP["fixes"].append(
        {"node": int(tag), "ux": int(ux), "uy": int(uy), "uz": int(uz), "rx": int(rx), "ry": int(ry), "rz": int(rz)}
    )
    _ORIG["fix"](tag, ux, uy, uz, rx, ry, rz)


def _wrap_mass(tag: int, m1: float, m2: float, m3: float, m4: float, m5: float, m6: float) -> None:
    _CAP["masses"].append({"node": int(tag), "m": [float(m1), float(m2), float(m3), float(m4), float(m5), float(m6)]})
    _ORIG["mass"](tag, m1, m2, m3, m4, m5, m6)


def _wrap_rigidDiaphragm(perp: int, master: int, *slaves: int) -> None:
    _CAP["rigid_diaphragms"].append({"perp": int(perp), "master": int(master), "slaves": [int(s) for s in slaves]})
    _ORIG["rigidDiaphragm"](perp, master, *slaves)


def _wrap_geomTransf(ttype: str, tag: int, *args: Any) -> None:
    _CAP["geom_transf"].append({"ttype": str(ttype), "tag": int(tag), "args": list(args)})
    _ORIG["geomTransf"](ttype, tag, *args)


def _wrap_element(etype: str, tag: int, i: int, j: int, *args: Any) -> None:
    _CAP["elements"].append(
        {"etype": str(etype), "tag": int(tag), "i": int(i), "j": int(j), "args": [*args]}
    )
    _ORIG["element"](etype, tag, i, j, *args)


def _wrap_elasticBeamColumn(tag: int, i: int, j: int, *args: Any) -> None:
    # Mirror element capture but record explicit etype
    _CAP["elements"].append(
        {"etype": "elasticBeamColumn", "tag": int(tag), "i": int(i), "j": int(j), "args": [*args]}
    )
    _ORIG["elasticBeamColumn"](tag, i, j, *args)


def _patch_ops_functions() -> None:
    # Save originals once
    if not _ORIG:
        _ORIG["node"] = _ops.node
        _ORIG["fix"] = _ops.fix
        _ORIG["mass"] = _ops.mass
        _ORIG["rigidDiaphragm"] = _ops.rigidDiaphragm
        _ORIG["geomTransf"] = _ops.geomTransf
        _ORIG["element"] = _ops.element
        # Some code calls the convenience factory directly
        _ORIG["elasticBeamColumn"] = getattr(_ops, "elasticBeamColumn", None)

    # Patch core funcs
    _ops.node = _wrap_node               # type: ignore[assignment]
    _ops.fix = _wrap_fix                 # type: ignore[assignment]
    _ops.mass = _wrap_mass               # type: ignore[assignment]
    _ops.rigidDiaphragm = _wrap_rigidDiaphragm  # type: ignore[assignment]
    _ops.geomTransf = _wrap_geomTransf   # type: ignore[assignment]
    _ops.element = _wrap_element         # type: ignore[assignment]
    # Optional factory
    if _ORIG.get("elasticBeamColumn") is not None:
        _ops.elasticBeamColumn = _wrap_elasticBeamColumn  # type: ignore[assignment]


def _restore_ops_functions() -> None:
    if not _ORIG:
        return
    _ops.node = _ORIG["node"]
    _ops.fix = _ORIG["fix"]
    _ops.mass = _ORIG["mass"]
    _ops.rigidDiaphragm = _ORIG["rigidDiaphragm"]
    _ops.geomTransf = _ORIG["geomTransf"]
    _ops.element = _ORIG["element"]
    if _ORIG.get("elasticBeamColumn") is not None:
        _ops.elasticBeamColumn = _ORIG["elasticBeamColumn"]  # type: ignore[assignment]


def _refresh_project_bindings() -> None:
    """
    Ensure project modules bind to the patched functions. This matters when:
    - modules were already imported before capture (e.g., run in same process)
    - modules import OpenSees funcs by name (alias), not via 'ops.<fn>'
    Strategy:
      * try to reload common modules; if not present, skip.
    """
    candidates = ["nodes", "supports", "diaphragms", "columns", "beams", "MODEL_translator"]
    for name in candidates:
        if name in sys.modules:
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                # Non-fatal; best-effort
                pass
    print("[CAPTURE] Refreshed project module bindings for capture.")


def enable_capture() -> None:
    reset_capture()
    _patch_ops_functions()
    _refresh_project_bindings()
    print("[CAPTURE] Enabled OpenSees capture wrappers.")


def disable_capture() -> None:
    _restore_ops_functions()
    print("[CAPTURE] Disabled OpenSees capture wrappers.")


@contextmanager
def capture_session():
    """
    Context manager to enable/disable capture cleanly.
    """
    try:
        enable_capture()
        yield
    finally:
        disable_capture()
