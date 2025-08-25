# -*- coding: utf-8 -*-
"""
explicit_static_probe.py

Purpose
-------
Apply a small static point load to a rigid-diaphragm master node in the
generated explicit model (out/explicit_model.py) to check basic stability:
 - Build domain from explicit file
 - Choose a diaphragm master (top by default)
 - Apply lateral load (default: +X) and run a short static analysis
 - Report master-node displacement and equilibrium of reactions

This is a *probe*, not a full load case. It helps answer: "Can my current
explicit model be subjected to an OpenSeesPy analysis and respond sensibly?"

Usage
-----
1) Ensure you generated the explicit model from artifacts:
     python generate_explicit_model.py --out out --explicit out/explicit_model.py

2) Run this probe (defaults: top diaphragm, X-direction, P=1.0, 10 steps):
     python explicit_static_probe.py

Examples
--------
# Load 5.0 in +Y on the 2nd diaphragm (index 1 top->bottom), 20 steps:
python explicit_static_probe.py --dir Y --P 5.0 --di 1 --steps 20

# Target a story name explicitly:
python explicit_static_probe.py --story "Roof" --P 2.0

Notes
-----
- Requires artifacts: out/diaphragms.json and out/nodes.json
- Uses constraints('Transformation') because diaphragms create MPCs
- Units are your model units; choose P accordingly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# OpenSeesPy imports
from openseespy.opensees import (
    wipe, getNodeTags, getEleTags, nodeDisp, timeSeries, pattern, load,
    constraints, numberer, system, test, algorithm, integrator, analysis, analyze,
    reactions, nodeReaction
)

# Keep consistent with repo config if present
try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = "out"


# -------------------------
# Utility / IO
# -------------------------
def _load_explicit_module(path: str):
    spec = importlib.util.spec_from_file_location("explicit_model_generated", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load explicit model from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# -------------------------
# Selection helpers
# -------------------------
def _select_diaphragm(
    diaphragms: List[Dict[str, Any]],
    nodes_map: Dict[int, Dict[str, float]],
    story: Optional[str],
    di_index: Optional[int],
) -> Dict[str, Any]:
    """
    Pick a diaphragm record:
      - If story provided: match by rec['story'] (case-sensitive exact)
      - elif di_index provided: sort by master Z (desc) and pick that index
      - else: pick the highest-Z master (top diaphragm)
    """
    if not diaphragms:
        raise ValueError("No diaphragms found in out/diaphragms.json")

    # Filter to only those whose master exists in nodes_map
    ds = [d for d in diaphragms if _as_int(d.get("master")) in nodes_map]
    if not ds:
        raise ValueError("Diaphragms present, but none of their master nodes exist in nodes.json.")

    if story is not None:
        for d in ds:
            if d.get("story") == story:
                return d
        raise ValueError(f"Requested story '{story}' not found among diaphragms.")

    # Build list with z for sorting
    with_z: List[Tuple[float, Dict[str, Any]]] = []
    for d in ds:
        m = _as_int(d.get("master"))
        z = _as_float(nodes_map[m]["z"])
        with_z.append((z, d))
    # Sort by Z descending (top first)
    with_z.sort(key=lambda t: t[0], reverse=True)

    if di_index is None:
        return with_z[0][1]
    if di_index < 0 or di_index >= len(with_z):
        raise ValueError(f"di index {di_index} out of range [0,{len(with_z)-1}]")
    return with_z[di_index][1]


def _nodes_map_from_nodes_json(nodes_json: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for rec in nodes_json.get("nodes", []):
        tag = _as_int(rec.get("tag"))
        out[tag] = {"x": _as_float(rec.get("x")), "y": _as_float(rec.get("y")), "z": _as_float(rec.get("z"))}
    return out


# -------------------------
# Core Probe
# -------------------------
def run_probe(
    explicit_path: str,
    artifacts_dir: str,
    story: Optional[str],
    di_index: Optional[int],
    direction: str,
    P: float,
    steps: int,
    tol: float,
    iters: int,
) -> Dict[str, Any]:
    """
    Build the explicit model, load a diaphragm master, apply a point load,
    and run a small static analysis. Returns a report dict.
    """
    if not os.path.exists(explicit_path):
        raise FileNotFoundError(f"Explicit model not found at: {explicit_path}")

    # Load artifacts required to pick the master node
    dpath = os.path.join(artifacts_dir, "diaphragms.json")
    npath = os.path.join(artifacts_dir, "nodes.json")
    if not os.path.exists(dpath):
        raise FileNotFoundError(f"Missing {dpath}")
    if not os.path.exists(npath):
        raise FileNotFoundError(f"Missing {npath}")

    dj = _load_json(dpath)
    nj = _load_json(npath)

    nodes_map = _nodes_map_from_nodes_json(nj)
    ds: List[Dict[str, Any]] = dj.get("diaphragms", [])
    drec = _select_diaphragm(ds, nodes_map, story, di_index)

    master = _as_int(drec.get("master"))
    mxyz = nodes_map[master]

    # Direction mapping
    dir_up = direction.upper()
    dof_map = {"X": 1, "Y": 2, "Z": 3}
    if dir_up not in dof_map:
        raise ValueError("direction must be one of X, Y, Z")
    dof = dof_map[dir_up]

    # Build domain from explicit model
    mod = _load_explicit_module(explicit_path)
    wipe()
    mod.build_model()

    # Analysis stack (Transformation for MPCs)
    constraints("Transformation")
    numberer("RCM")
    system("BandGeneral")
    test("NormDispIncr", tol, iters)
    algorithm("Newton")

    # Time series, pattern, and load at the master node
    timeSeries("Linear", 1)
    pattern("Plain", 1, 1)
    Fx = Fy = Fz = Mx = My = Mz = 0.0
    if dof == 1:
        Fx = P
    elif dof == 2:
        Fy = P
    else:
        Fz = P
    load(master, Fx, Fy, Fz, Mx, My, Mz)

    # Integrator & analysis
    dlam = 1.0 / max(1, steps)  # total factor = 1.0
    integrator("LoadControl", dlam)
    analysis("Static")

    # Solve
    rc = analyze(steps)

    # Displacements at master
    ux = nodeDisp(master, 1)
    uy = nodeDisp(master, 2)
    uz = nodeDisp(master, 3)
    rz = nodeDisp(master, 6)  # rotation about Z can matter for diaphragm

    # Reactions/equilibrium (sum only in loaded direction)
    eq_dir = dof  # 1=X, 2=Y, 3=Z
    sum_react = 0.0
    try:
        # Compute reactions and sum at support nodes
        reactions()
        # supports.json can give exact support nodes; falling back to "all nodes" is expensive,
        # so try loading supports.json if present:
        spath = os.path.join(artifacts_dir, "supports.json")
        support_nodes: List[int] = []
        if os.path.exists(spath):
            sj = _load_json(spath)
            for ent in sj.get("applied", []):
                support_nodes.append(_as_int(ent.get("node")))
        else:
            # Fallback: try all nodes (slower but robust)
            support_nodes = [int(t) for t in (getNodeTags() or [])]

        for nd in support_nodes:
            try:
                sum_react += float(nodeReaction(nd, eq_dir))
            except Exception:
                # Not a supported dof for that node or no reaction available
                pass
    except Exception:
        # Reactions not available; keep as 0 and mark in report
        pass

    # Report
    report: Dict[str, Any] = {
        "explicit_path": explicit_path,
        "artifacts_dir": artifacts_dir,
        "story": drec.get("story"),
        "selected_master": master,
        "master_xyz": mxyz,
        "direction": dir_up,
        "P": P,
        "steps": steps,
        "return_code": rc,
        "disp": {"ux": ux, "uy": uy, "uz": uz, "rz": rz},
        "sum_reaction_dir": sum_react,
    }

    # Basic evaluation
    stable = (rc == 0) and all(math.isfinite(v) for v in [ux, uy, uz, rz])
    report["stable"] = bool(stable)
    # Equilibrium check (sign depends on convention; compare magnitudes)
    report["equilibrium_ok"] = (abs(abs(sum_react) - abs(P)) / max(1.0, abs(P))) < 1e-2

    return report


# -------------------------
# CLI
# -------------------------
def _print_report(rep: Dict[str, Any]) -> None:
    print("\n=== Explicit Static Diaphragm Probe ===")
    print(f"Explicit file : {rep['explicit_path']}")
    print(f"Artifacts dir : {rep['artifacts_dir']}")
    print(f"Target story  : {rep['story']}")
    print(f"Master node   : {rep['selected_master']} @ {rep['master_xyz']}")
    print(f"Load          : {rep['P']} in +{rep['direction']} over {rep['steps']} steps")
    print(f"Return code   : {rep['return_code']} (0 means success)")
    d = rep["disp"]
    print(f"Displacements : Ux={d['ux']:.6g}, Uy={d['uy']:.6g}, Uz={d['uz']:.6g}, Rz={d['rz']:.6g}")
    print(f"Σ Reactions   : {rep['sum_reaction_dir']:.6g} (along {rep['direction']})")
    print(f"Stable        : {'YES' if rep['stable'] else 'NO'}")
    print(f"Equilibrium   : {'OK' if rep.get('equilibrium_ok') else 'WARN'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a point load to a diaphragm master node and run a static probe.")
    parser.add_argument("--explicit", default=None, help="Path to explicit_model.py (default: out/explicit_model.py)")
    parser.add_argument("--out", dest="artifacts_dir", default=OUT_DIR, help="Artifacts dir (default from config or 'out')")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--story", default=None, help="Story name to target (exact match)")
    group.add_argument("--di", type=int, default=None, help="Diaphragm index top->bottom (0-based)")
    parser.add_argument("--dir", default="X", help="Load direction: X, Y, or Z (default: X)")
    parser.add_argument("--P", type=float, default=1.0, help="Load magnitude in model units (default: 1.0)")
    parser.add_argument("--steps", type=int, default=10, help="Number of load steps (default: 10)")
    parser.add_argument("--tol", type=float, default=1e-8, help="Solver tolerance (default: 1e-8)")
    parser.add_argument("--iters", type=int, default=20, help="Max iterations per step (default: 20)")
    args = parser.parse_args()

    explicit_path = args.explicit or os.path.join(OUT_DIR, "explicit_model.py")
    rep = run_probe(
        explicit_path=explicit_path,
        artifacts_dir=args.artifacts_dir,
        story=args.story,
        di_index=args.di,
        direction=args.dir,
        P=args.P,
        steps=args.steps,
        tol=args.tol,
        iters=args.iters,
    )
    _print_report(rep)


if __name__ == "__main__":
    main()
