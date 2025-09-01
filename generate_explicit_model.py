# -*- coding: utf-8 -*-
"""
generate_explicit_model.py

Generates a standalone OpenSeesPy model script from Phase-2 artifacts.

Outputs (by default):
  - out/explicit_model.py    (legacy explicit file used by viewer/checkers)

New:
  - --pullover               → write out/model.py (PullOver-ready)
  - --nonlinear path.json    → optional overrides to emit forceBeamColumn members
                               with hinge aggregators and HingeEndpoint integration.

Artifacts consumed (contracts; unchanged):
  - out/nodes.json           (from emit_nodes.py)
  - out/supports.json        (from supports.py)
  - out/diaphragms.json      (from diaphragms.py)
  - out/columns.json         (from columns.py)
  - out/beams.json           (from beams.py)

Override JSON (optional, user-authored)
---------------------------------------
Schema (minimal; example):

{
  "hinge_sets": [
    {
      "name": "ColHingeV1",
      "elastic": { "tag": 4444, "E": 2.5e10, "A": 0.49, "Iy": 0.01, "Iz": 0.01, "G": 1.04e10, "J": 0.02 },
      "matMy":   { "tag": 6,  "My": 1.0e6, "theta": 0.005, "Lp": 0.10, "b": 0.01, "R0": 20, "cR1": 0.925, "cR2": 0.15,
                   "a1": 0.01, "a2": 1.0, "a3": 0.01, "a4": 1.0 },
      "matMz":   { "tag": 7,  "My": 1.0e6, "theta": 0.005, "Lp": 0.10, "b": 0.01, "R0": 20, "cR1": 0.925, "cR2": 0.15,
                   "a1": 0.01, "a2": 1.0, "a3": 0.01, "a4": 1.0 },
      "sec_i_tag": 40,
      "sec_j_tag": 41,
      "beamInt_tag": 33,
      "hingeLength": 0.20
    }
  ],
  "elements": [
    { "kind": "COLUMN", "by": { "tags": [36716140] }, "use_set": "ColHingeV1", "transf_tag": 111, "eleType": "forceBeamColumn" }
    // You may also target by line names:
    // { "kind": "BEAM", "by": { "lines": ["B1","B2"] }, "use_set": "ColHingeV1", "transf_tag": 222, "eleType": "forceBeamColumn" }
  ]
}

Rationale & integration
-----------------------
- We keep current elastic emission as default (your ETABS model is linear) and offer an
  opt-in path to elevate a subset to `forceBeamColumn` using your hinge recipe (identical
  to your PullOver sample) without touching artifacts themselves.
- GeomTransf tags remain: 111 = columns (x-axis), 222 = beams (z-axis).
- IMPORTANT: builder creation now uses the correct OpenSees flags:
  model('basic', '-ndm', ndm, '-ndf', ndf)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Set


# Config hooks (kept identical to repo)
try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = "out"


# -----------------------
# Helpers to read artifacts
# -----------------------
def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _exists_and_has(seq: Optional[List[Any]]) -> bool:
    return isinstance(seq, list) and len(seq) > 0


# -----------------------
# Override handling
# -----------------------
class HingeSet:
    """Parsed hinge set, enough to emit mats/sections/beamIntegration exactly once."""
    def __init__(self, data: Dict[str, Any]) -> None:
        self.name: str = str(data.get("name", "HingeSet"))
        self.elastic: Dict[str, Any] = dict(data.get("elastic") or {})
        self.matMy: Dict[str, Any] = dict(data.get("matMy") or {})
        self.matMz: Dict[str, Any] = dict(data.get("matMz") or {})
        self.sec_i_tag: int = int(data.get("sec_i_tag", 0))
        self.sec_j_tag: int = int(data.get("sec_j_tag", 0))
        self.beamInt_tag: int = int(data.get("beamInt_tag", 0))
        self.hingeLength: float = float(data.get("hingeLength", 0.0))

    def valid(self) -> bool:
        needed = ("tag", "E", "A", "Iy", "Iz", "G", "J")
        ok_elastic = all(k in self.elastic for k in needed)
        ok_m1 = all(k in self.matMy for k in ("tag", "My", "theta", "Lp", "b", "R0", "cR1", "cR2", "a1", "a2", "a3", "a4"))
        ok_m2 = all(k in self.matMz for k in ("tag", "My", "theta", "Lp", "b", "R0", "cR1", "cR2", "a1", "a2", "a3", "a4"))
        ok_tags = self.sec_i_tag > 0 and self.sec_j_tag > 0 and self.beamInt_tag > 0 and self.hingeLength > 0.0
        return ok_elastic and ok_m1 and ok_m2 and ok_tags


class NLTarget:
    """Which elements to apply a hinge set to, and how to emit them."""
    def __init__(self, data: Dict[str, Any]) -> None:
        self.kind: str = str(data.get("kind", "")).upper()  # "COLUMN" or "BEAM"
        by: Dict[str, Any] = dict(data.get("by") or {})
        self.by_tags: Set[int] = {int(t) for t in (by.get("tags") or [])}
        self.by_lines: Set[str] = {str(s) for s in (by.get("lines") or [])}
        self.use_set: str = str(data.get("use_set", ""))
        self.eleType: str = str(data.get("eleType", "forceBeamColumn"))
        self.transf_tag: int = int(data.get("transf_tag", 0))  # typically 111 or 222

    def matches(self, kind: str, tag: int, line: str) -> bool:
        if self.kind and self.kind != str(kind).upper():
            return False
        if self.by_tags and tag in self.by_tags:
            return True
        if self.by_lines and (line in self.by_lines):
            return True
        return False


class NLOverrides:
    def __init__(self) -> None:
        self.hinge_sets: Dict[str, HingeSet] = {}
        self.targets: List[NLTarget] = []

    @staticmethod
    def load(path: Optional[str]) -> "NLOverrides":
        out = NLOverrides()
        if not path:
            return out
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            return out

        for hs in data.get("hinge_sets", []):
            obj = HingeSet(hs)
            if obj.valid():
                out.hinge_sets[obj.name] = obj
        for tg in data.get("elements", []):
            out.targets.append(NLTarget(tg))
        return out

    def find(self, kind: str, tag: int, line: str) -> Optional[Tuple[HingeSet, NLTarget]]:
        for t in self.targets:
            if t.matches(kind, tag, line):
                hs = self.hinge_sets.get(t.use_set)
                if hs and hs.valid():
                    return hs, t
        return None

    def any(self) -> bool:
        return bool(self.hinge_sets) and bool(self.targets)


# -----------------------
# Emit explicit python file (string builder)
# -----------------------
def _emit_header(lines: List[str], ndm: int, ndf: int) -> None:
    lines.append("# -*- coding: utf-8 -*-")
    lines.append('"""Generated by generate_explicit_model.py — DO NOT EDIT BY HAND."""')
    lines.append("from openseespy.opensees import *  # noqa")
    lines.append("")
    lines.append("def build_model(ndm: int = 3, ndf: int = 6) -> None:")
    lines.append("    wipe()")
    # FIX: Use OpenSees flags for builder creation
    lines.append('    model("basic", "-ndm", ndm, "-ndf", ndf)')
    lines.append("")


def _emit_nodes(lines: List[str], nodes_json: Dict[str, Any]) -> None:
    nodes = nodes_json.get("nodes") or []
    if not nodes:
        lines.append("    # [nodes] No nodes.json; nothing to create.")
        return
    lines.append("    # --- Nodes ---")
    for n in nodes:
        tag = int(n["tag"])
        x, y, z = float(n["x"]), float(n["y"]), float(n["z"])
        lines.append(f"    node({tag}, {x:.9g}, {y:.9g}, {z:.9g})")
    lines.append(f"    # [nodes] Created {len(nodes)} node(s).")
    lines.append("")


def _emit_supports(lines: List[str], sup_json: Dict[str, Any]) -> None:
    recs = sup_json.get("applied") or []
    if not recs:
        return
    lines.append("    # --- Supports (fix) ---")
    for r in recs:
        tag = int(r["node"])
        m = [int(v) for v in r.get("mask", [0, 0, 0, 0, 0, 0])]
        m = (m + [0, 0, 0, 0, 0, 0])[:6]
        lines.append(f"    fix({tag}, {m[0]}, {m[1]}, {m[2]}, {m[3]}, {m[4]}, {m[5]})")
    lines.append(f"    # [supports] Applied {len(recs)} fixities.")
    lines.append("")


def _emit_diaphragms(lines: List[str], dg_json: Dict[str, Any]) -> None:
    recs = dg_json.get("diaphragms") or []
    if not recs:
        return
    lines.append("    # --- Rigid Diaphragms, master mass/fix ---")
    for d in recs:
        master = int(d["master"])
        mass = d.get("mass") or {}
        fix  = d.get("fix") or {}
        slaves = [int(s) for s in (d.get("slaves") or [])]
        # mass()
        M  = float(mass.get("M", 0.0))
        Izz = float(mass.get("Izz", 0.0))
        lines.append(f"    mass({master}, {M:.9g}, {M:.9g}, 0.0, 0.0, 0.0, {Izz:.9g})")
        # fix(master, 0,0,1,1,1,0)
        ux = int(fix.get("ux", 0)); uy = int(fix.get("uy", 0)); uz = int(fix.get("uz", 1))
        rx = int(fix.get("rx", 1)); ry = int(fix.get("ry", 1)); rz = int(fix.get("rz", 0))
        lines.append(f"    fix({master}, {ux}, {uy}, {uz}, {rx}, {ry}, {rz})")
        # rigidDiaphragm(3, master, *slaves)
        if slaves:
            s_list = ", ".join(str(s) for s in slaves)
            lines.append(f"    rigidDiaphragm(3, {master}, {s_list})")
    lines.append(f"    # [diaphragms] Created {len(recs)} diaphragm constraints.")
    lines.append("")


def _emit_transforms(lines: List[str], has_cols: bool, has_beams: bool) -> None:
    if not (has_cols or has_beams):
        return
    lines.append("    # --- Geometric transformations ---")
    if has_cols:
        lines.append("    geomTransf('Linear', 111, 1, 0, 0)  # Columns (X local)")
    if has_beams:
        lines.append("    geomTransf('Linear', 222, 0, 0, 1)  # Beams   (Z local)")
    lines.append("")


def _emit_nonlinear_defs(lines: List[str], ov: NLOverrides) -> None:
    """Emit unique hinge sets once (materials, aggregators, beamIntegration)."""
    if not ov.any():
        return
    lines.append("    # --- Nonlinear hinge sets (from --nonlinear) ---")
    for name, hs in ov.hinge_sets.items():
        e = hs.elastic
        # Elastic section wrapper (OpenSees 3D Elastic section: E,A,Iz,Iy,G,J)
        lines.append(f"    section('Elastic', {int(e['tag'])}, {float(e['E']):.9g}, {float(e['A']):.9g}, "
                     f"{float(e['Iz']):.9g}, {float(e['Iy']):.9g}, {float(e['G']):.9g}, {float(e['J']):.9g})")

        # Uniaxial materials (Steel02) for My and Mz using curvature slope E_curv = M / (theta/Lp)
        for label, m in (("My", hs.matMy), ("Mz", hs.matMz)):
            matTag = int(m["tag"])
            My = float(m["My"]); theta = float(m["theta"]); Lp = float(m["Lp"])
            Ecurv = My / (theta / Lp) if theta > 0.0 and Lp > 0.0 else 0.0
            b = float(m["b"]); R0 = float(m["R0"]); cR1 = float(m["cR1"]); cR2 = float(m["cR2"])
            a1 = float(m["a1"]); a2 = float(m["a2"]); a3 = float(m["a3"]); a4 = float(m["a4"])
            lines.append(f"    uniaxialMaterial('Steel02', {matTag}, {My:.9g}, {Ecurv:.9g}, {b:.9g}, "
                         f"{R0:.9g}, {cR1:.9g}, {cR2:.9g}, {a1:.9g}, {a2:.9g}, {a3:.9g}, {a4:.9g})")

        # Aggregator sections for end i and end j
        lines.append(f"    section('Aggregator', {hs.sec_i_tag}, "
                     f"{int(hs.matMy['tag'])}, 'My', {int(hs.matMz['tag'])}, 'Mz', '-section', {int(e['tag'])})")
        lines.append(f"    section('Aggregator', {hs.sec_j_tag}, "
                     f"{int(hs.matMy['tag'])}, 'My', {int(hs.matMz['tag'])}, 'Mz', '-section', {int(e['tag'])})")

        # Beam integration (HingeEndpoint)
        lines.append(f"    beamIntegration('HingeEndpoint', {hs.beamInt_tag}, "
                     f"{hs.sec_i_tag}, {hs.hingeLength:.9g}, {hs.sec_j_tag}, {hs.hingeLength:.9g}, {int(e['tag'])})")

        lines.append(f"    # [hinge_set] {name} → elastic={int(e['tag'])}, int={hs.beamInt_tag}")
    lines.append("")


def _emit_columns(lines: List[str], cols_json: Dict[str, Any], ov: NLOverrides) -> None:
    cols = cols_json.get("columns") or []
    if not cols:
        return
    lines.append("    # --- Columns ---")
    for c in cols:
        tag = int(c["tag"])
        i_node = int(c["i_node"]); j_node = int(c["j_node"])
        transf_tag = int(c.get("transf_tag", 111))
        A = float(c.get("A", 0.0)); E = float(c.get("E", 0.0)); G = float(c.get("G", 0.0))
        J = float(c.get("J", 0.0)); Iy = float(c.get("Iy", 0.0)); Iz = float(c.get("Iz", 0.0))
        line_name = str(c.get("line", "?"))

        picked = ov.find("COLUMN", tag, line_name)
        if picked:
            hs, tgt = picked
            tr = tgt.transf_tag or transf_tag or 111
            lines.append(f"    element('forceBeamColumn', {tag}, {i_node}, {j_node}, {tr}, {hs.beamInt_tag})")
        else:
            lines.append(f"    element('elasticBeamColumn', {tag}, {i_node}, {j_node}, "
                         f"{A:.9g}, {E:.9g}, {G:.9g}, {J:.9g}, {Iy:.9g}, {Iz:.9g}, {transf_tag})")
    lines.append(f"    # [columns] Created {len(cols)} columns.")
    lines.append("")


def _emit_beams(lines: List[str], beams_json: Dict[str, Any], ov: NLOverrides) -> None:
    bs = beams_json.get("beams") or []
    if not bs:
        return
    lines.append("    # --- Beams ---")
    for b in bs:
        tag = int(b["tag"])
        i_node = int(b["i_node"]); j_node = int(b["j_node"])
        transf_tag = int(b.get("transf_tag", 222))
        A = float(b.get("A", 0.0)); E = float(b.get("E", 0.0)); G = float(b.get("G", 0.0))
        J = float(b.get("J", 0.0)); Iy = float(b.get("Iy", 0.0)); Iz = float(b.get("Iz", 0.0))
        line_name = str(b.get("line", "?"))

        picked = ov.find("BEAM", tag, line_name)
        if picked:
            hs, tgt = picked
            tr = tgt.transf_tag or transf_tag or 222
            lines.append(f"    element('forceBeamColumn', {tag}, {i_node}, {j_node}, {tr}, {hs.beamInt_tag})")
        else:
            lines.append(f"    element('elasticBeamColumn', {tag}, {i_node}, {j_node}, "
                         f"{A:.9g}, {E:.9g}, {G:.9g}, {J:.9g}, {Iy:.9g}, {Iz:.9g}, {transf_tag})")
    lines.append(f"    # [beams] Created {len(bs)} beams.")
    lines.append("")


def generate(out_dir: str,
             explicit_path: Optional[str],
             *,
             pullover: bool = False,
             nonlinear_path: Optional[str] = None,
             ndm: int = 3,
             ndf: int = 6) -> str:
    """Generate explicit python file and return its path."""
    out_dir = out_dir or "out"
    os.makedirs(out_dir, exist_ok=True)

    # Load artifacts
    nodes_json = _read_json(os.path.join(out_dir, "nodes.json"))
    sup_json   = _read_json(os.path.join(out_dir, "supports.json"))
    dg_json    = _read_json(os.path.join(out_dir, "diaphragms.json"))
    cols_json  = _read_json(os.path.join(out_dir, "columns.json"))
    beams_json = _read_json(os.path.join(out_dir, "beams.json"))

    has_cols = _exists_and_has(cols_json.get("columns"))
    has_beams = _exists_and_has(beams_json.get("beams"))

    # Load optional overrides
    ov = NLOverrides.load(nonlinear_path)

    # Decide output path
    target_path = explicit_path or (os.path.join(out_dir, "model.py") if pullover else os.path.join(out_dir, "explicit_model.py"))

    # Build code
    L: List[str] = []
    _emit_header(L, ndm, ndf)
    _emit_nodes(L, nodes_json)
    _emit_supports(L, sup_json)
    _emit_diaphragms(L, dg_json)
    _emit_transforms(L, has_cols, has_beams)
    if ov.any():
        _emit_nonlinear_defs(L, ov)
    _emit_columns(L, cols_json, ov)
    _emit_beams(L, beams_json, ov)
    # End function
    if len(L) == 0 or not L[-1].strip():
        pass
    else:
        L.append("")

    # Write file
    with open(target_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"[explicit] Wrote {target_path}")
    return target_path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate explicit OpenSees model from artifacts.")
    p.add_argument("--out", default=str(OUT_DIR), help="Artifacts folder (default: out)")
    p.add_argument("--explicit", default=None, help="Output path for explicit model file.")
    p.add_argument("--pullover", action="store_true", help="Emit PullOver-ready 'model.py' into --out.")
    p.add_argument("--nonlinear", default=None, help="Path to nonlinear_overrides.json (optional).")
    p.add_argument("--ndm", type=int, default=3)
    p.add_argument("--ndf", type=int, default=6)
    args = p.parse_args()

    generate(args.out, args.explicit, pullover=args.pullover, nonlinear_path=args.nonlinear, ndm=args.ndm, ndf=args.ndf)


if __name__ == "__main__":
    main()
