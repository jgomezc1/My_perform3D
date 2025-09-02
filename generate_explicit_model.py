# -*- coding: utf-8 -*-
"""
generate_explicit_model.py

Generates a standalone OpenSeesPy model script from Phase-2 artifacts.

Outputs (by default):
  - out/explicit_model.py    (legacy explicit file used by viewer/checkers)

Flags:
  - --pullover               → write out/model.py (PullOver-ready)
  - --nonlinear path.json    → optional overrides to emit forceBeamColumn members
                               with hinge aggregators and HingeEndpoint integration.

Artifacts consumed (contracts; unchanged):
  - out/nodes.json           (from emit_nodes.py)
  - out/supports.json        (from supports.py)
  - out/diaphragms.json      (from diaphragms.py)
  - out/columns.json         (from columns.py)
  - out/beams.json           (from beams.py)

What's new
----------
1) Always emits `geomTransf(...)` exactly once for every transformation tag used.
2) Nonlinear overrides are now schema-tolerant:
   - Mode A ("emit"): JSON provides elastic section + uniaxial materials; we emit them.
   - Mode B ("use_existing"): JSON provides ready-to-use tags (sec_i_tag, sec_j_tag, beamInt_tag, etc.),
     and we will NOT emit materials/sections/integration—just reference those tags.
3) Clear diagnostics: logs of matched targets and reasons when an override is ignored.
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


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


# -----------------------
# Override handling
# -----------------------
class HingeSet:
    """
    A hinge "set" can be provided in two modes:

    Mode A: "emit" (fully-specified)
      {
        "name": "RC_Default",
        "mode": "emit",  # optional; defaults to "emit" if elastic+matMy+matMz are present
        "elastic": {"tag": 9001, "E":..., "A":..., "Iy":..., "Iz":..., "G":..., "J":...},
        "matMy": {"tag": 9101, "My":..., "theta":..., "Lp":..., "b":..., "R0":..., "cR1":..., "cR2":..., "a1":..., "a2":..., "a3":..., "a4":...},
        "matMz": {"tag": 9102, ... same keys as matMy ...},
        "sec_i_tag": 9201,
        "sec_j_tag": 9202,
        "beamInt_tag": 9301,
        "hingeLength": 0.15
      }

    Mode B: "use_existing" (reference pre-defined tags; no materials/sections emitted)
      {
        "name": "RC_Predef",
        "mode": "use_existing",
        "elastic_tag": 7001,    # optional (if needed by your aggregator)
        "sec_i_tag": 7101,
        "sec_j_tag": 7102,
        "beamInt_tag": 7201,
        "hingeLength": 0.15
      }
    """
    def __init__(self, data: Dict[str, Any]) -> None:
        self.name: str = str(data.get("name", "HingeSet"))
        # Mode inference
        m = str(data.get("mode", "")).strip().lower()
        self.mode: str = m if m in ("emit", "use_existing") else "emit"

        # Full emit data
        self.elastic: Dict[str, Any] = dict(data.get("elastic") or {})
        self.matMy: Dict[str, Any] = dict(data.get("matMy") or {})
        self.matMz: Dict[str, Any] = dict(data.get("matMz") or {})

        # Tags (both modes use these)
        self.sec_i_tag: int = _to_int(data.get("sec_i_tag"), 0)
        self.sec_j_tag: int = _to_int(data.get("sec_j_tag"), 0)
        self.beamInt_tag: int = _to_int(data.get("beamInt_tag"), 0)
        self.hingeLength: float = _to_float(data.get("hingeLength"), 0.0)

        # For use_existing mode
        self.elastic_tag: int = _to_int(data.get("elastic_tag"), _to_int(self.elastic.get("tag"), 0))

        # If mode unspecified, infer from presence of elastic+matMy+matMz
        if m == "":
            if self.elastic and self.matMy and self.matMz:
                self.mode = "emit"
            else:
                self.mode = "use_existing"

    def _has_full_emit_payload(self) -> bool:
        need_el = all(k in self.elastic for k in ("tag", "E", "A", "Iy", "Iz", "G", "J"))
        need_my = all(k in self.matMy   for k in ("tag","My","theta","Lp","b","R0","cR1","cR2","a1","a2","a3","a4"))
        need_mz = all(k in self.matMz   for k in ("tag","My","theta","Lp","b","R0","cR1","cR2","a1","a2","a3","a4"))
        return need_el and need_my and need_mz

    def valid(self, reasons: List[str]) -> bool:
        ok_core = (self.sec_i_tag > 0 and self.sec_j_tag > 0 and self.beamInt_tag > 0 and self.hingeLength > 0.0)
        if not ok_core:
            reasons.append(f"[hinge_set:{self.name}] core tags/hingeLength missing or invalid.")
            return False
        if self.mode == "emit":
            if not self._has_full_emit_payload():
                reasons.append(f"[hinge_set:{self.name}] mode=emit but elastic/matMy/matMz incomplete.")
                return False
        # use_existing requires no payload checks
        return True

    def will_emit_defs(self) -> bool:
        return self.mode == "emit"


class NLTarget:
    """Which elements to apply a hinge set to, and how to emit them."""
    def __init__(self, data: Dict[str, Any]) -> None:
        self.kind: str = str(data.get("kind", "")).upper().strip()  # "COLUMN" or "BEAM" (optional; if omitted => both)
        by: Dict[str, Any] = dict(data.get("by") or {})
        self.by_tags: Set[int] = { _to_int(t) for t in (by.get("tags") or []) }
        self.by_lines: Set[str] = { str(s) for s in (by.get("lines") or []) }
        self.use_set: str = str(data.get("use_set", "")).strip()
        self.eleType: str = str(data.get("eleType", "forceBeamColumn")).strip()
        self.transf_tag: int = _to_int(data.get("transf_tag"), 0)

    def matches(self, kind: str, tag: int, line: str) -> bool:
        if self.kind and self.kind != str(kind).upper():
            return False
        if self.by_tags and tag in self.by_tags:
            return True
        if self.by_lines and (line in self.by_lines):
            return True
        # If neither filter is provided, treat as "match all of kind" (if kind supplied)
        return (self.kind != "")


class NLOverrides:
    def __init__(self) -> None:
        self.hinge_sets: Dict[str, HingeSet] = {}
        self.targets: List[NLTarget] = []
        self._diagnostics: List[str] = []

    @staticmethod
    def load(path: Optional[str]) -> "NLOverrides":
        out = NLOverrides()
        if not path:
            return out
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception as e:
            out._diagnostics.append(f"[nonlinear] Failed to read overrides file: {e}")
            return out

        # Hinge sets
        for hs in data.get("hinge_sets", []):
            obj = HingeSet(hs)
            reasons: List[str] = []
            if obj.valid(reasons):
                out.hinge_sets[obj.name] = obj
            else:
                out._diagnostics.extend(reasons)

        # Targets
        for tg in data.get("elements", []):
            out.targets.append(NLTarget(tg))

        if not out.hinge_sets:
            out._diagnostics.append("[nonlinear] No valid hinge_sets parsed.")
        if not out.targets:
            out._diagnostics.append("[nonlinear] No element targets found.")

        return out

    def find(self, kind: str, tag: int, line: str) -> Optional[Tuple[HingeSet, NLTarget]]:
        for t in self.targets:
            if t.matches(kind, tag, line):
                hs = self.hinge_sets.get(t.use_set)
                if hs:
                    return hs, t
        return None

    def any(self) -> bool:
        return bool(self.hinge_sets) and bool(self.targets)

    def diagnostics(self) -> List[str]:
        return list(self._diagnostics)


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
        x, y, z = _to_float(n["x"]), _to_float(n["y"]), _to_float(n["z"])
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
        M  = _to_float(mass.get("M"))
        Izz = _to_float(mass.get("Izz"))
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


def _emit_nonlinear_defs(lines: List[str], ov: NLOverrides) -> None:
    """Emit hinge sets that require emission (mode='emit')."""
    if not ov.any():
        return
    to_emit = [hs for hs in ov.hinge_sets.values() if hs.will_emit_defs()]
    if not to_emit:
        lines.append("    # [nonlinear] Using pre-defined hinge tags; no materials/sections emitted.")
        lines.append("")
        return

    lines.append("    # --- Nonlinear hinge sets (from --nonlinear) ---")
    for hs in to_emit:
        e = hs.elastic
        # Elastic section wrapper (OpenSees 3D Elastic section: E,A,Iz,Iy,G,J)
        lines.append(f"    section('Elastic', {int(e['tag'])}, {_to_float(e['E']):.9g}, {_to_float(e['A']):.9g}, "
                     f"{_to_float(e['Iz']):.9g}, {_to_float(e['Iy']):.9g}, {_to_float(e['G']):.9g}, {_to_float(e['J']):.9g})")

        # Uniaxial materials (Steel02) for My and Mz using curvature slope E_curv = M / (theta/Lp)
        for label, m in (("My", hs.matMy), ("Mz", hs.matMz)):
            matTag = _to_int(m.get("tag"))
            My = _to_float(m.get("My")); theta = _to_float(m.get("theta")); Lp = _to_float(m.get("Lp"))
            Ecurv = My / (theta / Lp) if theta > 0.0 and Lp > 0.0 else 0.0
            b = _to_float(m.get("b")); R0 = _to_float(m.get("R0")); cR1 = _to_float(m.get("cR1")); cR2 = _to_float(m.get("cR2"))
            a1 = _to_float(m.get("a1")); a2 = _to_float(m.get("a2")); a3 = _to_float(m.get("a3")); a4 = _to_float(m.get("a4"))
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

        lines.append(f"    # [hinge_set] {hs.name} emitted (elastic={int(e['tag'])}, int={hs.beamInt_tag})")
    lines.append("")


# --- NEW: transformation emission tracking/emitter ---
def _emit_geom_if_needed(lines: List[str], tr: int, kind: str, emitted: Set[int]) -> None:
    """
    Ensure a geomTransf is emitted once for the given transformation tag.
    Orientation vector is inferred from the element kind:
      - COLUMN → (1, 0, 0)
      - BEAM   → (0, 0, 1)
    Unknown kinds default to BEAM vector for safety.
    """
    if tr in emitted:
        return
    if kind.upper() == "COLUMN":
        lines.append(f"    geomTransf('Linear', {int(tr)}, 1, 0, 0)")
    else:
        # Default/BEAM
        lines.append(f"    geomTransf('Linear', {int(tr)}, 0, 0, 1)")
    emitted.add(int(tr))


def _emit_columns(lines: List[str], cols_json: Dict[str, Any],
                  ov: NLOverrides, tr_emitted: Set[int],
                  counters: Dict[str, int]) -> None:
    cols = cols_json.get("columns") or []
    if not cols:
        return
    lines.append("    # --- Columns ---")
    for c in cols:
        tag = _to_int(c.get("tag"))
        i_node = _to_int(c.get("i_node")); j_node = _to_int(c.get("j_node"))
        transf_tag = _to_int(c.get("transf_tag")) or 111
        A = _to_float(c.get("A")); E = _to_float(c.get("E")); G = _to_float(c.get("G"))
        J = _to_float(c.get("J")); Iy = _to_float(c.get("Iy")); Iz = _to_float(c.get("Iz"))
        line_name = str(c.get("line", "?"))

        picked = ov.find("COLUMN", tag, line_name)
        if picked:
            hs, tgt = picked
            tr = _to_int(tgt.transf_tag) or transf_tag or 111
            _emit_geom_if_needed(lines, tr, "COLUMN", tr_emitted)

            # Emit nonlinear element (forceBeamColumn)
            lines.append(f"    # [nl] COLUMN tag {tag} ← hinge_set '{hs.name}'")
            lines.append(f"    element('forceBeamColumn', {tag}, {i_node}, {j_node}, {tr}, {hs.beamInt_tag})")
            counters["nl_columns"] += 1
        else:
            tr = transf_tag or 111
            _emit_geom_if_needed(lines, tr, "COLUMN", tr_emitted)
            lines.append(f"    element('elasticBeamColumn', {tag}, {i_node}, {j_node}, "
                         f"{A:.9g}, {E:.9g}, {G:.9g}, {J:.9g}, {Iy:.9g}, {Iz:.9g}, {tr})")
            counters["el_columns"] += 1
    lines.append(f"    # [columns] Created {len(cols)} columns.")
    lines.append("")


def _emit_beams(lines: List[str], beams_json: Dict[str, Any],
                ov: NLOverrides, tr_emitted: Set[int],
                counters: Dict[str, int]) -> None:
    bs = beams_json.get("beams") or []
    if not bs:
        return
    lines.append("    # --- Beams ---")
    for b in bs:
        tag = _to_int(b.get("tag"))
        i_node = _to_int(b.get("i_node")); j_node = _to_int(b.get("j_node"))
        transf_tag = _to_int(b.get("transf_tag")) or 222
        A = _to_float(b.get("A")); E = _to_float(b.get("E")); G = _to_float(b.get("G"))
        J = _to_float(b.get("J")); Iy = _to_float(b.get("Iy")); Iz = _to_float(b.get("Iz"))
        line_name = str(b.get("line", "?"))

        picked = ov.find("BEAM", tag, line_name)
        if picked:
            hs, tgt = picked
            tr = _to_int(tgt.transf_tag) or transf_tag or 222
            _emit_geom_if_needed(lines, tr, "BEAM", tr_emitted)

            lines.append(f"    # [nl] BEAM tag {tag} ← hinge_set '{hs.name}'")
            lines.append(f"    element('forceBeamColumn', {tag}, {i_node}, {j_node}, {tr}, {hs.beamInt_tag})")
            counters["nl_beams"] += 1
        else:
            tr = transf_tag or 222
            _emit_geom_if_needed(lines, tr, "BEAM", tr_emitted)
            lines.append(f"    element('elasticBeamColumn', {tag}, {i_node}, {j_node}, "
                         f"{A:.9g}, {E:.9g}, {G:.9g}, {J:.9g}, {Iy:.9g}, {Iz:.9g}, {tr})")
            counters["el_beams"] += 1
    lines.append(f"    # [beams] Created {len(bs)} beams.")
    lines.append("")


def _emit_footer(lines: List[str], counters: Dict[str, int], diag: List[str]) -> None:
    if diag:
        lines.append("    # --- Nonlinear diagnostics ---")
        for d in diag:
            lines.append(f"    # {d}")
        lines.append("")
    lines.append(f"    # [summary] NL beams={counters['nl_beams']}, NL columns={counters['nl_columns']}, "
                 f"EL beams={counters['el_beams']}, EL columns={counters['el_columns']}")
    lines.append("    # --- done ---")
    lines.append("")


def _build_explicit(ndm: int, ndf: int,
                    out_path: str,
                    nodes_path: str, supports_path: str, diaph_path: str,
                    cols_path: str, beams_path: str,
                    ov: NLOverrides) -> None:
    """Assemble the script into a list of lines and write it."""
    lines: List[str] = []
    _emit_header(lines, ndm, ndf)

    nodes_json = _read_json(nodes_path)
    sup_json   = _read_json(supports_path)
    d_json     = _read_json(diaphragms_path := diaph_path)
    cols_json  = _read_json(cols_path)
    beams_json = _read_json(beams_path)

    _emit_nodes(lines, nodes_json)
    _emit_supports(lines, sup_json)
    _emit_diaphragms(lines, d_json)
    _emit_nonlinear_defs(lines, ov)

    # Track which transformation tags have been emitted
    tr_emitted: Set[int] = set()

    # Counters
    counters = {"nl_beams": 0, "nl_columns": 0, "el_beams": 0, "el_columns": 0}

    # Emit elements (and per-tag geomTransf as needed)
    _emit_columns(lines, cols_json, ov, tr_emitted, counters)
    _emit_beams(lines, beams_json, ov, tr_emitted, counters)

    _emit_footer(lines, counters, ov.diagnostics())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[explicit] Wrote {out_path} (ndm={ndm}, ndf={ndf})")
    if tr_emitted:
        print(f"[explicit] Emitted {len(tr_emitted)} geomTransf tag(s): sample={list(sorted(tr_emitted))[:8]}")
    print(f"[explicit] Summary: NL beams={counters['nl_beams']}, NL columns={counters['nl_columns']}, "
          f"EL beams={counters['el_beams']}, EL columns={counters['el_columns']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate explicit OpenSeesPy model from artifacts.")
    ap.add_argument("--pullover", action="store_true",
                    help="Write PullOver-ready script to out/model.py (default is out/explicit_model.py)")
    ap.add_argument("--nonlinear", type=str, default=None,
                    help="Path to nonlinear overrides JSON (hinge sets + element targets)")
    ap.add_argument("--ndm", type=int, default=3)
    ap.add_argument("--ndf", type=int, default=6)
    args = ap.parse_args()

    out_file = "model.py" if args.pullover else "explicit_model.py"
    out_path = os.path.join(OUT_DIR, out_file)

    nodes_path    = os.path.join(OUT_DIR, "nodes.json")
    supports_path = os.path.join(OUT_DIR, "supports.json")
    diaph_path    = os.path.join(OUT_DIR, "diaphragms.json")
    cols_path     = os.path.join(OUT_DIR, "columns.json")
    beams_path    = os.path.join(OUT_DIR, "beams.json")

    ov = NLOverrides.load(args.nonlinear)
    _build_explicit(args.ndm, args.ndf, out_path,
                    nodes_path, supports_path, diaph_path,
                    cols_path, beams_path,
                    ov)


if __name__ == "__main__":
    main()
