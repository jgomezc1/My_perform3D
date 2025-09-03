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

Contracts (artifacts consumed):
  - out/nodes.json
  - out/supports.json
  - out/diaphragms.json
  - out/columns.json
  - out/beams.json

What this file guarantees
-------------------------
1) Every transformation tag referenced by any element is **defined exactly once**
   via `geomTransf('Linear', tag, ...)`.
2) If an element is missing a proper per-element `transf_tag` (or has a placeholder),
   we **derive a deterministic per-element tag**:
      BEAM   → 1000000000 + element_tag
      COLUMN → 1100000000 + element_tag
3) Nonlinear overrides (via --nonlinear) apply `forceBeamColumn` and still get
   a valid transform emitted.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Set


# Config hook
try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = "out"


# -----------------------
# Helpers
# -----------------------
def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


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
# Nonlinear overrides
# -----------------------
class HingeSet:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.name: str = str(data.get("name", "HingeSet"))
        m = str(data.get("mode", "")).strip().lower()
        self.mode: str = m if m in ("emit", "use_existing") else "emit"

        self.elastic: Dict[str, Any] = dict(data.get("elastic") or {})
        self.matMy: Dict[str, Any] = dict(data.get("matMy") or {})
        self.matMz: Dict[str, Any] = dict(data.get("matMz") or {})

        self.sec_i_tag: int = _to_int(data.get("sec_i_tag"), 0)
        self.sec_j_tag: int = _to_int(data.get("sec_j_tag"), 0)
        self.beamInt_tag: int = _to_int(data.get("beamInt_tag"), 0)
        self.hingeLength: float = _to_float(data.get("hingeLength"), 0.0)

        self.elastic_tag: int = _to_int(data.get("elastic_tag"), _to_int(self.elastic.get("tag"), 0))

        # Infer mode if unspecified
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
        if self.mode == "emit" and not self._has_full_emit_payload():
            reasons.append(f"[hinge_set:{self.name}] mode=emit but elastic/matMy/matMz incomplete.")
            return False
        return True

    def will_emit_defs(self) -> bool:
        return self.mode == "emit"


class NLTarget:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.kind: str = str(data.get("kind", "")).upper().strip()  # "COLUMN", "BEAM", or ""(match all)
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
        # if neither filter provided, match-all for provided kind
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

        for hs in data.get("hinge_sets", []):
            obj = HingeSet(hs)
            reasons: List[str] = []
            if obj.valid(reasons):
                out.hinge_sets[obj.name] = obj
            else:
                out._diagnostics.extend(reasons)

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
# Emit explicit python file
# -----------------------
def _emit_header(lines: List[str], ndm: int, ndf: int) -> None:
    lines.append("# -*- coding: utf-8 -*-")
    lines.append('"""Generated by generate_explicit_model.py — DO NOT EDIT BY HAND."""')
    lines.append("from openseespy.opensees import *  # noqa")
    lines.append("")
    lines.append("def build_model(ndm: int = 3, ndf: int = 6) -> None:")
    lines.append("    wipe()")
    lines.append('    model(\"basic\", \"-ndm\", ndm, \"-ndf\", ndf)')
    lines.append("")


def _emit_nodes(lines: List[str], nodes_json: Dict[str, Any]) -> None:
    nodes = nodes_json.get("nodes") or []
    if not nodes:
        lines.append("    # [nodes] No nodes.json; nothing to create.")
        return
    lines.append("    # --- Nodes ---")
    for n in nodes:
        tag = _to_int(n.get("tag"))
        x, y, z = _to_float(n.get("x")), _to_float(n.get("y")), _to_float(n.get("z"))
        lines.append(f"    node({tag}, {x:.9g}, {y:.9g}, {z:.9g})")
    lines.append(f"    # [nodes] Created {len(nodes)} node(s).")
    lines.append("")


def _emit_supports(lines: List[str], sup_json: Dict[str, Any]) -> None:
    recs = sup_json.get("applied") or []
    if not recs:
        return
    lines.append("    # --- Supports (fix) ---")
    for r in recs:
        tag = _to_int(r.get("node"))
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
        master = _to_int(d.get("master"))
        mass = d.get("mass") or {}
        fix  = d.get("fix") or {}
        slaves = [ _to_int(s) for s in (d.get("slaves") or []) ]
        M  = _to_float(mass.get("M"))
        Izz = _to_float(mass.get("Izz"))
        lines.append(f"    mass({master}, {M:.9g}, {M:.9g}, 0.0, 0.0, 0.0, {Izz:.9g})")
        ux = int(fix.get("ux", 0)); uy = int(fix.get("uy", 0)); uz = int(fix.get("uz", 1))
        rx = int(fix.get("rx", 1)); ry = int(fix.get("ry", 1)); rz = int(fix.get("rz", 0))
        lines.append(f"    fix({master}, {ux}, {uy}, {uz}, {rx}, {ry}, {rz})")
        if slaves:
            s_list = ", ".join(str(s) for s in slaves)
            lines.append(f"    rigidDiaphragm(3, {master}, {s_list})")
    lines.append(f"    # [diaphragms] Created {len(recs)} diaphragm constraints.")
    lines.append("")


def _emit_nonlinear_defs(lines: List[str], ov: NLOverrides) -> None:
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
        lines.append(
            f"    section('Elastic', {int(e['tag'])}, "
            f"{_to_float(e['E']):.9g}, {_to_float(e['A']):.9g}, {_to_float(e['Iz']):.9g}, "
            f"{_to_float(e['Iy']):.9g}, {_to_float(e['G']):.9g}, {_to_float(e['J']):.9g})"
        )
        for label, m in (("My", hs.matMy), ("Mz", hs.matMz)):
            matTag = _to_int(m.get("tag"))
            My = _to_float(m.get("My")); theta = _to_float(m.get("theta")); Lp = _to_float(m.get("Lp"))
            Ecurv = My / (theta / Lp) if theta > 0.0 and Lp > 0.0 else 0.0
            b = _to_float(m.get("b")); R0 = _to_float(m.get("R0")); cR1 = _to_float(m.get("cR1")); cR2 = _to_float(m.get("cR2"))
            a1 = _to_float(m.get("a1")); a2 = _to_float(m.get("a2")); a3 = _to_float(m.get("a3")); a4 = _to_float(m.get("a4"))
            lines.append(
                f"    uniaxialMaterial('Steel02', {matTag}, {My:.9g}, {Ecurv:.9g}, {b:.9g}, "
                f"{R0:.9g}, {cR1:.9g}, {cR2:.9g}, {a1:.9g}, {a2:.9g}, {a3:.9g}, {a4:.9g})"
            )
        lines.append(
            f"    section('Aggregator', {hs.sec_i_tag}, "
            f"{int(hs.matMy['tag'])}, 'My', {int(hs.matMz['tag'])}, 'Mz', '-section', {int(e['tag'])})"
        )
        lines.append(
            f"    section('Aggregator', {hs.sec_j_tag}, "
            f"{int(hs.matMy['tag'])}, 'My', {int(hs.matMz['tag'])}, 'Mz', '-section', {int(e['tag'])})"
        )
        lines.append(
            f"    beamIntegration('HingeEndpoint', {hs.beamInt_tag}, "
            f"{hs.sec_i_tag}, {hs.hingeLength:.9g}, {hs.sec_j_tag}, {hs.hingeLength:.9g}, {int(e['tag'])})"
        )
        lines.append(f"    # [hinge_set] {hs.name} emitted (elastic={int(e['tag'])}, int={hs.beamInt_tag})")
    lines.append("")


# Track which transformation tags we already emitted
def _emit_geom_if_needed(lines: List[str], tr: int, kind: str, emitted: Set[int]) -> None:
    if tr in emitted:
        return
    if kind.upper() == "COLUMN":
        lines.append(f"    geomTransf('Linear', {int(tr)}, 1, 0, 0)")
    else:
        lines.append(f"    geomTransf('Linear', {int(tr)}, 0, 0, 1)")
    emitted.add(int(tr))


def _derive_transf_tag(kind: str, ele_tag: int, existing_tag: int) -> int:
    """
    Ensure a per-element transformation tag even if JSON had a placeholder or 0.
    - For BEAM:   placeholder 0/222 → 1000000000 + ele_tag
    - For COLUMN: placeholder 0/111 → 1100000000 + ele_tag
    Otherwise return existing_tag.
    """
    if kind.upper() == "BEAM":
        if existing_tag in (0, 222):
            return 1000000000 + int(ele_tag)
    else:  # COLUMN
        if existing_tag in (0, 111):
            return 1100000000 + int(ele_tag)
    return int(existing_tag)


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
        transf_tag_raw = _to_int(c.get("transf_tag")) or 111
        tr = _derive_transf_tag("COLUMN", tag, transf_tag_raw)
        A = _to_float(c.get("A")); E = _to_float(c.get("E")); G = _to_float(c.get("G"))
        J = _to_float(c.get("J")); Iy = _to_float(c.get("Iy")); Iz = _to_float(c.get("Iz"))
        line_name = str(c.get("line", "?"))

        picked = ov.find("COLUMN", tag, line_name)
        if picked:
            hs, tgt = picked
            # An override may provide its own transf_tag; still derive a per-element default if it is 0.
            tr_override = _to_int(tgt.transf_tag) or tr
            tr = _derive_transf_tag("COLUMN", tag, tr_override)
            _emit_geom_if_needed(lines, tr, "COLUMN", tr_emitted)
            lines.append(f"    # [nl] COLUMN tag {tag} ← hinge_set '{hs.name}'")
            lines.append(f"    element('forceBeamColumn', {tag}, {i_node}, {j_node}, {tr}, {hs.beamInt_tag})")
            counters["nl_columns"] += 1
        else:
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
        transf_tag_raw = _to_int(b.get("transf_tag")) or 222
        tr = _derive_transf_tag("BEAM", tag, transf_tag_raw)
        A = _to_float(b.get("A")); E = _to_float(b.get("E")); G = _to_float(b.get("G"))
        J = _to_float(b.get("J")); Iy = _to_float(b.get("Iy")); Iz = _to_float(b.get("Iz"))
        line_name = str(b.get("line", "?"))

        picked = ov.find("BEAM", tag, line_name)
        if picked:
            hs, tgt = picked
            tr_override = _to_int(tgt.transf_tag) or tr
            tr = _derive_transf_tag("BEAM", tag, tr_override)
            _emit_geom_if_needed(lines, tr, "BEAM", tr_emitted)
            lines.append(f"    # [nl] BEAM tag {tag} ← hinge_set '{hs.name}'")
            lines.append(f"    element('forceBeamColumn', {tag}, {i_node}, {j_node}, {tr}, {hs.beamInt_tag})")
            counters["nl_beams"] += 1
        else:
            _emit_geom_if_needed(lines, tr, "BEAM", tr_emitted)
            lines.append(f"    element('elasticBeamColumn', {tag}, {i_node}, {j_node}, "
                         f"{A:.9g}, {E:.9g}, {G:.9g}, {J:.9g}, {Iy:.9g}, {Iz:.9g}, {tr})")
            counters["el_beams"] += 1
    lines.append(f"    # [beams] Created {len(bs)} beams.")
    lines.append("")


def _emit_header_and_defs(lines: List[str], ndm: int, ndf: int, ov: NLOverrides) -> None:
    _emit_header(lines, ndm, ndf)
    # Nonlinear sets that need emission
    _emit_nonlinear_defs(lines, ov)


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
    lines: List[str] = []
    _emit_header_and_defs(lines, ndm, ndf, ov)

    nodes_json = _read_json(nodes_path)
    sup_json   = _read_json(supports_path)
    d_json     = _read_json(diaph_path)
    cols_json  = _read_json(cols_path)
    beams_json = _read_json(beams_path)

    # Nodes / supports / diaphragms first
    _emit_nodes(lines, nodes_json)
    _emit_supports(lines, sup_json)
    _emit_diaphragms(lines, d_json)

    # Track emitted transforms
    tr_emitted: Set[int] = set()

    # Counters
    counters = {"nl_beams": 0, "nl_columns": 0, "el_beams": 0, "el_columns": 0}

    # Elements (with per-tag transforms)
    _emit_columns(lines, cols_json, ov, tr_emitted, counters)
    _emit_beams(lines, beams_json, ov, tr_emitted, counters)

    _emit_footer(lines, counters, ov.diagnostics())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[explicit] Wrote {out_path} (ndm={ndm}, ndf={ndf})")
    if tr_emitted:
        sample = list(sorted(tr_emitted))[:8]
        print(f"[explicit] Emitted {len(tr_emitted)} geomTransf tag(s). sample={sample}")


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
