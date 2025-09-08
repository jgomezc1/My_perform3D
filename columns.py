"""
Create COLUMN elements as OpenSeesPy elasticBeamColumn members with support for
ETABS rigid end zones (LENGTHOFFI/LENGTHOFFJ when RIGIDZONE=1). Writes Phase-2
artifact `columns.json`.

Rules (retained)
----------------
- **find-next-lower-story**: For a COLUMN line at story S, connect its endpoint
  'i' at story S to endpoint 'j' at the next lower story K where **both** i and j
  appear in that story's active points. If none is found, skip.
- Enforce local-axis convention (configurable): by default **i = bottom, j = top**.

New in this version
-------------------
- Auto-initialize OpenSees model if ndm/ndf are zero (no wipe).
- Splits members into up to **three segments** (rigid I, deformable mid, rigid J).
- Creates deterministic **intermediate nodes** at the offset boundaries.
- Per-segment **geomTransf** (one per element) derived from the element tag.
- Emits richer `columns.json` records with a `segment` field.

OpenSeesPy signatures (exact):
    ops.geomTransf('Linear', transf_tag, 1, 0, 0)
    ops.element('elasticBeamColumn', tag, nI, nJ,
                A, E, G, J, Iy, Iz, transf_tag)
"""
from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple, Optional
import json
import os
import hashlib

import openseespy.opensees as ops  # unified ops namespace

# Optional config hooks
try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = "out"

# Convention: i = bottom, j = top (swap if needed)
try:
    from config import ENFORCE_COLUMN_I_AT_BOTTOM  # type: ignore
except Exception:
    ENFORCE_COLUMN_I_AT_BOTTOM = True

# Rigid end scale (A, Iy, Iz, J are multiplied by this for rigid segments)
try:
    from config import RIGID_END_SCALE  # type: ignore
except Exception:
    RIGID_END_SCALE = 1.0e6

# Prefer project tagging helpers if available
try:
    from tagging import element_tag  # type: ignore
except Exception:
    def element_tag(kind: str, name: str, story_index: int) -> int:  # type: ignore
        s = f"{kind}|{name}|{story_index}".encode("utf-8")
        return int.from_bytes(hashlib.md5(s).digest()[:4], "big") & 0x7FFFFFFF

from rigid_end_utils import split_with_rigid_ends  # type: ignore


def _ensure_ops_model(ndm: int = 3, ndf: int = 6) -> None:
    """
    Ensure the OpenSees domain is initialized. If ndm/ndf are zero, set a basic 3D, 6-DOF model.
    This is idempotent and does not wipe an existing model.
    """
    try:
        cur_ndm = ops.getNDM()
        cur_ndf = ops.getNDF()
    except Exception:
        cur_ndm, cur_ndf = 0, 0
    if int(cur_ndm) == 0 or int(cur_ndf) == 0:
        ops.model("basic", "-ndm", ndm, "-ndf", ndf)
        print(f"[columns] Initialized OpenSees model: ndm={ndm}, ndf={ndf}")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _point_pid(p: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "tag", "point", "pid"):
        if key in p and p[key] is not None:
            return str(p[key])
    return None


def _active_points_map(story: Dict[str, Any]) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    out: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    aps = story.get("active_points") or {}
    for sname, pts in aps.items():
        for p in pts:
            pid = _point_pid(p)
            if not pid:
                print(f"[columns] WARN: active_point in '{sname}' missing 'id'/'tag'; skipped.")
                continue
            out[(pid, sname)] = (float(p["x"]), float(p["y"]), float(p["z"]))
    return out


def _point_exists(pid: str, sname: str, act_pt_map: Dict[Tuple[str, str], Tuple[float, float, float]]) -> bool:
    return (str(pid), sname) in act_pt_map


def _ensure_node_for(
    pid: str, sname: str, sidx: int, act_pt_map: Dict[Tuple[str, str], Tuple[float, float, float]],
    existing_nodes: Set[int]
) -> Optional[int]:
    key = (str(pid), sname)
    if key not in act_pt_map:
        return None
    tag = int(pid) * 1000 + int(sidx)
    if tag not in existing_nodes:
        x, y, z = act_pt_map[key]
        ops.node(tag, x, y, z)
        existing_nodes.add(tag)
    return tag


def _dedupe_last_section_wins(lines_for_story: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last: Dict[str, Dict[str, Any]] = {}
    for ln in lines_for_story:
        last[str(ln["name"])] = ln
    return list(last.values())


def define_columns(
    story_path: str = os.path.join(OUT_DIR, "story_graph.json"),
    raw_path: str = os.path.join(OUT_DIR, "parsed_raw.json"),
    *,
    b_sec: float = 0.40,   # width  [m]
    h_sec: float = 0.40,   # depth  [m]
    E_col: float = 2.50e10,  # [Pa]
    nu_col: float = 0.20
) -> List[int]:
    """
    Build COLUMN elements with the next-lower-story rule and rigid ends.
    Returns the list of created element tags. Also writes OUT_DIR/columns.json.
    """
    # Ensure OpenSees domain exists
    _ensure_ops_model(3, 6)

    story = _load_json(story_path)
    _raw = _load_json(raw_path)

    # Section properties
    G_col = E_col / (2.0 * (1.0 + nu_col))
    A_col = b_sec * h_sec
    Iy_col = (b_sec * h_sec**3) / 12.0
    Iz_col = (h_sec * b_sec**3) / 12.0
    J_col = b_sec * h_sec**3 / 3.0

    story_names: List[str] = list(story.get("story_order_top_to_bottom", []))  # top -> bottom
    story_index = {name: i for i, name in enumerate(story_names)}
    act_pt_map = _active_points_map(story)

    created: List[int] = []
    skips: List[str] = []
    emitted: List[Dict[str, Any]] = []

    try:
        existing_nodes: Set[int] = set(ops.getNodeTags())
    except Exception:
        existing_nodes = set()

    # Lines are per story; we build columns between consecutive stories where points reappear.
    active_lines: Dict[str, List[Dict[str, Any]]] = story.get("active_lines", {})
    for sname, lines in active_lines.items():
        sidx = story_index[sname]
        per_story = _dedupe_last_section_wins(lines)

        # For each COLUMN line at this story, find the next lower story that also contains both points.
        for ln in per_story:
            if str(ln.get("type", "")).upper() != "COLUMN":
                continue

            pid_i: str = str(ln["i"])  # ETABS i is upper
            pid_j: str = str(ln["j"])  # ETABS j is lower (on that story)

            # Find next lower story where BOTH points exist
            k_found: Optional[int] = None
            sK: Optional[str] = None
            for k in range(sidx + 1, len(story_names)):
                sK_candidate = story_names[k]
                if _point_exists(pid_i, sK_candidate, act_pt_map) and _point_exists(pid_j, sK_candidate, act_pt_map):
                    k_found = k
                    sK = sK_candidate
                    break
            if k_found is None or sK is None:
                skips.append(f"{ln.get('name','?')} @ '{sname}' skipped — no lower story with both endpoints")
                continue

            # Build (nTop, nBot) and their coords
            nTop = _ensure_node_for(pid_i, sname, sidx, act_pt_map, existing_nodes)   # upper
            nBot = _ensure_node_for(pid_j, sK, k_found, act_pt_map, existing_nodes)   # lower
            if nTop is None or nBot is None:
                skips.append(f"{ln.get('name','?')} @ '{sname}' skipped — endpoint nodes missing")
                continue

            pTop = act_pt_map[(pid_i, sname)]
            pBot = act_pt_map[(pid_j, sK)]

            # Orientation enforcement (i=bottom, j=top) if requested
            nI, nJ = (nBot, nTop)
            pI, pJ = (pBot, pTop)
            LoffI = float(ln.get("length_off_i", 0.0) or 0.0)
            LoffJ = float(ln.get("length_off_j", 0.0) or 0.0)
            line_name = str(ln.get("name", "?"))

            if not ENFORCE_COLUMN_I_AT_BOTTOM:
                # Keep ETABS i->j direction (i at S (top), j at K (bottom)):
                nI, nJ = (nTop, nBot)
                pI, pJ = (pTop, pBot)
            else:
                # ETABS i is upper; our convention is i=bottom. Swap offsets accordingly.
                LoffI, LoffJ = float(LoffJ), float(LoffI)

            parts = split_with_rigid_ends(
                kind="COLUMN", line_name=line_name, story_index=int(sidx),
                nI=nI, nJ=nJ, pI=pI, pJ=pJ, LoffI=LoffI, LoffJ=LoffJ
            )

            # Create elements for each segment
            for seg in parts['segments']:
                role = seg['role']
                i_tag, j_tag = seg['i'], seg['j']

                etag = element_tag("COLUMN", line_name + seg['suffix'], int(sidx))
                transf_tag = 1100000000 + etag
                ops.geomTransf('Linear', transf_tag, 1, 0, 0)

                if role.startswith("rigid"):
                    A = A_col * RIGID_END_SCALE
                    Iy = Iy_col * RIGID_END_SCALE
                    Iz = Iz_col * RIGID_END_SCALE
                    J = J_col * RIGID_END_SCALE
                else:
                    A, Iy, Iz, J = A_col, Iy_col, Iz_col, J_col

                # Ensure split interface nodes exist in the OpenSees domain
                coord_by_tag = {
                    int(parts['nodes']['nI']): parts['coords']['nI'],
                    int(parts['nodes']['nIm']): parts['coords']['nIm'],
                    int(parts['nodes']['nJm']): parts['coords']['nJm'],
                    int(parts['nodes']['nJ']): parts['coords']['nJ'],
                }
                for t in (i_tag, j_tag):
                    if t not in existing_nodes:
                        cx, cy, cz = coord_by_tag[int(t)]
                        ops.node(int(t), float(cx), float(cy), float(cz))
                        existing_nodes.add(int(t))

                ops.element('elasticBeamColumn', etag, i_tag, j_tag, A, E_col, G_col, J, Iy, Iz, transf_tag)
                created.append(etag)

                coords = parts['coords']
                emitted.append({
                    "tag": etag,
                    "segment": role,
                    "parent_line": line_name,
                    "story": sname,
                    "i_node": i_tag,
                    "j_node": j_tag,
                    "section": ln.get("section"),
                    "transf_tag": transf_tag,
                    "A": A, "E": E_col, "G": G_col, "J": J, "Iy": Iy, "Iz": Iz,
                    "length_off_i": LoffI, "length_off_j": LoffJ,
                })

    if skips:
        print("[columns] Skips:")
        for s in skips:
            print(" -", s)
    print(f"[columns] Created {len(created)} column segments (including rigid ends).")

    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "columns.json"), "w", encoding="utf-8") as f:
            json.dump({"columns": emitted, "counts": {"created": len(created)}, "skips": skips}, f, indent=2)
        print(f"[columns] Wrote {OUT_DIR}/columns.json")
    except Exception as e:
        print(f"[columns] WARN: failed to write columns.json: {e}")

    return created
