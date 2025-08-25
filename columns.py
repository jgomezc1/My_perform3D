# columns.py
"""
Create COLUMN elements as OpenSeesPy elasticBeamColumn members using the following
rules, and emit Phase-2 artifact `columns.json`.

Local axis orientation: by default enforce i=bottom, j=top for columns (configurable).
"find-next-lower-story" rule:

For each COLUMN LINEASSIGN at story S:
  - Create exactly one segment from:
        upper node: endpoint "i" at story S
        lower node: endpoint "j" at the next lower story K > S where BOTH endpoints
                    ("i" and "j") exist in that story's active points.
  - If no such lower story exists (no occurrence of BOTH endpoints below), skip and report.
  - Intermediate stories without endpoints are skipped by design.

OpenSeesPy signatures (exact):
    geomTransf('Linear', 111, 1, 0, 0)
    element('elasticBeamColumn', tag, nI, nJ,
            A, E, G, J, Iy, Iz, transfTag)
"""
from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple, Optional
import json
import os
import hashlib

from openseespy.opensees import geomTransf, element, node, getNodeTags

# Optional config hooks
try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = "out"

# Configuration switch: enforce column local axis i=bottom, j=top
try:
    from config import ENFORCE_COLUMN_I_AT_BOTTOM  # type: ignore
except Exception:
    ENFORCE_COLUMN_I_AT_BOTTOM = True

# Prefer project tagging helpers if available
try:
    from tagging import element_tag  # type: ignore
except Exception:
    def element_tag(kind: str, name: str, story_index: int) -> int:  # type: ignore
        s = f"{kind}|{name}|{story_index}".encode("utf-8")
        return int.from_bytes(hashlib.md5(s).digest()[:4], "big") & 0x7FFFFFFF


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _point_pid(p: Dict[str, Any]) -> Optional[str]:
    """
    Safely extract a point identifier from a Phase-1 active_point record.
    Priority: 'id' → 'tag' → ('point', 'pid') if present. Returns None if none found.
    """
    for key in ("id", "tag", "point", "pid"):
        if key in p and p[key] is not None:
            return str(p[key])
    return None


def _active_points_map(story: Dict[str, Any]) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """
    Build lookup: (point_id, story_name) -> (x, y, z), skipping records without a usable id.
    """
    out: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for sname, pts in story.get("active_points", {}).items():
        for p in pts:
            pid = _point_pid(p)
            if pid is None:
                # Graceful skip; bad record should not crash the build.
                print(f"[columns] WARN: active_point in '{sname}' missing 'id'/'tag'; skipped.")
                continue
            out[(pid, sname)] = (float(p["x"]), float(p["y"]), float(p["z"]))
    return out


def _point_exists(pid: str, sname: str, act_pt_map: Dict[Tuple[str, str], Tuple[float, float, float]]) -> bool:
    return (str(pid), sname) in act_pt_map


def _ensure_node_for(
    pid: str, sname: str, sidx: int, act_pt_map: Dict[Tuple[str, str], Tuple[float, float, float]],
    existing_nodes: Set[int]
) -> int:
    """
    Ensure a node for (pid, sname) exists; create it if missing using active_points coords.
    """
    tag = int(pid) * 1000 + int(sidx)
    if tag not in existing_nodes:
        x, y, z = act_pt_map[(str(pid), sname)]
        node(tag, x, y, z)
        existing_nodes.add(tag)
    return tag


def _dedupe_last_section_wins(lines_for_story: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last: Dict[str, Dict[str, Any]] = {}
    for ln in lines_for_story:
        last[str(ln["name"])] = ln
    return list(last.values())


def define_columns(
    story_path: str = os.path.join(OUT_DIR, "story_graph.json"),
    raw_path: str   = os.path.join(OUT_DIR, "parsed_raw.json"),
    *,
    b_sec: float = 0.40,   # width  [m]
    h_sec: float = 0.40,   # depth  [m]
    E_col: float = 2.50e10,  # [Pa]
    nu_col: float = 0.20
) -> List[int]:
    """
    Build COLUMN elements with the "find-next-lower-story" rule.
    Returns the list of created element tags. Also writes OUT_DIR/columns.json.
    """
    story = _load_json(story_path)
    _raw  = _load_json(raw_path)

    # Transform and properties
    geomTransf('Linear', 111, 1, 0, 0)
    transf_tag = 111

    G_col  = E_col / (2.0 * (1.0 + nu_col))
    A_col  = b_sec * h_sec
    Iy_col = (b_sec * h_sec**3) / 12.0
    Iz_col = (h_sec * b_sec**3) / 12.0
    J_col  =  b_sec * h_sec**3 / 3.0

    story_names: List[str] = list(story.get("story_order_top_to_bottom", []))  # top -> bottom
    story_index = {name: i for i, name in enumerate(story_names)}
    act_pt_map  = _active_points_map(story)

    created: List[int] = []
    skips: List[str] = []
    emitted: List[Dict[str, Any]] = []

    existing_nodes: Set[int] = set(getNodeTags())

    active_lines: Dict[str, List[Dict[str, Any]]] = story.get("active_lines", {})
    for sname, lines in active_lines.items():
        sidx = story_index[sname]
        per_story = _dedupe_last_section_wins(lines)

        for ln in per_story:
            if str(ln.get("type","")).upper() != "COLUMN":
                continue

            pid_i = str(ln["i"])  # ETABS "i" (upper endpoint)
            pid_j = str(ln["j"])  # ETABS "j" (lower endpoint)

            # Find next lower story K where BOTH endpoints exist
            k_found: Optional[int] = None
            for k in range(sidx + 1, len(story_names)):
                sK = story_names[k]
                if _point_exists(pid_i, sK, act_pt_map) and _point_exists(pid_j, sK, act_pt_map):
                    k_found = k
                    break
            if k_found is None:
                skips.append(f"{ln.get('name','?')} @ '{sname}' skipped — no lower story with both endpoints")
                continue

            # Create or get nodes
            if (pid_i, sname) not in act_pt_map:
                skips.append(f"{ln.get('name','?')} @ '{sname}' skipped — upper endpoint '{pid_i}' not in active_points[{sname}]")
                continue
            if (pid_j, story_names[k_found]) not in act_pt_map:
                skips.append(f"{ln.get('name','?')} @ '{sname}' skipped — lower endpoint '{pid_j}' not in active_points[{story_names[k_found]}]")
                continue

            n_top    = _ensure_node_for(pid_i, sname, sidx, act_pt_map, existing_nodes)
            n_bottom = _ensure_node_for(pid_j, story_names[k_found], k_found, act_pt_map, existing_nodes)

            # Deterministic element tag (use upper story index for stability)
            tag = element_tag("COLUMN", str(ln.get("name","?")), int(sidx))

            # Orientation enforcement
            if ENFORCE_COLUMN_I_AT_BOTTOM:
                e_nI, e_nJ = n_bottom, n_top
                orientation = "i=bottom,j=top"
            else:
                e_nI, e_nJ = n_top, n_bottom
                orientation = "i=top,j=bottom"

            # Create element
            element('elasticBeamColumn', tag, e_nI, e_nJ,
                    A_col, E_col, G_col, J_col, Iy_col, Iz_col, transf_tag)
            created.append(tag)

            emitted.append({
                "tag": tag,
                "story_top": sname,
                "story_bottom": story_names[k_found],
                "line": str(ln.get("name","?")),
                "i_node": e_nI,
                "j_node": e_nJ,
                "orientation": orientation,
                "section": ln.get("section"),
                "transf_tag": transf_tag,
                "A": A_col, "E": E_col, "G": G_col, "J": J_col,
                "Iy": Iy_col, "Iz": Iz_col
            })

    if ENFORCE_COLUMN_I_AT_BOTTOM:
        print(f"[columns] Orientation: enforced i=bottom, j=top on {len(created)} column elements.")

    if skips:
        print("[columns] Skips:")
        for s in skips:
            print(" -", s)
    print(f"[columns] Created {len(created)} column elements.")

    # Emit artifact
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "columns.json"), "w", encoding="utf-8") as f:
            json.dump({"columns": emitted, "counts": {"created": len(created)}, "skips": skips}, f, indent=2)
        print(f"[columns] Wrote {OUT_DIR}/columns.json")
    except Exception as e:
        print(f"[columns] WARN: failed to write columns.json: {e}")

    return created
