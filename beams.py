"""
Create BEAM elements as OpenSeesPy elasticBeamColumn members with support for
ETABS rigid end zones (LENGTHOFFI/LENGTHOFFJ when RIGIDZONE=1). Writes Phase-2
artifact `beams.json`.

New in this version
-------------------
- Splits members into up to **three segments** (rigid I, deformable mid, rigid J).
- Creates deterministic **intermediate nodes** at the offset boundaries.
- Per-segment **geomTransf** (one per element) derived from the element tag.
- Emits richer `beams.json` records with a `segment` field ('rigid_i'|'deformable'|'rigid_j').

Assumptions (unchanged)
-----------------------
- Deterministic node tags:
      node_tag = point_int * 1000 + story_index
  where story_index is 0 for the **top** story and increases downward.
- Deterministic element tags via tagging.element_tag(kind, name, story_index). If
  tagging.py is not available, we fall back to an internal stable hash.
- Orientation vector: vecXZ = (0, 0, 1) for beams.

Schema note
-----------
Previous `beams.json` contained one entry per line/element. We now emit
**one entry per created segment** while preserving prior fields. New fields:
`segment` (role), `parent_line`, and optional `i_coords`/`j_coords`. Downstream
consumers expecting one-per-line should filter `segment == "deformable"`.
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

# Prefer project tagging helpers if available
try:
    from tagging import node_tag_grid, element_tag  # type: ignore
except Exception:
    node_tag_grid = None  # type: ignore

    def element_tag(kind: str, name: str, story_index: int) -> int:  # type: ignore
        s = f"{kind}|{name}|{story_index}".encode("utf-8")
        return int.from_bytes(hashlib.md5(s).digest()[:4], "big") & 0x7FFFFFFF

# Rigid end scale (A, Iy, Iz, J are multiplied by this for rigid segments)
try:
    from config import RIGID_END_SCALE  # type: ignore
except Exception:
    RIGID_END_SCALE = 1.0e6

from rigid_end_utils import split_with_rigid_ends  # type: ignore


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
                print(f"[beams] WARN: active_point in '{sname}' missing 'id'/'tag'; skipped.")
                continue
            out[(pid, sname)] = (float(p["x"]), float(p["y"]), float(p["z"]))
    return out


def _dedupe_last_section_wins(lines_for_story: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last: Dict[str, Dict[str, Any]] = {}
    for ln in lines_for_story:
        last[str(ln["name"])] = ln  # overwrite => last wins
    return list(last.values())


def _ensure_node_for(
    pid: str, sname: str, sidx: int, act_pt_map: Dict[Tuple[str, str], Tuple[float, float, float]],
    existing_nodes: Set[int],
) -> Optional[int]:
    """
    Ensure a node for (pid, sname) exists; create it if missing using active_points coords.
    Returns the node tag, or None if the point is absent from active_points.
    """
    key = (str(pid), sname)
    if key not in act_pt_map:
        return None
    tag = int(pid) * 1000 + int(sidx)
    if tag not in existing_nodes:
        x, y, z = act_pt_map[key]
        node(tag, x, y, z)
        existing_nodes.add(tag)
    return tag


def define_beams(
    story_path: str = os.path.join(OUT_DIR, "story_graph.json"),
    raw_path: str   = os.path.join(OUT_DIR, "parsed_raw.json"),
    *,
    # --- Placeholder properties (override as needed; units consistent with your model) ---
    b_sec: float = 0.40,   # width  [m]
    h_sec: float = 0.50,   # depth  [m]
    E_beam: float = 2.50e10,  # [Pa]
    nu_beam: float = 0.20
) -> List[int]:
    """
    Builds BEAM elements per-story with "last section wins" within each story.
    Supports rigid ends via LENGTHOFFI/LENGTHOFFJ in story_graph.
    Returns the list of created element tags. Also writes OUT_DIR/beams.json.
    """
    story = _load_json(story_path)
    _raw  = _load_json(raw_path)  # parity/debugging

    # --- Section / Material ---
    G_beam  = E_beam / (2.0 * (1.0 + nu_beam))
    A_beam  = b_sec * h_sec
    Iy_beam = (b_sec * h_sec**3) / 12.0
    Iz_beam = (h_sec * b_sec**3) / 12.0
    J_beam  =  b_sec * h_sec**3 / 3.0

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
            if str(ln.get("type", "")).upper() != "BEAM":
                continue

            pid_i: str = str(ln["i"])
            pid_j: str = str(ln["j"])

            nI = _ensure_node_for(pid_i, sname, sidx, act_pt_map, existing_nodes)
            nJ = _ensure_node_for(pid_j, sname, sidx, act_pt_map, existing_nodes)
            if nI is None or nJ is None:
                skips.append(f"{ln.get('name','?')} @ '{sname}' skipped — endpoint(s) not present on this story")
                continue

            pI = act_pt_map[(pid_i, sname)]
            pJ = act_pt_map[(pid_j, sname)]

            LoffI = float(ln.get("length_off_i", 0.0) or 0.0)
            LoffJ = float(ln.get("length_off_j", 0.0) or 0.0)

            line_name = str(ln.get("name", "?"))
            parts = split_with_rigid_ends(
                kind="BEAM", line_name=line_name, story_index=int(sidx),
                nI=nI, nJ=nJ, pI=pI, pJ=pJ, LoffI=LoffI, LoffJ=LoffJ
            )

            # Create elements per returned segments
            for seg in parts['segments']:
                role = seg['role']
                i_tag, j_tag = seg['i'], seg['j']

                # Stable element tag within BEAM range using line_name+suffix
                etag = element_tag("BEAM", line_name + seg['suffix'], int(sidx))

                # Per-element transformation: Linear with vecXZ = (0,0,1)
                transf_tag = 1000000000 + etag
                geomTransf('Linear', transf_tag, 0, 0, 1)

                # Properties (inflate if rigid)
                if role.startswith("rigid"):
                    A = A_beam * RIGID_END_SCALE
                    Iy = Iy_beam * RIGID_END_SCALE
                    Iz = Iz_beam * RIGID_END_SCALE
                    J  = J_beam  * RIGID_END_SCALE
                else:
                    A, Iy, Iz, J = A_beam, Iy_beam, Iz_beam, J_beam

                element('elasticBeamColumn', etag, i_tag, j_tag, A, E_beam, G_beam, J, Iy, Iz, transf_tag)
                created.append(etag)

                # Emit record
                coords = parts['coords']
                emitted.append({
                    "tag": etag,
                    "segment": role,
                    "parent_line": line_name,
                    "story": sname,
                    "line": line_name,  # backward compat
                    "i_node": i_tag,
                    "j_node": j_tag,
                    "i_coords": coords['nI'] if i_tag in (parts['nodes']['nI'], parts['nodes']['nIm']) else None,
                    "j_coords": coords['nJ'] if j_tag in (parts['nodes']['nJ'], parts['nodes']['nJm']) else None,
                    "section": ln.get("section"),
                    "transf_tag": transf_tag,
                    "A": A, "E": E_beam, "G": G_beam, "J": J, "Iy": Iy, "Iz": Iz,
                    "length_off_i": LoffI, "length_off_j": LoffJ,
                })

    if skips:
        print("[beams] Skips:")
        for s in skips:
            print(" -", s)
    print(f"[beams] Created {len(created)} beam segments (including rigid ends).")

    # Emit artifact
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "beams.json"), "w", encoding="utf-8") as f:
            json.dump({"beams": emitted, "counts": {"created": len(created)}, "skips": skips}, f, indent=2)
        print(f"[beams] Wrote {OUT_DIR}/beams.json")
    except Exception as e:
        print(f"[beams] WARN: failed to write beams.json: {e}")

    return created
