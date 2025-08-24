# story_builder.py
"""
Builds the STORY-centered view needed for Phase 2.

Rules implemented:
- Story elevations:
  * File order is top->bottom.
  * The base story is the one with ELEV == 0 (if none, the last entry is treated as base).
  * Elevations accumulate via HEIGHT unless an explicit ELEV is provided for a story.

- Point Z rule:
  * If a point in $ POINT COORDINATES has a third value `d`, then its absolute Z at story S is:
        Z = Z_story(S) - d
    This applies ALWAYS.
  * If a point does not have a third value, then:
        Z = Z_story(S)

- Active lines:
  * `$ LINE CONNECTIVITIES` define topology (i, j).
  * `$ LINE ASSIGNS` define per-story activation and section.
  * For beams, "last section wins" per story/line.
  * For columns, we only prepare the per-story activation; actual vertical matching (S to next lower
    story with the same endpoints present) is done in columns.py.

Output schema:
{
  "story_order_top_to_bottom": [story_name_top,...,story_name_bottom],
  "story_elev": {story_name: z_abs, ...},
  "active_points": {
     story_name: [
       {"id": pid, "x": x, "y": y, "z": z_abs, "explicit_z": bool, "diaphragm": ..., "springprop": ...},
       ...
     ],
     ...
  },
  "active_lines": {
     story_name: [
       {"name": line_name, "type": "BEAM"|"COLUMN", "i": pid_i, "j": pid_j, "section": maybe_str},
       ...
     ],
     ...
  },

  # Kept for compatibility (always empty now because explicit-Z is story-dependent):
  "free_points_xyz": []
}
"""
from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict
from config import EPS


def compute_story_elevations(stories: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    stories: list of dicts in ETABS file order (top -> bottom), each with:
      - name: str
      - height: float | None
      - elev: float | None (explicit)
    Returns a dict {story_name: absolute_Z}.
    """
    if not stories:
        return {}

    # Choose base story
    base_idx = None
    for i, s in enumerate(stories):
        if s["elev"] is not None and abs(s["elev"]) < EPS:
            base_idx = i
            break
    if base_idx is None:
        base_idx = len(stories) - 1  # default: last entry as base

    elev: Dict[str, float] = {}
    base = stories[base_idx]
    base_z = base["elev"] if base["elev"] is not None else 0.0
    elev[base["name"]] = base_z

    # Upward from base (towards top): Z(upper) = Z(lower) + HEIGHT(upper), unless ELEV overrides
    last_idx = base_idx
    for i in range(base_idx - 1, -1, -1):
        s = stories[i]
        dz = s["height"] or 0.0
        z = elev[stories[last_idx]["name"]] + dz
        if s["elev"] is not None:
            z = s["elev"]
        elev[s["name"]] = z
        last_idx = i

    # Downward from base (towards bottom): Z(lower) = Z(upper) - HEIGHT(upper), unless ELEV overrides
    for i in range(base_idx + 1, len(stories)):
        upper = stories[i - 1]
        s = stories[i]
        dz = (upper["height"] or 0.0)
        z = elev[upper["name"]] - dz
        if s["elev"] is not None:
            z = s["elev"]
        elev[s["name"]] = z

    return elev


def build_story_graph(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    raw: dict from parse_e2k with keys:
      - stories: [{name, height, elev}, ...] (top->bottom order)
      - points:  {pid: {"x": float, "y": float, "has_three": bool, "third": float|None}, ...}
      - point_assigns: [{"point": pid, "story": name, "diaphragm":?, "springprop":?}, ...]
      - lines:   {lname: {"name": lname, "kind": "BEAM"|"COLUMN", "i": pid, "j": pid}, ...}
      - line_assigns: [{"line": lname, "story": name, "section":?}, ...]

    Returns a STORY-centered dict with:
      - story_order_top_to_bottom
      - story_elev
      - active_points[story] -> [{id,x,y,z,explicit_z,diaphragm,springprop}]
      - active_lines[story]  -> [{name,type,i,j,section}]
      - free_points_xyz      -> []  (explicit-Z is story-dependent now; no global free nodes)
    """
    stories: List[Dict[str, Any]] = raw["stories"]
    points: Dict[str, Dict[str, Any]] = raw["points"]
    point_assigns: List[Dict[str, Any]] = raw["point_assigns"]
    lines: Dict[str, Dict[str, Any]] = raw["lines"]
    line_assigns: List[Dict[str, Any]] = raw["line_assigns"]

    story_names = [s["name"] for s in stories]  # top -> bottom
    story_elev = compute_story_elevations(stories)
    story_set = set(story_elev.keys())

    # Build active points per story with the NEW Z rule
    # (SPRINGPROP is ignored for Z computation)
    active_points = defaultdict(list)
    for a in point_assigns:
        story = a["story"]
        if story not in story_set:
            continue
        pid = a["point"]
        prec = points.get(pid)
        if not prec:
            continue
        has_three = bool(prec.get("has_three"))
        d = prec.get("third") if has_three else None
        if has_three and d is not None:
            z = story_elev[story] - d
            explicit_flag = True
        else:
            z = story_elev[story]
            explicit_flag = False

        active_points[story].append({
            "id": pid,
            "x": prec["x"],
            "y": prec["y"],
            "z": z,
            "explicit_z": explicit_flag,
            "diaphragm": a.get("diaphragm"),
            "springprop": a.get("springprop"),
        })

    # Active lines per story (dedupe by line name; "last section wins")
    la_by_story = defaultdict(list)
    for la in line_assigns:
        la_by_story[la["story"]].append(la)

    active_lines = {}
    for sname, assigns in la_by_story.items():
        per_line = {}
        for la in assigns:
            rec = lines.get(la["line"])
            if not rec:
                continue
            key = la["line"]
            if key not in per_line:
                per_line[key] = {
                    "name": rec["name"],
                    "type": rec["kind"],
                    "i": rec["i"],
                    "j": rec["j"],
                    "section": la.get("section"),
                }
            else:
                if la.get("section") is not None:
                    per_line[key]["section"] = la["section"]
        active_lines[sname] = list(per_line.values())

    # With the new Z rule, explicit-Z is NOT a global free coordinate—it's story-dependent.
    # We keep this key for compatibility, but it's intentionally empty.
    free_points_xyz: List[Dict[str, Any]] = []

    return {
        "story_order_top_to_bottom": story_names,
        "story_elev": story_elev,
        "active_points": dict(active_points),
        "active_lines": dict(active_lines),
        "free_points_xyz": free_points_xyz,
    }
