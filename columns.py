# columns.py
"""
Create COLUMN elements as OpenSeesPy elasticBeamColumn members using the NEW
Local axis orientation: enforced i=bottom, j=top for columns (configurable).
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

Placeholders for section properties are provided and can be replaced later.
"""
from __future__ import annotations
import json
from typing import Dict, Any, List, Set, Tuple
from openseespy.opensees import geomTransf, element
from tagging import node_tag_grid, element_tag

# Configuration switch: enforce column local axis i=bottom, j=top
try:
    from config import ENFORCE_COLUMN_I_AT_BOTTOM  # type: ignore
except Exception:
    ENFORCE_COLUMN_I_AT_BOTTOM = True


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_point_sets_per_story(story: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Return {story_name: set_of_point_ids_present} for quick membership checks."""
    pts_by_story: Dict[str, Set[str]] = {}
    for sname, pts in story["active_points"].items():
        pts_by_story[sname] = {p["id"] for p in pts}
    return pts_by_story


def define_columns(
    story_path: str = "out/story_graph.json",
    raw_path: str   = "out/parsed_raw.json",
    *,
    # --- Placeholder properties (replace as your pipeline mat/sect evolves) ---
    E_col: float = 25.0e9,
    nu_col: float = 0.2,
    b_col: float = 0.30,
    h_col: float = 0.30,
) -> List[int]:
    """
    Build COLUMN elements following the "find-next-lower-story" rule.

    For each (story S, column line L):
      - Anchor the upper node at S (endpoint "i") if BOTH endpoints exist at S.
      - Search downwards for the next story K > S where BOTH endpoints exist.
      - Create a single segment between the node at S and the node at K,
        skipping any stories in between that don't contain both endpoints.

    IMPORTANT (orientation):
      OpenSees defines the local longitudinal axis along i → j.
      With ENFORCE_COLUMN_I_AT_BOTTOM=True, we assign i=bottom (lower-Z) and j=top (higher-Z)
      for columns, which flips the original "top->bottom" order to match structural conventions.

    Returns the list of created element tags.
    """
    story = _load_json(story_path)
    _raw  = _load_json(raw_path)  # kept for parity/debug

    # Geometric transformation (exact)
    geomTransf('Linear', 111, 1, 0, 0)

    # --- Placeholder Material and Section Properties ---
    G_col  = E_col / (2.0 * (1.0 + nu_col))
    A_col  = b_col * h_col
    Iy_col = (b_col * h_col**3) / 12.0
    Iz_col = (h_col * b_col**3) / 12.0
    J_col  =  b_col * h_col**3 / 3.0

    story_names = story["story_order_top_to_bottom"]  # top -> bottom
    story_index = {s: i for i, s in enumerate(story_names)}  # name -> index
    pts_by_story = _build_point_sets_per_story(story)

    created: List[int] = []
    skips: List[str] = []
    reoriented = 0  # columns with i->j enforced bottom->top

    def story_has_both(sname: str, pid_i: str, pid_j: str) -> bool:
        sset = pts_by_story.get(sname, set())
        return (pid_i in sset) and (pid_j in sset)

    # For each story, for each column line assigned there, find the next lower story
    # where BOTH endpoints exist. Create the segment (S -> K).
    for sname, lines in story["active_lines"].items():
        sidx = story_index[sname]
        # search only if not bottom-most story (no lower stories exist)
        if sidx >= len(story_names) - 1:
            for ln in lines:
                if ln["type"] == "COLUMN":
                    skips.append(f"{ln['name']} at lowest story '{sname}' — no lower story exists")
            continue

        for ln in lines:
            if ln["type"] != "COLUMN":
                continue

            pid_i: str = ln["i"]
            pid_j: str = ln["j"]

            # Upper node must exist in story S
            if not story_has_both(sname, pid_i, pid_j):
                # If both endpoints are not present at S, we cannot anchor the upper node as defined.
                skips.append(f"{ln['name']} at '{sname}' skipped — endpoints not both present at upper story")
                continue

            # Find next lower story K where BOTH endpoints exist
            k_found = None
            for k in range(sidx + 1, len(story_names)):
                kname = story_names[k]
                if story_has_both(kname, pid_i, pid_j):
                    k_found = k
                    break

            if k_found is None:
                skips.append(f"{ln['name']} at '{sname}' skipped — no lower story with both endpoints")
                continue

            # Build tags and enforce bottom->top local axis if configured
            n_top    = node_tag_grid(pid_i, sidx)     # upper node at S, endpoint 'i'
            n_bottom = node_tag_grid(pid_j, k_found)  # lower node at K, endpoint 'j'
            tag = element_tag("COLUMN", ln["name"], sidx)

            # Create element with desired orientation
            if ENFORCE_COLUMN_I_AT_BOTTOM:
                e_nI, e_nJ = n_bottom, n_top  # i=bottom, j=top
                reoriented += 1
            else:
                e_nI, e_nJ = n_top, n_bottom # original top->bottom
            element('elasticBeamColumn', tag, e_nI, e_nJ,
                    A_col, E_col, G_col, J_col, Iy_col, Iz_col, 111)
            created.append(tag)

    if ENFORCE_COLUMN_I_AT_BOTTOM:
        print(f"[columns] Orientation: enforced i=bottom, j=top on {reoriented} column elements.")

    # Diagnostics
    if skips:
        print("[columns] Skips:")
        for s in skips:
            print(" -", s)
    print(f"[columns] Created {len(created)} column elements.")

    return created
