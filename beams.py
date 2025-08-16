# beams.py
"""
Create BEAM elements as OpenSeesPy elasticBeamColumn members using the
**per-story / last section wins** behavior.

Definitions implemented:

1) Per-story:
   For each STORY S, if a BEAM line (from $ LINE CONNECTIVITIES) is assigned on S
   (via $ LINE ASSIGNS), we create exactly ONE beam element on story S that connects
   the two endpoints' nodes on S.

2) Last section wins (within a story):
   story_builder.py already condenses assigns so that for each (story, line) the
   FINAL record has the "winning" section. Here we *reassert* that behavior by
   re-deduping the per-story list (just in case) so the final occurrence per line
   in a story is the one used. If multiple entries appear for the same line within
   a story, the last one encountered in file order decides the section.

3) Node creation:
   Nodes should typically exist after define_nodes(). For robustness, this file
   can also create any missing node on demand using the coordinates in
   story_graph.json (active_points already carry x,y,z resolved with the new rule:
       Z = Z_story - d  (if the point has a third value d)
       Z = Z_story      (otherwise)
   ).

OpenSeesPy signatures (exact):
    geomTransf('Linear', 222, 0, 0, 1)
    element('elasticBeamColumn', tag, nI, nJ,
            A_beam, E_beam, G_beam, J_beam, Iy_beam, Iz_beam, 222)

Section properties here are placeholders; swap in real ones when you wire sections.
"""
from __future__ import annotations
import json
from typing import Dict, Any, List, Set, Tuple
from openseespy.opensees import geomTransf, element, node, getNodeTags
from tagging import node_tag_grid, element_tag


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _active_points_map(story: Dict[str, Any]) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """
    Build a quick lookup: (point_id, story_name) -> (x, y, z)
    """
    out: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for sname, pts in story["active_points"].items():
        for p in pts:
            out[(p["id"], sname)] = (p["x"], p["y"], p["z"])
    return out


def _dedupe_last_section_wins(lines_for_story: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given a list of line dicts for a story (as in story_graph["active_lines"][S]),
    return a list where only the **last** occurrence of each line name remains.
    """
    last: Dict[str, Dict[str, Any]] = {}
    for ln in lines_for_story:
        last[ln["name"]] = ln  # overwrite => last wins
    return list(last.values())


def define_beams(
    story_path: str = "out/story_graph.json",
    raw_path: str   = "out/parsed_raw.json",
    *,
    # --- Placeholder properties (override as needed; units consistent with your model) ---
    b_sec: float = 0.40,   # width
    h_sec: float = 0.50,   # depth
    E_beam: float = 2.50e10,
    nu_beam: float = 0.20
) -> List[int]:
    """
    Builds BEAM elements per-story with "last section wins" within each story.
    Returns the list of created element tags.
    """
    story = _load_json(story_path)
    _raw  = _load_json(raw_path)  # kept for parity/debugging; not required here

    # Geometric transformation (exact)
    geomTransf('Linear', 222, 0, 0, 1)

    # --- Placeholder Material and Section Properties ---
    G_beam  = E_beam / (2.0 * (1.0 + nu_beam))
    A_beam  = b_sec * h_sec
    Iy_beam = (b_sec * h_sec**3) / 12.0
    Iz_beam = (h_sec * b_sec**3) / 12.0
    J_beam  =  b_sec * h_sec**3 / 3.0

    story_names = story["story_order_top_to_bottom"]  # top -> bottom
    story_index = {name: i for i, name in enumerate(story_names)}
    act_pt_map  = _active_points_map(story)  # (pid, sname) -> (x,y,z)

    created: List[int] = []
    skips: List[str] = []

    # Cache existing nodes to avoid duplicate creations
    existing_nodes: Set[int] = set(getNodeTags())

    def ensure_node_for(pid: str, sname: str) -> int | None:
        """
        Ensure a node for (pid, sname) exists; create it if missing using active_points coords.
        Returns the node tag, or None if the point does not exist in active_points for that story.
        """
        idx = story_index[sname]
        tag = node_tag_grid(pid, idx)
        if tag in existing_nodes:
            return tag
        key = (pid, sname)
        if key not in act_pt_map:
            return None  # point not active on this story
        x, y, z = act_pt_map[key]
        node(tag, x, y, z)
        existing_nodes.add(tag)
        return tag

    # Iterate stories, apply per-story / last-section-wins, and create beams
    for sname, lines in story["active_lines"].items():
        sidx = story_index[sname]
        # Reassert "last section wins" at this layer as a final guard:
        per_story = _dedupe_last_section_wins(lines)

        for ln in per_story:
            if ln["type"] != "BEAM":
                continue

            pid_i: str = ln["i"]
            pid_j: str = ln["j"]

            # Both endpoints must exist on this story (per-story behavior)
            nI = ensure_node_for(pid_i, sname)
            nJ = ensure_node_for(pid_j, sname)

            if nI is None or nJ is None:
                skips.append(f"{ln['name']} @ '{sname}' skipped — endpoint(s) not present on this story")
                continue

            # Deterministic element tag (stable)
            tag = element_tag("BEAM", ln["name"], sidx)

            # Create element
            element('elasticBeamColumn', tag, nI, nJ,
                    A_beam, E_beam, G_beam, J_beam, Iy_beam, Iz_beam, 222)
            created.append(tag)

    # Diagnostics
    if skips:
        print("[beams] Skips:")
        for s in skips:
            print(" -", s)
    print(f"[beams] Created {len(created)} beam elements.")

    return created
