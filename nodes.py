# nodes.py
"""
Define OpenSeesPy nodes from Phase-1 artifacts using the NEW Z rule.

Key points:
- There are no global "free" explicit-Z nodes anymore. If a point has a third value `d`,
  then at each story S where it appears:
      Z = Z_story(S) - d
  We therefore create a grid node per (point, story) as needed, using the grid tag rule.

- Tagging:
  Grid node tag: int(point_id) * 1000 + story_index (top=0).
  (We keep node_tag_free available in tagging.py for other cases, but do not use it here.)
"""
from __future__ import annotations
import json
from typing import Dict, Any, Set
from openseespy.opensees import node
from tagging import node_tag_grid  # node_tag_free unused here


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def define_nodes(story_path: str = "out/story_graph.json",
                 raw_path: str = "out/parsed_raw.json") -> Set[int]:
    story = _load_json(story_path)
    # raw currently unused; kept for debugging parity/consistency
    _ = _load_json(raw_path)

    story_names = story["story_order_top_to_bottom"]  # top -> bottom
    story_index = {name: i for i, name in enumerate(story_names)}

    created: Set[int] = set()

    # Create nodes for all active points at each story (Z is already resolved in story_graph)
    for sname, pts in story["active_points"].items():
        idx = story_index[sname]
        for p in pts:
            pid, x, y, z = p["id"], p["x"], p["y"], p["z"]
            tag = node_tag_grid(pid, idx)
            if tag not in created:
                node(tag, x, y, z)
                created.add(tag)

    return created
