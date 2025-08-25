# -*- coding: utf-8 -*-
"""
emit_nodes.py

Emit the nodes artifact (nodes.json) representing the actual OpenSees node set:
- Grid nodes derived from story_graph.json (Phase-1, Z already resolved there)
- Diaphragm master nodes derived from diaphragms.json (Phase-2)

Deterministic tag:
    node_tag = point_id * 1000 + story_index   (story_index = 0 at top, increasing downward)

Z rule used by this emitter (aligned with current repo):
  - Prefer the pre-resolved absolute Z stored in story_graph.json at active_points[*]["z"].
  - If that key is missing, compute:
        Z = story_elev[story] - offset
    where offset = active_points[*]["explicit_z"] if present, else active_points[*]["z"] (legacy offset), else 0.0.

This keeps nodes.json faithful to the OpenSees domain created by nodes.py, and still
guards against future omissions by falling back to the documented rule.

Output schema (stable, v1):
{
  "nodes": [
    {"tag": 101000, "x": 0.0, "y": 0.0, "z": 30.0, "story": "Roof", "story_index": 0,
     "kind": "grid", "source_point_id": "101"},
    {"tag": 9001, "x": 12.34, "y": 8.76, "z": 20.0, "story": "Story-3", "story_index": 2,
     "kind": "diaphragm_master"}
  ],
  "counts": {"total": 123, "grid": 120, "master": 3},
  "version": 1
}

Python: 3.11+
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _story_index_map(story_graph: Dict[str, Any]) -> Tuple[List[str], Dict[str, int]]:
    order = story_graph.get("story_order_top_to_bottom") or []
    return order, {name: i for i, name in enumerate(order)}


def _point_vertical_offset(p: Dict[str, Any]) -> float:
    """
    Phase-1 'explicit_z' (when present) and legacy 'z' can appear as vertical
    offsets with respect to the story elevation (not absolute world Z).
    """
    v = p.get("explicit_z")
    if isinstance(v, (int, float)):
        return float(v)
    v = p.get("z")
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def _grid_nodes_from_story_graph(story_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build deterministic grid nodes from active_points per story.
    Prefer pre-resolved absolute Z in the story_graph; otherwise compute Z = story_elev - offset.
    """
    out: List[Dict[str, Any]] = []
    order, sidx = _story_index_map(story_graph)
    elev_by_story: Dict[str, float] = story_graph.get("story_elev") or {}

    active_points: Dict[str, List[Dict[str, Any]]] = story_graph.get("active_points") or {}
    for sname, pts in active_points.items():
        idx = sidx.get(sname)
        if idx is None:
            continue
        z_story = float(elev_by_story.get(sname, 0.0))
        for p in pts:
            pid_raw = p.get("id", p.get("tag"))
            pid_str = str(pid_raw) if pid_raw is not None else ""
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            tag = pid * 1000 + idx
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))

            # Prefer absolute Z already stored in story_graph
            z_field = p.get("z")
            if isinstance(z_field, (int, float)):
                z = float(z_field)
            else:
                z = z_story - _point_vertical_offset(p)

            out.append(
                {
                    "tag": tag,
                    "x": x,
                    "y": y,
                    "z": z,
                    "story": sname,
                    "story_index": idx,
                    "kind": "grid",
                    "source_point_id": pid_str,
                }
            )
    return out


def _build_coord_map(nodes: List[Dict[str, Any]]) -> Dict[int, Tuple[float, float, float]]:
    m: Dict[int, Tuple[float, float, float]] = {}
    for n in nodes:
        m[int(n["tag"])] = (float(n["x"]), float(n["y"]), float(n["z"]))
    return m


def _master_nodes_from_diaphragms(
    diaphragms: Dict[str, Any],
    story_graph: Dict[str, Any],
    grid_nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Reconstruct diaphragm master nodes:
    - tag is provided in diaphragms.json as 'master'
    - x,y computed as centroid of slave node coordinates
    - z taken from story_elev (or mean slave z if story elevation missing)
    """
    out: List[Dict[str, Any]] = []
    if not diaphragms:
        return out

    order, sidx = _story_index_map(story_graph)
    elev_by_story: Dict[str, float] = story_graph.get("story_elev") or {}
    coord_map = _build_coord_map(grid_nodes)

    for rec in diaphragms.get("diaphragms", []):
        sname = rec.get("story")
        if sname not in sidx:
            continue
        idx = sidx[sname]
        slaves = rec.get("slaves") or []
        if not slaves:
            continue
        xs, ys, zs = [], [], []
        for t in slaves:
            try:
                tag = int(t)
            except Exception:
                continue
            if tag in coord_map:
                x, y, z = coord_map[tag]
                xs.append(x)
                ys.append(y)
                zs.append(z)

        if not xs:
            continue

        x_c = sum(xs) / len(xs)
        y_c = sum(ys) / len(ys)
        z_c = float(elev_by_story.get(sname, (sum(zs) / len(zs))))

        try:
            master_tag = int(rec.get("master"))
        except Exception:
            continue

        out.append(
            {
                "tag": master_tag,
                "x": x_c,
                "y": y_c,
                "z": z_c,
                "story": sname,
                "story_index": idx,
                "kind": "diaphragm_master",
            }
        )

    return out


def emit_nodes_json(out_dir: str = "out") -> str:
    """
    Build and save nodes.json to 'out_dir'. Returns the output path.
    """
    sg = _load_json(os.path.join(out_dir, "story_graph.json"))
    dg = _load_json(os.path.join(out_dir, "diaphragms.json"))

    grid_nodes = _grid_nodes_from_story_graph(sg)
    master_nodes = _master_nodes_from_diaphragms(dg, sg, grid_nodes)

    all_nodes = {int(n["tag"]): n for n in grid_nodes}
    for m in master_nodes:
        all_nodes[int(m["tag"])] = m  # overwrite if collision (shouldn't happen)

    nodes_list = [all_nodes[k] for k in sorted(all_nodes.keys())]
    out = {
        "nodes": nodes_list,
        "counts": {"total": len(nodes_list), "grid": len(grid_nodes), "master": len(master_nodes)},
        "version": 1,
    }

    out_path = os.path.join(out_dir, "nodes.json")
    _save_json(out_path, out)
    print(f"[ARTIFACTS] Wrote nodes.json with {out['counts']['total']} nodes "
          f"({out['counts']['grid']} grid, {out['counts']['master']} masters) at: {out_path}")
    return out_path


if __name__ == "__main__":
    try:
        from config import OUT_DIR as _DEFAULT_OUT
    except Exception:
        _DEFAULT_OUT = "out"
    emit_nodes_json(_DEFAULT_OUT)
