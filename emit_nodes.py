# -*- coding: utf-8 -*-
"""
emit_nodes.py

Emit the nodes artifact (nodes.json) representing the actual OpenSees node set:
- Grid nodes derived from story_graph.json (Phase-1, Z already resolved there)
- Diaphragm master nodes derived from diaphragms.json (Phase-2)
- Intermediate interface nodes registered by rigid-end splitting
  via _intermediate_nodes.json, merged here without changing nodes.json schema.

Deterministic tag:
    grid/master: node_tag = point_id * 1000 + story_index   (story_index = 0 at top, increasing downward)
    intermediate (rigid interfaces): bounded 32-bit deterministic hash in the band
        [1_500_000_000, 2_100_000_000), with even/odd = I/J to avoid overlap.
        Collisions are resolved by +2 probing (still deterministic for a given set).

Z rule used by this emitter (aligned with current repo):
  - Prefer the pre-resolved absolute Z stored in story_graph.json at active_points[*]["z"].
  - If that key is missing, compute:
        Z = story_elev[story] - offset
    where offset = active_points[*]["explicit_z"] if present, else active_points[*]["z"] (legacy offset), else 0.0.

Output schema (stable, v1):
{
  "nodes": [...],
  "counts": {"total": N, "grid": G, "master": M},
  "version": 1
}

Python: 3.11+
"""
from __future__ import annotations

import json
import math
import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple


# -----------------------
# JSON I/O helpers
# -----------------------
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


# -----------------------
# Story helpers
# -----------------------
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


# -----------------------
# Grid/master reconstruction
# -----------------------
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


# -----------------------
# Intermediate node registry (32-bit bounded tags)
# -----------------------
_INT32_MAX = 2_147_483_647
# Reserve a safe band for interface nodes. We'll fill it with even/odd = I/J.
_IF_BASE = 1_500_000_000
_IF_SLOTS = 300_000_000  # *2 for I/J => up to ~2.1e9


def _registry_path(out_dir: str) -> str:
    # Internal, auxiliary file merged into nodes.json by emit_nodes_json()
    return os.path.join(out_dir, "_intermediate_nodes.json")


def _load_registry(out_dir: str) -> Dict[str, Any]:
    path = _registry_path(out_dir)
    data = _load_json(path)
    if "nodes" not in data:
        data["nodes"] = []
    return data


def _save_registry(out_dir: str, data: Dict[str, Any]) -> None:
    path = _registry_path(out_dir)
    _save_json(path, data)
    print(f"[ARTIFACTS] Updated intermediate node registry at: {path}")


def _interface_slot(min_tag: int, max_tag: int, end: str) -> int:
    """
    Deterministic 64-bit hash -> slot in [0, _IF_SLOTS).
    """
    key = f"{min_tag}|{max_tag}|{str(end).upper()}".encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()  # 64-bit
    return int.from_bytes(h, "big") % _IF_SLOTS


def _interface_tag32(i_tag: int, j_tag: int, end: str, used: Optional[set[int]] = None) -> int:
    """
    Compute a 32-bit bounded deterministic tag for interface nodes.
    Even tags -> I-end (add 0), odd tags -> J-end (add 1).
    Linear probing by +2 on collision.
    """
    a, b = sorted((int(i_tag), int(j_tag)))
    end_u = str(end).upper()
    end_code = 0 if end_u == "I" else 1
    slot = _interface_slot(a, b, end_u)
    used_set = used or set()
    for step in range(_IF_SLOTS):
        cand = _IF_BASE + ((slot + step) % _IF_SLOTS) * 2 + end_code
        if cand not in used_set:
            return cand
    raise RuntimeError("[emit_nodes] Interface tag range exhausted; too many collisions.")


def _find_by_source(reg: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    for n in reg.get("nodes", []):
        if str(n.get("source")) == source:
            return n
    return None


def register_intermediate_node(
    out_dir: str,
    i_tag: int,
    j_tag: int,
    end: str,
    x: float,
    y: float,
    z: float,
    story_index: int,
    story_name: str,
    kind: str = "rigid_interface",
) -> int:
    """
    Register (idempotently) an intermediate interface node. Returns the 32-bit tag.

    Behavior:
      - If an entry with the same 'source' exists and has an out-of-range tag,
        it is rewritten to the 32-bit band (collision-safe).
      - Otherwise, a new entry is appended with a fresh deterministic tag.
    """
    end_u = str(end).upper()
    assert end_u in ("I", "J"), "end must be 'I' or 'J'"

    reg = _load_registry(out_dir)
    existing_by_tag = {int(n.get("tag")): n for n in reg.get("nodes", []) if isinstance(n.get("tag"), int)}
    used_tags: set[int] = set(existing_by_tag.keys())
    source_key = f"interface({int(i_tag)},{int(j_tag)},{end_u})"

    # If same source exists, reuse/repair
    existing = _find_by_source(reg, source_key)
    if existing is not None:
        try:
            etag = int(existing.get("tag"))
        except Exception:
            etag = -1
        if 0 < etag <= _INT32_MAX:
            # Valid and in-range -> reuse as-is
            return etag
        # Out-of-range or invalid -> repair with a bounded tag
        new_tag = _interface_tag32(int(i_tag), int(j_tag), end_u, used=used_tags)
        existing["tag"] = int(new_tag)
        existing["x"] = float(x)
        existing["y"] = float(y)
        existing["z"] = float(z)
        existing["story"] = str(story_name)
        existing["story_index"] = int(story_index)
        existing["kind"] = str(kind)
        _save_registry(out_dir, reg)
        print(f"[emit_nodes] Repaired oversized interface node tag -> {new_tag} for {source_key}")
        return new_tag

    # Brand new registration
    tag = _interface_tag32(int(i_tag), int(j_tag), end_u, used=used_tags)

    node_rec = {
        "tag": int(tag),
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "story": str(story_name),
        "story_index": int(story_index),
        "kind": str(kind),
        "source": source_key,
    }
    reg["nodes"].append(node_rec)
    reg["counts"] = {"intermediate": len(reg["nodes"])}
    reg["version"] = 1
    _save_registry(out_dir, reg)
    print(f"[emit_nodes] Registered intermediate node tag={tag} at ({x:.4f},{y:.4f},{z:.4f}) story='{story_name}'")
    return tag


# -----------------------
# Emit nodes.json (merged)
# -----------------------
def emit_nodes_json(out_dir: str = "out") -> str:
    """
    Build and save nodes.json to 'out_dir'. Returns the output path.
    Merges grid + masters + any registered intermediate nodes (schema unchanged).
    """
    sg = _load_json(os.path.join(out_dir, "story_graph.json"))
    dg = _load_json(os.path.join(out_dir, "diaphragms.json"))
    reg = _load_registry(out_dir)

    grid_nodes = _grid_nodes_from_story_graph(sg)
    master_nodes = _master_nodes_from_diaphragms(dg, sg, grid_nodes)
    interm_nodes: List[Dict[str, Any]] = reg.get("nodes", [])

    all_nodes: Dict[int, Dict[str, Any]] = {int(n["tag"]): n for n in grid_nodes}
    for m in master_nodes:
        all_nodes[int(m["tag"])] = m  # overwrite if collision (shouldn't happen)
    for k in interm_nodes:
        # Only accept in-range, integer tags when merging
        try:
            kt = int(k["tag"])
        except Exception:
            continue
        if 0 < kt <= _INT32_MAX:
            all_nodes[kt] = k

    nodes_list = [all_nodes[k] for k in sorted(all_nodes.keys())]
    out = {
        "nodes": nodes_list,
        "counts": {
            "total": len(nodes_list),
            "grid": len(grid_nodes),
            "master": len(master_nodes),
        },
        "version": 1,
    }

    out_path = os.path.join(out_dir, "nodes.json")
    _save_json(out_path, out)
    print(
        "[ARTIFACTS] Wrote nodes.json with "
        f"{out['counts']['total']} nodes ({out['counts']['grid']} grid, "
        f"{out['counts']['master']} masters, +{len(interm_nodes)} intermediates merged) at: {out_path}"
    )
    return out_path


if __name__ == "__main__":
    try:
        from config import OUT_DIR as _DEFAULT_OUT
    except Exception:
        _DEFAULT_OUT = "out"
    emit_nodes_json(_DEFAULT_OUT)
