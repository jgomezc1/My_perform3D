# diaphragms.py
"""
Rigid diaphragm creation for OpenSeesPy (hard-default: centroid master node).

Detection rule (per story):
- Use story_graph.json → active_points[story].
- Exclude points with explicit_z=True (they're not on the story plane).
- If all remaining points have a DIAPHRAGM name (optionally validated against
  parsed_raw.json['diaphragm_names']), the story is a single rigid diaphragm group.

Master-node rule (hard default):
- Compute the XY centroid of the story's diaphragm points.
- Create a NEW OpenSees node at (cx, cy, cz) where cz is the average Z of the
  points (they should coincide with the story plane).
- Use this new node as the master; all existing story nodes become slaves.

OpenSees call:
- Prefer rigidDiaphragm(3, master, *slaves) (perpDirn=3 for XY).
- Fall back to rigidDiaphragm(master, *slaves) if wrapper lacks the first arg.

Determinism:
- New node tags allocated as max(existing_tags)+1 at creation time.
- We also persist a deterministic summary to out/diaphragms.json for the viewer.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Set
import json
import os

from openseespy.opensees import (
    rigidDiaphragm as _ops_rigidDiaphragm,
    node as _ops_node,
    getNodeTags as _ops_getNodeTags,
    fix as _ops_fix,  # <-- ADDED
)

# Tolerance (available if expanded checks are needed)
try:
    from config import EPS  # type: ignore
except Exception:
    EPS = 1e-9

# Output dir (if present in config, use it; otherwise default to ./out)
try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _call_rigid_diaphragm(master: int, slaves: List[int]) -> None:
    """Try modern signature first (perpDirn=3), then legacy. Log both cases."""
    try:
        _ops_rigidDiaphragm(3, master, *slaves)
        print(f"[diaphragms] rigidDiaphragm(3, {master}, {', '.join(map(str, slaves))})")
    except TypeError:
        print("[diaphragms] Falling back to rigidDiaphragm(master, *slaves).")
        _ops_rigidDiaphragm(master, *slaves)
        print(f"[diaphragms] rigidDiaphragm({master}, {', '.join(map(str, slaves))})")


def _centroid_xy(points_xy: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = max(len(points_xy), 1)
    sx = sum(x for x, _ in points_xy)
    sy = sum(y for _, y in points_xy)
    return sx / n, sy / n


def define_rigid_diaphragms(
    story_path: str = "out/story_graph.json",
    raw_path: str   = "out/parsed_raw.json"
) -> List[Tuple[str, int, List[int]]]:
    """
    Identify and create rigid diaphragms per story, using a NEW centroid master node.

    Returns
    -------
    List[(story_name, master_tag, [slave_tags...])]
    """
    story = _load_json(story_path)
    raw   = _load_json(raw_path)

    story_names: List[str] = story.get("story_order_top_to_bottom", [])
    story_index = {s: i for i, s in enumerate(story_names)}
    by_story_pts: Dict[str, List[Dict[str, Any]]] = story.get("active_points", {})
    valid_names: Set[str] = set(raw.get("diaphragm_names", []))  # optional; may be empty

    created: List[Tuple[str, int, List[int]]] = []
    skips: List[str] = []

    # Determine next available node tag for creating centroid nodes
    existing_tags = list(map(int, _ops_getNodeTags() or []))
    next_tag = (max(existing_tags) + 1) if existing_tags else 1

    for sname in story_names:
        pts = by_story_pts.get(sname, [])
        if not pts:
            continue

        # Exclude points not on the story plane
        plane_pts = [p for p in pts if not p.get("explicit_z", False)]
        if not plane_pts:
            skips.append(f"{sname}: no plane points after explicit-Z exclusion")
            continue

        # All must have a diaphragm name
        names = {p.get("diaphragm") for p in plane_pts}
        if None in names or "" in names:
            skips.append(f"{sname}: not all points have DIAPHRAGM assigned")
            continue
        if len(names) != 1:
            skips.append(f"{sname}: multiple diaphragms present ({sorted(list(names))}); skipped")
            continue

        # Gather slave node tags and coordinates
        idx = story_index[sname]
        tags_coords: List[Tuple[int, float, float, float]] = []
        for p in plane_pts:
            # Prefer explicit tag if Phase-1 carried it; otherwise reconstruct deterministically
            tag = int(p.get("tag")) if "tag" in p else int(p["id"]) * 1000 + int(idx)
            tags_coords.append((tag, float(p["x"]), float(p["y"]), float(p["z"])))

        if len(tags_coords) < 2:
            skips.append(f"{sname}: fewer than 2 nodes → cannot define rigidDiaphragm")
            continue

        # Compute centroid on XY; Z from average of plane points
        xs_ys = [(x, y) for (_, x, y, _) in tags_coords]
        cx, cy = _centroid_xy(xs_ys)
        cz = sum(z for (_, _, _, z) in tags_coords) / len(tags_coords)

        # Create NEW master node at centroid with a deterministic new tag
        master_tag = next_tag
        next_tag += 1
        _ops_node(master_tag, cx, cy, cz)

        # Constrain master node DOFs per rigid-diaphragm convention:
        # UX, UY, RZ = free (0); UZ, RX, RY = fixed (1)
        try:
            _ops_fix(master_tag, 0, 0, 1, 1, 1, 0)
            print(f"[diaphragms] fix({master_tag}, 0,0,1,1,1,0)")
        except Exception as e:
            print(f"[diaphragms] WARN: fix() failed for master {master_tag}: {e}")

        slave_tags = [t for (t, _, _, _) in tags_coords]
        print(f"[diaphragms] {sname}: created centroid master node tag={master_tag} "
              f"at ({cx:.3f},{cy:.3f},{cz:.3f}); slaves={len(slave_tags)}")

        # Apply rigid diaphragm constraint (prefer perpDirn=3 signature)
        try:
            _call_rigid_diaphragm(master_tag, slave_tags)
            created.append((sname, master_tag, slave_tags))
        except Exception as e:
            skips.append(f"{sname}: rigidDiaphragm failed: {e}")

    # Persist deterministic metadata for the viewer
    try:
        out_dir = str(OUT_DIR) if OUT_DIR is not None else "out"
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "version": 1,
            "stories_top_to_bottom": story_names,
            "diaphragms": [
                {"story": s, "master": m, "slaves": sl, "master_fix": [0,0,1,1,1,0]}
                for (s, m, sl) in created
            ],
        }
        with open(os.path.join(out_dir, "diaphragms.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[diaphragms] Wrote {os.path.join(out_dir, 'diaphragms.json')}")
    except Exception as e:
        print(f"[diaphragms] WARN: failed to write diaphragms.json: {e}")

    # Logs
    if created:
        print(f"[diaphragms] Created {len(created)} rigid diaphragm(s).")
    else:
        print("[diaphragms] No rigid diaphragms created.")
    if skips:
        print("[diaphragms] Skips:")
        for s in skips:
            print(" -", s)

    return created
