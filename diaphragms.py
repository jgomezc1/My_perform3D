# diaphragms.py
"""
Rigid diaphragm creation for OpenSeesPy.

Bug fix (requested):
- Treat the diaphragm label **"DISCONNECTED"** (any case) as **no diaphragm**.
- Additionally, **any story that contains restraint/support nodes must NOT get a diaphragm**.

General rules (kept consistent with the project):
- Stories are processed in the order given by story_graph.json["story_order_top_to_bottom"]
  (index 0 = top story).
- Candidate nodes per story come from story_graph.json["active_points"][story], with
  points having explicit_z==True excluded (they are not on the story plane).
- (All-or-nothing per story) A rigid diaphragm is created only if **every candidate**
  on that story carries a *valid diaphragm name* that is **not** "DISCONNECTED".
  We do not split a story into multiple diaphragm groups.
- Master node is a NEW node at the XY centroid (Z = mean candidates' z).
- Constraint: rigidDiaphragm 3 <master> <slaves...>  (3 = plane ⟂ to Z)

Outputs:
- Returns a list of (story_name, master_tag, [slave_tags...]).
- Writes a compact JSON summary to OUT_DIR/diaphragms.json for the viewer.
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Set
import json
import os

from openseespy.opensees import (
    rigidDiaphragm as _ops_rigidDiaphragm,
    node as _ops_node,
    getNodeTags as _ops_getNodeTags,
)

# Optional config hooks
try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = "out"

try:
    from config import EPS  # type: ignore
except Exception:
    EPS = 1e-9


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _call_rigid(master: int, slaves: List[int]) -> None:
    """Try modern signature (perpDirn=3) then fall back to legacy."""
    try:
        _ops_rigidDiaphragm(3, master, *slaves)
        print(f"[diaphragms] rigidDiaphragm(3, {master}, {', '.join(map(str, slaves))})")
    except TypeError:
        print("[diaphragms] NOTE: falling back to rigidDiaphragm(master, *slaves)")
        _ops_rigidDiaphragm(master, *slaves)
        print(f"[diaphragms] rigidDiaphragm({master}, {', '.join(map(str, slaves))})")


def _centroid_xy(pts_xy: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = max(len(pts_xy), 1)
    sx = sum(x for x, _ in pts_xy)
    sy = sum(y for _, y in pts_xy)
    return sx / n, sy / n


def _story_indices_with_supports(supports_path: str, story_count: int) -> Set[int]:
    """Return the set of story indices (0=top) that contain restraint nodes.

    Mapping relies on the deterministic node-tag rule:
        tag = point_int * 1000 + story_index
    Hence story_index = tag % 1000.
    """
    idxs: Set[int] = set()
    if not os.path.exists(supports_path):
        return idxs
    try:
        data = _read_json(supports_path)
        for rec in data.get("applied", []):
            tag = int(rec.get("node"))
            sidx = tag % 1000
            # only keep sensible indices
            if 0 <= sidx < max(story_count, 1):
                idxs.add(sidx)
    except Exception as e:
        print(f"[diaphragms] WARN: failed reading supports: {e}")
    return idxs


def define_rigid_diaphragms(
    story_path: str = os.path.join(OUT_DIR, "story_graph.json"),
    raw_path: str   = os.path.join(OUT_DIR, "parsed_raw.json"),
    supports_path: str = os.path.join(OUT_DIR, "supports.json"),
) -> List[Tuple[str, int, List[int]]]:
    """Identify and create rigid diaphragms per story (single group per story).

    Returns:
        List of (story_name, master_tag, [slave_tags...])
    """
    # ---- Load Phase-1 artifacts ----
    sg = _read_json(story_path)
    pr = _read_json(raw_path)

    story_order: List[str] = list(sg.get("story_order_top_to_bottom", []))
    story_elev: Dict[str, float] = {str(k): float(v) for k, v in sg.get("story_elev", {}).items()}
    active_points: Dict[str, List[Dict[str, Any]]] = sg.get("active_points", {})

    # Known diaphragm names from the .e2k (if present)
    known_diaph: Set[str] = set(str(x).strip() for x in pr.get("diaphragm_names", []))

    # Stories that have supports: EXCLUDE from diaphragm creation
    idx_with_supports = _story_indices_with_supports(supports_path, story_count=len(story_order))

    # Build mapping story -> index (0 = top)
    story_index: Dict[str, int] = {s: i for i, s in enumerate(story_order)}

    created: List[Tuple[str, int, List[int]]] = []
    skips: List[str] = []

    # Pre-existing tags for master creation
    try:
        existing_tags = list(map(int, _ops_getNodeTags()))
    except Exception:
        existing_tags = []
    next_tag_base = max(existing_tags) + 1 if existing_tags else 1

    for sname in story_order:
        pts = list(active_points.get(sname, []))
        if not pts:
            skips.append(f"{sname}: no active points")
            continue

        # Exclude points off the story plane
        plane_pts = [p for p in pts if not bool(p.get("explicit_z", False))]
        if not plane_pts:
            skips.append(f"{sname}: all points have explicit_z=True")
            continue

        sidx = story_index[sname]

        # -- Rule 0: If story has supports, skip diaphragm --
        if sidx in idx_with_supports:
            skips.append(f"{sname}: has restraint/support nodes → no rigid diaphragm")
            continue

        # -- Rule 1: Validate diaphragm labels (treat 'DISCONNECTED' as no-diaphragm) --
        labels = []
        disconnected_only = True
        all_valid_named = True
        for p in plane_pts:
            lbl_raw = p.get("diaphragm")
            lbl = str(lbl_raw).strip() if lbl_raw is not None else ""
            if lbl.upper() == "DISCONNECTED" or lbl == "":
                # Treat as unlabeled; keep disconnected_only flag true unless a valid label shows up
                labels.append(None)
            else:
                labels.append(lbl)
                disconnected_only = False
                if known_diaph and (lbl not in known_diaph):
                    all_valid_named = False  # unknown label
        if disconnected_only:
            skips.append(f"{sname}: DIAPH='DISCONNECTED' → no rigid diaphragm")
            continue

        # Require *all* candidates to have a valid (non-empty, not DISCONNECTED) label
        if not all(lbl is not None for lbl in labels):
            skips.append(f"{sname}: mixed or missing diaphragm labels → no rigid diaphragm (all-or-nothing rule)")
            continue
        if not all_valid_named:
            skips.append(f"{sname}: found label not present in diaphragm_names → skip")
            continue

        # Gather slave node tags and coordinates
        tags_coords: List[Tuple[int, float, float, float]] = []
        for p in plane_pts:
            # Deterministic tag rule: tag = point_int*1000 + story_index
            p_id = int(p.get("tag", p["id"]))
            tag = int(p_id) * 1000 + int(sidx) if "tag" not in p else int(p["tag"])
            tags_coords.append((tag, float(p["x"]), float(p["y"]), float(p["z"])))

        if len(tags_coords) < 2:
            skips.append(f"{sname}: fewer than 2 nodes → cannot define rigid diaphragm")
            continue

        # Compute centroid (XY), take mean Z (should equal story plane)
        xs = [x for _, x, _, _ in tags_coords]
        ys = [y for _, _, y, _ in tags_coords]
        zs = [z for _, _, _, z in tags_coords]
        cx, cy = _centroid_xy(list(zip(xs, ys)))
        cz = sum(zs) / len(zs)

        # Create master node with a fresh tag
        master_tag = next_tag_base
        next_tag_base += 1
        _ops_node(master_tag, cx, cy, cz)

        # Slaves are the existing story nodes
        slave_tags = [t for (t, _, _, _) in tags_coords]

        # Apply rigid diaphragm constraint (XY plane => perpDirn=3)
        _call_rigid(master_tag, slave_tags)

        created.append((sname, master_tag, slave_tags))

    # Persist viewer metadata
    out_json = {
        "diaphragms": [
            {
                "story": sname,
                "master": master,
                "slaves": slaves,
            }
            for (sname, master, slaves) in created
        ]
    }
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "diaphragms.json"), "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2)
    except Exception as e:
        print(f"[diaphragms] WARN: failed to write {OUT_DIR}/diaphragms.json: {e}")

    # Logging
    if created:
        print(f"[diaphragms] Created {len(created)} rigid diaphragm(s).")
    else:
        print("[diaphragms] No rigid diaphragms created.")
    if skips:
        print("[diaphragms] Skips:")
        for s in skips:
            print(" -", s)

    return created
