# diaphragms.py
"""
Rigid diaphragm creation for OpenSeesPy.

Bug fix (kept):
- Treat the diaphragm label **"DISCONNECTED"** (any case) as **no diaphragm**.
- Additionally, **any story that contains restraint/support nodes must NOT get a diaphragm**.

Behavior (this module now also handles master-node BCs and mass):
- For each eligible story (all-or-nothing labeling rule), create ONE **master node**
  at the (x,y)-centroid of the story’s active points and tie all story nodes to it
  with `rigidDiaphragm 3 master slaves...` (plane ⟂ Z; ties UX, UY, RZ).
- Apply **fix(master, 0, 0, 1, 1, 1, 0)** so masters cannot move vertically (UZ)
  nor rock about X,Y (RX, RY). In-plane DOFs (UX, UY, RZ) remain free.
- Assign a **lumped translational mass** to the master:
      M = ρ * t * A
  where A is the convex-hull area of the candidate points on that story,
  t is the assumed slab thickness, and ρ is the concrete mass density.
  Rotational inertia about Z is set to **Izz = RZ_MASS_FACTOR * M** (simple proxy).

Config (overrides via config.py if present):
- OUT_DIR: output folder (default "out")
- EPS: tolerance (default 1e-9) — not critical here
- SLAB_THICKNESS: float meters (default 0.10)
- CONCRETE_DENSITY: float kg/m^3 (default 2500.0)
- RZ_MASS_FACTOR: float dimensionless (default 100.0)

Outputs:
- Returns a list of (story_name, master_tag, [slave_tags...]).
- Writes a compact JSON summary to OUT_DIR/diaphragms.json for the viewer.
  Each record also includes "mass" and "izz" for transparency.
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Set
import json
import os

from openseespy.opensees import (
    rigidDiaphragm as _ops_rigidDiaphragm,
    node as _ops_node,
    getNodeTags as _ops_getNodeTags,
    fix as _ops_fix,
    mass as _ops_mass,
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

try:
    from config import SLAB_THICKNESS  # type: ignore
except Exception:
    SLAB_THICKNESS = 0.10  # m

try:
    from config import CONCRETE_DENSITY  # type: ignore
except Exception:
    CONCRETE_DENSITY = 2500.0  # kg/m^3

try:
    from config import RZ_MASS_FACTOR  # type: ignore
except Exception:
    RZ_MASS_FACTOR = 100.0  # Izz = factor * M


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


def _cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Monotone chain convex hull. Returns vertices in CCW order (first==last not repeated)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    # Concatenate without duplicating first/last point
    return lower[:-1] + upper[:-1]


def _polygon_area(pts_ccw: List[Tuple[float, float]]) -> float:
    """Signed area (positive for CCW)."""
    n = len(pts_ccw)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = pts_ccw[i]
        x2, y2 = pts_ccw[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


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
            if 0 <= sidx < max(story_count, 1):
                idxs.add(sidx)
    except Exception as e:
        print(f"[diaphragms] WARN: failed reading supports: {e}")
    return idxs


def define_rigid_diaphragms(
    story_path: str = os.path.join(OUT_DIR, "story_graph.json"),
    raw_path: str = os.path.join(OUT_DIR, "parsed_raw.json"),
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

    # We'll also accumulate meta for JSON (mass, inertia, area)
    meta: List[Dict[str, Any]] = []

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

        # Compute convex-hull area for mass proxy
        hull = _convex_hull(list(zip(xs, ys)))
        area = _polygon_area(hull)  # m^2
        M = CONCRETE_DENSITY * SLAB_THICKNESS * area  # kg (lumped translational mass)
        Izz = RZ_MASS_FACTOR * M                       # crude proxy for polar inertia

        # Create master node with a fresh tag
        master_tag = next_tag_base
        next_tag_base += 1
        _ops_node(master_tag, cx, cy, cz)

        # Apply lumped mass and out-of-plane fixities to the master
        try:
            _ops_mass(master_tag, M, M, 0.0, 0.0, 0.0, Izz)
            print(f"[diaphragms] mass(master={master_tag}, M={M:.3f}, Izz={Izz:.3f}) (t={SLAB_THICKNESS}, ρ={CONCRETE_DENSITY}, A={area:.3f})")
        except Exception as e:
            print(f"[diaphragms] WARN: failed applying mass to master {master_tag}: {e}")

        try:
            _ops_fix(master_tag, 0, 0, 1, 1, 1, 0)
            print(f"[diaphragms] fix(master={master_tag}, ux=0, uy=0, uz=1, rx=1, ry=1, rz=0)")
        except Exception as e:
            print(f"[diaphragms] WARN: failed applying fix to master {master_tag}: {e}")

        # Slaves are the existing story nodes
        slave_tags = [t for (t, _, _, _) in tags_coords]

        # Apply rigid diaphragm constraint (XY plane => perpDirn=3)
        _call_rigid(master_tag, slave_tags)

        created.append((sname, master_tag, slave_tags))
        meta.append({
            "story": sname,
            "master": master_tag,
            "slaves": slave_tags,
            "mass": {"M": M, "Izz": Izz, "A": area, "t": SLAB_THICKNESS, "rho": CONCRETE_DENSITY}
        })

    # Persist viewer metadata
    out_json = {"diaphragms": meta}
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
