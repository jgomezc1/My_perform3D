# diaphragms.py
"""
Rigid diaphragm creation for OpenSeesPy, with master-node **mass()** and **fix()**
application, and emission of Phase-2 artifact `diaphragms.json`.

Rules
-----
- Treat the diaphragm label **"DISCONNECTED"** (any case) as **no diaphragm**.
- Any story that contains restraint/support nodes must **NOT** get a diaphragm.
- All-or-nothing per story: create a SINGLE diaphragm per story only if every
  candidate point on that story has a valid diaphragm label (not DISCONNECTED).
- Master node is a NEW node at the XY centroid (Z = mean of candidates' z).
- Constraint: rigidDiaphragm 3 <master> <slaves...>  (3 = plane ⟂ to Z)
- Fixities on master: fix(master, 0, 0, 1, 1, 1, 0)  (free UX, UY, RZ)
- Mass on master: mass(master, M, M, 0, 0, 0, Izz) with
      M = ρ * t * A     and     Izz = RZ_MASS_FACTOR * M

Config (overridable in config.py)
---------------------------------
OUT_DIR: str = "out"
SLAB_THICKNESS: float = 0.10        # m
CONCRETE_DENSITY: float = 2500.0    # kg/m^3
RZ_MASS_FACTOR: float = 100.0       # Izz = factor * M
EPS: float = 1e-9
PLANE_TOL: float = 1e-6             # z-plane tolerance for attaching intermediates

Outputs
-------
Writes OUT_DIR/diaphragms.json with one record per created diaphragm:
{
  "story": "Story-1",
  "master": 9001,
  "slaves": [101000, 102000, ...],
  "mass": {"M": ..., "Izz": ..., "A": ..., "t": ..., "rho": ..., "applied": true/false},
  "fix":  {"ux": 0, "uy": 0, "uz": 1, "rx": 1, "ry": 1, "rz": 0, "applied": true/false}
}
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
    model as _ops_model,
    wipe as _ops_wipe,
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

try:
    from config import PLANE_TOL  # type: ignore
except Exception:
    PLANE_TOL = 1e-6  # allowable |z - story_elev| to attach intermediate nodes


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
    """Simple centroid of XY points."""
    if not pts_xy:
        return (0.0, 0.0)
    sx = sum(p[0] for p in pts_xy)
    sy = sum(p[1] for p in pts_xy)
    n = float(len(pts_xy))
    return (sx / n, sy / n)


def _cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew’s monotone chain, returns hull in CCW order, no duplicate last point."""
    pts = sorted(set(pts))
    if len(pts) <= 1:
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
    return lower[:-1] + upper[:-1]


def _polygon_area(pts_ccw: List[Tuple[float, float]]) -> float:
    if len(pts_ccw) < 3:
        return 0.0
    a = 0.0
    for (x1, y1), (x2, y2) in zip(pts_ccw, pts_ccw[1:] + [pts_ccw[0]]):
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
    except Exception:
        return idxs
    sup_nodes = data.get("supports", [])
    for s in sup_nodes:
        try:
            t = int(s.get("tag"))
            idxs.add(t % 1000)
        except Exception:
            continue
    # Defensive clamp
    idxs = set(i for i in idxs if 0 <= i < story_count)
    return idxs


def _ensure_ops_model(ndm: int = 3, ndf: int = 6) -> None:
    """Ensure a valid OpenSees model exists so node/mass/fix/rigidDiaphragm calls succeed."""
    try:
        _ops_wipe()
    except Exception:
        pass
    try:
        _ops_model("basic", "-ndm", ndm, "-ndf", ndf)
        print(f"[diaphragms] Initialized OpenSees model: ndm={ndm}, ndf={ndf}")
    except Exception as e:
        print(f"[diaphragms] WARN: failed to initialize OpenSees model: {e}")


def define_rigid_diaphragms(
    story_path: str = os.path.join(OUT_DIR, "story_graph.json"),
    raw_path: str = os.path.join(OUT_DIR, "parsed_raw.json"),
    supports_path: str = os.path.join(OUT_DIR, "supports.json"),
) -> List[Tuple[str, int, List[int]]]:
    """Identify and create rigid diaphragms per story (single group per story)."""
    _ensure_ops_model()

    # Inputs
    try:
        sg = _read_json(story_path)
    except Exception as e:
        print(f"[diaphragms] ERROR reading {story_path}: {e}")
        return []

    try:
        pr = _read_json(raw_path)
    except Exception:
        pr = {}

    story_order: List[str] = sg.get("story_order_top_to_bottom", [])
    story_elev: Dict[str, float] = sg.get("story_elev", {})
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

    # Accumulate meta for JSON (mass, inertia, area, fix)
    meta: List[Dict[str, Any]] = []

    for sname in story_order:
        pts = list(active_points.get(sname, []))
        if not pts:
            skips.append(f"{sname}: no active points")
            continue

        sidx = story_index.get(sname, None)
        if sidx is None:
            skips.append(f"{sname}: missing story index")
            continue

        if sidx in idx_with_supports:
            skips.append(f"{sname}: story has supports → skip diaphragm")
            continue

        # Filter to candidates on same plane (z ≈ story_elev)
        z0 = float(story_elev.get(sname, 0.0))
        plane_pts: List[Dict[str, Any]] = []
        for p in pts:
            try:
                z = float(p.get("z", z0))
            except Exception:
                z = z0
            if abs(z - z0) <= EPS:
                plane_pts.append(p)

        if not plane_pts:
            skips.append(f"{sname}: no candidates on story plane z={z0}")
            continue

        # Diaphragm labels
        labels: List[str | None] = []
        all_valid_named = True
        for p in plane_pts:
            label = str(p.get("diaphragm", "")).strip()
            if not label or label.lower() == "disconnected":
                labels.append(None)
                all_valid_named = False
            else:
                labels.append(label)
                if known_diaph and (label not in known_diaph):
                    all_valid_named = False

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
            p_id = int(p.get("tag", p.get("id")))
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
        area = _polygon_area(hull)

        # Create master node
        master_tag = next_tag_base
        next_tag_base += 1
        try:
            _ops_node(master_tag, cx, cy, cz)
            print(f"[diaphragms] master node {master_tag} @ ({cx:.3f},{cy:.3f},{cz:.3f}) for story '{sname}'")
        except Exception as e:
            skips.append(f"{sname}: failed to create master node: {e}")
            continue

        # Mass & fixity on master
        M = CONCRETE_DENSITY * SLAB_THICKNESS * area
        Izz = RZ_MASS_FACTOR * M
        mass_applied = False
        fix_applied = False
        try:
            _ops_mass(master_tag, M, M, 0.0, 0.0, 0.0, Izz)
            mass_applied = True
        except Exception as e:
            print(f"[diaphragms] WARN mass({master_tag}) failed: {e}")
        try:
            _ops_fix(master_tag, 0, 0, 1, 1, 1, 0)
            fix_applied = True
        except Exception as e:
            print(f"[diaphragms] WARN fix({master_tag}) failed: {e}")

        # Constraint
        slave_tags = sorted(t for t, *_ in tags_coords if t != master_tag)
        try:
            _call_rigid(master_tag, slave_tags)
        except Exception as e:
            skips.append(f"{sname}: rigidDiaphragm failed: {e}")
            continue

        created.append((sname, master_tag, slave_tags))
        meta.append({
            "story": sname,
            "master": master_tag,
            "slaves": slave_tags,
            "mass": {"M": M, "Izz": Izz, "A": area, "t": SLAB_THICKNESS, "rho": CONCRETE_DENSITY, "applied": mass_applied},
            "fix":  {"ux": 0, "uy": 0, "uz": 1, "rx": 1, "ry": 1, "rz": 0, "applied": fix_applied}
        })

    # Persist viewer metadata
    out_json = {"diaphragms": meta}
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "diaphragms.json"), "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2)
        print(f"[diaphragms] Wrote {OUT_DIR}/diaphragms.json")
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

    # Attach intermediate rigid-interface nodes into slaves (z-plane filtered)
    try:
        added = attach_intermediate_nodes_to_rds(OUT_DIR)
        print(f"[diaphragms] Attached {added} intermediate node(s) to diaphragms.")
    except Exception as e:
        print(f"[diaphragms] attach: WARN: {e}")

    return created


def attach_intermediate_nodes_to_rds(out_dir: str = OUT_DIR, inter_file: str = "_intermediate_nodes.json") -> int:
    """
    Post-process OUT_DIR/diaphragms.json to attach nodes with kind="rigid_interface"
    to the slaves list of the diaphragm that matches their `story` field **and**
    whose z-coordinate lies on the story plane within PLANE_TOL.

    - Does NOT change schema.
    - Idempotent and deterministic (de-duplicates and sorts).
    - Returns the number of attachments performed (new tags actually added).
    """
    diaph_path = os.path.join(out_dir, "diaphragms.json")
    inter_path = os.path.join(out_dir, inter_file)
    story_path = os.path.join(out_dir, "story_graph.json")

    if not os.path.exists(diaph_path):
        print(f"[diaphragms] attach: {diaph_path} not found; nothing to do.")
        return 0
    if not os.path.exists(inter_path):
        print(f"[diaphragms] attach: {inter_path} not found; nothing to do.")
        return 0
    if not os.path.exists(story_path):
        print(f"[diaphragms] attach: {story_path} not found; cannot z-filter; nothing to do.")
        return 0

    try:
        with open(diaph_path, "r", encoding="utf-8") as f:
            diaph = json.load(f)
        with open(inter_path, "r", encoding="utf-8") as f:
            inter = json.load(f)
        with open(story_path, "r", encoding="utf-8") as f:
            sg = json.load(f)
    except Exception as e:
        print(f"[diaphragms] attach: failed to read inputs: {e}")
        return 0

    di_list = diaph.get("diaphragms") or []
    nodes = inter.get("nodes") or []
    story_elev: Dict[str, float] = sg.get("story_elev", {})

    # Build story -> set(tags) from intermediate nodes, z-plane filtered
    by_story: Dict[str, Set[int]] = {}
    skipped_offplane = 0
    for n in nodes:
        try:
            if str(n.get("kind", "")).strip().lower() != "rigid_interface":
                continue
            s = str(n.get("story", "")).strip()
            if not s or s not in story_elev:
                continue
            t = int(n.get("tag"))
            z = float(n.get("z"))
            if abs(z - float(story_elev[s])) > PLANE_TOL:
                skipped_offplane += 1
                continue
        except Exception:
            continue
        by_story.setdefault(s, set()).add(t)

    if skipped_offplane:
        print(f"[diaphragms] attach: skipped {skipped_offplane} intermediate node(s) off the story plane (> {PLANE_TOL}).")

    if not by_story:
        print("[diaphragms] attach: no intermediate nodes to attach (after z-plane filter).")
        return 0

    added = 0
    for d in di_list:
        sname = str(d.get("story", "")).strip()
        if not sname or sname not in by_story:
            continue
        slaves = list(map(int, d.get("slaves") or []))
        before = set(slaves)
        after = before | by_story[sname]
        if after != before:
            added += len(after - before)
            d["slaves"] = sorted(after)

    try:
        with open(diaph_path, "w", encoding="utf-8") as f:
            json.dump({"diaphragms": di_list, **{k: v for k, v in diaph.items() if k != "diaphragms"}}, f, indent=2)
        print(f"[diaphragms] attach: updated {diaph_path} (+{added} tag(s)).")
    except Exception as e:
        print(f"[diaphragms] attach: failed to write {diaph_path}: {e}")
        return 0

    return added
