# supports.py
"""
Apply displacement boundary conditions from ETABS POINTASSIGN ... RESTRAINT
to the current OpenSeesPy domain.

Reads ETABS .e2k directly (lightweight parser) so we don't have to rerun Phase-1.
If E2K is unavailable, optionally falls back to `out/parsed_raw.json` if it
contains 'restraint' in point_assigns (future-proof).

Mapping:
  ETABS DOFs: "UX UY UZ RX RY RZ"  →  fix(nodeTag, 1/0,...,1/0) in this order.

Node tag convention:
  tag = int(point_id) * 1000 + story_index
where story_index is zero for the top story and increases downward, following
story_graph["story_order_top_to_bottom"].

Usage:
  from supports import define_point_restraints_from_e2k
  define_point_restraints_from_e2k()
"""
from __future__ import annotations

import re
import json
import os
from typing import Dict, Any, List, Tuple

from openseespy.opensees import (
    getNodeTags as _ops_getNodeTags,
    fix as _ops_fix,
)

# Config
try:
    from config import OUT_DIR, E2K_PATH  # type: ignore
except Exception:
    OUT_DIR, E2K_PATH = "out", None


_RE_POINTASSIGN_RESTRAINT = re.compile(
    r'POINT\s*ASSIGN\S*\s+"(?P<pt>\d+)"\s+"(?P<story>[^"]+)"\s+RESTRAINT\s+"(?P<dofs>[^"]+)"',
    re.IGNORECASE,
)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dofs_to_mask(tokens: List[str]) -> Tuple[int, int, int, int, int, int]:
    # Order is UX, UY, UZ, RX, RY, RZ
    order = ("UX", "UY", "UZ", "RX", "RY", "RZ")
    toks = {t.upper() for t in tokens}
    return tuple(1 if k in toks else 0 for k in order)  # type: ignore[return-value]


def _read_restraints_from_e2k(e2k_path: str) -> List[Tuple[str, str, Tuple[int,int,int,int,int,int]]]:
    """Return list of (point_id, story_name, mask) from .e2k."""
    out: List[Tuple[str, str, Tuple[int,int,int,int,int,int]]] = []
    with open(e2k_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = _RE_POINTASSIGN_RESTRAINT.search(line)
            if not m:
                continue
            pt = m.group("pt")
            story = m.group("story")
            tokens = m.group("dofs").strip().split()
            mask = _dofs_to_mask(tokens)
            out.append((pt, story, mask))
    return out


def _read_restraints_from_parsed_raw(raw_path: str) -> List[Tuple[str, str, Tuple[int,int,int,int,int,int]]]:
    """Fallback: read restraints from out/parsed_raw.json if present there."""
    if not os.path.exists(raw_path):
        return []
    raw = _load_json(raw_path)
    out: List[Tuple[str, str, Tuple[int,int,int,int,int,int]]] = []
    for pa in raw.get("point_assigns", []):
        story = pa.get("story")
        pt = pa.get("point")
        extra = pa.get("extra") or {}
        rest = pa.get("restraint") or extra.get("restraint")
        if not (story and pt and rest):
            continue
        tokens = str(rest).strip().split()
        mask = _dofs_to_mask(tokens)
        out.append((str(pt), str(story), mask))
    return out


def define_point_restraints_from_e2k(
    e2k_path: str | None = None,
    story_graph_path: str = None,
    raw_path: str = None,
) -> List[Tuple[int, Tuple[int,int,int,int,int,int]]]:
    """
    Parse ETABS RESTRAINT point-assigns and apply OpenSees 'fix' to existing nodes.

    Returns:
        List of (node_tag, mask) actually applied.
    """
    story_graph_path = story_graph_path or os.path.join(OUT_DIR or "out", "story_graph.json")
    raw_path = raw_path or os.path.join(OUT_DIR or "out", "parsed_raw.json")
    sg = _load_json(story_graph_path)
    order = sg.get("story_order_top_to_bottom") or sg.get("story_order_top_to_bottom".lower())
    if not order:
        raise RuntimeError("story_graph.json missing story_order_top_to_bottom.")
    story_idx = {s: i for i, s in enumerate(order)}

    # Source 1: E2K
    path = e2k_path or E2K_PATH
    pairs: List[Tuple[str, str, Tuple[int,int,int,int,int,int]]] = []
    if path and os.path.exists(path):
        pairs = _read_restraints_from_e2k(path)

    # Fallback: parsed_raw.json
    if not pairs:
        pairs = _read_restraints_from_parsed_raw(raw_path)

    if not pairs:
        print("[supports] No RESTRAINT entries found in E2K or parsed_raw.json. Nothing to apply.")
        return []

    existing = set(int(t) for t in (_ops_getNodeTags() or []))
    applied: List[Tuple[int, Tuple[int,int,int,int,int,int]]] = []
    skipped = 0
    for pt, story, mask in pairs:
        if story not in story_idx:
            print(f"[supports] WARN: Story '{story}' not found in story_graph; skipping point {pt}.")
            continue
        tag = int(pt) * 1000 + story_idx[story]
        if tag not in existing:
            # If node doesn't exist (e.g., explicit_z or point inactive on story), skip safely.
            print(f"[supports] Skip: node tag {tag} (pt={pt}, story={story}) not in domain.")
            skipped += 1
            continue
        try:
            _ops_fix(tag, *mask)
            print(f"[supports] fix({tag}, {','.join(map(str,mask))})")
            applied.append((tag, mask))
        except Exception as e:
            print(f"[supports] ERROR applying fix({tag}, {mask}): {e}")

    # Persist a small QA artifact
    try:
        out_dir = OUT_DIR or "out"
        os.makedirs(out_dir, exist_ok=True)
        qa = {
            "version": 1,
            "applied": [{"node": t, "mask": m} for t, m in applied],
            "skipped": skipped,
        }
        with open(os.path.join(out_dir, "supports.json"), "w", encoding="utf-8") as f:
            json.dump(qa, f, indent=2, ensure_ascii=False)
        print(f"[supports] Wrote {os.path.join(out_dir, 'supports.json')}")
    except Exception as e:
        print(f"[supports] WARN: could not write supports.json: {e}")

    return applied
