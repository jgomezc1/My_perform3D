"""rigid_end_utils.py
Utilities to compute and create rigid-end offset nodes and segments for beams/columns.
This module is intentionally lightweight and has **no repo-global state**; it only
creates OpenSees nodes when needed and returns a structured description that
callers can use to emit artifacts consistently.

Returned structure from `split_with_rigid_ends`:
{
  'nodes':  {'nI': int, 'nIm': Optional[int], 'nJm': Optional[int], 'nJ': int},
  'coords': {'nI': (x,y,z), 'nIm': Optional[(x,y,z)], 'nJm': Optional[(x,y,z)], 'nJ': (x,y,z)},
  'segments': [
      {'role': 'rigid_i'|'deformable'|'rigid_j', 'i': int, 'j': int, 'suffix': '#RI'|'#MID'|'#RJ'}
  ]
}
"""
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Set
from math import sqrt
from openseespy.opensees import node, getNodeTags

# Use project tagging if available
try:
    from tagging import node_tag_free  # type: ignore
except Exception:
    import hashlib
    def node_tag_free(key: str) -> int:  # type: ignore
        s = hashlib.sha1(key.encode('utf-8')).hexdigest()
        return 900_000_000 + int(s[:8], 16) % 90_000_000

def _unit_dir(pI: Tuple[float,float,float], pJ: Tuple[float,float,float]) -> Tuple[float,float,float]:
    dx = pJ[0]-pI[0]; dy = pJ[1]-pI[1]; dz = pJ[2]-pI[2]
    L = (dx*dx + dy*dy + dz*dz) ** 0.5
    if L == 0.0:
        return (0.0, 0.0, 0.0)
    return (dx/L, dy/L, dz/L)

def offset_point(p: Tuple[float,float,float], q: Tuple[float,float,float], Loff: float) -> Tuple[float,float,float]:
    """Point at distance Loff from p towards q."""
    ux, uy, uz = _unit_dir(p, q)
    return (p[0] + ux*Loff, p[1] + uy*Loff, p[2] + uz*Loff)

def ensure_offset_node(tag_key: str, coords: Tuple[float,float,float], existing_nodes: Set[int]) -> int:
    """Create a node with a deterministic tag derived from tag_key in the free range."""
    tag = node_tag_free(tag_key)
    if tag not in existing_nodes:
        node(tag, *coords)
        existing_nodes.add(tag)
    return tag

def split_with_rigid_ends(
    *,
    kind: str,
    line_name: str,
    story_index: int,
    nI: int, nJ: int,
    pI: Tuple[float,float,float], pJ: Tuple[float,float,float],
    LoffI: float, LoffJ: float
) -> Dict[str, Any]:
    """Compute optional intermediate nodes and segments for rigid ends."""
    # Normalize/Clamp offsets
    dx = pJ[0]-pI[0]; dy = pJ[1]-pI[1]; dz = pJ[2]-pI[2]
    Lfull = (dx*dx + dy*dy + dz*dz) ** 0.5
    LoffI = max(0.0, float(LoffI or 0.0))
    LoffJ = max(0.0, float(LoffJ or 0.0))
    if Lfull > 0.0 and (LoffI + LoffJ) >= 0.9999*Lfull:
        # shrink evenly to leave a tiny deformable core
        scale = 0.4999 * Lfull / max(LoffI + LoffJ, 1e-9)
        LoffI *= scale; LoffJ *= scale

    nodes = {'nI': nI, 'nIm': None, 'nJm': None, 'nJ': nJ}
    coords = {'nI': pI, 'nIm': None, 'nJm': None, 'nJ': pJ}
    segments: List[Dict[str, Any]] = []

    existing_nodes: Set[int] = set(getNodeTags())

    if LoffI > 0.0:
        cIm = offset_point(pI, pJ, LoffI)
        tag_key = f"OFF|{kind}|{line_name}|{story_index}|I"
        nIm = ensure_offset_node(tag_key, cIm, existing_nodes)
        nodes['nIm'] = nIm; coords['nIm'] = cIm
        segments.append({'role': 'rigid_i', 'i': nI, 'j': nIm, 'suffix': '#RI'})
        left_i = nIm
    else:
        left_i = nI

    if LoffJ > 0.0:
        cJm = offset_point(pJ, pI, LoffJ)  # from J towards I
        tag_key = f"OFF|{kind}|{line_name}|{story_index}|J"
        nJm = ensure_offset_node(tag_key, cJm, existing_nodes)
        nodes['nJm'] = nJm; coords['nJm'] = cJm
        right_j = nJm
    else:
        right_j = nJ

    # main deformable
    segments.append({'role': 'deformable', 'i': left_i, 'j': right_j, 'suffix': '#MID'})

    if LoffJ > 0.0:
        segments.append({'role': 'rigid_j', 'i': right_j, 'j': nJ, 'suffix': '#RJ'})

    return {'nodes': nodes, 'coords': coords, 'segments': segments}
