# tagging.py
"""
Stable ID utilities for nodes/elements so model diffs remain clean across runs.
"""
from __future__ import annotations
import hashlib

# Keep plenty of room between point-derived tags and generated IDs
STORY_MULTIPLIER = 1000          # node = point_int * STORY_MULTIPLIER + story_index
FREE_OFFSET      = 900_000_000   # free (explicit-Z) nodes live in a high range

def _stable_int(text: str) -> int:
    """Deterministic 32-bit-ish int from text (portable across runs/OS/Python)."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def point_int(pid: str) -> int:
    """Prefer numeric point ids; fall back to stable hash if needed."""
    try:
        return int(pid)
    except Exception:
        return _stable_int(f"PT|{pid}") % 10_000_000  # keep compact

def node_tag_grid(pid: str, story_index: int) -> int:
    """Node tag for a grid point at a specific story index (top=0)."""
    return point_int(pid) * STORY_MULTIPLIER + story_index

def node_tag_free(pid: str) -> int:
    """Node tag for an explicit-Z (free) point (story-agnostic)."""
    return FREE_OFFSET + point_int(pid)

def element_tag(kind: str, line_name: str, story_index: int) -> int:
    """
    Stable element tag for BEAM/COLUMN at a given story slice.
    Two disjoint ranges to keep types visually distinct in debugging:
      - BEAM   : 1_000_000..1_900_000_000
      - COLUMN : 2_000_000..2_900_000_000
    """
    base = 1_000_000 if kind.upper() == "BEAM" else 2_000_000
    return base + (_stable_int(f"{kind.upper()}|{line_name}|{story_index}") % 900_000_000)
