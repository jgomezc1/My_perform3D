# e2k_parser.py
"""
Phase 1: Robust parser for ETABS .e2k essentials.
Extracts:
- STORIES (top->bottom listing; we will compute absolute elevations later)
- POINT COORDINATES
- POINT ASSIGNS
- LINE CONNECTIVITIES
- LINE ASSIGNS
Implements the user's SPRINGPROP rule for interpreting the third point number.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List
from collections import defaultdict

SECTION_HDR = re.compile(r'^\s*\$[^\n]*\n', re.IGNORECASE | re.MULTILINE)

def _extract_section(text: str, title_regex: str) -> str:
    m = re.search(title_regex + r'[^\n]*\n', text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    n = re.search(r'^\s*\$[^\n]*\n', text[start:], flags=re.MULTILINE)
    end = start + n.start() if n else len(text)
    return text[start:end]

def parse_e2k(text: str) -> Dict[str, Any]:
    # STORIES: top->bottom listing in file
    stories_txt = _extract_section(text, r'^\s*\$ STORIES')
    story_lines = [ln for ln in stories_txt.splitlines() if ln.strip()]
    story_pat = re.compile(
        r'^\s*STORY\s+"([^"]+)"'                     # name
        r'(?:\s+HEIGHT\s+([-+]?\d+(?:\.\d+)?))?'     # height
        r'(?:\s+ELEV\s+([-+]?\d+(?:\.\d+)?))?'       # explicit elev
        r'(?:\s+SIMILARTO\s+"([^"]+)")?'             # similar_to
        r'(?:\s+MASTERSTORY\s+"([^"]+)")?',          # masterstory flag
        re.IGNORECASE
    )
    stories: List[Dict[str, Any]] = []
    for ln in story_lines:
        m = story_pat.match(ln)
        if m:
            stories.append({
                "name": m.group(1),
                "height": float(m.group(2)) if m.group(2) else None,
                "elev": float(m.group(3)) if m.group(3) else None,
                "similar_to": m.group(4),
                "masterstory": m.group(5),
            })

    # POINT COORDINATES
    pt_txt = _extract_section(text, r'^\s*\$ POINT COORDINATES')
    pt_lines = [ln for ln in pt_txt.splitlines() if ln.strip()]
    pt_pat = re.compile(
        r'^\s*POINT\s+"([^"]+)"\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)(?:\s+([-+]?\d+(?:\.\d+)?))?',
        re.IGNORECASE
    )
    points: Dict[str, Dict[str, Any]] = {}
    for ln in pt_lines:
        m = pt_pat.match(ln)
        if not m:
            continue
        pid = m.group(1)
        points[pid] = {
            "x": float(m.group(2)),
            "y": float(m.group(3)),
            "third": float(m.group(4)) if m.group(4) else None,
            "has_three": m.group(4) is not None,
        }

    # POINT ASSIGNS
    pa_txt = _extract_section(text, r'^\s*\$ POINT ASSIGNS')
    pa_lines = [ln for ln in pa_txt.splitlines() if ln.strip()]
    pa_head = re.compile(r'^\s*POINTASSIGN\s+"([^"]+)"\s+"([^"]+)"(.*)$', re.IGNORECASE)
    token = re.compile(r'\b(DIAPHRAGM|SPRINGPROP|POINTMASS|RESTRAINT|FRAMEPROP|JOINTPATTERN|SPCONSTRAINT)\b\s+"([^"]+)"', re.IGNORECASE)
    point_assigns: List[Dict[str, Any]] = []
    for ln in pa_lines:
        m = pa_head.match(ln)
        if not m:
            continue
        pid, story, tail = m.group(1), m.group(2), m.group(3) or ""
        found = {k.upper(): v for k, v in token.findall(tail)}
        point_assigns.append({
            "point": pid,
            "story": story,
            "diaphragm": found.get("DIAPHRAGM"),
            "springprop": found.get("SPRINGPROP"),
            "extra": {k: v for k, v in found.items() if k not in ("DIAPHRAGM", "SPRINGPROP")},
        })

    # LINE CONNECTIVITIES
    lc_txt = _extract_section(text, r'^\s*\$ LINE CONNECTIVITIES')
    lc_lines = [ln for ln in lc_txt.splitlines() if ln.strip()]
    lc_pat = re.compile(r'^\s*LINE\s+"([^"]+)"\s+([A-Z]+)\s+"([^"]+)"\s+"([^"]+)"', re.IGNORECASE)
    lines: Dict[str, Dict[str, Any]] = {}
    for ln in lc_lines:
        m = lc_pat.match(ln)
        if not m:
            continue
        lines[m.group(1)] = {
            "name": m.group(1),
            "kind": m.group(2).upper(),
            "i": m.group(3),
            "j": m.group(4),
        }

    # LINE ASSIGNS
    la_txt = _extract_section(text, r'^\s*\$ LINE ASSIGNS')
    la_lines = [ln for ln in la_txt.splitlines() if ln.strip()]
    la_head = re.compile(r'^\s*LINEASSIGN\s+"([^"]+)"\s+"([^"]+)"(.*)$', re.IGNORECASE)
    la_token = re.compile(r'\b(SECTION|SECT|FRAMEPROP|PIER|SPANDREL|LOCALAXIS|RELEASE)\b\s+"([^"]+)"', re.IGNORECASE)
    line_assigns: List[Dict[str, Any]] = []
    for ln in la_lines:
        m = la_head.match(ln)
        if not m:
            continue
        lname, story, tail = m.group(1), m.group(2), m.group(3) or ""
        found = {k.upper(): v for k, v in la_token.findall(tail)}
        section = found.get("SECTION") or found.get("SECT") or found.get("FRAMEPROP")
        line_assigns.append({
            "line": lname,
            "story": story,
            "section": section,
            "extra": found
        })

    return {
        "stories": stories,
        "points": points,
        "point_assigns": point_assigns,
        "lines": lines,
        "line_assigns": line_assigns,
    }
