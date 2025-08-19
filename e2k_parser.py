# e2k_parser.py
"""
Phase 1: Robust parser for ETABS .e2k essentials.

Extracts:
- STORIES (top->bottom listing; we will compute absolute elevations later)
- POINT COORDINATES
- POINT ASSIGNS  (now recognizes DIAPH and DIAPHRAGM)
- LINE CONNECTIVITIES
- LINE ASSIGNS
- DIAPHRAGM NAMES

Notes
-----
ETABS examples observed in the wild:
    POINTASSIGN "56" "01_P2_m170" DIAPH "D1"
Some exports use DIAPHRAGM instead of DIAPH. We accept BOTH and normalize to
the unified key 'diaphragm' in the output.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List

SECTION_HDR = re.compile(r'^\s*\$[^\n]*\n', re.IGNORECASE | re.MULTILINE)


def _extract_section(text: str, title_regex: str) -> str:
    """
    Return the text between a section header matching title_regex and the next
    section header, or the end of file if none.
    """
    m = re.search(title_regex + r'[^\n]*\n', text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    n = re.search(r'^\s*\$[^\n]*\n', text[start:], flags=re.MULTILINE)
    end = start + n.start() if n else len(text)
    return text[start:end]


def parse_e2k(text: str) -> Dict[str, Any]:
    """
    Parse ETABS .e2k text into a normalized dict used by Phase-1.

    Returns
    -------
    {
      "stories":        [ { "name", "height", "elev", "similar_to", "masterstory" }, ... ],
      "points":         { pid: { "x", "y", "third", "has_three" }, ... },
      "point_assigns":  [ { "point", "story", "diaphragm", "springprop", "extra" }, ... ],
      "lines":          { lname: { "name", "kind", "i", "j" }, ... },
      "line_assigns":   [ { "line", "story", "section", "extra" }, ... ],
      "diaphragm_names":[ "D1", "D2", ... ]
    }
    """
    # STORIES
    stories_txt = _extract_section(text, r'^\s*\$ STORIES')
    story_lines = [ln for ln in stories_txt.splitlines() if ln.strip()]
    story_pat = re.compile(
        r'^\s*STORY\s+"([^"]+)"'                     # name
        r'(?:\s+HEIGHT\s+([-+]?\d+(?:\.\d+)?))?'     # height
        r'(?:\s+ELEV\s+([-+]?\d+(?:\.\d+)?))?'       # explicit elev
        r'(?:\s+SIMILARTO\s+"([^"]+)")?'             # similar_to
        r'(?:\s+MASTERSTORY\s+"([^"]+)")?',          # masterstory
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

    # POINT ASSIGNS  (recognize DIAPH and DIAPHRAGM)
    pa_txt = _extract_section(text, r'^\s*\$ POINT ASSIGNS')
    pa_lines = [ln for ln in pa_txt.splitlines() if ln.strip()]
    pa_head = re.compile(r'^\s*POINTASSIGN\s+"([^"]+)"\s+"([^"]+)"(.*)$', re.IGNORECASE)
    # Tokens of the form: TOKEN "value"
    token = re.compile(
        r'\b('
        r'DIAPHRAGM|DIAPH|'         # diaphragm synonyms
        r'SPRINGPROP|POINTMASS|RESTRAINT|FRAMEPROP|JOINTPATTERN|SPCONSTRAINT'
        r')\b\s+"([^"]+)"',
        re.IGNORECASE
    )
    point_assigns: List[Dict[str, Any]] = []
    for ln in pa_lines:
        m = pa_head.match(ln)
        if not m:
            continue
        pid, story, tail = m.group(1), m.group(2), m.group(3) or ""
        # Build a dict of tokens; if duplicates appear, the last one wins
        found: Dict[str, str] = {}
        for k, v in token.findall(tail):
            found[k.upper()] = v
        diaphragm = found.get("DIAPHRAGM") or found.get("DIAPH")  # normalize
        point_assigns.append({
            "point": pid,
            "story": story,
            "diaphragm": diaphragm,
            "springprop": found.get("SPRINGPROP"),
            "extra": {k: v for k, v in found.items() if k not in ("DIAPHRAGM", "DIAPH", "SPRINGPROP")},
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

    # DIAPHRAGM NAMES
    dn_txt = _extract_section(text, r'^\s*\$ DIAPHRAGM NAMES')
    dn_lines = [ln for ln in dn_txt.splitlines() if ln.strip()]
    dn_pat = re.compile(r'^\s*DIAPHRAGM\s+"([^"]+)"', re.IGNORECASE)
    diaphragm_names: List[str] = []
    for ln in dn_lines:
        m = dn_pat.match(ln)
        if m:
            diaphragm_names.append(m.group(1))

    return {
        "stories": stories,
        "points": points,
        "point_assigns": point_assigns,
        "lines": lines,
        "line_assigns": line_assigns,
        "diaphragm_names": diaphragm_names,
    }
