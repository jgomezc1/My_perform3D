"""
Phase 1: Robust parser for ETABS .e2k essentials.

Extracts:
- STORIES (top->bottom listing; we will compute absolute elevations later)
- POINT COORDINATES
- POINT ASSIGNS  (now recognizes DIAPH and DIAPHRAGM)
- LINE CONNECTIVITIES
- LINE ASSIGNS
  * section/frameprop
  * longitudinal rigid ends: LENGTHOFFI, LENGTHOFFJ
  * nodal offsets: OFFSETXI, OFFSETYI, OFFSETZI, OFFSETXJ, OFFSETYJ, OFFSETZJ
- DIAPHRAGM NAMES

Notes
-----
ETABS examples observed in the wild:
    POINTASSIGN "56" "01_P2_m170" DIAPH "D1"
Some exports use DIAPHRAGM instead of DIAPH. We accept BOTH and normalize to
the unified key 'diaphragm' in the output.

For $ LINE ASSIGNS we now parse both:
  - QUOTED tokens:  SECTION/SECT/FRAMEPROP, PIER, SPANDREL, LOCALAXIS, RELEASE
  - NUMERIC (unquoted) tokens: LENGTHOFFI/J, OFFSETS I/J for X/Y/Z
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional


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


def _to_float_or_none(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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
      "line_assigns":   [
                          {
                            "line", "story", "section",
                            "length_off_i", "length_off_j",
                            "offsets_i": {"x","y","z"}, "offsets_j": {"x","y","z"},
                            "extra"
                          }, ...
                        ],
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
                "height": _to_float_or_none(m.group(2)),
                "elev": _to_float_or_none(m.group(3)),
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
            "third": _to_float_or_none(m.group(4)),
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

    # Header: LINEASSIGN "<line>" "<story>" <tail>
    la_head = re.compile(r'^\s*LINEASSIGN\s+"([^"]+)"\s+"([^"]+)"(.*)$', re.IGNORECASE)

    # (A) QUOTED tokens: TOKEN "value"
    la_token_quoted = re.compile(
        r'\b(SECTION|SECT|FRAMEPROP|PIER|SPANDREL|LOCALAXIS|RELEASE)\b\s+"([^"]+)"',
        re.IGNORECASE
    )

    # (B) NUMERIC (unquoted) tokens: TOKEN number
    #   - LENGTHOFFI, LENGTHOFFJ
    #   - OFFSETXI, OFFSETYI, OFFSETZI, OFFSETXJ, OFFSETYJ, OFFSETZJ
    la_token_numeric = re.compile(
        r'\b('
        r'LENGTHOFFI|LENGTHOFFJ|'
        r'OFFSETXI|OFFSETYI|OFFSETZI|'
        r'OFFSETXJ|OFFSETYJ|OFFSETZJ'
        r')\b\s+([-+]?\d+(?:\.\d+)?)',
        re.IGNORECASE
    )

    line_assigns: List[Dict[str, Any]] = []

    for ln in la_lines:
        m = la_head.match(ln)
        if not m:
            continue
        lname, story, tail = m.group(1), m.group(2), m.group(3) or ""

        # Collect quoted tokens
        found_str: Dict[str, str] = {k.upper(): v for k, v in la_token_quoted.findall(tail)}
        section = found_str.get("SECTION") or found_str.get("SECT") or found_str.get("FRAMEPROP")

        # Collect numeric tokens
        found_num: Dict[str, float] = {}
        for k, v in la_token_numeric.findall(tail):
            found_num[k.upper()] = float(v)

        # Normalize offsets
        length_off_i = found_num.get("LENGTHOFFI")
        length_off_j = found_num.get("LENGTHOFFJ")

        offsets_i = {
            "x": found_num.get("OFFSETXI"),
            "y": found_num.get("OFFSETYI"),
            "z": found_num.get("OFFSETZI"),
        }
        # Remove keys with None to keep JSON clean
        offsets_i = {k: v for k, v in offsets_i.items() if v is not None}

        offsets_j = {
            "x": found_num.get("OFFSETXJ"),
            "y": found_num.get("OFFSETYJ"),
            "z": found_num.get("OFFSETZJ"),
        }
        offsets_j = {k: v for k, v in offsets_j.items() if v is not None}

        entry: Dict[str, Any] = {
            "line": lname,
            "story": story,
            "section": section,
        }
        if length_off_i is not None:
            entry["length_off_i"] = length_off_i
        if length_off_j is not None:
            entry["length_off_j"] = length_off_j
        if offsets_i:
            entry["offsets_i"] = offsets_i
        if offsets_j:
            entry["offsets_j"] = offsets_j

        # Preserve extras for any future tokens (quoted + numeric)
        # If keys collide, numeric wins (more specific for these fields)
        extra: Dict[str, Any] = {}
        extra.update(found_str)
        extra.update(found_num)
        # If we elevated some extras to top-level, it's fine to keep them here too for traceability.
        if extra:
            entry["extra"] = extra

        line_assigns.append(entry)

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
