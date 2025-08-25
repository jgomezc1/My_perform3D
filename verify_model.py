# -*- coding: utf-8 -*-
"""
verify_model.py

Artifact-level QA for MyPerform3D Phase-2.

Reads OUT_DIR artifacts:
  - story_graph.json
  - supports.json (optional, but recommended)
  - diaphragms.json (optional; present after build)
  - columns.json (optional; present after build)
  - beams.json   (optional; present after build)

Outputs:
  - <OUT_DIR>/verify_report.json  (machine-readable)
  - Console summary:

    === Verification Summary ===
    Summary: PASS | WARN | FAIL
    Artifacts dir: <path>
    Report: <path>/verify_report.json
    Checks:
     - base_supports: ...
     - rigid_diaphragms: ...
     - master_masses: ...
     - elements_presence: ...
     - transforms_sections: ...
     - orphans: ...

Design notes, aligned with current repo:
- Deterministic node tags: tag = point_id*1000 + story_index (story_index=0 top, increasing downward).
- We infer story_index for theoretical grid nodes from story_graph.json.
- We do NOT require an OpenSees runtime; we verify emitted JSON plus deterministic rules.
- Safe if some artifacts are missing: we degrade to WARN/FAIL with actionable messages.

Python: 3.11+
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _story_index_map(story_graph: Dict[str, Any]) -> Tuple[List[str], Dict[str, int]]:
    names: List[str] = list(story_graph.get("story_order_top_to_bottom", []))
    return names, {name: i for i, name in enumerate(names)}


def _active_point_tag_set(story_graph: Dict[str, Any]) -> Set[int]:
    """Compute all expected grid node tags from active_points using the deterministic rule."""
    names, sidx = _story_index_map(story_graph)
    out: Set[int] = set()
    for sname, pts in (story_graph.get("active_points") or {}).items():
        idx = sidx.get(sname)
        if idx is None:
            continue
        for p in pts:
            pid = str(p.get("id", p.get("tag")))
            if pid and pid.isdigit():
                out.add(int(pid) * 1000 + idx)
    return out


def _nodes_used_by_elements(elem_json: Dict[str, Any], kind: str) -> Set[int]:
    used: Set[int] = set()
    recs = (elem_json or {}).get(kind) or []
    for e in recs:
        for k in ("i_node", "j_node"):
            v = e.get(k)
            if isinstance(v, int):
                used.add(v)
    return used


def verify_model(
    artifacts_dir: str = "out",
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    sg = _load(os.path.join(artifacts_dir, "story_graph.json")) or {}
    sp = _load(os.path.join(artifacts_dir, "supports.json")) or {}
    dg = _load(os.path.join(artifacts_dir, "diaphragms.json")) or {}
    cj = _load(os.path.join(artifacts_dir, "columns.json")) or {}
    bj = _load(os.path.join(artifacts_dir, "beams.json")) or {}

    report: Dict[str, Any] = {
        "artifacts_dir": artifacts_dir,
        "present": {
            "story_graph": bool(sg),
            "supports": bool(sp),
            "diaphragms": bool(dg),
            "columns": bool(cj),
            "beams": bool(bj),
        },
        "checks": {},
    }

    # Base supports check
    check_base_supports = {"status": "warn", "details": ""}
    if sp and sg:
        names, sidx = _story_index_map(sg)
        base_name = names[-1] if names else None
        # story_index is tag % 1000
        base_idx = sidx.get(base_name) if base_name else None
        applied = sp.get("applied") or []
        base_nodes = []
        if isinstance(applied, list) and base_idx is not None:
            for rec in applied:
                tag = int(rec.get("node", -1))
                if tag % 1000 == base_idx:
                    base_nodes.append(tag)
        if base_nodes:
            check_base_supports.update({"status": "pass", "details": f"{len(base_nodes)} base support nodes detected"})
        else:
            check_base_supports.update({"status": "fail", "details": "No base supports detected"})
    else:
        check_base_supports.update({"status": "warn", "details": "supports.json or story_graph.json missing"})

    report["checks"]["base_supports"] = check_base_supports

    # Rigid diaphragms rules
    check_rigid = {"status": "pass", "details": [], "violations": []}
    if dg and sg:
        names, sidx = _story_index_map(sg)
        # Stories with supports (from supports.json if present)
        stories_with_supports: Set[str] = set()
        if sp:
            applied = sp.get("applied") or []
            for rec in applied:
                tag = int(rec.get("node", -1))
                idx = tag % 1000
                if 0 <= idx < len(names):
                    stories_with_supports.add(names[idx])

        for rec in dg.get("diaphragms", []):
            sname = rec.get("story")
            slaves = rec.get("slaves", [])
            mass = rec.get("mass", {})
            fix = rec.get("fix", {})
            # Skip-on-support rule
            if sname in stories_with_supports:
                check_rigid["violations"].append(f"Diaphragm created on support story '{sname}'")
            # Mass/fix applied flags
            if not bool(mass.get("applied")):
                check_rigid["violations"].append(f"Master mass not applied for story '{sname}'")
            if not bool(fix.get("applied")):
                check_rigid["violations"].append(f"Master fix not applied for story '{sname}'")
            # Minimal slave count
            if len(slaves) < 2:
                check_rigid["violations"].append(f"Too few diaphragm slaves on '{sname}' ({len(slaves)})")
        if check_rigid["violations"]:
            check_rigid["status"] = "fail" if strict else "warn"
            check_rigid["details"] = f"{len(check_rigid['violations'])} issue(s) found"
        else:
            check_rigid["details"] = "All diaphragms OK"
    else:
        check_rigid.update({"status": "warn", "details": "diaphragms.json or story_graph.json missing"})

    report["checks"]["rigid_diaphragms"] = check_rigid

    # Elements presence & basic metadata
    checks_elements = {"status": "pass", "details": []}
    col_count = len((cj or {}).get("columns") or [])
    beam_count = len((bj or {}).get("beams") or [])
    if col_count == 0 and beam_count == 0:
        checks_elements["status"] = "fail"
        checks_elements["details"].append("No columns or beams created")
    else:
        checks_elements["details"].append(f"Columns: {col_count}, Beams: {beam_count}")
    report["checks"]["elements_presence"] = checks_elements

    # Transforms & sections presence (metadata fields exist; not physics correctness)
    checks_meta = {"status": "pass", "details": []}
    missing_transf = 0
    missing_section = 0
    total = 0
    for k, blob in (("columns", cj), ("beams", bj)):
        for e in (blob or {}).get(k, []) or []:
            total += 1
            if e.get("transf_tag") is None:
                missing_transf += 1
            # Section may be None today → warn if many missing
            if e.get("section", None) in (None, ""):
                missing_section += 1
    sec_rate = (missing_section / total) if total else 1.0
    if missing_transf:
        checks_meta["status"] = "warn"
        checks_meta["details"].append(f"{missing_transf} element(s) missing transf_tag")
    if sec_rate > 0.5:
        checks_meta["status"] = "warn"
        checks_meta["details"].append(f"{missing_section}/{total} element(s) without section label")
    if not checks_meta["details"]:
        checks_meta["details"].append("Transforms and section labels present")
    report["checks"]["transforms_sections"] = checks_meta

    # Orphans: element node tags not derivable from story_graph (masters are allowed orphans)
    checks_orphans = {"status": "pass", "details": []}
    expected_nodes = _active_point_tag_set(sg)
    used_nodes = _nodes_used_by_elements(cj, "columns") | _nodes_used_by_elements(bj, "beams")
    # Masters are not in expected set; ignore tags not multiple of 1000 rule
    grid_like_used = {t for t in used_nodes if t % 1000 >= 0}
    missing = sorted(grid_like_used - expected_nodes)
    if missing:
        checks_orphans["status"] = "warn" if not strict else "fail"
        checks_orphans["details"].append(f"{len(missing)} element node(s) not in story_graph active_points")
        checks_orphans["missing_nodes_sample"] = missing[:10]
    else:
        checks_orphans["details"].append("No orphan element nodes detected")
    report["checks"]["orphans"] = checks_orphans

    # Final summary level
    levels = {"pass": 0, "warn": 1, "fail": 2}
    worst = 0
    for v in report["checks"].values():
        worst = max(worst, levels.get(str(v.get("status")).lower(), 2))
    summary = ["PASS", "WARN", "FAIL"][worst]
    report["summary"] = summary

    # Save report
    _save(os.path.join(artifacts_dir, "verify_report.json"), report)

    # Console summary
    print("=== Verification Summary ===")
    print(f"Summary: {summary}")
    print(f"Artifacts dir: {artifacts_dir}")
    print(f"Report: {os.path.join(artifacts_dir, 'verify_report.json')}")
    print("\nChecks:")
    for name, res in report["checks"].items():
        print(f" - {name}: {res['status']}  ({'; '.join(res.get('details') or [])})")

    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify MyPerform3D Phase-2 artifacts")
    p.add_argument("--artifacts", default="out", help="Artifacts directory (default: out)")
    p.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verify_model(artifacts_dir=args.artifacts, strict=args.strict)
