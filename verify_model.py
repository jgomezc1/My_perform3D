# -*- coding: utf-8 -*-
"""
verify_model.py

Artifact-level QA for MyPerform3D Phase-2.

Reads OUT_DIR artifacts:
  - story_graph.json
  - supports.json (optional)
  - diaphragms.json (optional)
  - columns.json (optional)
  - beams.json   (optional)
  - nodes.json   (optional)

Outputs:
  - <OUT_DIR>/verify_report.json  (machine-readable)
  - Console summary.

Checks added/updated in this version:
  - endpoints_exist: all element endpoints exist in nodes.json
  - orphans: does NOT flag registered rigid-interface nodes
  - nodes_presence: ignores rigid-interface nodes in "extra_nodes"
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
    """
    Collect endpoint tags used by elements in an artifact blob.
    Accepts common keys: i_node/j_node, i/j, node_i/node_j, ni/nj.
    """
    used: Set[int] = set()
    recs = (elem_json or {}).get(kind) or []
    for e in recs:
        # Primary keys used in beams/columns emitters
        if isinstance(e.get("i_node"), int):
            used.add(int(e["i_node"]))
        if isinstance(e.get("j_node"), int):
            used.add(int(e["j_node"]))

        # Secondary fallbacks
        for ki, kj in (("i", "j"), ("node_i", "node_j"), ("ni", "nj"), ("I", "J")):
            if ki in e and isinstance(e[ki], int):
                used.add(int(e[ki]))
            if kj in e and isinstance(e[kj], int):
                used.add(int(e[kj]))
    return used


def _point_offset(p: Dict[str, Any]) -> float:
    v = p.get("explicit_z")
    if isinstance(v, (int, float)):
        return float(v)
    v = p.get("z")
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def verify_model(
    artifacts_dir: str = "out",
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    artifacts_dir_str = str(artifacts_dir)

    sg = _load(os.path.join(artifacts_dir_str, "story_graph.json")) or {}
    sp = _load(os.path.join(artifacts_dir_str, "supports.json")) or {}
    dg = _load(os.path.join(artifacts_dir_str, "diaphragms.json")) or {}
    cj = _load(os.path.join(artifacts_dir_str, "columns.json")) or {}
    bj = _load(os.path.join(artifacts_dir_str, "beams.json")) or {}
    nj = _load(os.path.join(artifacts_dir_str, "nodes.json")) or {}

    report: Dict[str, Any] = {
        "artifacts_dir": artifacts_dir_str,
        "present": {
            "story_graph": bool(sg),
            "supports": bool(sp),
            "diaphragms": bool(dg),
            "columns": bool(cj),
            "beams": bool(bj),
            "nodes": bool(nj),
        },
        "checks": {},
    }

    # --- story_elev_order ---
    elev_order = {"status": "pass", "details": []}
    if sg:
        names, _ = _story_index_map(sg)
        elev = sg.get("story_elev") or {}
        vals = [float(elev.get(n, 0.0)) for n in names]
        ok = all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))
        if not ok:
            elev_order["status"] = "fail" if strict else "warn"
            elev_order["details"].append("story_elev not monotone from top to bottom")
            elev_order["sequence"] = [{"story": n, "elev": v} for n, v in zip(names, vals)]
        else:
            elev_order["details"].append("story_elev monotone (top->bottom)")
    else:
        elev_order["status"] = "warn"
        elev_order["details"].append("story_graph.json missing")
    report["checks"]["story_elev_order"] = elev_order

    # --- base_supports ---
    check_base_supports = {"status": "warn", "details": ""}
    if sp and sg:
        names, sidx = _story_index_map(sg)
        base_name = names[-1] if names else None
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

    # --- rigid_diaphragms ---
    check_rigid = {"status": "pass", "details": [], "violations": []}
    if dg and sg:
        names, sidx = _story_index_map(sg)
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
            if sname in stories_with_supports:
                check_rigid["violations"].append(f"Diaphragm created on support story '{sname}'")
            if not bool(mass.get("applied")):
                check_rigid["violations"].append(f"Master mass not applied for story '{sname}'")
            if not bool(fix.get("applied")):
                check_rigid["violations"].append(f"Master fix not applied for story '{sname}'")
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

    # --- nodes_presence ---
    nodes_check = {"status": "warn", "details": []}
    expected_grid = _active_point_tag_set(sg) if sg else set()
    have_nodes = {int(n.get("tag")) for n in (nj.get("nodes") or [])} if nj else set()
    kinds_by_tag = {int(n.get("tag")): str(n.get("kind", "")) for n in (nj.get("nodes") or [])} if nj else {}
    masters_in_dg = {int(rec.get("master")) for rec in (dg.get("diaphragms") or [])} if dg else set()
    interface_tags = {t for t, k in kinds_by_tag.items() if k == "rigid_interface"}

    if nj:
        missing_grid = sorted(list(expected_grid - have_nodes))
        # Do not count registered interface nodes as "extra"
        extra_nodes = sorted(list((have_nodes - expected_grid - masters_in_dg) - interface_tags))
        total = len(nj.get("nodes") or [])
        grid_count = sum(1 for n in (nj.get("nodes") or []) if n.get("kind") == "grid")
        master_count = sum(1 for n in (nj.get("nodes") or []) if n.get("kind") == "diaphragm_master")
        nodes_check["details"].append(f"total={total}, grid={grid_count}, master={master_count}")
        if missing_grid:
            nodes_check["status"] = "fail" if strict else "warn"
            nodes_check["details"].append(f"{len(missing_grid)} expected grid node(s) missing from nodes.json")
            nodes_check["missing_grid_sample"] = missing_grid[:10]
        if extra_nodes:
            # informational
            nodes_check["details"].append(f"{len(extra_nodes)} extra node(s) (not grid/master); sample shown")
            nodes_check["extra_nodes_sample"] = extra_nodes[:10]
        if not missing_grid and not extra_nodes:
            nodes_check["status"] = "pass"
    else:
        nodes_check["details"].append("nodes.json missing")
        nodes_check["status"] = "warn"
    report["checks"]["nodes_presence"] = nodes_check

    # --- node_z_consistency ---
    zcheck = {"status": "pass", "details": [], "mismatches": []}
    if nj and sg:
        names, sidx = _story_index_map(sg)
        elev_by_story: Dict[str, float] = sg.get("story_elev") or {}
        ap: Dict[str, List[Dict[str, Any]]] = sg.get("active_points") or {}
        offset_map: Dict[Tuple[str, str], float] = {}
        for sname, pts in ap.items():
            for p in pts:
                pid_raw = p.get("id", p.get("tag"))
                pid_str = str(pid_raw) if pid_raw is not None else ""
                if pid_str:
                    v = p.get("explicit_z")
                    if isinstance(v, (int, float)):
                        offset_map[(sname, pid_str)] = float(v)
                    else:
                        v2 = p.get("z")
                        offset_map[(sname, pid_str)] = float(v2) if isinstance(v2, (int, float)) else 0.0

        mismatches = 0
        sample: List[Dict[str, Any]] = []
        for n in nj.get("nodes", []):
            if n.get("kind") != "grid":
                continue
            sname = n.get("story")
            pid = n.get("source_point_id")
            if sname is None or pid is None:
                continue
            z_expected = float(elev_by_story.get(sname, 0.0)) - float(offset_map.get((sname, pid), 0.0))
            z_actual = float(n.get("z", 0.0))
            if abs(z_actual - z_expected) > 1e-6:
                mismatches += 1
                if len(sample) < 10:
                    sample.append({"tag": n.get("tag"), "story": sname, "pid": pid,
                                   "z_actual": z_actual, "z_expected": z_expected})
        if mismatches:
            zcheck["status"] = "fail" if strict else "warn"
            zcheck["details"].append(f"{mismatches} grid node(s) with Z mismatch vs story_elev - offset")
            zcheck["mismatches"] = sample
        else:
            zcheck["details"].append("All grid node Z values consistent with story_elev - offset")
    else:
        zcheck["status"] = "warn"
        zcheck["details"].append("nodes.json or story_graph.json missing")
    report["checks"]["node_z_consistency"] = zcheck

    # --- elements_presence & transforms_sections ---
    checks_elements = {"status": "pass", "details": []}
    col_count = len((cj or {}).get("columns") or [])
    beam_count = len((bj or {}).get("beams") or [])
    if col_count == 0 and beam_count == 0:
        checks_elements["status"] = "fail"
        checks_elements["details"].append("No columns or beams created")
    else:
        checks_elements["details"].append(f"Columns: {col_count}, Beams: {beam_count}")
    report["checks"]["elements_presence"] = checks_elements

    checks_meta = {"status": "pass", "details": []}
    missing_transf = 0
    missing_section = 0
    total = 0
    for k, blob in (("columns", cj), ("beams", bj)):
        for e in (blob or {}).get(k, []) or []:
            total += 1
            if e.get("transf_tag") is None:
                missing_transf += 1
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

    # --- endpoints_exist (NEW) ---
    endpoints_check = {"status": "pass", "details": []}
    if nj:
        have_nodes = {int(n.get("tag")) for n in (nj.get("nodes") or [])}
        missing: List[Dict[str, Any]] = []

        def _scan(label: str, blob: Dict[str, Any]) -> None:
            recs = (blob or {}).get(label) or []
            for idx, e in enumerate(recs, start=1):
                i_tag = e.get("i_node")
                j_tag = e.get("j_node")
                if not isinstance(i_tag, int) or not isinstance(j_tag, int):
                    # Try fallbacks
                    for ki, kj in (("i", "j"), ("node_i", "node_j"), ("ni", "nj"), ("I", "J")):
                        if ki in e and kj in e and isinstance(e[ki], int) and isinstance(e[kj], int):
                            i_tag, j_tag = int(e[ki]), int(e[kj])
                            break
                if isinstance(i_tag, int) and isinstance(j_tag, int):
                    bad = [t for t in (i_tag, j_tag) if t not in have_nodes]
                    if bad:
                        missing.append({"index": idx, "kind": label, "missing": bad})

        _scan("beams", bj)
        _scan("columns", cj)

        if missing:
            endpoints_check["status"] = "fail"
            endpoints_check["details"].append(f"{len(missing)} element(s) reference node(s) absent from nodes.json")
            endpoints_check["sample"] = missing[:10]
        else:
            endpoints_check["details"].append("All element endpoints exist in nodes.json")
    else:
        endpoints_check["status"] = "fail" if strict else "warn"
        endpoints_check["details"].append("nodes.json missing; cannot verify endpoints")
    report["checks"]["endpoints_exist"] = endpoints_check

    # --- orphans (updated to ignore registered interface nodes) ---
    checks_orphans = {"status": "pass", "details": []}
    expected_nodes = _active_point_tag_set(sg)
    used_nodes = _nodes_used_by_elements(cj, "columns") | _nodes_used_by_elements(bj, "beams")
    have_nodes = {int(n.get("tag")) for n in (nj.get("nodes") or [])} if nj else set()
    # Only consider as orphan those used nodes that are neither in story_graph nor in nodes.json
    missing = sorted((used_nodes - expected_nodes) - have_nodes)
    if missing:
        checks_orphans["status"] = "warn" if not strict else "fail"
        checks_orphans["details"].append(f"{len(missing)} element node(s) not in story_graph or nodes.json")
        checks_orphans["missing_nodes_sample"] = missing[:10]
    else:
        checks_orphans["details"].append("No orphan element nodes detected")
    report["checks"]["orphans"] = checks_orphans

    # --- summary ---
    levels = {"pass": 0, "warn": 1, "fail": 2}
    worst = 0
    for v in report["checks"].values():
        worst = max(worst, levels.get(str(v.get("status")).lower(), 2))
    summary = ["PASS", "WARN", "FAIL"][worst]
    report["summary"] = summary

    out_path = os.path.join(artifacts_dir_str, "verify_report.json")
    _save(out_path, report)

    print("=== Verification Summary ===")
    print(f"Summary: {summary}")
    print(f"Artifacts dir: {artifacts_dir_str}")
    print(f"Report: {out_path}")
    print("\nChecks:")
    for name, res in report["checks"].items():
        details = res.get("details") or []
        if isinstance(details, list):
            details_str = "; ".join(details)
        else:
            details_str = str(details)
        print(f" - {name}: {res['status']}  ({details_str})")

    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify MyPerform3D Phase-2 artifacts")
    p.add_argument("--artifacts", default="out", help="Artifacts directory (default: out)")
    p.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verify_model(artifacts_dir=args.artifacts, strict=args.strict)
