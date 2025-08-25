# -*- coding: utf-8 -*-
"""
verify_domain_vs_artifacts.py

Builds the OpenSees model under a capture session and compares the **live domain**
against Phase-1/2 artifacts. Writes a machine-readable report:

  out/verify_runtime_report.json

All path fields stored in JSON are strings to avoid WindowsPath serialization issues.
Includes robust write with retry/fallback to avoid Windows file locks.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple, Set

from ops_capture import capture_session, get_capture, save_capture


TOL = 1e-6
# Rigid diaphragm master fix pattern from repo spec: fix(master, 0,0,1,1,1,0)
DIAPH_FIX = (0, 0, 1, 1, 1, 0)


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_write_json(path: str, data: Dict[str, Any], *, attempts: int = 6, delay: float = 0.25) -> str:
    """
    Write JSON to 'path' using a temp file and atomic replace. On Windows, if the
    target is locked by another process, retry a few times; if still failing,
    fall back to a unique filename with PID suffix. Returns the actual file path.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    # Try to replace with retries
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            if i == attempts - 1:
                break
            time.sleep(delay)
    # Fallback: unique name
    fallback = path.replace(".json", f".{os.getpid()}.json")
    with open(fallback, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[WARN] Could not overwrite locked file: {path}. Wrote fallback: {fallback}")
    # Best-effort cleanup of tmp
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass
    return fallback


def _nodes_dict(nodes_json: Dict[str, Any]) -> Dict[int, Tuple[float, float, float]]:
    out: Dict[int, Tuple[float, float, float]] = {}
    for n in nodes_json.get("nodes", []):
        try:
            tag = int(n["tag"])
            out[tag] = (float(n["x"]), float(n["y"]), float(n["z"]))
        except Exception:
            continue
    return out


def _supports_dict(supports_json: Dict[str, Any]) -> Dict[int, Tuple[int, int, int, int, int, int]]:
    out: Dict[int, Tuple[int, int, int, int, int, int]] = {}
    for r in supports_json.get("applied", []):
        try:
            tag = int(r["node"])
            out[tag] = (
                int(r.get("ux", 0)),
                int(r.get("uy", 0)),
                int(r.get("uz", 0)),
                int(r.get("rx", 0)),
                int(r.get("ry", 0)),
                int(r.get("rz", 0)),
            )
        except Exception:
            continue
    return out


def _diaphragms_dict(diaphragms_json: Dict[str, Any]) -> Dict[int, Set[int]]:
    """
    Map master -> set(slaves) from artifacts (perp is always 3 by spec).
    """
    out: Dict[int, Set[int]] = {}
    for d in diaphragms_json.get("diaphragms", []):
        try:
            m = int(d["master"])
            slaves = set(int(s) for s in d.get("slaves", []))
            out[m] = slaves
        except Exception:
            continue
    return out


def _diaphragm_fix_spec(diaphragms_json: Dict[str, Any]) -> Dict[int, Tuple[int, int, int, int, int, int]]:
    """
    master -> fix tuple from artifacts, but only where fix.applied == True.
    If flags are omitted, we still require 'applied' to be True, and we fill
    unspecified fields with the repo spec defaults (0,0,1,1,1,0).
    """
    out: Dict[int, Tuple[int, int, int, int, int, int]] = {}
    for d in diaphragms_json.get("diaphragms", []):
        fx = d.get("fix") or {}
        if not fx.get("applied"):
            continue
        try:
            mtag = int(d["master"])
        except Exception:
            continue
        tpl = (
            int(fx.get("ux", 0)),
            int(fx.get("uy", 0)),
            int(fx.get("uz", 1)),
            int(fx.get("rx", 1)),
            int(fx.get("ry", 1)),
            int(fx.get("rz", 0)),
        )
        out[mtag] = tpl
    return out


def _master_mass_spec(diaphragms_json: Dict[str, Any]) -> Dict[int, Tuple[float, float]]:
    """
    master -> (M, Izz) from artifacts for stories where 'applied'==True.
    """
    out: Dict[int, Tuple[float, float]] = {}
    for d in diaphragms_json.get("diaphragms", []):
        if not (d.get("mass") and d["mass"].get("applied")):
            continue
        try:
            mtag = int(d["master"])
            M = float(d["mass"]["M"])
            Izz = float(d["mass"]["Izz"])
            out[mtag] = (M, Izz)
        except Exception:
            continue
    return out


def _union_element_pairs(*json_blobs: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """
    Create set of undirected element endpoint pairs from beams/columns artifacts.
    """
    pairs: Set[Tuple[int, int]] = set()
    for blob in json_blobs:
        for k in ("beams", "columns"):
            for e in blob.get(k, []) or []:
                try:
                    i = int(e["i_node"]); j = int(e["j_node"])
                    a, b = (i, j) if i <= j else (j, i)
                    pairs.add((a, b))
                except Exception:
                    continue
    return pairs


def _transf_tags(*json_blobs: Dict[str, Any]) -> Set[int]:
    tags: Set[int] = set()
    for blob in json_blobs:
        for k in ("beams", "columns"):
            for e in blob.get(k, []) or []:
                v = e.get("transf_tag")
                if v is not None:
                    try:
                        tags.add(int(v))
                    except Exception:
                        continue
    return tags


def compare_runtime_vs_artifacts(artifacts_dir: str, stage: str = "all", strict: bool = False) -> Dict[str, Any]:
    artifacts_dir_str = str(artifacts_dir)

    # Build under capture
    with capture_session():
        import importlib
        M = importlib.import_module("MODEL_translator")
        M.build_model(stage)

    cap = get_capture()
    save_capture(artifacts_dir_str)

    # Load artifacts
    nodes_json = _load_json(os.path.join(artifacts_dir_str, "nodes.json"))
    supports_json = _load_json(os.path.join(artifacts_dir_str, "supports.json"))
    diaph_json = _load_json(os.path.join(artifacts_dir_str, "diaphragms.json"))
    cols_json = _load_json(os.path.join(artifacts_dir_str, "columns.json"))
    beams_json = _load_json(os.path.join(artifacts_dir_str, "beams.json"))

    report: Dict[str, Any] = {"artifacts_dir": artifacts_dir_str, "checks": {}}

    # --- nodes_set & coords ---
    nodes_art = _nodes_dict(nodes_json)
    nodes_cap = {int(n["tag"]): (float(n["x"]), float(n["y"]), float(n["z"])) for n in cap["nodes"]}
    art_tags = set(nodes_art.keys())
    cap_tags = set(nodes_cap.keys())
    missing_in_cap = sorted(list(art_tags - cap_tags))
    extra_in_cap = sorted(list(cap_tags - art_tags))

    coords_mismatch = []
    for tag in sorted(art_tags & cap_tags):
        xa, ya, za = nodes_art[tag]
        xc, yc, zc = nodes_cap[tag]
        if abs(xa - xc) > TOL or abs(ya - yc) > TOL or abs(za - zc) > TOL:
            if len(coords_mismatch) < 10:
                coords_mismatch.append({"tag": tag, "art": [xa, ya, za], "cap": [xc, yc, zc]})

    status = "pass"
    details = [f"art={len(art_tags)}, cap={len(cap_tags)}"]
    if missing_in_cap:
        status = "fail" if strict else "warn"
        details.append(f"{len(missing_in_cap)} node(s) in artifacts missing in runtime")
    if extra_in_cap:
        status = "warn" if status == "pass" else status
        details.append(f"{len(extra_in_cap)} extra node(s) present in runtime not in artifacts")
    if coords_mismatch:
        status = "fail" if strict else "warn"
        details.append(f"{len(coords_mismatch)} node coord mismatch (sample shown)")

    report["checks"]["nodes_set"] = {
        "status": status,
        "details": details,
        "missing_in_runtime_sample": missing_in_cap[:10],
        "extra_in_runtime_sample": extra_in_cap[:10],
        "coord_mismatch_sample": coords_mismatch,
    }

    # --- supports ---
    sup_art = _supports_dict(supports_json)
    sup_cap_all = {int(f["node"]): (f["ux"], f["uy"], f["uz"], f["rx"], f["ry"], f["rz"]) for f in cap["fixes"]}

    # Identify diaphragm masters to filter them out of the supports comparison
    dia_art = _diaphragms_dict(diaph_json)  # master -> slaves
    dia_masters = set(dia_art.keys())

    # Remove captured fixes on diaphragm masters if they match the diaphragm fix pattern
    sup_cap = {
        n: tpl for n, tpl in sup_cap_all.items()
        if not (n in dia_masters and tpl == DIAPH_FIX)
    }

    # Now compare only supports listed in artifacts
    missing_supports = sorted(list(set(sup_art.keys()) - set(sup_cap.keys())))
    # "extra" supports are those present at runtime but not listed in supports.json,
    # AFTER excluding diaphragm master fixities
    extra_supports = sorted(list(set(sup_cap.keys()) - set(sup_art.keys())))

    mismatched = []
    for node in sorted(set(sup_art.keys()) & set(sup_cap.keys())):
        if sup_art[node] != sup_cap[node]:
            if len(mismatched) < 10:
                mismatched.append({"node": node, "art": sup_art[node], "cap": sup_cap[node]})

    status = "pass"
    details = [f"art={len(sup_art)}, cap={len(sup_cap_all)} (filtered cap for supports compare={len(sup_cap)})"]
    if missing_supports:
        status = "fail" if strict else "warn"
        details.append(f"{len(missing_supports)} supports missing in runtime")
    if extra_supports:
        status = "warn" if status == "pass" else status
        details.append(f"{len(extra_supports)} extra supports present in runtime (non-diaphragm)")
    if mismatched:
        status = "fail" if strict else "warn"
        details.append(f"{len(mismatched)} support flag mismatch (sample shown)")
    report["checks"]["supports"] = {
        "status": status,
        "details": details,
        "missing_in_runtime_sample": missing_supports[:10],
        "extra_in_runtime_sample": extra_supports[:10],
        "mismatch_sample": mismatched,
    }

    # --- diaphragms ---
    # Evaluate master/slaves AND also validate master fixities here
    dia_cap_map: Dict[int, Set[int]] = {}
    perp_violations: List[Dict[str, Any]] = []
    for rec in cap["rigid_diaphragms"]:
        if rec.get("perp") != 3:
            perp_violations.append(rec)
        m = int(rec["master"])
        slaves = set(int(s) for s in rec.get("slaves", []))
        if m in dia_cap_map:
            dia_cap_map[m] |= slaves
        else:
            dia_cap_map[m] = slaves

    missing_masters = sorted(list(set(dia_art.keys()) - set(dia_cap_map.keys())))
    extra_masters = sorted(list(set(dia_cap_map.keys()) - set(dia_art.keys())))
    slave_mismatches = []
    for m in sorted(set(dia_art.keys()) & set(dia_cap_map.keys())):
        a = dia_art[m]
        c = dia_cap_map[m]
        if a != c:
            if len(slave_mismatches) < 10:
                slave_mismatches.append({"master": m, "art_diff": sorted(list(a ^ c))})

    # Master fixity validation (IMPORTANT: use diaph_json here)
    fx_spec = _diaphragm_fix_spec(diaph_json)  # only where fix.applied == True
    cap_fix_map = {int(f["node"]): (f["ux"], f["uy"], f["uz"], f["rx"], f["ry"], f["rz"]) for f in cap["fixes"]}

    missing_fix = []
    fix_mismatch = []
    for m, spec_tpl in fx_spec.items():
        if m not in cap_fix_map:
            missing_fix.append(m)
        else:
            cap_tpl = cap_fix_map[m]
            if cap_tpl != spec_tpl:
                if len(fix_mismatch) < 10:
                    fix_mismatch.append({"master": m, "cap": cap_tpl, "spec": spec_tpl})

    status = "pass"
    details = [f"masters art={len(dia_art)}, cap={len(dia_cap_map)}"]
    if perp_violations:
        status = "fail" if strict else "warn"
        details.append(f"{len(perp_violations)} rigidDiaphragm perp!=3")
    if missing_masters:
        status = "fail" if strict else "warn"
        details.append(f"{len(missing_masters)} master(s) missing in runtime")
    if extra_masters:
        status = "warn" if status == "pass" else status
        details.append(f"{len(extra_masters)} extra master(s) present in runtime")
    if slave_mismatches:
        status = "fail" if strict else "warn"
        details.append(f"{len(slave_mismatches)} master(s) with slave set mismatch (sample shown)")
    if missing_fix:
        status = "fail" if strict else "warn"
        details.append(f"{len(missing_fix)} master(s) missing required diaphragm fixity")
    if fix_mismatch:
        status = "fail" if strict else "warn"
        details.append(f"{len(fix_mismatch)} master(s) with diaphragm fix mismatch (sample shown)")
    report["checks"]["diaphragms"] = {
        "status": status,
        "details": details,
        "perp_violations_sample": perp_violations[:3],
        "missing_masters_sample": missing_masters[:10],
        "extra_masters_sample": extra_masters[:10],
        "slave_mismatch_sample": slave_mismatches,
        "missing_fix_sample": missing_fix[:10],
        "fix_mismatch_sample": fix_mismatch,
    }

    # --- master masses ---
    mass_spec = _master_mass_spec(diaph_json)  # mtag -> (M, Izz)
    mass_cap = {int(m["node"]): [float(x) for x in m["m"]] for m in cap["masses"]}
    missing_mass = sorted(list(set(mass_spec.keys()) - set(mass_cap.keys())))
    bad_mass = []
    for mtag, (M, Izz) in mass_spec.items():
        if mtag not in mass_cap:
            continue
        v = mass_cap[mtag]
        ok = (
            abs(v[0] - M) <= TOL
            and abs(v[1] - M) <= TOL
            and abs(v[2] - 0.0) <= TOL
            and abs(v[3] - 0.0) <= TOL
            and abs(v[4] - 0.0) <= TOL
            and abs(v[5] - Izz) <= TOL
        )
        if not ok and len(bad_mass) < 10:
            bad_mass.append({"node": mtag, "cap": v, "spec": [M, M, 0.0, 0.0, 0.0, Izz]})
    status = "pass"
    details = [f"masters with mass spec={len(mass_spec)}, captured masses={len(mass_cap)}"]
    if missing_mass:
        status = "fail" if strict else "warn"
        details.append(f"{len(missing_mass)} master(s) with missing runtime mass")
    if bad_mass:
        status = "fail" if strict else "warn"
        details.append(f"{len(bad_mass)} master(s) with mass mismatch (sample shown)")
    report["checks"]["master_masses"] = {
        "status": status,
        "details": details,
        "missing_mass_sample": missing_mass[:10],
        "mismatch_sample": bad_mass,
    }

    # --- elements & transforms ---
    art_pairs = _union_element_pairs(beams_json, cols_json)
    cap_pairs: Set[Tuple[int, int]] = set()
    for e in cap["elements"]:
        try:
            i = int(e["i"]); j = int(e["j"])
            a, b = (i, j) if i <= j else (j, i)
            cap_pairs.add((a, b))
        except Exception:
            continue
    missing_pairs = sorted(list(art_pairs - cap_pairs))
    extra_pairs = sorted(list(cap_pairs - art_pairs))

    status = "pass"
    details = [f"pairs art={len(art_pairs)}, cap={len(cap_pairs)}"]
    if missing_pairs:
        status = "fail" if strict else "warn"
        details.append(f"{len(missing_pairs)} element pair(s) missing in runtime")
    if extra_pairs:
        status = "warn" if status == "pass" else status
        details.append(f"{len(extra_pairs)} extra element pair(s) present in runtime")
    report["checks"]["elements"] = {
        "status": status,
        "details": details,
        "missing_pairs_sample": missing_pairs[:10],
        "extra_pairs_sample": extra_pairs[:10],
    }

    art_transf = _transf_tags(beams_json, cols_json)
    cap_transf = set(int(t["tag"]) for t in cap["geom_transf"])
    missing_transf = sorted(list(art_transf - cap_transf))
    status = "pass" if not missing_transf else ("fail" if strict else "warn")
    details = [f"art transf={len(art_transf)}, cap transf={len(cap_transf)}"]
    if missing_transf:
        details.append(f"missing transf tags: {missing_transf[:10]}")
    report["checks"]["transf_present"] = {"status": status, "details": details}

    # Final summary
    levels = {"pass": 0, "warn": 1, "fail": 2}
    worst = max(levels.get(report["checks"][k]["status"], 2) for k in report["checks"])
    summary = ["PASS", "WARN", "FAIL"][worst]
    report["summary"] = summary

    # Robust write (handles Windows locks)
    out_path = os.path.join(artifacts_dir_str, "verify_runtime_report.json")
    final_path = _safe_write_json(out_path, report)

    print("=== Runtime vs Artifacts Verification ===")
    print(f"Summary: {summary}")
    print(f"Artifacts dir: {artifacts_dir_str}")
    print(f"Report: {final_path}")
    print("\nChecks:")
    for name, res in report["checks"].items():
        det = res.get("details", [])
        if isinstance(det, list):
            det = "; ".join(det)
        print(f" - {name}: {res['status']}  ({det})")

    return report


def _parse_args():
    p = argparse.ArgumentParser(description="Verify live OpenSees domain vs artifacts")
    p.add_argument("--artifacts", default="out", help="Artifacts directory (default: out)")
    p.add_argument("--stage", default="all", choices=["nodes", "columns", "beams", "all"],
                   help="Build stage to verify (default: all)")
    p.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compare_runtime_vs_artifacts(args.artifacts, args.stage, args.strict)
