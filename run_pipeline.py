# -*- coding: utf-8 -*-
"""
run_pipeline.py

One-shot orchestrator for MyPerform3D:
  1) Phase-1 parsing (phase1_run.py)
  2) Phase-2 build (MODEL_translator.build_model)
  3) Static artifact verification (verify_model.verify_model)
  4) Runtime-vs-artifacts verification (verify_domain_vs_artifacts.py)  <-- now via subprocess

Writes:
  - out/verify_report.json
  - out/domain_capture.json
  - out/verify_runtime_report.json
  - out/pipeline_summary.json

All paths written to JSON are strings (avoid WindowsPath serialization issues).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict


def _load(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_phase1() -> None:
    print("[PIPELINE] Phase-1: running phase1_run.py ...")
    subprocess.run([sys.executable, "phase1_run.py"], check=True)
    print("[PIPELINE] Phase-1: done.")


def run_phase2(stage: str) -> None:
    print(f"[PIPELINE] Phase-2: building model with stage='{stage}' ...")
    import importlib
    M = importlib.import_module("MODEL_translator")
    M.build_model(stage)
    print("[PIPELINE] Phase-2: done.")


def run_static_verify(out_dir: str, strict: bool) -> Dict[str, Any]:
    print("[PIPELINE] Static verify: verify_model.py ...")
    import importlib
    vm = importlib.import_module("verify_model")
    rep: Dict[str, Any] = vm.verify_model(artifacts_dir=str(out_dir), strict=strict)
    print("[PIPELINE] Static verify: done.")
    return rep


def run_runtime_verify(out_dir: str, stage: str, strict: bool) -> Dict[str, Any]:
    """
    Run runtime verification in a FRESH subprocess so capture sees patched ops
    before any project modules import OpenSees (prevents bound-alias bypass).
    """
    print("[PIPELINE] Runtime verify: verify_domain_vs_artifacts.py (subprocess) ...")
    cmd = [sys.executable, "verify_domain_vs_artifacts.py", "--artifacts", str(out_dir), "--stage", stage]
    if strict:
        cmd.append("--strict")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"[PIPELINE] Runtime verify returned non-zero exit code: {res.returncode}")
    # Load the generated report
    rep_path = os.path.join(str(out_dir), "verify_runtime_report.json")
    rep = _load(rep_path) or {"summary": "FAIL"}
    print("[PIPELINE] Runtime verify: done.")
    return rep


def summarize(static_rep: Dict[str, Any], runtime_rep: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    levels = {"PASS": 0, "WARN": 1, "FAIL": 2}
    s = static_rep.get("summary", "FAIL")
    r = runtime_rep.get("summary", "FAIL")
    worst = max(levels.get(str(s), 2), levels.get(str(r), 2))
    summary = ["PASS", "WARN", "FAIL"][worst]
    out_dir_str = str(out_dir)
    summary_blob = {
        "artifacts_dir": out_dir_str,
        "static_verify": {"summary": s, "report": str(os.path.join(out_dir_str, "verify_report.json"))},
        "runtime_verify": {"summary": r, "report": str(os.path.join(out_dir_str, "verify_runtime_report.json"))},
        "summary": summary,
    }
    _save(os.path.join(out_dir_str, "pipeline_summary.json"), summary_blob)
    print("\n=== Pipeline Summary ===")
    print(f"Artifacts dir: {out_dir_str}")
    print(f"Static verify : {s}  ({summary_blob['static_verify']['report']})")
    print(f"Runtime verify: {r}  ({summary_blob['runtime_verify']['report']})")
    print(f"Overall       : {summary}")
    return summary_blob


def main() -> None:
    p = argparse.ArgumentParser(description="Run MyPerform3D pipeline end-to-end")
    p.add_argument("--stage", default="all", choices=["nodes", "columns", "beams", "all"],
                   help="Build stage for Phase-2 (default: all)")
    p.add_argument("--strict", action="store_true", help="Treat WARN as FAIL in verifiers")
    p.add_argument("--skip-phase1", action="store_true", help="Skip Phase-1 run (reuse existing artifacts)")
    p.add_argument("--skip-static", action="store_true", help="Skip static verification")
    p.add_argument("--skip-domain", action="store_true", help="Skip runtime (captured) verification")
    p.add_argument("--out", default=None, help="Artifacts directory; defaults to config.OUT_DIR or 'out'")
    args = p.parse_args()

    # Resolve OUT_DIR and normalize to str
    out_dir = args.out
    if not out_dir:
        try:
            from config import OUT_DIR as _OUT_DIR
        except Exception:
            _OUT_DIR = "out"
        out_dir = _OUT_DIR
    out_dir = str(out_dir)

    if not args.skip_phase1:
        run_phase1()
    else:
        print("[PIPELINE] Skipping Phase-1 per flag.")

    run_phase2(args.stage)

    static_report: Dict[str, Any] = {"summary": "SKIPPED"}
    runtime_report: Dict[str, Any] = {"summary": "SKIPPED"}

    if not args.skip_static:
        static_report = run_static_verify(out_dir, args.strict)
    else:
        print("[PIPELINE] Skipping static verification per flag.")

    if not args.skip_domain:
        runtime_report = run_runtime_verify(out_dir, args.stage, args.strict)
    else:
        print("[PIPELINE] Skipping runtime verification per flag.")

    summarize(static_report, runtime_report, out_dir)


if __name__ == "__main__":
    main()
