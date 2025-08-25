# -*- coding: utf-8 -*-
"""
explicit_runtime_check.py

Purpose
-------
Sanity-check that the generated out/explicit_model.py can be subjected to an
OpenSeesPy analysis. This does NOT run your project loads/records; it ensures:
  - The explicit model builds a valid OpenSees domain
  - Constraints/numbering/system can be configured
  - A zero-increment static "analyze(0)" runs cleanly
  - (Optional) Eigenvalue extraction is possible

IMPORTANT
---------
If your model defines rigid diaphragms (multi-point constraints), you MUST use:
    constraints('Transformation')
Using constraints('Plain') with MPCs prints:
    "PlainHandler::handle() - constraint matrix not identity ..."
and may appear to hang due to excessive warnings.

Usage
-----
1) Ensure you generated the explicit model:
     python generate_explicit_model.py --out out --explicit out/explicit_model.py
2) Run this checker:
     python explicit_runtime_check.py            # uses out/explicit_model.py
   or
     python explicit_runtime_check.py --explicit out/explicit_model.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Any, Dict, List, Optional, Tuple

# OpenSeesPy
from openseespy.opensees import (
    wipe, getNodeTags, getEleTags, constraints, numberer, system, test,
    algorithm, integrator, analysis, analyze, eigen
)

# Keep consistent with config if present
try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = "out"


def _load_explicit_module(path: str):
    spec = importlib.util.spec_from_file_location("explicit_model_generated", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load explicit model from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _safe(fn, *args, **kwargs) -> Tuple[bool, Optional[str]]:
    try:
        fn(*args, **kwargs)
        return True, None
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"


def _summary_block(title: str, rows: List[Tuple[str, str]]) -> str:
    width = max(len(k) for k, _ in rows) if rows else 20
    out = [title]
    for k, v in rows:
        out.append(f" - {k:<{width}} : {v}")
    return "\n".join(out)


def check_explicit(explicit_path: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {"explicit_path": explicit_path, "checks": [], "status": "FAIL"}

    if not os.path.exists(explicit_path):
        report["checks"].append(("explicit_model.py presence", "FAIL (file not found)"))
        return report

    # 1) Import module
    try:
        mod = _load_explicit_module(explicit_path)
        report["checks"].append(("import explicit_model.py", "OK"))
    except Exception as e:
        report["checks"].append(("import explicit_model.py", f"FAIL ({e})"))
        return report

    # 2) Build domain
    try:
        wipe()
        mod.build_model()  # default ndm=3, ndf=6 inside the explicit file
        report["checks"].append(("build_model()", "OK"))
    except Exception as e:
        report["checks"].append(("build_model()", f"FAIL ({e})"))
        return report

    # 3) Basic domain counts
    try:
        nodes = getNodeTags()
        eles = getEleTags()
        n_nodes = len(nodes) if nodes is not None else 0
        n_eles = len(eles) if eles is not None else 0
        report["checks"].append(("node count", f"{n_nodes}"))
        report["checks"].append(("element count", f"{n_eles}"))
        if n_nodes == 0:
            report["checks"].append(("domain sanity", "FAIL (no nodes)"))
            return report
        if n_eles == 0:
            report["checks"].append(("domain sanity", "WARN (no elements)"))
    except Exception as e:
        report["checks"].append(("domain query", f"FAIL ({e})"))
        return report

    # 4) Configure analysis stack (use Transformation for MPC/diaphragms)
    handler = "Transformation"
    ok, err = _safe(constraints, handler)
    if not ok:
        # Fallback (not recommended if MPCs exist)
        handler = "Plain"
        ok2, err2 = _safe(constraints, handler)
        if not ok2:
            report["checks"].append((f"constraints('{handler}')", f"FAIL ({err2})"))
            return report
    report["checks"].append((f"constraints('{handler}')", "OK"))

    ok, err = _safe(numberer, "RCM")
    report["checks"].append(("numberer('RCM')", "OK" if ok else f"FAIL ({err})"))
    if not ok:
        return report

    ok, err = _safe(system, "BandGeneral")
    report["checks"].append(("system('BandGeneral')", "OK" if ok else f"FAIL ({err})"))
    if not ok:
        return report

    ok, err = _safe(test, "NormDispIncr", 1.0e-8, 10)
    report["checks"].append(("test('NormDispIncr',1e-8,10)", "OK" if ok else f"FAIL ({err})"))
    if not ok:
        return report

    ok, err = _safe(algorithm, "Newton")
    report["checks"].append(("algorithm('Newton')", "OK" if ok else f"FAIL ({err})"))
    if not ok:
        return report

    ok, err = _safe(integrator, "LoadControl", 0.0)  # zero-step integrator
    report["checks"].append(("integrator('LoadControl',0.0)", "OK" if ok else f"FAIL ({err})"))
    if not ok:
        return report

    ok, err = _safe(analysis, "Static")
    report["checks"].append(("analysis('Static')", "OK" if ok else f"FAIL ({err})"))
    if not ok:
        return report

    # 5) Try a zero-step analysis to ensure everything assembles
    try:
        rc = analyze(0)  # assemble without stepping
        if rc == 0:
            report["checks"].append(("analyze(0)", "OK"))
        else:
            report["checks"].append(("analyze(0)", f"WARN (returned {rc})"))
    except Exception as e:
        report["checks"].append(("analyze(0)", f"FAIL ({e})"))
        return report

    # 6) Optional eigen try (works even without explicit loads)
    try:
        vals = eigen("-fullGenLapack", 3)
        if isinstance(vals, (list, tuple)) and len(vals) > 0:
            report["checks"].append(("eigen(3)", f"OK (first λ={vals[0]:.6g})"))
        else:
            report["checks"].append(("eigen(3)", "WARN (no eigenvalues returned)"))
    except Exception as e:
        # Eigen can fail if mass is singular/incomplete; this is a WARN, not FAIL
        report["checks"].append(("eigen(3)", f"WARN ({e.__class__.__name__}: {e})"))

    # Decide status
    fails = [c for c in report["checks"] if "FAIL" in c[1]]
    if fails:
        report["status"] = "FAIL"
    else:
        warns = [c for c in report["checks"] if "WARN" in c[1]]
        report["status"] = "WARN" if warns else "PASS"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for out/explicit_model.py analysis readiness.")
    parser.add_argument("--explicit", default=None, help="Path to explicit_model.py (default: out/explicit_model.py)")
    args = parser.parse_args()

    explicit_path = args.explicit or os.path.join(OUT_DIR, "explicit_model.py")
    rep = check_explicit(explicit_path)

    rows = [(k, v) for (k, v) in rep["checks"]]
    print("\n=== Explicit Model Analysis Readiness ===")
    print(f"Explicit file : {rep['explicit_path']}")
    print(_summary_block("Checks", rows))
    print(f"Summary       : {rep['status']}")


if __name__ == "__main__":
    main()
