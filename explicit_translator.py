# -*- coding: utf-8 -*-
"""
explicit_translator.py

Adapter so the Streamlit viewer can render a FULLY-EXPLICIT OpenSeesPy model.
The app expects a module exposing: build_model(stage: str = "all", artifacts_dir: str | None = None) -> None

This adapter simply imports the generated explicit model from OUT_DIR (default: 'out/explicit_model.py')
and delegates to its build_model(). No artifacts are read here; the explicit model already contains all
the OpenSeesPy calls (nodes, supports, diaphragms, geomTransf, elements).

Usage in your workflow:
1) Generate explicit model:
   python generate_explicit_model.py --out out --explicit out/explicit_model.py

2) Streamlit viewer:
   streamlit run model_viewer_APP.py
   -> In the sidebar, use "Browse Files" and pick this file: explicit_translator.py

The viewer will import build_model() from this adapter, which in turn calls the explicit model.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Optional

# Keep in sync with config if present
try:
    from config import OUT_DIR  # type: ignore
except Exception:
    OUT_DIR = "out"


def _import_explicit_module(path: str):
    """Dynamically import the generated explicit model module from a path."""
    spec = importlib.util.spec_from_file_location("explicit_model_generated", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load explicit model from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(stage: str = "all", artifacts_dir: Optional[str] = None) -> None:
    """
    Signature expected by the Streamlit app.
    - stage is ignored (explicit model is already complete)
    - artifacts_dir is ignored (explicit model doesn’t read JSONs)

    We locate 'out/explicit_model.py' (or <artifacts_dir>/explicit_model.py if provided)
    and call its build_model().
    """
    explicit_dir = artifacts_dir or OUT_DIR
    explicit_path = os.path.join(explicit_dir, "explicit_model.py")

    if not os.path.exists(explicit_path):
        raise FileNotFoundError(
            f"Explicit model not found at {explicit_path}. "
            f"Generate it first with: python generate_explicit_model.py --out {explicit_dir}"
        )

    mod = _import_explicit_module(explicit_path)

    # Defensive: ensure the explicit file exposes the expected entry point
    if not hasattr(mod, "build_model"):
        raise AttributeError(f"{explicit_path} does not define build_model().")

    # Delegate. The explicit model has default ndm=3, ndf=6.
    mod.build_model()
