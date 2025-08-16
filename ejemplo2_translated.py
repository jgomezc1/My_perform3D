# -*- coding: utf-8 -*-
"""
Ejemplo_translated.py
Self-contained builder for the Ejemplo model that:
- Uses Ejemplo.e2k as input (config-free)
- Writes Phase-1 artifacts to ./out_Ejemplo/
- Auto-runs Phase-1 if artifacts are missing or stale
- Exposes build_model(stage=...) for the Streamlit app

Stages:
  'nodes'            -> nodes only
  'columns'          -> nodes + columns
  'beams' or 'all'   -> nodes + columns + beams
"""

from __future__ import annotations
from pathlib import Path
import importlib
import time

# OpenSees
from openseespy.opensees import wipe, model

# Phase-1 & config (we’ll override these in-memory for Ejemplo only)
import config as _cfg
import phase1_run as _p1

# Phase-2
from nodes import define_nodes
from columns import define_columns
from beams import define_beams

# --------- Adjust these if you like ----------
E2K_FILE = Path("Ejemplo.e2k")
OUT_DIR  = Path("out_Ejemplo")
# --------------------------------------------


def _artifact_paths():
    story_path = OUT_DIR / "story_graph.json"
    raw_path   = OUT_DIR / "parsed_raw.json"
    return story_path, raw_path


def _needs_phase1(e2k_path: Path, out_dir: Path) -> bool:
    """Return True if Phase-1 must be (re)run: missing artifacts or older than the .e2k."""
    story_path, raw_path = _artifact_paths()
    if not story_path.exists() or not raw_path.exists():
        return True
    try:
        e2k_mtime   = e2k_path.stat().st_mtime
        story_mtime = story_path.stat().st_mtime
        raw_mtime   = raw_path.stat().st_mtime
    except FileNotFoundError:
        return True
    # Rebuild if either artifact is older than the source .e2k
    return (story_mtime < e2k_mtime) or (raw_mtime < e2k_mtime)


def _run_phase1_for_ejemplo(verbose: bool = True) -> None:
    """
    Run Phase-1 for Ejemplo into OUT_DIR, without permanently touching your global config.
    We temporarily monkey-patch config.E2K_PATH and config.OUT_DIR, reload phase1_run, and run it.
    """
    if verbose:
        print(f"[Ejemplo] Phase-1: E2K_FILE = {E2K_FILE.resolve()}")
        print(f"[Ejemplo] Phase-1: OUT_DIR  = {OUT_DIR.resolve()}")

    if not E2K_FILE.exists():
        raise FileNotFoundError(f"[Ejemplo] Input .e2k not found at: {E2K_FILE.resolve()}")

    OUT_DIR.mkdir(exist_ok=True)

    # Save originals to restore later
    _orig_e2k = getattr(_cfg, "E2K_PATH", None)
    _orig_out = getattr(_cfg, "OUT_DIR", None)

    try:
        # Patch config for this one run
        _cfg.E2K_PATH = E2K_FILE
        _cfg.OUT_DIR  = OUT_DIR
        _cfg.OUT_DIR.mkdir(exist_ok=True)

        # Reload phase1_run so it binds the updated config values
        importlib.reload(_p1)
        _p1.main()

        if verbose:
            sp, rp = _artifact_paths()
            print("[Ejemplo] Phase-1 complete.")
            print(f"  - {rp}")
            print(f"  - {sp}")

    finally:
        # Restore original config (don’t affect your main pipeline)
        if _orig_e2k is not None:
            _cfg.E2K_PATH = _orig_e2k
        if _orig_out is not None:
            _cfg.OUT_DIR = _orig_out


def _preflight_phase1():
    """Ensure Phase-1 artifacts exist and are fresh; run Phase-1 if needed."""
    if _needs_phase1(E2K_FILE, OUT_DIR):
        _run_phase1_for_ejemplo(verbose=True)


def _reset_model():
    wipe()
    model("basic", "-ndm", 3, "-ndf", 6)


def build_model(stage: str = "all") -> None:
    """
    Build the Ejemplo model in stages, using Ejemplo-specific artifacts in out_Ejemplo/.
    - Auto-runs Phase-1 if needed
    - Always builds nodes first
    """
    _preflight_phase1()
    story_path, raw_path = map(str, _artifact_paths())

    _reset_model()

    # Nodes first
    define_nodes(story_path=story_path, raw_path=raw_path)

    if stage.lower() == "nodes":
        print("[EJEMPLO] Built NODES only.")
        return

    # Columns (uses the 'next lower story with both endpoints' rule)
    define_columns(story_path=story_path, raw_path=raw_path)

    if stage.lower() in ("all", "beams"):
        # Beams (per-story; last section wins inside beams.py)
        define_beams(story_path=story_path, raw_path=raw_path)

    print(f"[EJEMPLO] Model built with stage={stage}.")
