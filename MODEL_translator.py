# -*- coding: utf-8 -*-
"""
MODEL_translator.py
Builds the OpenSees model for visualization inside the Streamlit app.

Staged build supported via build_model(stage=...):
  - 'nodes'            : nodes + point restraints + rigid diaphragms (+ nodes.json)
  - 'columns'          : nodes + restraints + diaphragms + columns (+ nodes.json)
  - 'beams' or 'all'   : nodes + restraints + diaphragms + columns + beams (+ nodes.json)

Change log:
- 2025-08-25: Build order updated to define supports BEFORE diaphragms so that
  diaphragms.py can read out/supports.json and skip stories with supports.
- 2025-08-25: Emit nodes.json after diaphragms creation (includes grid + masters).
"""
from __future__ import annotations

from openseespy.opensees import wipe, model

from nodes import define_nodes
from supports import define_point_restraints_from_e2k
from diaphragms import define_rigid_diaphragms
from columns import define_columns
from beams import define_beams
from emit_nodes import emit_nodes_json

# Default artifacts directory for emit_nodes_json; fall back to "out"
try:
    from config import OUT_DIR as _DEFAULT_OUT
except Exception:
    _DEFAULT_OUT = "out"


def build_model(stage: str = "all") -> None:
    """
    Build the OpenSees model according to the requested stage.

    Order (supports before diaphragms, then emit nodes.json):
      1) Nodes
      2) Point restraints (supports)        -> out/supports.json
      3) Rigid diaphragms (uses supports)   -> out/diaphragms.json
      4) Emit nodes.json (grid + masters)   -> out/nodes.json
      5) Columns                            -> out/columns.json
      6) Beams (if stage in {'all','beams'})-> out/beams.json
    """
    wipe()
    # 3D, 6-DOF nodes (UX, UY, UZ, RX, RY, RZ)
    model("basic", "-ndm", 3, "-ndf", 6)

    # 1) Nodes
    define_nodes()

    # 2) Point restraints (from ETABS POINTASSIGN ... RESTRAINT)
    define_point_restraints_from_e2k()

    # 3) Diaphragms (creates centroid masters and ties slaves; masters are fixed in UZ,RX,RY)
    define_rigid_diaphragms()

    # 4) Emit nodes.json (includes masters; requires story_graph.json + diaphragms.json)
    emit_nodes_json(_DEFAULT_OUT)

    if stage.lower() == "nodes":
        print("[MODEL] Built NODES + RESTRAINTS + DIAPHRAGMS + NODES.JSON.")
        return

    # 5) Elements
    define_columns()

    # 6) Beams if requested/all
    if stage.lower() in ("all", "beams"):
        define_beams()

    print(f"[MODEL] Model built with stage={stage} (nodes.json emitted).")


if __name__ == "__main__":
    build_model()
