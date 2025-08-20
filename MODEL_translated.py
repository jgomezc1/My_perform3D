# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
MODEL_translated.py
Builds the OpenSees model for visualization inside the Streamlit app.

Staged build supported via build_model(stage=...):
  - 'nodes'            : nodes + rigid diaphragms + point restraints
  - 'columns'          : nodes + diaphragms + restraints + columns
  - 'beams' or 'all'   : nodes + diaphragms + restraints + columns + beams
"""
from openseespy.opensees import wipe, model
from nodes import define_nodes
from columns import define_columns
from beams import define_beams
from diaphragms import define_rigid_diaphragms
from supports import define_point_restraints_from_e2k


def build_model(stage: str = "all") -> None:
    """
    Build the OpenSees model according to the requested stage.
    """
    wipe()
    # 3D, 6-DOF nodes (UX, UY, UZ, RX, RY, RZ)
    model('basic', '-ndm', 3, '-ndf', 6)

    # 1) Nodes
    define_nodes()

    # 2) Diaphragms (creates centroid masters and ties slaves; masters are fixed in UZ,RX,RY)
    define_rigid_diaphragms()

    # 3) Point restraints (from ETABS POINTASSIGN ... RESTRAINT)
    # Safe to call even if no RESTRAINT entries exist; it will no-op.
    define_point_restraints_from_e2k()

    if stage.lower() == "nodes":
        print("[MODEL] Built NODES + DIAPHRAGMS + RESTRAINTS.")
        return

    # 4) Elements
    define_columns()

    if stage.lower() in ("all", "beams"):
        define_beams()

    print(f"[MODEL] Model built with stage={stage}.")


if __name__ == "__main__":
    build_model()
