# -*- coding: utf-8 -*-
"""
Ejemplo_translated.py
Builds the OpenSees model for visualization inside the Streamlit app.

Staged build supported via build_model(stage=...):
  - 'nodes'            : nodes + rigid diaphragms (centroid master node)
  - 'columns'          : nodes + diaphragms + columns
  - 'beams' or 'all'   : nodes + diaphragms + columns + beams
"""
from openseespy.opensees import wipe, model
from nodes import define_nodes
from columns import define_columns
from beams import define_beams
from diaphragms import define_rigid_diaphragms  # NEW


def build_model(stage: str = "all") -> None:
    """Build the OpenSees model in memory.

    Parameters
    ----------
    stage : str
        'nodes'   -> nodes + rigid diaphragms
        'columns' -> nodes + rigid diaphragms + columns
        'beams' or 'all' -> nodes + rigid diaphragms + columns + beams
    """
    # Reset & initialize OpenSees model
    wipe()
    model("basic", "-ndm", 3, "-ndf", 6)

    # 1) Nodes first
    define_nodes()

    # 2) Rigid diaphragms (centroid master node, built only from nodes)
    define_rigid_diaphragms()

    if stage.lower() == "nodes":
        print("[EJEMPLO] Built NODES + DIAPHRAGMS.")
        return

    # 3) Elements
    define_columns()

    if stage.lower() in ("all", "beams"):
        define_beams()

    print(f"[EJEMPLO] Model built with stage={stage}.")


if __name__ == "__main__":
    build_model()
