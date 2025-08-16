# -*- coding: utf-8 -*-
"""
kosmos_translated.py
Builds the OpenSees model for visualization inside the Streamlit app.

Staged build supported via build_model(stage=...):
  - 'nodes'            : only nodes
  - 'columns'          : nodes + columns
  - 'beams' or 'all'   : nodes + columns + beams
"""
from openseespy.opensees import wipe, model
from nodes import define_nodes
from columns import define_columns
from beams import define_beams


def build_model(stage: str = "all") -> None:
    """Build the OpenSees model in memory.

    Parameters
    ----------
    stage : str
        'nodes'   -> only nodes
        'columns' -> nodes + columns
        'beams' or 'all' -> nodes + columns + beams
    """
    # 1) Reset & initialize OpenSees model
    wipe()
    model("basic", "-ndm", 3, "-ndf", 6)

    # 2) Always build nodes first
    define_nodes()

    # 3) Stage-dependent elements
    if stage.lower() == "nodes":
        print("[KOSMOS] Built NODES only.")
        return

    # Columns first so vertical continuity is easy to inspect
    define_columns()

    if stage.lower() in ("all", "beams"):
        define_beams()

    print(f"[KOSMOS] Model built with stage={stage}.")
    

if __name__ == "__main__":
    build_model()