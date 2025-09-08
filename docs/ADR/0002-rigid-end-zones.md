# ADR-0002 — Rigid End Zones as 3-Segment Members

## Context
ETABS allows definition of rigid end zones using the `LENGTHOFFI` / `LENGTHOFFJ` attributes (when `RIGIDZONE=1`).  
To reflect this in OpenSeesPy, elements must be split into up to three parts:
- A short rigid elastic segment at the I end (if `LENGTHOFFI > 0`)
- A central deformable segment
- A short rigid elastic segment at the J end (if `LENGTHOFFJ > 0`)

The rigid segments must use inflated section properties to approximate infinite stiffness.  
Intermediate nodes must be created to connect these segments. These nodes must be deterministically tagged and included in `nodes.json` before any elements reference them.

A bug was discovered (2025-09-04):  
- Intermediate nodes were generated in memory but **not propagated into `nodes.json`**.  
- As a result, `beams.json` / `columns.json` referenced non-existent nodes, causing `generate_explicit_model.py` to fail.

## Decision
- Split any member with `LENGTHOFFI/J` into 2 or 3 segments depending on offsets.
- Use a utility function (`rigid_end_utils.split_with_rigid_ends`) to compute intermediate node positions along the member vector.
- Route all new node creation through a **single registrar** (`emit_nodes.register_intermediate_node(...)`) that:
  - Assigns deterministic tags based on parent line + offset distance + story index
  - Deduplicates if the same node is requested multiple times
  - Updates the in-memory node registry
- Ensure `nodes.json` is written with these intermediate nodes **before** emitting `beams.json` / `columns.json`.
- Each segment gets a unique `geomTransf` tag derived from its element tag.
- Rigid segments use inflated section properties (e.g., scale factors ×1e6).

## Consequences
- **Positive**
  - Models align with ETABS rigid zone semantics.
  - No element will reference a missing node if the registrar is always used.
  - Deterministic tagging allows reproducibility across runs.
- **Negative**
  - Elements-per-member increase → larger `explicit_model.py`.
  - Downstream consumers expecting one element per ETABS line must filter for `segment=="deformable"`.
  - Slight complexity increase in verification layer.

## Status
Accepted — 2025-09-04.

## Notes
- Invariant: Before emitting beams/columns, **all i/j nodes must already exist in `nodes.json`**.
- Verification: `verify_model.py --check endpoints_exist` will fail fast if dangling references exist.
- Future extension: similar approach will be used for nodal offsets (`OFFSETX/Y/Z`) and `-jntOffset` in `geomTransf`.
