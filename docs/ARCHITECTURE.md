# System Architecture (My_perform3D)

## Overview
Goal: translate ETABS `.e2k` into an OpenSeesPy model + machine-readable artifacts, then verify and probe the model.

**Phases**
1. **Phase-1 (Parse & Story Graph)**
   - `e2k_parser.py` → normalized dict
   - `story_builder.py` → `out/story_graph.json` (stories, elevations, active points/lines)
2. **Phase-2 (Domain + Artifacts)**
   - `nodes.py` + `emit_nodes.py` → `out/nodes.json` (domain nodes incl. diaphragm masters)
   - `diaphragms.py` → `out/diaphragms.json` (master/slaves, mass/fix)
   - `beams.py`, `columns.py` → `out/beams.json`, `out/columns.json` (segmenting for rigid ends)
   - Supports/BCs → `out/supports.json`
3. **Explicit Model & Checks**
   - `generate_explicit_model.py` → `out/explicit_model.py`
   - `explicit_runtime_check.py` / `explicit_static_probe.py` → smoke/probe analyses
   - `verify_model.py`, `verify_domain_vs_artifacts.py` → parity checks

**Pipelines**
- `phase1_run.py`, `run_pipeline.py` orchestrate end-to-end generation.

## Dataflow
`.e2k` → `e2k_parser.py` → `story_builder.py` → `story_graph.json`
→ (`diaphragms.py`, `nodes.py`/`emit_nodes.py`) → `nodes.json`, `diaphragms.json`
→ (`beams.py`, `columns.py`, `supports.py`) → `beams.json`, `columns.json`, `supports.json`
→ `generate_explicit_model.py` → `explicit_model.py` → checks/probes

## Key Conventions
- Node tag determinism (see PROMPT / NONNEGOTIABLES).
- Rigid end logic handled via `rigid_end_utils.split_with_rigid_ends(...)`.
- Per-element `geomTransf` (unique tag derived from element tag).
### Node creation ordering (critical)
- Any process that creates new nodes (e.g., rigid-end splitting) must **register nodes first** into the nodes registry → flushed to `out/nodes.json`.
- Only then may `beams.json` / `columns.json` be emitted, ensuring their `i_node`/`j_node` exist.
- Preferred API: `emit_nodes.register_intermediate_node(story, x, y, z, source="rigid_end") -> tag`
  - Deterministic tag strategy required (see NONNEGOTIABLES).

## External Interfaces
- OpenSeesPy API for model build, constraints, analysis.
- JSON artifacts in `out/` consumed by generator and viewers.

## Failure Modes to Watch
- Missing intermediate nodes when splitting for rigid ends.
- Diaphragm creation on stories with supports.
- Per-element transforms missing in `explicit_model.py`.
