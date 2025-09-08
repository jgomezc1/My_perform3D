# Roadmap (Near-Term)

## 0–2 weeks
- Wave-1 (Correctness): Route all intermediate node creation through a registry and flush them into `nodes.json` **before** element emission. Add verification for “no dangling endpoints.”
- Wave-2 (Determinism): Introduce a deterministic tag rule for intermediate nodes (e.g., base on parent line’s endpoints + normalized offset distance + story_index).
- Wave-3 (Refactor Safety): Add unit tests to `verify_model.py` that fail when elements reference non-existent nodes.
- Ensure `generate_explicit_model.py` emits per-element `geomTransf` for **all** beams/columns.
- Confirm intermediate nodes are included in `nodes.json` before any element references them.
- Strengthen `verify_domain_vs_artifacts.py` for segment continuity and transform presence.

## 2–6 weeks
- Add `-jntOffset` support to `geomTransf` when nodal offsets (OFFSETX/Y/Z) are present.
- Parameterize sections/materials (remove placeholders) via parsed frame/section properties.
- Expand probes: eigen checks per story; small lateral pushover by story master.

## Nice-to-have
- CI job that runs `explicit_runtime_check.py` on PRs.
