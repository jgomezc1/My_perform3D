# Session Handoff (Rolling Notes)

---
## [2025-09-09]
**Branch & Commit:** feature/offsets @ 34264e2

**Last Session**
- Rigid ends implemented. Intermediate nodes now registered in nodes.json.

**Next Up**
- Attach intermediate nodes (kind=intermediate) at each story to that story’s rigid diaphragm.


## [2025-09-08]
**Branch & Commit:** feature/offsets @ 34264e2

**Last Session**
- Rigid ends implemented. Intermediate nodes now registered in nodes.json.

**Next Up**
- Attach intermediate nodes (kind=intermediate) at each story to that story’s rigid diaphragm.

## [2025-09-04]
**Branch & Commit:** feature/offsets @ 77f6b85

**Last Session**
- Implemented rigid-end splitting for beams/columns when `LENGTHOFFI/J` present.
- Created intermediate nodes (white nodes) at rigid/deformable interfaces.

**Problem Discovered**
- Intermediate nodes were **not** propagated into `nodes.json`.
- Downstream, `generate_explicit_model.py` references non-existent nodes → model fails.

**Next Up (Wave-1: correctness)**
- Ensure all intermediate nodes created during splitting are **registered** into `nodes.json` before any element emission.
- Add verification: “no element endpoint is missing in `nodes.json`”.

**Risks / Blockers**
- Double-creation of the same intermediate node if multiple lines share geometry.
- Tagging collisions if tag derivation for intermediate nodes isn’t deterministic.

