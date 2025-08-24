\# Architecture (High-Signal Map)



\## Data Flow

ETABS `.e2k` → \*\*Phase-1\*\* (`e2k\_parser.py` → `story\_builder.py`) → artifacts in `out/`:

\- `parsed\_raw.json`, `story\_graph.json`, plus CSV sanity files.



\*\*Phase-2 (domain)\*\* via `MODEL\_translator.build\_model(stage)`:

1\) `nodes.define\_nodes()` builds grid nodes (tag = `point\_id\*1000 + story\_index`).

2\) `diaphragms.define\_rigid\_diaphragms()` creates master @ XY centroid; \*\*skips\*\* stories with supports and the `"DISCONNECTED"` label; writes `out/diaphragms.json`.

3\) `supports.define\_point\_restraints\_from\_e2k()` maps ETABS `RESTRAINT` → `fix(...)`; uses the grid-tag rule; optional fallback to `parsed\_raw.json`.

4\) `columns.define\_columns()` vertical segments with “find-next-lower-story”; i=bottom, j=top enforced.

5\) `beams.define\_beams()` per-story, “last section wins”.



Viewer `model\_viewer\_APP.py`: plots nodes/elements; overlays diaphragm masters and BCs from `out/\*.json`.



\## Files (purpose)

\- `e2k\_parser.py`: tolerant parser for stories/points/assigns/lines/diaphragms.

\- `story\_builder.py`: computes story elevations, active points/lines per story.

\- `nodes.py`, `diaphragms.py`, `supports.py`, `columns.py`, `beams.py`: domain builders.

\- `MODEL\_translator.py`: orchestrates build stages.

\- `model\_viewer\_APP.py`: Streamlit viewer.



\## Conventions

\- \*\*Tag rule\*\*: `node\_tag = int(point\_id)\*1000 + story\_index` (top=0 downward).

\- \*\*Stages\*\*: `nodes` → `columns` → `all`.

\- \*\*Diaphragms\*\*: all-or-nothing per story; DISCONNECTED = none; stories with supports = none.



