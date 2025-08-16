# phase1_run.py
"""
CLI entrypoint for Phase 1.
Usage:
    python phase1_run.py
Outputs:
    out/parsed_raw.json
    out/story_graph.json
    out/story_table.csv
    out/point_matrix.csv
"""
import json
from pathlib import Path
import pandas as pd

from config import E2K_PATH, OUT_DIR
from e2k_parser import parse_e2k
from story_builder import build_story_graph

def main():
    text = Path(E2K_PATH).read_text(encoding="utf-8", errors="ignore")
    raw = parse_e2k(text)
    story = build_story_graph(raw)

    # Write compact JSON (easier to diff; no giant .py files)
    (OUT_DIR / "parsed_raw.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "story_graph.json").write_text(json.dumps(story, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV sanity: stories
    rows = [{
        "story": s["name"],
        "height": s["height"],
        "elev_explicit": s["elev"],
        "elev_computed": story["story_elev"].get(s["name"]),
    } for s in raw["stories"]]
    pd.DataFrame(rows).to_csv(OUT_DIR / "story_table.csv", index=False)

    # CSV sanity: point vs story activation
    story_order = story["story_order_top_to_bottom"]
    pt_to_stories = {}
    for a in raw["point_assigns"]:
        pt_to_stories.setdefault(a["point"], set()).add(a["story"])
    matrix = []
    for pid in sorted(raw["points"].keys(), key=lambda x: (len(x), x)):
        row = {"point": pid}
        assigned = pt_to_stories.get(pid, set())
        for s in story_order:
            row[s] = 1 if s in assigned else 0
        matrix.append(row)
    pd.DataFrame(matrix).to_csv(OUT_DIR / "point_matrix.csv", index=False)

    print("Phase 1 complete.")
    print(f"  - {OUT_DIR/'parsed_raw.json'}")
    print(f"  - {OUT_DIR/'story_graph.json'}")
    print(f"  - {OUT_DIR/'story_table.csv'}")
    print(f"  - {OUT_DIR/'point_matrix.csv'}")

if __name__ == "__main__":
    main()
