# Artifact Contracts (JSON)

All artifacts live under `out/`. Schemas are **stable** unless explicitly changed.

## 1) story_graph.json (excerpt)
```json
{
  "story_order_top_to_bottom": ["Roof","Story-1","Story-2"],
  "story_elev": {"Roof":30.0,"Story-1":20.0,"Story-2":10.0},
  "active_points": {
    "Roof": [{ "id":"101", "x":0.0, "y":0.0, "z":30.0, "diaphragm":"D1" }]
  },
  "active_lines": {
    "Roof": [
      { "name":"L1", "type":"BEAM", "i":"101", "j":"102",
        "length_off_i":0.30, "length_off_j":0.20, "section":"B400x500" }
    ]
  }
}

## 2) nodes.json-nodes 
`kind ∈ {"grid","diaphragm_master","intermediate","generated"}`

- `grid`: original grid/point nodes from ETABS.
- `diaphragm_master`: story master nodes.
- `intermediate`: nodes created by rigid-end splits at I/J interfaces.
- `generated`: any algorithmic node not directly tied to a raw ETABS point.
```json
{
  "nodes": [
    {"tag":101000,"x":0.0,"y":0.0,"z":30.0,"story":"Roof","story_index":0,"kind":"grid","source_point_id":"101"},
    {"tag":9001,"x":12.3,"y":8.7,"z":30.0,"story":"Roof","story_index":0,"kind":"diaphragm_master"}
  ],
  "counts":{"total":123,"grid":120,"master":3},
  "version":1
}

### Mandatory existence rule (hard invariant)
- **Every** `i_node` / `j_node` referenced by `beams.json` or `columns.json` **must** exist in `nodes.json`.
- The build fails verification if any reference is missing.
"counts":{"total":123,"grid":120,"master":2,"intermediate":1}

## 3) diaphragms.json (excerpt)
```json
{
  "diaphragms":[
    {
      "story":"Roof","master":9001,"slaves":[101000,102000],
      "mass":{"M":12345.0,"Izz":1234500.0,"A":400.0,"t":0.1,"rho":2500.0,"applied":true},
      "fix":{"ux":0,"uy":0,"uz":1,"rx":1,"ry":1,"rz":0,"applied":true}
    }
  ]
}

## 4) beams.json (one entry per segment)
```json
{
  "beams": [
    {
      "tag": 210001, "segment":"rigid_i","parent_line":"L1","story":"Roof",
      "i_node":101000,"j_node":501000,"transf_tag":1210001,
      "A": 0.2,"E":2.5e10,"G":1.04e10,"J":0.01,"Iy":0.004,"Iz":0.002,
      "length_off_i":0.30,"length_off_j":0.20
    },
    { "tag": 210002, "segment":"deformable", "...": "..." },
    { "tag": 210003, "segment":"rigid_j", "...": "..." }
  ],
  "counts":{"created":3},
  "skips":[]
}

## 5) columns.json (same pattern as beams.json)

## 6) supports.json (typical)
```json
{
  "applied":[ {"node": 301003, "fix":[1,1,1,1,1,1]} ],
  "notes":"Tags imply story via tag%1000."
}

