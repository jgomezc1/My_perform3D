# view_utils_App.py
"""
Minimal plotting/utility helpers for the Streamlit viewer.

Public API:
- create_interactive_plot(nodes, elements, options) -> plotly.graph_objs.Figure

Where:
    nodes    : Dict[int, Tuple[float, float, float]]
    elements : Dict[int, Tuple[int, int]]   # element_tag -> (ni, nj)

Notes:
- We auto-separate "beams" vs "columns" by orientation (dominant axis).
- Optional: draw local longitudinal axes (i -> j) as 3D cones at member midpoints.
- No external deps beyond Plotly.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, List
import math
from plotly import graph_objects as go

Vec3 = Tuple[float, float, float]

# Fallback EPS if config is absent
try:
    from config import EPS  # shared project tolerance
except Exception:
    EPS = 1e-9


def _axis_ranges(nodes: Dict[int, Vec3], pad_ratio: float = 0.05) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]:
    if not nodes:
        return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    xs, ys, zs = zip(*[nodes[n] for n in nodes])
    def pad(lo, hi):
        span = max(hi - lo, 1.0)
        p = span * pad_ratio
        return lo - p, hi + p
    return pad(min(xs), max(xs)), pad(min(ys), max(ys)), pad(min(zs), max(zs))


def _dominant_axis(p1: Vec3, p2: Vec3) -> str:
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    dz = abs(p2[2] - p1[2])
    m = max(dx, dy, dz)
    if m <= EPS:
        return "other"
    if m == dz: return "Z"
    if m == dx: return "X"
    if m == dy: return "Y"
    return "other"


def _segment_lists(
    nodes: Dict[int, Vec3],
    elements: Dict[int, Tuple[int, int]],
) -> Tuple[List[float], List[float], List[float], List[str], List[str],
           List[float], List[float], List[float], List[str], List[str]]:
    """
    Build two polyline traces (beams vs columns) by separating with None breaks.
    Returns:
        beams_x, beams_y, beams_z, beams_text, beams_hover,
        cols_x,  cols_y,  cols_z,  cols_text,  cols_hover
    """
    beams_x: List[float]; beams_y: List[float]; beams_z: List[float]
    beams_x, beams_y, beams_z, beams_text, beams_hover = [], [], [], [], []
    cols_x: List[float]; cols_y: List[float]; cols_z: List[float]
    cols_x, cols_y, cols_z, cols_text, cols_hover = [], [], [], [], []

    for etag in sorted(elements.keys()):
        ni, nj = elements[etag]
        if ni not in nodes or nj not in nodes:
            continue
        p1, p2 = nodes[ni], nodes[nj]
        dom = _dominant_axis(p1, p2)
        xseq = [p1[0], p2[0], None]
        yseq = [p1[1], p2[1], None]
        zseq = [p1[2], p2[2], None]
        txt  = f"Ele {etag} | nI={ni}, nJ={nj}"
        hov  = f"<b>Element</b> {etag}<br>nI={ni} → nJ={nj}<br>Δx={p2[0]-p1[0]:.3f}, Δy={p2[1]-p1[1]:.3f}, Δz={p2[2]-p1[2]:.3f}"

        if dom == "Z":
            cols_x += xseq; cols_y += yseq; cols_z += zseq
            cols_text += [txt, txt, ""]
            cols_hover += [hov, hov, ""]
        else:
            beams_x += xseq; beams_y += yseq; beams_z += zseq
            beams_text += [txt, txt, ""]
            beams_hover += [hov, hov, ""]

    return beams_x, beams_y, beams_z, beams_text, beams_hover, cols_x, cols_y, cols_z, cols_text, cols_hover


def _median(values: List[float]) -> float:
    if not values: return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _local_axes_trace(
    nodes: Dict[int, Vec3],
    elements: Dict[int, Tuple[int, int]],
    frac: float = 0.25
):
    """
    Build a single Cone trace that shows local longitudinal axes (i -> j)
    for all given elements. Each arrow is anchored at the element midpoint.

    Parameters
    ----------
    frac : float
        The arrow length is frac * median(element_length). Must be > 0.
        Using the median makes arrows robust against a few very long members.
    """
    if not elements or not nodes or frac <= 0.0:
        return None

    # First pass: gather lengths
    lengths: List[float] = []
    for _, (ni, nj) in elements.items():
        if ni not in nodes or nj not in nodes:
            continue
        x1, y1, z1 = nodes[ni]
        x2, y2, z2 = nodes[nj]
        L = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        if L > EPS:
            lengths.append(L)

    if not lengths:
        return None

    Lmed = _median(lengths)
    axis_len = max(Lmed * frac, EPS * 100.0)

    xs: List[float]; ys: List[float]; zs: List[float]
    us: List[float]; vs: List[float]; ws: List[float]
    xs, ys, zs, us, vs, ws = [], [], [], [], [], []

    # Second pass: midpoints + directions (i -> j), normalized then scaled
    for _, (ni, nj) in elements.items():
        if ni not in nodes or nj not in nodes:
            continue
        x1, y1, z1 = nodes[ni]
        x2, y2, z2 = nodes[nj]
        dx, dy, dz = (x2-x1), (y2-y1), (z2-z1)
        L = math.sqrt(dx*dx + dy*dy + dz*dz)
        if L <= EPS:
            continue
        # midpoint as anchor
        xm = 0.5 * (x1 + x2)
        ym = 0.5 * (y1 + y2)
        zm = 0.5 * (z1 + z2)
        xs.append(xm); ys.append(ym); zs.append(zm)
        # normalized direction scaled to a common display length
        s = axis_len / L
        us.append(dx * s); vs.append(dy * s); ws.append(dz * s)

    if not xs:
        return None

    # One batched cone trace for performance
    cone = go.Cone(
        x=xs, y=ys, z=zs,
        u=us, v=vs, w=ws,
        anchor="tail",          # arrows point in +u,+v,+w from the anchor (midpoint)
        showscale=False,
        name="Local x (i→j)"
        # We keep default sizemode ("scaled") and omit sizeref to let Plotly scale sensibly.
    )
    return cone


def create_interactive_plot(
    nodes: Dict[int, Vec3],
    elements: Dict[int, Tuple[int, int]],
    options: Dict[str, Any] | None = None
) -> go.Figure:
    """
    Build a tidy Plotly 3D scene with separated traces for beams vs columns.

    options:
      - show_axes: bool
      - show_grid: bool
      - show_nodes: bool
      - show_local_axes: bool        <-- NEW (default False)
      - local_axis_frac: float       <-- NEW (default 0.25)
      - node_size: int
      - beam_thickness: int
      - column_thickness: int
    """
    options = options or {}
    show_axes = bool(options.get("show_axes", True))
    show_grid = bool(options.get("show_grid", True))
    show_nodes = bool(options.get("show_nodes", True))
    show_local_axes = bool(options.get("show_local_axes", False))
    local_axis_frac = float(options.get("local_axis_frac", 0.25))
    node_size = int(options.get("node_size", 3))
    lw_beam = int(options.get("beam_thickness", 2))
    lw_col  = int(options.get("column_thickness", 3))

    data_traces = []

    # Nodes (markers) — only if requested
    if show_nodes and nodes:
        n_tags = sorted(nodes.keys())
        nx = [nodes[i][0] for i in n_tags]
        ny = [nodes[i][1] for i in n_tags]
        nz = [nodes[i][2] for i in n_tags]
        ntext = [f"Node {i}" for i in n_tags]
        nhover = [f"<b>Node</b> {i}<br>x={nodes[i][0]:.3f}, y={nodes[i][1]:.3f}, z={nodes[i][2]:.3f}" for i in n_tags]
        nodes_trace = go.Scatter3d(
            x=nx, y=ny, z=nz,
            mode="markers",
            text=ntext,
            hovertext=nhover,
            hoverinfo="text",
            marker=dict(size=node_size),
            name="Nodes"
        )
        data_traces.append(nodes_trace)

    # Elements (two polyline traces)
    (bx, by, bz, btxt, bhov,
     cx, cy, cz, ctxt, chov) = _segment_lists(nodes, elements)

    if bx:  # Beams / Others
        beams_trace = go.Scatter3d(
            x=bx, y=by, z=bz,
            mode="lines",
            text=btxt,
            hovertext=bhov,
            hoverinfo="text",
            line=dict(width=lw_beam),
            name="Beams/Other (X/Y)"
        )
        data_traces.append(beams_trace)

    if cx:  # Columns
        cols_trace = go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode="lines",
            text=ctxt,
            hovertext=chov,
            hoverinfo="text",
            line=dict(width=lw_col),
            name="Columns (Z)"
        )
        data_traces.append(cols_trace)

    # Local longitudinal axes (i -> j), batched as one cone trace
    if show_local_axes:
        cone = _local_axes_trace(nodes, elements, frac=max(0.01, local_axis_frac))
        if cone is not None:
            data_traces.append(cone)

    xr, yr, zr = _axis_ranges(nodes)
    scene = dict(
        xaxis=dict(title="X", showgrid=show_grid, zeroline=False, range=xr),
        yaxis=dict(title="Y", showgrid=show_grid, zeroline=False, range=yr),
        zaxis=dict(title="Z", showgrid=show_grid, zeroline=False, range=zr),
        aspectmode="data"
    )

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        scene=scene,
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
        title="OpenSees Domain"
    )

    if not show_axes:
        fig.update_scenes(xaxis=dict(visible=False),
                          yaxis=dict(visible=False),
                          zaxis=dict(visible=False))
    return fig
