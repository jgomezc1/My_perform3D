# view_utils_App.py
"""
Minimal plotting/utility helpers for the Streamlit viewer.

Public API:
- create_interactive_plot(nodes, elements, options) -> plotly.graph_objs.Figure

Where:
    nodes          : Dict[int, Tuple[float, float, float]]
    elements       : Dict[int, Tuple[int, int]]   # element_tag -> (ni, nj)
Options:
    show_axes, show_grid, show_nodes
    show_local_axes, local_axis_frac
    show_master_nodes, master_nodes (Iterable[int]), master_node_size
    # Boundary-condition (supports) overlay:
    show_supports: bool
    supports_by_node: Dict[int, Tuple[int,int,int,int,int,int]]
    supports_dofs: Dict[str,bool]  # keys: UX, UY, UZ, RX, RY, RZ
    supports_size: float           # glyph scale
    supports_exclude: Iterable[int]  # node tags to ignore (e.g., master nodes)

Notes:
- Beams vs columns separated by dominant axis (Z -> columns).
- Local longitudinal axes (i -> j) optional, batched as a single Cone trace.
- Diaphragm master nodes are rendered as distinct markers, independent of "show_nodes".
- Boundary conditions overlay:
    * Translational (UX/UY/UZ): **triangles** in the plane perpendicular to the constrained axis.
      - UX: triangle in YZ-plane
      - UY: triangle in XZ-plane
      - UZ: triangle in XY-plane
    * Rotational (RX/RY/RZ): **'x' symbols** (two short crossing line segments) in the plane perpendicular to the rotation axis.
- No external deps beyond Plotly.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, List, Iterable, Optional, Set
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
        cols_x,  cols_y,  cols_z, cols_text,  cols_hover
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
        hov  = (f"<b>Element</b> {etag}<br>nI={ni} → nJ={nj}"
                f"<br>Δx={p2[0]-p1[0]:.3f}, Δy={p2[1]-p1[1]:.3f}, Δz={p2[2]-p1[2]:.3f}")
        if dom == "Z":
            cols_x += xseq; cols_y += yseq; cols_z += zseq
            cols_text += [txt, "", ""]
            cols_hover += [hov, "", ""]
        else:
            beams_x += xseq; beams_y += yseq; beams_z += zseq
            beams_text += [txt, "", ""]
            beams_hover += [hov, "", ""]

    return beams_x, beams_y, beams_z, beams_text, beams_hover, cols_x, cols_y, cols_z, cols_text, cols_hover


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    m = n // 2
    return (s[m] if n % 2 == 1 else 0.5 * (s[m-1] + s[m]))


def _local_axes_trace(
    nodes: Dict[int, Vec3],
    elements: Dict[int, Tuple[int, int]],
    frac: float = 0.25
):
    """
    Build a single Cone trace that shows local longitudinal axes (i -> j)
    for all given elements. Each arrow is anchored at the element midpoint.
    """
    if not elements or not nodes or frac <= 0.0:
        return None

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

    for _, (ni, nj) in elements.items():
        if ni not in nodes or nj not in nodes:
            continue
        x1, y1, z1 = nodes[ni]
        x2, y2, z2 = nodes[nj]
        dx, dy, dz = (x2-x1), (y2-y1), (z2-z1)
        L = math.sqrt(dx*dx + dy*dy + dz*dz)
        if L <= EPS:
            continue
        xm = 0.5 * (x1 + x2); ym = 0.5 * (y1 + y2); zm = 0.5 * (z1 + z2)
        s = axis_len / L
        xs.append(xm); ys.append(ym); zs.append(zm)
        us.append(dx * s); vs.append(dy * s); ws.append(dz * s)

    if not xs:
        return None

    cone = go.Cone(
        x=xs, y=ys, z=zs,
        u=us, v=vs, w=ws,
        anchor="tail",
        showscale=False,
        name="Local x (i→j)"
    )
    return cone


# ---------- Boundary-condition overlay helpers ----------
def _triangle(center: Vec3, axis: str, r: float) -> Tuple[List[float], List[float], List[float]]:
    """
    Return a closed triangle polyline centered at node, lying in the plane
    perpendicular to 'axis'. The triangle is wireframe (Scatter3d lines).
      - UX: triangle in YZ-plane
      - UY: triangle in XZ-plane
      - UZ: triangle in XY-plane
    """
    x, y, z = center
    if axis == "X":  # YZ-plane
        p1 = (x, y - r, z - r)
        p2 = (x, y + r, z - r)
        p3 = (x, y,     z + r)
    elif axis == "Y":  # XZ-plane
        p1 = (x - r, y, z - r)
        p2 = (x + r, y, z - r)
        p3 = (x,     y, z + r)
    else:  # "Z" -> XY-plane
        p1 = (x - r, y - r, z)
        p2 = (x + r, y - r, z)
        p3 = (x,     y + r, z)
    xs = [p1[0], p2[0], p3[0], p1[0], None]
    ys = [p1[1], p2[1], p3[1], p1[1], None]
    zs = [p1[2], p2[2], p3[2], p1[2], None]
    return xs, ys, zs

def _x_symbol(center: Vec3, axis: str, r: float) -> Tuple[List[float], List[float], List[float]]:
    """
    Return an 'x' symbol (two short crossing line segments) centered at node,
    lying in the plane perpendicular to 'axis'.
      - RX: 'x' in YZ-plane
      - RY: 'x' in XZ-plane
      - RZ: 'x' in XY-plane
    """
    x, y, z = center
    if axis == "X":  # YZ-plane
        a1 = (x, y - r, z - r); b1 = (x, y + r, z + r)
        a2 = (x, y - r, z + r); b2 = (x, y + r, z - r)
    elif axis == "Y":  # XZ-plane
        a1 = (x - r, y, z - r); b1 = (x + r, y, z + r)
        a2 = (x - r, y, z + r); b2 = (x + r, y, z - r)
    else:  # "Z" -> XY-plane
        a1 = (x - r, y - r, z); b1 = (x + r, y + r, z)
        a2 = (x - r, y + r, z); b2 = (x + r, y - r, z)

    xs = [a1[0], b1[0], None, a2[0], b2[0], None]
    ys = [a1[1], b1[1], None, a2[1], b2[1], None]
    zs = [a1[2], b1[2], None, a2[2], b2[2], None]
    return xs, ys, zs

def _supports_traces(
    nodes: Dict[int, Vec3],
    supports_by_node: Dict[int, Tuple[int,int,int,int,int,int]],
    dofs: Dict[str, bool],
    size: float = 0.25,
    exclude: Optional[Iterable[int]] = None,
) -> List[go.Scatter3d]:
    """
    Build per-DOF traces for supports.
      - **Translational (UX/UY/UZ): triangles** in plane ⟂ to axis.
      - **Rotational (RX/RY/RZ): 'x' symbols** in plane ⟂ to axis.
    Excludes any node tags provided in 'exclude'.
    """
    if not nodes or not supports_by_node:
        return []

    excl: Set[int] = set(exclude or [])

    # Per-DOF buffers (polyline coordinates with None breaks)
    tri = { "UX": ([], [], []), "UY": ([], [], []), "UZ": ([], [], []) }
    xsy = { "RX": ([], [], []), "RY": ([], [], []), "RZ": ([], [], []) }

    # Auto scale: use a fraction of global bbox size
    xr, yr, zr = _axis_ranges(nodes)
    bbox = max(xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0])
    L = max(bbox * max(size, 0.01), 1e-6)
    r_tri = 0.5 * L     # triangle "radius"
    r_x   = 0.5 * L     # x-symbol half-length

    for n, mask in supports_by_node.items():
        if n in excl or n not in nodes:
            continue
        cx, cy, cz = nodes[n]
        ux, uy, uz, rx, ry, rz = mask

        # Translational: triangles
        if dofs.get("UX", True) and ux:
            xs, ys, zs = _triangle((cx, cy, cz), "X", r_tri)
            X, Y, Z = tri["UX"]; X += xs; Y += ys; Z += zs
        if dofs.get("UY", True) and uy:
            xs, ys, zs = _triangle((cx, cy, cz), "Y", r_tri)
            X, Y, Z = tri["UY"]; X += xs; Y += ys; Z += zs
        if dofs.get("UZ", True) and uz:
            xs, ys, zs = _triangle((cx, cy, cz), "Z", r_tri)
            X, Y, Z = tri["UZ"]; X += xs; Y += ys; Z += zs

        # Rotational: 'x' symbols
        if dofs.get("RX", True) and rx:
            xs, ys, zs = _x_symbol((cx, cy, cz), "X", r_x)
            X, Y, Z = xsy["RX"]; X += xs; Y += ys; Z += zs
        if dofs.get("RY", True) and ry:
            xs, ys, zs = _x_symbol((cx, cy, cz), "Y", r_x)
            X, Y, Z = xsy["RY"]; X += xs; Y += ys; Z += zs
        if dofs.get("RZ", True) and rz:
            xs, ys, zs = _x_symbol((cx, cy, cz), "Z", r_x)
            X, Y, Z = xsy["RZ"]; X += xs; Y += ys; Z += zs

    traces: List[go.Scatter3d] = []

    # Build triangle traces for UX/UY/UZ
    for key, (xs, ys, zs) in tri.items():
        if dofs.get(key, False) and xs:
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                hoverinfo="skip", name=f"BC {key} (tri)",
                line=dict(width=3)
            ))

    # Build 'x' traces for RX/RY/RZ
    for key, (xs, ys, zs) in xsy.items():
        if dofs.get(key, False) and xs:
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                hoverinfo="skip", name=f"BC {key} (x)",
                line=dict(width=3)
            ))

    return traces


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
      - show_local_axes: bool
      - local_axis_frac: float
      - show_master_nodes: bool
      - master_nodes: Iterable[int]
      - master_node_size: int
      - node_size: int
      - beam_thickness: int
      - column_thickness: int
      - show_supports: bool
      - supports_by_node: Dict[...]
      - supports_dofs: Dict[str,bool]
      - supports_size: float
      - supports_exclude: Iterable[int]
    """
    options = options or {}
    show_axes = bool(options.get("show_axes", True))
    show_grid = bool(options.get("show_grid", True))
    show_nodes = bool(options.get("show_nodes", True))
    show_local_axes = bool(options.get("show_local_axes", False))
    local_axis_frac = float(options.get("local_axis_frac", 0.25))

    show_mn = bool(options.get("show_master_nodes", True))
    mn_tags: Iterable[int] = options.get("master_nodes", []) or []
    mn_size = int(options.get("master_node_size", 8))

    node_size = int(options.get("node_size", 3))
    lw_beam = int(options.get("beam_thickness", 2))
    lw_col  = int(options.get("column_thickness", 3))

    # Supports overlay options
    show_supports = bool(options.get("show_supports", False))
    supports_by_node = options.get("supports_by_node", {}) or {}
    supports_dofs = options.get("supports_dofs", {"UX": True,"UY":True,"UZ":True,"RX":True,"RY":True,"RZ":True})
    supports_size = float(options.get("supports_size", 0.25))
    supports_exclude = options.get("supports_exclude", set()) or set()

    data_traces = []

    # Nodes (markers)
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

    # Local longitudinal axes (i -> j)
    if show_local_axes:
        cone = _local_axes_trace(nodes, elements, frac=max(0.01, local_axis_frac))
        if cone is not None:
            data_traces.append(cone)

    # Diaphragm Master Nodes — independent of show_nodes
    if show_mn and mn_tags:
        mn_tags_list = [int(t) for t in mn_tags if int(t) in nodes]
        if mn_tags_list:
            mx = [nodes[t][0] for t in mn_tags_list]
            my = [nodes[t][1] for t in mn_tags_list]
            mz = [nodes[t][2] for t in mn_tags_list]
            mtext = [f"MN {t}" for t in mn_tags_list]
            mhover = [f"<b>Master Node</b> {t}<br>x={nodes[t][0]:.3f}, y={nodes[t][1]:.3f}, z={nodes[t][2]:.3f}" for t in mn_tags_list]
            masters_trace = go.Scatter3d(
                x=mx, y=my, z=mz,
                mode="markers",
                text=mtext,
                hovertext=mhover,
                hoverinfo="text",
                marker=dict(size=mn_size, color="red"),
                name="Diaphragm Masters"
            )
            data_traces.append(masters_trace)

    # Supports overlay (triangles for U*, 'x' for R*)
    if show_supports and supports_by_node:
        bc_traces = _supports_traces(
            nodes=nodes,
            supports_by_node=supports_by_node,
            dofs=supports_dofs,
            size=supports_size,
            exclude=supports_exclude,
        )
        data_traces.extend(bc_traces)

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
