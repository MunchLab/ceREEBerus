"""Metro-style views of the vertex and edge maps returned by ceREEBerus ILPs.

This module intentionally does not solve an ILP. Pass a solved ``Assignment``
or a fitted ``Interleave``; the plotted strands come directly from its matrices.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _assignment(result):
    assignment = getattr(result, "assignment", result)
    if assignment is None or not all(hasattr(assignment, name) for name in ("phi", "psi", "F", "G")):
        raise ValueError("Pass a solved Assignment or a fitted Interleave object.")
    return assignment


def _mapped_pairs(matrix, *, allow_unmapped=False, strict=True):
    """Return (source, target, block, column) from nonzero ILP columns."""
    pairs = []
    for block in matrix.get_all_block_indices():
        labeled = matrix[block]
        values = np.asarray(labeled.get_array())
        for j, source in enumerate(labeled.cols):
            selected = np.flatnonzero(np.isclose(values[:, j], 1))
            if not len(selected) and allow_unmapped:
                continue
            if len(selected) != 1:
                if strict:
                    raise ValueError(
                        f"Block {block!r}, column {j} ({source!r}) has "
                        f"{len(selected)} selected rows; solve the ILP first."
                    )
                continue
            pairs.append((source, labeled.rows[int(selected[0])], block, j))
    return pairs


def _position(graph, vertex):
    if vertex in getattr(graph, "pos_f", {}):
        return np.asarray(graph.pos_f[vertex], dtype=float)
    return np.asarray((0.0, graph.f[vertex] * getattr(graph, "delta", 1)), dtype=float)


def _edge_curve(graph, edge, bend=0.0):
    p = _position(graph, edge[0])
    q = _position(graph, edge[1])
    t = np.linspace(0, 1, 30)
    midpoint = (p + q) / 2
    perpendicular = np.array((-(q - p)[1], (q - p)[0]))
    length = np.linalg.norm(perpendicular)
    if length:
        midpoint += bend * perpendicular / length
    return ((1 - t) ** 2)[:, None] * p + (2 * t * (1 - t))[:, None] * midpoint + (t ** 2)[:, None] * q


def draw_metro_map(result, direction="phi", key="0", ax=None, *, label_stations=True,
                   max_legend=16, strict=True):
    """Draw one ILP map over its destination graph.

    ``phi`` shows F -> G^n (or F^n -> G^2n with ``key='n'``);
    ``psi`` shows G -> F^n (or G^n -> F^2n). Each source edge is a
    colored strand on its mapped destination edge. A station lists the source
    vertices mapped to it. Parallel strands are spaced to expose overlap.

    Returns ``(figure, axes)``. The original graphs and matrices are untouched.
    """
    assignment = _assignment(result)
    if direction not in ("phi", "psi") or key not in ("0", "n"):
        raise ValueError("direction must be 'phi' or 'psi', and key must be '0' or 'n'.")
    target = (assignment.G if direction == "phi" else assignment.F)("n" if key == "0" else "2n")
    # Auxiliary n maps can have zero columns at the added boundary levels.
    vertex_pairs = _mapped_pairs(getattr(assignment, direction)(key, "V"),
                                 allow_unmapped=key == "n", strict=strict)
    edge_pairs = _mapped_pairs(getattr(assignment, direction)(key, "E"),
                               allow_unmapped=key == "n", strict=strict)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    # Baseline network, including edge multiplicity where NetworkX exposes it.
    for edge in target.edges():
        xy = _edge_curve(target, edge)
        ax.plot(xy[:, 0], xy[:, 1], color="#cbd5e1", linewidth=5, zorder=1)

    # A distinct color per source edge. Indices preserve repeated edge labels.
    palette = plt.get_cmap("tab20")
    grouped = defaultdict(list)
    for index, (source, dest, block, column) in enumerate(edge_pairs):
        grouped[dest].append((index, source, block, column))
    handles = []
    all_x = [float(_position(target, v)[0]) for v in target.nodes]
    span = max(all_x) - min(all_x) if all_x else 1.0
    lane_gap = max(span * 0.025, 0.025)
    for dest, strands in grouped.items():
        if not isinstance(dest, tuple) or len(dest) < 2 or not target.has_edge(dest[0], dest[1]):
            raise ValueError(f"Mapped destination edge {dest!r} is absent from the target graph.")
        base = _edge_curve(target, dest)
        p, q = base[0], base[-1]
        perpendicular = np.array((-(q - p)[1], (q - p)[0]))
        length = np.linalg.norm(perpendicular)
        if length:
            perpendicular /= length
        for lane, (index, source, _block, _column) in enumerate(strands):
            offset = (lane - (len(strands) - 1) / 2) * lane_gap
            xy = base + offset * perpendicular
            color = palette(index % 20)
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.8,
                    solid_capstyle="round", zorder=3)
            if index < max_legend:
                handles.append(Line2D([], [], color=color, linewidth=3, label=f"{source}"))

    stations = defaultdict(list)
    for source, dest, _block, _column in vertex_pairs:
        if dest not in target:
            raise ValueError(f"Mapped destination vertex {dest!r} is absent from the target graph.")
        stations[dest].append(source)
    for vertex in target.nodes:
        x, y = _position(target, vertex)
        mapped = stations[vertex]
        ax.scatter([x], [y], s=95 if mapped else 36, facecolor="white",
                   edgecolor="#172554" if mapped else "#94a3b8", linewidth=1.7, zorder=5)
        if label_stations:
            sources = ", ".join(map(str, mapped)) if mapped else "—"
            ax.annotate(f"{vertex}\n← {sources}", (x, y), xytext=(7, 5),
                        textcoords="offset points", fontsize=8, zorder=6)

    src_name, dst_name = ("F", "G") if direction == "phi" else ("G", "F")
    target_key = "n" if key == "0" else "2n"
    ax.set_title(f"{direction}: {src_name}{'' if key == '0' else '^n'} → {dst_name}^{target_key}")
    ax.set_ylabel("Function value")
    ax.set_xticks([])
    ax.margins(0.18)
    ax.set_aspect("auto")
    if handles:
        ax.legend(handles=handles, title="Source edges", loc="best", fontsize=7,
                  title_fontsize=8, framealpha=0.9)
    return fig, ax


def draw_interleaving_metro_maps(result, key="0", *, figsize=(16, 7), **kwargs):
    """Draw both ILP directions side by side; returns ``(figure, axes)``."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    draw_metro_map(result, "phi", key, ax=axes[0], **kwargs)
    draw_metro_map(result, "psi", key, ax=axes[1], **kwargs)
    return fig, axes
