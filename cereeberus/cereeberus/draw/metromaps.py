"""Plot the maps in a fitted mapper-graph interleaving.

``draw_metromaps`` reads the graphs and solved ILP matrices stored by
:class:`~cereeberus.distance.interleave.Interleave`. It does not fit maps or
change their assignments.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .draw import _format_node_label

__all__ = ["draw_metromaps"]


def _label(node):
    return str(_format_node_label(node))


def _branches(graph):
    """Group edges into directed paths between endpoints and junctions."""
    edges = list(graph.edges(keys=True))
    regular = {v for v in graph if graph.in_degree(v) == graph.out_degree(v) == 1}
    outgoing = {v: next(iter(graph.out_edges(v, keys=True))) for v in regular}
    order = lambda e: (graph.f[e[0]], _label(e[0]), graph.f[e[1]], _label(e[1]), str(e[2]))
    starts = sorted((e for e in edges if e[0] not in regular), key=order)
    branches, seen = [], set()
    for start in starts + sorted(edges, key=order):
        if start in seen:
            continue
        branch, edge = [], start
        while edge not in seen:
            branch.append(edge)
            seen.add(edge)
            if edge[1] not in regular:
                break
            edge = outgoing[edge[1]]
        branches.append(branch)
    return branches


def _pairs(matrix, *, allow_empty=False):
    """Read one selected destination per source column of an ILP map."""
    pairs = []
    for level in matrix.get_all_block_indices():
        block = matrix[level]
        values = np.asarray(block.get_array())
        for column, source in enumerate(block.cols):
            rows = np.flatnonzero(np.isclose(values[:, column], 1))
            if allow_empty and len(rows) == 0:
                continue
            if len(rows) != 1:
                raise ValueError("The interleaving maps must be solved before drawing.")
            pairs.append((source, block.rows[int(rows[0])]))
    return pairs


def _position(graph, vertex):
    if vertex in getattr(graph, "pos_f", {}):
        return np.asarray(graph.pos_f[vertex], dtype=float)
    return np.asarray((0, graph.f[vertex] * getattr(graph, "delta", 1)), dtype=float)


def _curve(graph, edge):
    a, b = _position(graph, edge[0]), _position(graph, edge[1])
    t = np.linspace(0, 1, 30)[:, None]
    return (1 - t) * a + t * b


def _styles(graph, edge_pairs, color_by):
    if color_by == "edge":
        return [(i, f"{_label(e[0])}→{_label(e[1])}" +
             (f" ({e[2]})" if len(e) > 2 and e[2] != 0 else ""))
            for i, (e, _) in enumerate(edge_pairs)]
    branches = _branches(graph)
    lookup = {edge: i for i, branch in enumerate(branches) for edge in branch}
    labels = [f"B{i}: {_label(path[0][0])}→{_label(path[-1][1])}"
              for i, path in enumerate(branches)]
    return [(lookup[edge], labels[lookup[edge]]) for edge, _ in edge_pairs]


def _draw_pair(assignment, direction, key, axes, *, color_by, label_stations, max_legend):
    source_name, target_name = ("F", "G") if direction == "phi" else ("G", "F")
    source_family = assignment.F if direction == "phi" else assignment.G
    target_family = assignment.G if direction == "phi" else assignment.F
    source = source_family(key)
    target_key = "n" if key == "0" else "2n"
    target = target_family(target_key)
    edge_pairs = _pairs(getattr(assignment, direction)(key, "E"), allow_empty=key == "n")
    vertex_pairs = _pairs(getattr(assignment, direction)(key, "V"), allow_empty=key == "n")
    styles = _styles(source, edge_pairs, color_by)
    palette = plt.get_cmap("tab20")
    source_ax, target_ax = axes

    source.draw(ax=source_ax, with_labels=True, node_size=360)
    for i, (edge, _) in enumerate(edge_pairs):
        xy = _curve(source, edge)
        source_ax.plot(xy[:, 0], xy[:, 1], color=palette(styles[i][0] % 20),
                       linewidth=3, zorder=1)

    for edge in target.edges(keys=True):
        xy = _curve(target, edge)
        target_ax.plot(xy[:, 0], xy[:, 1], color="#cbd5e1", linewidth=5, zorder=1)

    grouped = defaultdict(list)
    for i, (source_edge, target_edge) in enumerate(edge_pairs):
        if not target.has_edge(target_edge[0], target_edge[1]):
            raise ValueError(f"Mapped edge {target_edge!r} is absent from the target graph.")
        grouped[target_edge].append(i)
    x_values = [_position(target, v)[0] for v in target]
    gap = max((max(x_values) - min(x_values)) * .025, .025) if x_values else .025
    handles, seen = [], set()
    for target_edge, indices in grouped.items():
        base = _curve(target, target_edge)
        vector = base[-1] - base[0]
        normal = np.array((-vector[1], vector[0]))
        norm = np.linalg.norm(normal)
        if norm:
            normal /= norm
        for lane, i in enumerate(indices):
            group, label = styles[i]
            xy = base + (lane - (len(indices) - 1) / 2) * gap * normal
            color = palette(group % 20)
            target_ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.8, zorder=3)
            if group not in seen and len(handles) < max_legend:
                handles.append(Line2D([], [], color=color, linewidth=3, label=label))
                seen.add(group)

    mapped_vertices = defaultdict(list)
    for source_vertex, target_vertex in vertex_pairs:
        if target_vertex not in target:
            raise ValueError(f"Mapped vertex {target_vertex!r} is absent from the target graph.")
        mapped_vertices[target_vertex].append(source_vertex)
    values = list(target.f.values())
    low, span = min(values), max(1, max(values) - min(values))
    for vertex in target:
        x, y = _position(target, vertex)
        mapped = mapped_vertices[vertex]
        target_ax.scatter([x], [y], s=130 if mapped else 75,
                          color=plt.get_cmap("viridis")((target.f[vertex] - low) / span),
                          edgecolor="#172554" if mapped else "#94a3b8", zorder=5)
        target_ax.annotate(_label(vertex), (x, y), ha="center", va="center", fontsize=7, zorder=6)
        if mapped and label_stations:
            target_ax.annotate("← " + ", ".join(map(_label, mapped)), (x, y),
                               xytext=(9, 8), textcoords="offset points", fontsize=7)

    source_ax.set_title(f"{source_name}{'' if key == '0' else '^n'} · source")
    target_ax.set_title(f"{target_name}^{target_key} · mapped target")
    for ax in axes:
        ax.set_xticks([])
        ax.margins(.16)
        ax.set_ylabel("Function value" if ax is source_ax else "")
    if handles:
        target_ax.legend(handles=handles, title=f"Source {'branches' if color_by == 'branch' else 'edges'}",
                         loc="best", fontsize=7, title_fontsize=8, framealpha=.9)


def draw_metromaps(interleaving, *, direction="both", key="0", color_by="branch",
                   figsize=None, label_stations=True, max_legend=16):
    """Return a Matplotlib figure showing the maps in a fitted ``Interleave``.

    ``direction='both'`` gives two rows: F beside G^n, then G beside F^n.
    Use ``'phi'`` or ``'psi'`` for one row. ``key='n'`` draws the auxiliary
    maps F^n→G^2n and G^n→F^2n. Source branches share colors by default;
    ``color_by='edge'`` assigns one color to each source edge.
    """
    from ..distance.interleave import Interleave

    if not isinstance(interleaving, Interleave):
        raise TypeError("Pass an Interleave object.")
    if interleaving.assignment is None:
        raise ValueError("Call interleaving.fit() before drawing its maps.")
    if direction not in ("both", "phi", "psi"):
        raise ValueError("direction must be 'both', 'phi', or 'psi'.")
    if key not in ("0", "n"):
        raise ValueError("key must be '0' or 'n'.")
    if color_by not in ("branch", "edge"):
        raise ValueError("color_by must be 'branch' or 'edge'.")
    directions = ("phi", "psi") if direction == "both" else (direction,)
    fig, axes = plt.subplots(len(directions), 2, figsize=figsize or (16, 7 * len(directions)),
                             squeeze=False, sharey="row", constrained_layout=True)
    for row, map_name in enumerate(directions):
        _draw_pair(interleaving.assignment, map_name, key, axes[row],
                   color_by=color_by, label_stations=label_stations,
                   max_legend=max_legend)
    fig.suptitle("Solved interleaving maps")
    return fig
