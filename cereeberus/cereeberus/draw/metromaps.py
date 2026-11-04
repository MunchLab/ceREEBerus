"""Draw the maps stored in a fitted Mapper graph interleaving.

The ``draw_metromaps`` function reads the original and smoothed graphs
and the solved vertex and edge map matrices from an ``Interleave`` object. It
places each source graph beside its target and uses matching colors to trace
source edges or branches through the map. The metro-map style is a visual
metaphor; this module does not run a MetroSets layout algorithm or fit the ILP.
"""

from collections import defaultdict
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

from cereeberus import Interleave

from .draw import _format_node_label

__all__ = ["draw_metromaps", "color_by_edge"]


def _label(node):
    """Format a graph node using the labels shown by ``ReebGraph.draw``.

    Parameters:
        node (hashable): A Mapper graph node identifier. Internal tuple
            identifiers such as ``("mapper_subd", 3)`` are shortened.

    Returns:
        str: The displayed node label, such as ``"s3"``.
    """
    return str(_format_node_label(node))


def _branches(graph):
    """Partition source edges into directed branches for coloring.

    A branch continues through each vertex with exactly one incoming and one
    outgoing edge. It ends at a leaf or junction. Edge keys preserve distinct
    parallel routes, and sorting gives stable branch numbers for a fixed graph.

    Parameters:
        graph (MapperGraph): Source graph with directed, keyed edges and a
            function-value dictionary ``graph.f``.

    Returns:
        list[list[tuple]]: Maximal branches, each containing its directed
        ``(source, target, key)`` edges in traversal order. Every edge occurs
        in exactly one branch.
    """
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
    """Extract source-to-target assignments from a solved ILP map matrix.

    Each source object is a column in one function-value block. Its selected
    row identifies the target object. A column with multiple selected rows,
    or no selected row when ``allow_empty`` is false, is rejected.

    Parameters:
        matrix (LabeledBlockMatrix): A vertex or edge map returned by
            ``Assignment.phi`` or ``Assignment.psi``.
        allow_empty (bool): Skip zero columns in auxiliary ``key='n'`` maps.
            Defaults to ``False``.

    Returns:
        list[tuple]: ``(source, target)`` pairs in block and column order.

    Raises:
        ValueError: If a required column has anything other than one
            selected target.
    """
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
    """Read a vertex's position from the graph's own drawing layout.

    Parameters:
        graph (MapperGraph): Graph containing the vertex and its ``pos_f``
            drawing positions, as used by ``ReebGraph.draw``.
        vertex (hashable): Vertex identifier in ``graph``.

    Returns:
        numpy.ndarray: ``[x, y]`` position from ``graph.pos_f``.

    Raises:
        KeyError: If the vertex has no drawing position. Call
            ``graph.set_pos_from_f()`` to refresh the graph layout.
    """
    return np.asarray(graph.pos_f[vertex], dtype=float)


def _edge_segment(graph, edge):
    """Return the two endpoint positions of an edge for plotting.

    Parameters:
        graph (MapperGraph): Graph containing the edge.
        edge (tuple): Directed edge ``(source, target, key)``. Only its first
            two entries are used for the geometry.

    Returns:
        numpy.ndarray: Array of shape ``(2, 2)`` containing the source
        and target drawing positions.
    """
    return np.vstack((_position(graph, edge[0]), _position(graph, edge[1])))


def _styles(graph, edge_pairs, color_by):
    """Choose the color group and legend label for each source edge.

    Parameters:
        graph (MapperGraph): Source graph used to identify branches.
        edge_pairs (list[tuple]): ``(source_edge, target_edge)`` assignments
            extracted from the ILP edge map.
        color_by (str): ``'branch'`` to share colors along maximal branches,
            or ``'edge'`` to color each source edge separately.

    Returns:
        list[tuple[int, str]]: Color-group number and legend label for each
        edge pair, in the same order as ``edge_pairs``.
    """
    if color_by == "edge":
        return [(i, f"{_label(e[0])}→{_label(e[1])}" +
             (f" ({e[2]})" if len(e) > 2 and e[2] != 0 else ""))
            for i, (e, _) in enumerate(edge_pairs)]
    branches = _branches(graph)
    lookup = {edge: i for i, branch in enumerate(branches) for edge in branch}
    labels = [f"B{i}: {_label(path[0][0])}→{_label(path[-1][1])}"
              for i, path in enumerate(branches)]
    return [(lookup[edge], labels[lookup[edge]]) for edge, _ in edge_pairs]


def _group_colors(styles, color_scheme, warn_color_reuse):
    """Assign colors to the source groups shown in one map direction.

    Parameters:
        styles (list[tuple[int, str]]): Group number and label for each
            mapped source edge.
        color_scheme (str): Name of a Matplotlib colormap.
        warn_color_reuse (bool): Warn when a listed palette has fewer colors
            than the number of displayed groups.

    Returns:
        dict[int, tuple]: RGBA color keyed by source group number. Listed
        palettes cycle after their final color; continuous palettes sample
        evenly across their range.
    """
    palette = plt.get_cmap(color_scheme)
    groups = sorted({group for group, _ in styles})
    if isinstance(palette, ListedColormap):
        if warn_color_reuse and len(groups) > palette.N:
            warnings.warn(
                f"Color scheme {palette.name!r} has {palette.N} colors for "
                f"{len(groups)} source groups; colors will repeat. Choose a "
                "different color_scheme or use color_by='branch'.",
                UserWarning,
                stacklevel=3,
            )
        return {group: palette(i % palette.N) for i, group in enumerate(groups)}
    return {group: palette((i + 0.5) / max(len(groups), 1))
            for i, group in enumerate(groups)}


def _resolve_edge(graph, edge):
    """Resolve a requested source edge to its keyed graph edge.

    Parameters:
        graph (MapperGraph): Source graph for one map direction.
        edge (tuple): ``(u, v, key)`` or ``(u, v)`` when exactly one edge
            joins the two vertices.

    Returns:
        tuple: Keyed ``(u, v, key)`` edge in ``graph``.

    Raises:
        ValueError: If the edge is missing or an unkeyed pair is ambiguous.
    """
    if not isinstance(edge, tuple) or len(edge) not in (2, 3):
        raise ValueError("edge must be (source, target) or (source, target, key).")
    if len(edge) == 3:
        if graph.has_edge(*edge):
            return edge
        raise ValueError(f"Source edge {edge!r} is absent from the graph.")
    u, v = edge
    if not graph.has_edge(u, v):
        raise ValueError(f"Source edge {edge!r} is absent from the graph.")
    keys = list(graph[u][v])
    if len(keys) != 1:
        raise ValueError("Parallel edges require a (source, target, key) tuple.")
    return (u, v, keys[0])


def _draw_pair(assignment, direction, key, axes, *, color_by, color_scheme,
               warn_color_reuse, label_stations, max_legend, selected_edge=None,
               highlight_color="crimson"):
    """Draw one source graph beside the target of one interleaving map.

    This helper reads both vertex and edge ILP maps, overlays colored source
    edges and mapped target strands, and labels destination stations.

    Parameters:
        assignment (Assignment): Solved assignment stored by a fitted
            ``Interleave`` object.
        direction (str): ``'phi'`` for F to G, or ``'psi'`` for G to F.
        key (str): ``'0'`` for the main map or ``'n'`` for the auxiliary map.
        axes (sequence[matplotlib.axes.Axes]): Two axes in source, target
            order. The helper draws into these axes.
        color_by (str): ``'branch'`` or ``'edge'`` source coloring.
        color_scheme (str): Matplotlib colormap name for the source groups.
        warn_color_reuse (bool): Warn if a listed palette repeats colors.
        label_stations (bool): Whether to annotate mapped source vertices
            beside target stations.
        max_legend (int): Maximum number of colored groups in the legend.
        selected_edge (tuple or None): If given, color only this keyed source
            edge and its selected target strand.
        highlight_color (str): Matplotlib color used for ``selected_edge``.

    Returns:
        None: The supplied axes are modified in place.
    """
    source_name, target_name = ("F", "G") if direction == "phi" else ("G", "F")
    source_family = assignment.F if direction == "phi" else assignment.G
    target_family = assignment.G if direction == "phi" else assignment.F
    source = source_family(key)
    target_key = "n" if key == "0" else "2n"
    target = target_family(target_key)
    edge_pairs = _pairs(getattr(assignment, direction)(key, "E"), allow_empty=key == "n")
    vertex_pairs = _pairs(getattr(assignment, direction)(key, "V"), allow_empty=key == "n")
    styles = _styles(source, edge_pairs, color_by)
    colors = (_group_colors(styles, color_scheme, warn_color_reuse)
              if selected_edge is None else {})
    if selected_edge is not None and selected_edge not in {edge for edge, _ in edge_pairs}:
        raise ValueError(f"Source edge {selected_edge!r} has no selected ILP image.")
    source_ax, target_ax = axes

    source.draw(ax=source_ax, with_labels=True, node_size=360)
    for i, (edge, _) in enumerate(edge_pairs):
        if selected_edge is not None and edge != selected_edge:
            continue
        xy = _edge_segment(source, edge)
        color = highlight_color if selected_edge is not None else colors[styles[i][0]]
        source_ax.plot(xy[:, 0], xy[:, 1], color=color,
                       linewidth=5 if selected_edge is not None else 3,
                       zorder=4 if selected_edge is not None else 1)

    for edge in target.edges(keys=True):
        xy = _edge_segment(target, edge)
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
        base = _edge_segment(target, target_edge)
        vector = base[-1] - base[0]
        normal = np.array((-vector[1], vector[0]))
        norm = np.linalg.norm(normal)
        if norm:
            normal /= norm
        for lane, i in enumerate(indices):
            if selected_edge is not None and edge_pairs[i][0] != selected_edge:
                continue
            group, label = styles[i]
            xy = base + (lane - (len(indices) - 1) / 2) * gap * normal
            color = highlight_color if selected_edge is not None else colors[group]
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
        target_ax.legend(handles=handles, title=("Selected edge" if selected_edge is not None
                                                 else f"Source {'branches' if color_by == 'branch' else 'edges'}"),
                         loc="best", fontsize=7, title_fontsize=8, framealpha=.9)


def draw_metromaps(interleaving, *, direction="both", key="0", color_by="branch",
                   color_scheme="tab20", warn_color_reuse=True, figsize=None,
                   label_stations=True, max_legend=16):
    """Draw a fitted interleaving as paired source and target graphs.

    Each row places a colored source Mapper graph next to its smoothed target.
    Colored strands on the target show the selected ILP edge images; target
    stations indicate the selected vertex images. The function only reads the
    fitted result and does not solve or modify the interleaving.

    Parameters:
        interleaving (Interleave): An ``Interleave(F, G)`` object after a
            successful ``fit()`` call. Its ``assignment`` supplies the graphs
            and solved vertex and edge maps.
        direction (str): ``'both'`` (default) makes two rows, F beside G^n
            and G beside F^n. ``'phi'`` or ``'psi'`` makes one corresponding row.
        key (str): ``'0'`` (default) shows the main maps F→G^n and G→F^n.
            ``'n'`` shows the auxiliary maps F^n→G^2n and G^n→F^2n.
        color_by (str): ``'branch'`` (default) gives each maximal source path
            between leaves or junctions one color. ``'edge'`` gives each source
            edge its own color. This affects only the display.
        color_scheme (str): Matplotlib colormap name for source groups.
            Defaults to ``'tab20'``, whose 20 colors repeat when more than 20
            branches or edges are shown in one map direction. A continuous
            colormap such as ``'hsv'`` samples colors across its range.
        warn_color_reuse (bool): Emit a ``UserWarning`` when a listed color
            scheme has fewer colors than source groups. Defaults to ``True``.
        figsize (tuple[float, float] or None): Matplotlib figure size in
            inches. If ``None``, uses ``(16, 7)`` for one row or ``(16, 14)``
            for both rows.
        label_stations (bool): Whether to annotate target vertices with the
            source vertices mapped there. Defaults to ``True``.
        max_legend (int): Maximum number of source color groups displayed in
            each target-panel legend. Defaults to 16; groups beyond this limit
            are still drawn.

    Returns:
        matplotlib.figure.Figure: Figure containing two axes per map
        direction, ordered source then target. Access them through
        ``figure.axes``.

    Raises:
        TypeError: If ``interleaving`` is not an ``Interleave`` object.
        ValueError: If the object has not been fitted or an option is invalid.
    """
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
                   color_by=color_by, color_scheme=color_scheme,
                   warn_color_reuse=warn_color_reuse,
                   label_stations=label_stations,
                   max_legend=max_legend)
    fig.suptitle("Solved interleaving maps")
    return fig


def color_by_edge(interleaving, edge, *, direction="phi", key="0",
                  color="crimson", figsize=(16, 7), label_stations=False):
    """Highlight one ILP source edge and its mapped smoothed-target strand.

    The selected source edge is drawn in ``color`` beside the edge selected
    for it by the solved ILP map. Other graph edges remain in the neutral
    background. The target is G^n for ``direction='phi'`` and F^n for
    ``direction='psi'`` when ``key='0'``. This shows the ILP edge image in the
    smoothed graph; it does not infer a path in the unsmoothed target graph.

    Parameters:
        interleaving (Interleave): Fitted object containing a solved assignment.
        edge (tuple): Directed source edge ``(u, v, key)``. ``(u, v)`` is also
            accepted when no parallel edge has the same endpoints.
        direction (str): ``'phi'`` for an edge of F or ``'psi'`` for an edge of
            G. Defaults to ``'phi'``.
        key (str): ``'0'`` for the main map, or ``'n'`` for an auxiliary map.
            Defaults to ``'0'``.
        color (str): Matplotlib color for the highlighted edge and strand.
            Defaults to ``'crimson'``.
        figsize (tuple[float, float]): Figure size in inches.
        label_stations (bool): Whether to show all vertex correspondences at
            target stations. Defaults to ``False`` for a focused edge view.

    Returns:
        matplotlib.figure.Figure: Two-panel source and target figure.

    Raises:
        TypeError: If ``interleaving`` is not an ``Interleave`` object.
        ValueError: If the object is unfitted, the direction or key is invalid,
            the edge is missing or ambiguous, or it has no selected ILP image.
    """
    if not isinstance(interleaving, Interleave):
        raise TypeError("Pass an Interleave object.")
    if interleaving.assignment is None:
        raise ValueError("Call interleaving.fit() before drawing its maps.")
    if direction not in ("phi", "psi"):
        raise ValueError("direction must be 'phi' or 'psi'.")
    if key not in ("0", "n"):
        raise ValueError("key must be '0' or 'n'.")
    source_family = (interleaving.assignment.F if direction == "phi"
                     else interleaving.assignment.G)
    selected_edge = _resolve_edge(source_family(key), edge)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True,
                             constrained_layout=True)
    _draw_pair(interleaving.assignment, direction, key, axes,
               color_by="edge", color_scheme="tab20", warn_color_reuse=False,
               label_stations=label_stations, max_legend=1,
               selected_edge=selected_edge, highlight_color=color)
    fig.suptitle(f"{direction}: selected edge and ILP image")
    return fig
