"""Draw fitted interleaving maps with matching source and target colors."""

from collections import defaultdict
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize

from cereeberus import Interleave

from .draw import _format_node_label

__all__ = ["draw_metromaps", "color_by_arc", "color_by_edge"]


def _label(node):
    """Format a node using the same labels as ReebGraph.draw."""
    return str(_format_node_label(node))


def _arcs(graph):
    """Partition keyed edges into maximal directed arcs.

    Continue through vertices with one incoming and one outgoing edge.
    Sorting makes arc numbers stable for a fixed graph.
    """
    edges = list(graph.edges(keys=True))
    regular = {v for v in graph if graph.in_degree(v) == graph.out_degree(v) == 1}
    outgoing = {v: next(iter(graph.out_edges(v, keys=True))) for v in regular}
    order = lambda e: (graph.f[e[0]], _label(e[0]), graph.f[e[1]], _label(e[1]), str(e[2]))
    starts = sorted((e for e in edges if e[0] not in regular), key=order)
    arcs, seen = [], set()
    for start in starts + sorted(edges, key=order):
        if start in seen:
            continue
        arc, edge = [], start
        while edge not in seen:
            arc.append(edge)
            seen.add(edge)
            if edge[1] not in regular:
                break
            edge = outgoing[edge[1]]
        arcs.append(arc)
    return arcs


def _pairs(matrix, *, allow_empty=False):
    """Read (source, target) pairs from the selected matrix rows.

    Require one selected row per column, except for empty auxiliary columns
    when allow_empty is True.
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


def _edge_segment(graph, edge):
    """Read the two endpoint positions of a keyed edge from graph.pos_f."""
    return np.asarray([graph.pos_f[v] for v in edge[:2]], dtype=float)


def _styles(graph, edge_pairs, color_by):
    """Return a color-group number and legend label for each source edge."""
    if color_by == "edge":
        return [(i, f"{_label(e[0])}→{_label(e[1])}" +
             (f" ({e[2]})" if len(e) > 2 and e[2] != 0 else ""))
            for i, (e, _) in enumerate(edge_pairs)]
    arcs = _arcs(graph)
    lookup = {edge: i for i, arc in enumerate(arcs) for edge in arc}
    labels = [f"A{i}: {_label(path[0][0])}→{_label(path[-1][1])}"
              for i, path in enumerate(arcs)]
    return [(lookup[edge], labels[lookup[edge]]) for edge, _ in edge_pairs]


def _group_colors(styles, color_scheme, warn_color_reuse):
    """Assign colors to groups, cycling listed palettes when necessary.

    Continuous palettes are sampled evenly across the displayed groups.
    """
    palette = plt.get_cmap(color_scheme)
    groups = sorted({group for group, _ in styles})
    if isinstance(palette, ListedColormap):
        if warn_color_reuse and len(groups) > palette.N:
            warnings.warn(
                f"Color scheme {palette.name!r} has {palette.N} colors for "
                f"{len(groups)} source groups; colors will repeat. Choose a "
                "different color_scheme or use color_by='arc'.",
                UserWarning,
                stacklevel=3,
            )
        return {group: palette(i % palette.N) for i, group in enumerate(groups)}
    return {group: palette((i + 0.5) / max(len(groups), 1))
            for i, group in enumerate(groups)}


def _resolve_edge(graph, edge):
    """Resolve a source edge, requiring its key when parallel edges exist."""
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


def _validated_assignment(interleaving, direction, key, *, allow_both=False):
    """Validate the fitted interleaving, map direction, and smoothing level."""
    if not isinstance(interleaving, Interleave):
        raise TypeError("Pass an Interleave object.")
    if interleaving.assignment is None:
        raise ValueError("Call interleaving.fit() before drawing its maps.")
    directions = ("both", "phi", "psi") if allow_both else ("phi", "psi")
    if direction not in directions:
        choices = ", ".join(repr(value) for value in directions)
        raise ValueError(f"direction must be {choices}.")
    if key not in ("0", "n"):
        raise ValueError("key must be '0' or 'n'.")
    return interleaving.assignment


def _graph_title(assignment, which, exponent=None):
    """Use the input graph name (or F/G) and numeric smoothing exponent."""
    graph = (assignment.F if which == "F" else assignment.G)("0")
    name = str(graph.name or which)
    return name if exponent is None else f"{name}^{exponent}"


def _draw_pair(assignment, direction, key, axes, *, color_by, color_scheme,
               warn_color_reuse, show_vert_maps, max_legend, level_set_colors=False, selected_edges=None,
               selected_label=None, highlight_color="crimson"):
    """Draw source edges and their selected target images into two axes."""
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
              if selected_edges is None else {})
    if selected_edges is not None:
        missing = selected_edges - {edge for edge, _ in edge_pairs}
        if missing:
            raise ValueError(f"Source edges {missing!r} have no selected ILP image.")
    source_ax, target_ax = axes
    heights = [xy[1] for graph in (source, target) for xy in graph.pos_f.values()]
    height_norm = Normalize(min(heights), max(heights))

    def vertex_color(height):
        return plt.get_cmap("viridis")(height_norm(height)) if level_set_colors else "white"

    source.draw(ax=source_ax, with_labels=True, node_size=240)
    for line in source_ax.lines:
        line.set_color("#222222")
    for text in source_ax.texts:
        text.set_fontsize(8)
        text.set_zorder(6)
    for collection in source_ax.collections:
        collection.set_zorder(5)
        collection.set_array(None)
        collection.set_facecolor([vertex_color(source.pos_f[v][1]) for v in source])
        collection.set_edgecolor("#222222")
    for i, (edge, _) in enumerate(edge_pairs):
        if selected_edges is not None and edge not in selected_edges:
            continue
        xy = _edge_segment(source, edge)
        color = highlight_color if selected_edges is not None else colors[styles[i][0]]
        source_ax.plot(xy[:, 0], xy[:, 1], color=color,
                       linewidth=3.5 if selected_edges is not None else 3,
                       zorder=4 if selected_edges is not None else 1)

    for edge in target.edges(keys=True):
        xy = _edge_segment(target, edge)
        target_ax.plot(xy[:, 0], xy[:, 1], color="#222222", linewidth=1.5, zorder=1)

    grouped = defaultdict(list)
    for i, (source_edge, target_edge) in enumerate(edge_pairs):
        if not target.has_edge(*target_edge):
            raise ValueError(f"Mapped edge {target_edge!r} is absent from the target graph.")
        grouped[target_edge].append(i)
    x_values = [target.pos_f[v][0] for v in target]
    gap = max((max(x_values) - min(x_values)) * .006, .006) if x_values else .006
    handles, seen = [], set()
    for target_edge, indices in grouped.items():
        base = _edge_segment(target, target_edge)
        vector = base[-1] - base[0]
        for lane, i in enumerate(indices):
            if selected_edges is not None and edge_pairs[i][0] not in selected_edges:
                continue
            group, label = styles[i]
            offset = np.array([(lane - (len(indices) - 1) / 2) * gap, 0])
            # Separate shared tracks horizontally and join them at stations.
            xy = np.vstack((base[0], base[0] + 0.15 * vector + offset,
                            base[1] - 0.15 * vector + offset, base[1]))
            color = highlight_color if selected_edges is not None else colors[group]
            target_ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.6, zorder=3,
                           solid_capstyle="round", solid_joinstyle="round")
            if group not in seen and len(handles) < max_legend:
                handles.append(Line2D([], [], color=color, linewidth=3, label=label))
                seen.add(group)

    mapped_vertices = defaultdict(list)
    for source_vertex, target_vertex in vertex_pairs:
        if target_vertex not in target:
            raise ValueError(f"Mapped vertex {target_vertex!r} is absent from the target graph.")
        mapped_vertices[target_vertex].append(source_vertex)
    for vertex in target:
        x, y = np.asarray(target.pos_f[vertex], dtype=float)
        mapped = mapped_vertices[vertex]
        target_ax.scatter([x], [y], s=240,
                          color=vertex_color(y),
                          edgecolor="#222222", zorder=5)
        target_ax.annotate(_label(vertex), (x, y), ha="center", va="center", fontsize=8, zorder=6)
        if mapped and show_vert_maps:
            target_ax.annotate("← " + ", ".join(map(_label, mapped)), (x, y),
                               xytext=(9, 8), textcoords="offset points", fontsize=7)

    source_ax.set_title(_graph_title(assignment, source_name,
                                     None if key == "0" else assignment.n))
    target_ax.set_title(_graph_title(assignment, target_name,
                                     assignment.n if key == "0" else 2 * assignment.n))
    for ax in axes:
        ax.set_xticks([])
        ax.margins(x=0.08, y=0.05)
        ax.set_ylabel("Function value")
    if handles:
        target_ax.legend(handles=handles, title=(selected_label if selected_edges is not None
                                                 else f"Source {'arcs' if color_by == 'arc' else 'edges'}"),
                         loc="best", fontsize=7, title_fontsize=8, framealpha=.9)


def draw_metromaps(interleaving, *, direction="both", key="0", color_by="arc",
                   color_scheme="tab10", warn_color_reuse=True, figsize=None,
                   show_vert_maps=True, max_legend=16, level_set_colors=False, sharey=True):
    """Draw the maps of a fitted interleaving as paired graphs.

    Source edges and their selected target images share a color. Several
    images on one target edge are drawn as parallel tracks joined at stations.
    This uses the graphs' saved positions and does not solve or change the
    interleaving. Vertices have white fills and dark outlines by default.
    Set level_set_colors=True to
    color them by height, using the same color scale within each pair.

    Parameters:
        interleaving (Interleave): Object after a successful fit() call.
        direction (str): 'both', 'phi' (F to G^n), or 'psi' (G to F^n).
            Default is 'both'.
        key (str): '0' for the main maps, or 'n' for the auxiliary maps
            F^n to G^2n and G^n to F^2n. Default is '0'.
        color_by (str): 'arc' groups edges between leaves or junctions;
            'edge' colors each edge separately. Default is 'arc'.
        color_scheme (str): Matplotlib colormap. Default is 'tab10'. Listed
            palettes repeat colors; continuous palettes spread them evenly.
        warn_color_reuse (bool): Warn when a listed palette repeats colors.
            Default is True.
        figsize (tuple or None): Figure size in inches. Default is (16, 9)
            per row, with one row for each direction.
        show_vert_maps (bool): Annotate target nodes with their mapped
            source nodes. Default is True.
        max_legend (int): Maximum legend entries per row. Default is 16.
        level_set_colors (bool): Color vertices by function height instead
            of using neutral fills. Default is False.
        sharey (bool): Align the source and target y-axis ranges. If False,
            each panel uses its own range. Default is True.

    Returns:
        matplotlib.figure.Figure: Two axes per direction, source then target.

    Raises:
        TypeError: If interleaving is not an Interleave object.
        ValueError: If the maps are unfitted or the options are invalid.
    """
    assignment = _validated_assignment(interleaving, direction, key, allow_both=True)
    if color_by not in ("arc", "edge"):
        raise ValueError("color_by must be 'arc' or 'edge'.")
    directions = ("phi", "psi") if direction == "both" else (direction,)
    fig, axes = plt.subplots(len(directions), 2, figsize=figsize or (16, 9 * len(directions)),
                             squeeze=False, sharey="row" if sharey else False, constrained_layout=True)
    for row, map_name in enumerate(directions):
        _draw_pair(assignment, map_name, key, axes[row],
                   color_by=color_by, color_scheme=color_scheme,
                   warn_color_reuse=warn_color_reuse,
                   show_vert_maps=show_vert_maps,
                   max_legend=max_legend, level_set_colors=level_set_colors)
    return fig


def _draw_selected(assignment, direction, key, edges, label, color, figsize,
                   show_vert_maps, level_set_colors=False, sharey=True, color_by="edge"):
    """Draw selected source edges and their target images."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=sharey,
                             constrained_layout=True)
    _draw_pair(
        assignment, direction, key, axes, color_by=color_by,
        color_scheme="tab10", warn_color_reuse=False,
        show_vert_maps=show_vert_maps, max_legend=1, level_set_colors=level_set_colors,
        selected_edges=edges, selected_label=label, highlight_color=color,
    )
    return fig


def color_by_edge(interleaving, edge, *, direction="phi", key="0",
                  color="crimson", figsize=(16, 9), show_vert_maps=False,
                  level_set_colors=False, sharey=True):
    """Highlight one source edge and its selected target image.

    Parameters:
        interleaving (Interleave): Object after a successful fit() call.
        edge (tuple): Directed (u, v, key) edge. A pair (u, v) is accepted
            when there is only one edge between those nodes.
        direction (str): 'phi' selects an edge of F; 'psi' selects an edge
            of G. Default is 'phi'.
        key (str): '0' for the main map or 'n' for an auxiliary map.
            Default is '0'.
        color (str): Highlight color. Default is 'crimson'.
        figsize (tuple): Figure size in inches. Default is (16, 9).
        show_vert_maps (bool): Show target-node correspondences.
            Default is False.
        level_set_colors (bool): Color vertices by height. Default is False.
        sharey (bool): Align source and target y-axis ranges. Default is True.

    Returns:
        matplotlib.figure.Figure: Source and smoothed target panels.

    Raises:
        TypeError: If interleaving is not an Interleave object.
        ValueError: If options are invalid or the edge has no selected image.
    """
    assignment = _validated_assignment(interleaving, direction, key)
    source_family = assignment.F if direction == "phi" else assignment.G
    selected_edge = _resolve_edge(source_family(key), edge)
    return _draw_selected(
        assignment, direction, key, {selected_edge}, "Selected edge",
        color, figsize, show_vert_maps, level_set_colors, sharey,
    )


def color_by_arc(interleaving, arc, *, direction="phi", key="0",
                 color="crimson", figsize=(16, 9), show_vert_maps=False,
                 level_set_colors=False, sharey=True):
    """Highlight a source arc and all its selected target images.

    Arc coloring groups individual edge assignments; it does not compute
    an arc-level map or a path in the unsmoothed target graph.

    Parameters:
        interleaving (Interleave): Object after a successful fit() call.
        arc (int or tuple): Zero-based arc number (0 for A0), or a
            directed edge in the arc. Include its key for parallel edges.
        direction (str): 'phi' selects an arc of F; 'psi' selects an arc
            of G. Default is 'phi'.
        key (str): '0' for the main map or 'n' for an auxiliary map.
            Default is '0'.
        color (str): Highlight color. Default is 'crimson'.
        figsize (tuple): Figure size in inches. Default is (16, 9).
        show_vert_maps (bool): Show target-node correspondences.
            Default is False.
        level_set_colors (bool): Color vertices by height. Default is False.
        sharey (bool): Align source and target y-axis ranges. Default is True.

    Returns:
        matplotlib.figure.Figure: Source and smoothed target panels.

    Raises:
        TypeError: If interleaving is not an Interleave object.
        ValueError: If options are invalid or an arc edge has no image.
    """
    assignment = _validated_assignment(interleaving, direction, key)
    source_family = assignment.F if direction == "phi" else assignment.G
    source = source_family(key)
    arcs = _arcs(source)
    if isinstance(arc, int) and not isinstance(arc, bool):
        arc_index = arc
        if not 0 <= arc_index < len(arcs):
            raise ValueError(f"Arc number {arc_index} is out of range.")
    else:
        edge = _resolve_edge(source, arc)
        arc_index = next(i for i, path in enumerate(arcs) if edge in path)
    return _draw_selected(
        assignment, direction, key, set(arcs[arc_index]),
        f"Selected arc A{arc_index}", color, figsize, show_vert_maps,
        level_set_colors, sharey,
        color_by="arc",
    )
