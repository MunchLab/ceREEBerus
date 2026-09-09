import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx


def _format_node_label(node):
    """Format node labels for cleaner plotting output.

    Internal mapper subdivision nodes are stored as tuples like
    ("mapper_subd", k). Render these compactly as "s{k}".
    """
    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], int):
        if node[0] == "mapper_subd":
            return f"s{node[1]}"
        if node[0] == "reeb_auto":
            return f"r{node[1]}"
    return node


def dict_to_list(d):
    return list(d.values())


def line_loop_index(R):
    """Determine if edges between two nodes should be lines or loops

    Args:
        R (Reeb Graph): Reeb Graph

    Returns:
        2-element tuple containing

        - **line_index (list)** : list of indices for edges to be drawn as lines
        - **loop_index (list)** : list of indices for edges to be drawn as loops
    """
    edge_list = list(R.edges(keys=True))
    n = len(R.edges)
    loop_index = []
    line_index = []
    for i in range(0, n):
        if edge_list[i][2] == 1:
            loop_index.append(edge_list.index(edge_list[i][0:2] + (0,)))
            loop_index.append(i)
            line_index.remove(edge_list.index(edge_list[i][0:2] + (0,)))
        else:
            line_index.append(i)

    return (line_index, loop_index)


def slope_intercept(pt0, pt1):
    """Compute slope and intercept to be used in the bezier curve function

    Args:
        pt0 (ordered pair): first point
        pt1 (ordered pair): second point

    Returns:
        2-element tuple containing

        - **m (float)** : slope
        - **b (float)** : intercept
    """
    m = (pt0[1] - pt1[1]) / (pt0[0] - pt1[0])
    b = pt0[1] - m * pt0[0]
    return (m, b)


def bezier_curve(pt0, midpt, pt1):
    """Compute bezier curves for plotting two edges between a single set of nodes

    Args:
        pt0 (ordered pair): first point
        midpt (ordered pair): midpoint for bezier curve to pass through
        pt1 (ordered pair): second point

    Returns:
        points (np array): array of points to be used in plotting
    """

    x1, y1, x2, y2 = (pt0[0], pt0[1], midpt[0], midpt[1])
    a1, b1 = slope_intercept(pt0, midpt)
    a2, b2 = slope_intercept(midpt, pt1)
    points = []

    for i in range(0, 100):
        if x1 == x2:
            continue
        else:
            a, b = slope_intercept((x1, y1), (x2, y2))
        x = i * (x2 - x1) / 100 + x1
        y = a * x + b
        points.append((x, y))
        x1 += (midpt[0] - pt0[0]) / 100
        y1 = a1 * x1 + b1
        x2 += (pt1[0] - midpt[0]) / 100
        y2 = a2 * x2 + b2
    return points


# def reeb_plot(
#     R, with_labels=True, with_colorbar=False, cpx=0.1, cpy=0.1, ax=None, **kwargs
# ):
#     """Main plotting function for the Reeb Graph Class

#     Parameters:
#         R (Reeb Graph): object of Reeb Graph class
#         with_labels (bool): parameter to control whether or not to plot labels
#         with_colorbar (bool): parameter to control whether or not to plot colorbar
#         cp (float): parameter to control curvature of loops in the plotting function. For vertical Reeb graph, only mess with cpx.

#     """
#     if ax is None:
#         fig, ax = plt.subplots()

#     viridis = mpl.colormaps["viridis"].resampled(16)

#     n = len(R.nodes)

#     edge_list = list(R.edges)
#     line_index, loop_index = line_loop_index(R)

#     # Some weird plotting to make the colored and labeled nodes work.
#     # Taking the list of function values from the pos_f dicationary since the infinite node should already have a position set.
#     color_map = [R.pos_f[v][1] for v in R.nodes]
#     pathcollection = nx.draw_networkx_nodes(
#         R, R.pos_f, node_color=color_map, ax=ax, **kwargs
#     )
#     if with_labels:
#         label_map = {node: _format_node_label(node) for node in R.nodes}
#         nx.draw_networkx_labels(
#             R, pos=R.pos_f, labels=label_map, font_color="black", ax=ax
#         )
#     if with_colorbar:
#         plt.colorbar(pathcollection)

#     for i in line_index:
#         node0 = edge_list[i][0]
#         node1 = edge_list[i][1]
#         x_pos = (R.pos_f[node0][0], R.pos_f[node1][0])
#         y_pos = (R.pos_f[node0][1], R.pos_f[node1][1])
#         ax.plot(x_pos, y_pos, color="grey", zorder=0)

#     for i in loop_index:
#         node0 = edge_list[i][0]
#         node1 = edge_list[i][1]
#         xmid = (R.pos_f[node0][0] + R.pos_f[node1][0]) / 2
#         xmid0 = xmid - cpx * xmid
#         xmid1 = xmid + cpx * xmid
#         ymid = (R.pos_f[node0][1] + R.pos_f[node1][1]) / 2
#         ymid0 = ymid - cpy * ymid
#         ymid1 = ymid + cpy * ymid
#         curve = bezier_curve(R.pos_f[node0], (xmid0, ymid0), R.pos_f[node1])
#         c = np.array(curve)
#         ax.plot(c[:, 0], c[:, 1], color="grey", zorder=0)
#         curve = bezier_curve(R.pos_f[node0], (xmid1, ymid1), R.pos_f[node1])
#         c = np.array(curve)
#         ax.plot(c[:, 0], c[:, 1], color="grey", zorder=0)

#     ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)


def _draw_edges(R, ax, cpx=0.1, cpy=0.1):
    """Draw the edges of a Reeb-like graph onto ``ax``.

    Simple edges are drawn as straight lines; parallel (multi-)edges are
    drawn as a pair of bezier loops. This is shared between ``reeb_plot``
    and ``pie_plot`` so both draw edges identically.

    Parameters:
        R (Reeb Graph): object of Reeb Graph class
        ax (matplotlib.axes.Axes): axis to draw on
        cpx, cpy (float): curvature parameters for the loop edges
    """
    edge_list = list(R.edges)
    line_index, loop_index = line_loop_index(R)

    for i in line_index:
        node0 = edge_list[i][0]
        node1 = edge_list[i][1]
        x_pos = (R.pos_f[node0][0], R.pos_f[node1][0])
        y_pos = (R.pos_f[node0][1], R.pos_f[node1][1])
        ax.plot(x_pos, y_pos, color="grey", zorder=0)

    for i in loop_index:
        node0 = edge_list[i][0]
        node1 = edge_list[i][1]
        xmid = (R.pos_f[node0][0] + R.pos_f[node1][0]) / 2
        xmid0 = xmid - cpx * xmid
        xmid1 = xmid + cpx * xmid
        ymid = (R.pos_f[node0][1] + R.pos_f[node1][1]) / 2
        ymid0 = ymid - cpy * ymid
        ymid1 = ymid + cpy * ymid
        curve = bezier_curve(R.pos_f[node0], (xmid0, ymid0), R.pos_f[node1])
        c = np.array(curve)
        ax.plot(c[:, 0], c[:, 1], color="grey", zorder=0)
        curve = bezier_curve(R.pos_f[node0], (xmid1, ymid1), R.pos_f[node1])
        c = np.array(curve)
        ax.plot(c[:, 0], c[:, 1], color="grey", zorder=0)


def reeb_plot(
    R, with_labels=True, with_colorbar=False, cpx=0.1, cpy=0.1, ax=None, **kwargs
):
    """Main plotting function for the Reeb Graph Class

    Parameters:
        R (Reeb Graph): object of Reeb Graph class
        with_labels (bool): parameter to control whether or not to plot labels
        with_colorbar (bool): parameter to control whether or not to plot colorbar
        cp (float): parameter to control curvature of loops in the plotting function. For vertical Reeb graph, only mess with cpx.

    """
    if ax is None:
        fig, ax = plt.subplots()

    viridis = mpl.colormaps["viridis"].resampled(16)

    n = len(R.nodes)

    # Some weird plotting to make the colored and labeled nodes work.
    # Taking the list of function values from the pos_f dicationary since the infinite node should already have a position set.
    color_map = [R.pos_f[v][1] for v in R.nodes]
    pathcollection = nx.draw_networkx_nodes(
        R, R.pos_f, node_color=color_map, ax=ax, **kwargs
    )
    if with_labels:
        label_map = {node: _format_node_label(node) for node in R.nodes}
        nx.draw_networkx_labels(
            R, pos=R.pos_f, labels=label_map, font_color="black", ax=ax
        )
    if with_colorbar:
        plt.colorbar(pathcollection)

    _draw_edges(R, ax, cpx=cpx, cpy=cpy)

    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)


def node_label_counts(R, labels, categories=None):
    """Count, per vertex, how many of its assigned data points fall into
    each category of ``labels``.

    Requires ``R`` to have been built with :func:`cereeberus.computeMapper`,
    which records a ``node_points`` attribute on the graph mapping each
    vertex to the list of original point indices assigned to it.

    Parameters:
        R (MapperGraph): a mapper graph built via ``computeMapper``.
        labels (sequence): a category label for every point in the point
            cloud (or distance matrix) that was used to build ``R``. Indexed
            the same way as that point cloud (or, if ``pointcloud`` was
            ``None``, the same way as the rows/columns of the
            ``distance_matrix`` you built the graph with).
        categories (list, optional): the ordered list of category values to
            report counts for. Defaults to ``sorted(set(labels))``.

    Returns:
        dict: node -> {category: count}
    """
    if not hasattr(R, "node_points"):
        raise ValueError(
            "This graph has no per-node point membership to compute label "
            "percentages from. Build it with cereeberus.computeMapper() to "
            "use node_label_counts/pie_plot."
        )

    if categories is None:
        categories = sorted(set(labels))

    counts = {}
    for v in R.nodes:
        pts = R.node_points.get(v, [])
        counts[v] = {c: 0 for c in categories}
        for p in pts:
            lbl = labels[p]
            if lbl in counts[v]:
                counts[v][lbl] += 1
    return counts


def _pie_image(sizes, colors, edgecolor="black", linewidth=0.6, dpi=100, px=120):
    """Render a single pie chart to an RGBA image array, for use as a small
    'picture' glyph placed at a node's position.

    Parameters:
        sizes (list): wedge sizes (counts or percentages) for this node.
        colors (list): a color per entry in ``sizes``, same order.

    Returns:
        np.ndarray: an (px, px, 4) RGBA image array.
    """
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    pie_ax = fig.add_axes([0, 0, 1, 1])
    pie_ax.set_aspect("equal")
    if sum(sizes) == 0:
        # No points landed on this node; draw a neutral placeholder circle
        # rather than letting matplotlib.pie raise on an all-zero input.
        pie_ax.pie(
            [1], colors=["lightgrey"],
            wedgeprops={"edgecolor": edgecolor, "linewidth": linewidth},
        )
    else:
        pie_ax.pie(
            sizes, colors=colors,
            wedgeprops={"edgecolor": edgecolor, "linewidth": linewidth},
        )
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img



def pie_plot(R,labels,categories=None,colors=None,zoom=0.15,size_by_points=False,min_zoom=0.08,max_zoom=0.3,with_edges=True,with_legend=True,with_labels=True,cpx=0.1,cpy=0.1,ax=None):
    """Plot a mapper graph with each vertex drawn as a small pie-chart
    picture showing the percentage breakdown of ``labels`` among the
    original data points assigned to that vertex.

    Requires ``R`` to have been built with :func:`cereeberus.computeMapper`
    (which records per-node point membership); see ``node_label_counts``.

    Parameters:
        R (MapperGraph): a mapper graph built via ``computeMapper``.
        labels (sequence): a category label for every point in the point
            cloud (or distance matrix) used to build ``R``.
        categories (list, optional): ordered category values to plot.
            Defaults to ``sorted(set(labels))``.
        colors (dict, optional): mapping from category value to a
            matplotlib color. Defaults to the "tab10" colormap.
        zoom (float): size of each pie-chart glyph, as a fraction of its
            rendered base size. The right value depends on how tightly
            packed your graph's node positions are: dense graphs need a
            smaller zoom to avoid neighboring pies overlapping (which can
            create odd-looking visual artifacts where circles overlap).
            Start around 0.1-0.2 and adjust to taste.
        size_by_points (bool): if True, scale each pie's size by the number of
            points assigned to that node. This can be useful for emphasizing
            nodes that represent more data, but can also make the graph harder
            to read if the size differences are extreme. If True, ``min_zoom``
            and ``max_zoom`` control the range of zoom values.
        min_zoom, max_zoom (float): when ``size_by_points`` is True, the
            minimum and maximum zoom values to use for the smallest and largest
            nodes, respectively. 
        with_edges (bool): whether to draw the underlying graph edges.
        with_legend (bool): whether to add a legend mapping colors to
            category values.
        with_labels (bool): whether to draw each node's index/name as a
            text label on top of its pie glyph, matching reeb_plot.
        cpx, cpy (float): curvature parameters for multi-edges, as in
            ``reeb_plot``.
        ax (matplotlib.axes.Axes, optional)

    Returns:
        matplotlib.axes.Axes
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    if ax is None:
        fig, ax = plt.subplots()

    if categories is None:
        categories = sorted(set(labels))

    if colors is None:
        cmap = mpl.colormaps["tab10"]
        colors = {c: cmap(i % 10) for i, c in enumerate(categories)}

    counts = node_label_counts(R, labels, categories=categories)

    if size_by_points:
        totals = {v: sum(counts[v].values()) for v in R.nodes}
        t_min, t_max = min(totals.values()), max(totals.values())

        def _node_zoom(v):
            if t_max == t_min:
                return zoom
            else:
                #sqrt scaling to make the area of the pie area scale with the number of points
                frac = (np.sqrt(totals[v]) - np.sqrt(t_min)) / (np.sqrt(t_max) - np.sqrt(t_min))
                return min_zoom + frac * (max_zoom - min_zoom)

    else:
        def _node_zoom(v):
            return zoom
        
    if with_edges:
        _draw_edges(R, ax, cpx=cpx, cpy=cpy)

    for v in R.nodes:
        sizes = [counts[v][c] for c in categories]
        img = _pie_image(sizes, colors=[colors[c] for c in categories])
        imagebox = OffsetImage(img, zoom=_node_zoom(v))
        ab = AnnotationBbox(imagebox, R.pos_f[v], frameon=False, pad=0)
        ax.add_artist(ab)

    if with_labels:
        label_map = {node: _format_node_label(node) for node in R.nodes}
        nx.draw_networkx_labels(
            R, pos=R.pos_f, labels=label_map, font_color="black", ax=ax
        )

    if with_legend:
        handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="", markersize=8,
                markerfacecolor=colors[c], markeredgecolor="black", label=str(c),
            )
            for c in categories
        ]
        ax.legend(handles=handles, title="Category", loc="best")

    ax.relim()
    ax.autoscale_view()
    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    return ax