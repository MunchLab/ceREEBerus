"""Compute Reeb graphs of lower-star filtered simplicial complexes."""

import numpy as np

from ..reeb.lowerstar import LowerStar


class _DenseUnionFind:
    """Small array-backed union-find used by the exact slice sweep."""

    __slots__ = ("parent", "rank")

    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = bytearray(size)

    def find(self, item):
        parent = self.parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            item, parent[item] = parent[item], root
        return root

    def union(self, left, right):
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        rank = self.rank
        if rank[left] < rank[right]:
            left, right = right, left
        self.parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1


def computeReeb(
    K: LowerStar,
    verbose=False,
    *,
    set_pos=True,
    validate=True,
    keep_regular_nodes=False,
):
    """Compute the Reeb graph of a lower-star filtered simplicial complex.

    The function on the complex is the piecewise-linear extension of the
    values assigned to its vertices.  The complex can have any dimension and
    need not be a manifold: boundaries, edges shared by three or more
    triangles, and singular vertices are all allowed.

    By default the returned graph contains only the nodes where the topology
    of the level sets changes (minima, maxima, merges and splits), plus
    isolated components.  Pass ``keep_regular_nodes=True`` to also get a node
    for every level-set component at every vertex height, with a midpoint
    node on each edge between them.

    Parameters
    ----------
    K : LowerStar
        The filtered complex.  Only the values on the vertices are used; the
        filtration of every other simplex must be the maximum of its vertex
        values (this is what :class:`LowerStar` maintains).
    verbose : bool, default=False
        Print progress every 1000 vertex heights.
    set_pos : bool, default=True
        Compute node positions once at the end, so the result can be drawn
        with :meth:`ReebGraph.draw` immediately.  Set to False to skip the
        layout when only the graph itself is needed, for example in batch
        jobs on large complexes.
    validate : bool, default=True
        Check that every simplex's filtration equals the maximum of its
        vertex values.  Vertex values are always checked to be finite.  Set
        to False to skip this check on large complexes that are known to be
        valid.
    keep_regular_nodes : bool, default=False
        If True, keep one node for every level-set component at every vertex
        height, and add a midpoint node named ``"e_0"``, ``"e_1"``, ... on
        each edge between consecutive heights.  This is the output format of
        cereeberus 0.1.20 and earlier.  If False, those nodes (one edge in,
        one edge out) are left out, which gives the same Reeb graph up to
        subdivision with far fewer nodes.

    Returns
    -------
    ReebGraph
        The Reeb graph.  Edges point from the lower to the higher function
        value, and parallel edges (loops) are kept.  By default, nodes are
        named ``0, 1, 2, ...`` and their function values are vertex values.

    Raises
    ------
    TypeError
        If ``K`` is not a :class:`LowerStar`.
    ValueError
        If a vertex value is not finite, the lower-star condition fails
        (with ``validate=True``), the complex is malformed, or, with
        ``keep_regular_nodes=True``, no floating-point number lies strictly
        between two consecutive vertex values.

    Notes
    -----
    The algorithm sweeps upward through the distinct vertex values.  At each
    value it computes the connected components of the level set with a
    union-find over the vertices at that height and the edges crossing it;
    it does the same for one level strictly between consecutive values.
    Each component between two values becomes an edge, attached to the
    unique components it meets at the values above and below.

    Each triangle meets a level set in a convex (hence connected) set, so
    joining the points where a level crosses a triangle's boundary gives
    exactly the level-set components.  Only vertices, edges and triangles
    are needed, even for higher-dimensional input: a level set of any
    simplex is a convex polytope whose edges lie in the sliced triangles.

    Equal vertex values are treated as exact ties (flat regions); no
    tolerance is applied, so round values yourself first if you want nearby
    values treated as equal.  The result is the Reeb graph of the abstract
    complex given; for meshes, duplicated vertices (for example along
    texture seams) should be merged before building ``K``.

    Examples
    --------
    >>> from cereeberus import LowerStar, computeReeb
    >>> K = LowerStar()
    >>> K.insert([0, 1, 2])
    True
    >>> K.insert([1, 3])
    True
    >>> K.insert([2, 3])
    True
    >>> K.assign_filtration([0], 0.0)
    >>> K.assign_filtration([1], 3.0)
    >>> K.assign_filtration([2], 5.0)
    >>> K.assign_filtration([3], 7)
    >>> R = computeReeb(K)
    >>> R.number_of_nodes(), R.number_of_edges()
    (3, 3)
    >>> R.draw()

    To get the subdivided output (the format of cereeberus 0.1.20):

    >>> R_full = computeReeb(K, keep_regular_nodes=True)
    >>> R_full.number_of_nodes(), R_full.number_of_edges()
    (10, 10)
    """
    from ..reeb.reebgraph import ReebGraph

    if not isinstance(K, LowerStar):
        raise TypeError("K must be a LowerStar")

    # A scalar PL Reeb graph is determined by the input's 2-skeleton.
    # Do not materialize higher-dimensional simplices just to discard them.
    simplices = list(K.get_skeleton(2))
    if not simplices:
        return ReebGraph()
    input_dimension = K.dimension()

    vertex_labels = sorted(simplex[0] for simplex, _ in simplices if len(simplex) == 1)
    label_to_vertex = {label: index for index, label in enumerate(vertex_labels)}
    vertex_values = [float(K.filtration([label])) for label in vertex_labels]
    if not all(np.isfinite(value) for value in vertex_values):
        raise ValueError("All vertex function values must be finite")

    distinct_values = sorted(set(vertex_values))
    value_to_level = {value: level for level, value in enumerate(distinct_values)}
    vertex_level = [value_to_level[value] for value in vertex_values]
    level_count = len(distinct_values)
    vertex_count = len(vertex_labels)

    if validate:
        # Higher cells do not affect the sweep, but their filtration must still
        # satisfy the advertised input invariant. Stream this validation pass.
        for simplex, filtration in K.get_simplices():
            expected = max(vertex_values[label_to_vertex[label]] for label in simplex)
            if not np.isfinite(filtration) or float(filtration) != expected:
                raise ValueError(
                    f"Simplex {simplex} violates the lower-star invariant: "
                    f"filtration={filtration!r}, expected={expected!r}"
                )

    edges = []
    triangles_by_label = []
    for simplex, _ in simplices:
        if len(simplex) == 2:
            edges.append(tuple(label_to_vertex[label] for label in simplex))
        elif len(simplex) == 3:
            triangles_by_label.append(
                tuple(label_to_vertex[label] for label in simplex)
            )

    edge_id = {tuple(sorted(edge)): index for index, edge in enumerate(edges)}
    if len(edge_id) != len(edges):
        raise ValueError("The simplex tree contains duplicate edges")

    edge_low_level = []
    edge_high_level = []
    edge_low_vertex = []
    edge_high_vertex = []
    edge_start = [[] for _ in range(level_count)]
    edge_end = [[] for _ in range(level_count)]
    horizontal_edges = [[] for _ in range(level_count)]
    for eid, (left, right) in enumerate(edges):
        left_level, right_level = vertex_level[left], vertex_level[right]
        if left_level <= right_level:
            low_vertex, high_vertex = left, right
            low_level, high_level = left_level, right_level
        else:
            low_vertex, high_vertex = right, left
            low_level, high_level = right_level, left_level
        edge_low_vertex.append(low_vertex)
        edge_high_vertex.append(high_vertex)
        edge_low_level.append(low_level)
        edge_high_level.append(high_level)
        if low_level == high_level:
            horizontal_edges[low_level].append(eid)
        else:
            edge_start[low_level].append(eid)
            edge_end[high_level].append(eid)

    triangles = []
    face_start = [[] for _ in range(level_count)]
    face_end = [[] for _ in range(level_count)]
    for vertices in triangles_by_label:
        try:
            boundary = (
                edge_id[tuple(sorted((vertices[0], vertices[1])))],
                edge_id[tuple(sorted((vertices[0], vertices[2])))],
                edge_id[tuple(sorted((vertices[1], vertices[2])))],
            )
        except KeyError as exc:
            raise ValueError(
                "The input does not contain the faces of every triangle"
            ) from exc
        fid = len(triangles)
        triangles.append((vertices, boundary))
        levels = [vertex_level[vertex] for vertex in vertices]
        face_start[min(levels)].append(fid)
        face_end[max(levels)].append(fid)

    vertices_at_level = [[] for _ in range(level_count)]
    for vertex, level in enumerate(vertex_level):
        vertices_at_level[level].append(vertex)

    def component_map(tokens, unions):
        """Return token->component and deterministic component token lists."""
        token_index = {token: index for index, token in enumerate(tokens)}
        uf = _DenseUnionFind(len(tokens))
        for group in unions:
            if len(group) < 2:
                continue
            first = token_index[group[0]]
            for token in group[1:]:
                uf.union(first, token_index[token])
        root_groups = {}
        for index, token in enumerate(tokens):
            root_groups.setdefault(uf.find(index), []).append(token)
        components = sorted(root_groups.values(), key=lambda group: min(group))
        token_component = {
            token: component
            for component, group in enumerate(components)
            for token in group
        }
        return token_component, components

    def exact_components(level, active_edge_ids, active_face_ids):
        # Vertex tokens are [0,n); a strict edge-crossing token is n + eid.
        tokens = list(vertices_at_level[level])
        tokens.extend(vertex_count + eid for eid in sorted(active_edge_ids))
        unions = []
        for eid in horizontal_edges[level]:
            unions.append(edges[eid])
        for fid in sorted(active_face_ids):
            _, boundary = triangles[fid]
            intersection = []
            for eid in boundary:
                low_level = edge_low_level[eid]
                high_level = edge_high_level[eid]
                if low_level == level == high_level:
                    intersection.extend(edges[eid])
                elif low_level == level:
                    intersection.append(edge_low_vertex[eid])
                elif high_level == level:
                    intersection.append(edge_high_vertex[eid])
                elif low_level < level < high_level:
                    intersection.append(vertex_count + eid)
            if intersection:
                # A repeated vertex token is harmless but wastes union calls.
                unions.append(list(dict.fromkeys(intersection)))
        return component_map(tokens, unions)

    def regular_components(level, active_edge_ids, active_face_ids):
        # Token eid denotes the unique intersection of edge eid with any level
        # strictly between distinct_values[level] and [level + 1].
        tokens = sorted(active_edge_ids)
        unions = []
        for fid in sorted(active_face_ids):
            _, boundary = triangles[fid]
            crossings = [
                eid
                for eid in boundary
                if edge_low_level[eid] <= level < edge_high_level[eid]
            ]
            if crossings:
                unions.append(crossings)
        _, components = component_map(tokens, unions)
        return components

    def unique_component(edge_ids, level, token_component, upper):
        components = set()
        for eid in edge_ids:
            endpoint_level = edge_high_level[eid] if upper else edge_low_level[eid]
            if endpoint_level == level:
                token = edge_high_vertex[eid] if upper else edge_low_vertex[eid]
            else:
                token = vertex_count + eid
            components.add(token_component[token])
        if len(components) != 1:
            direction = "upper" if upper else "lower"
            raise RuntimeError(
                f"A regular fiber component has {len(components)} {direction} "
                "endpoint components; the input may not be a simplicial complex"
            )
        return components.pop()

    R = ReebGraph()
    R.graph.update(
        algorithm="exact_slice_sweep_2skeleton",
        input_dimension=input_dimension,
        sweep_dimension=min(input_dimension, 2),
        keep_regular_nodes=keep_regular_nodes,
        exact_ties=True,
        supports_nonmanifold=True,
        input_vertices=vertex_count,
        input_edges=len(edges),
        input_triangles=len(triangles),
        input_levels=level_count,
    )
    next_node = 0
    next_half_edge = 0
    active_edges = set()
    active_faces = set()
    # Each pending regular band is (source Reeb node, its crossing edge tokens).
    pending = []

    for level, value in enumerate(distinct_values):
        active_faces.update(face_start[level])
        active_edges.difference_update(edge_end[level])
        token_component, exact_groups = exact_components(
            level, active_edges, active_faces
        )

        incoming = [[] for _ in exact_groups]
        for source, band in pending:
            component = unique_component(band, level, token_component, upper=True)
            incoming[component].append(source)

        active_edges.update(edge_start[level])
        active_faces.difference_update(face_end[level])
        outgoing_bands = (
            regular_components(level, active_edges, active_faces)
            if level + 1 < level_count
            else []
        )
        outgoing = [[] for _ in exact_groups]
        for band in outgoing_bands:
            component = unique_component(band, level, token_component, upper=False)
            outgoing[component].append(band)

        new_pending = []
        for component in range(len(exact_groups)):
            sources = incoming[component]
            bands = outgoing[component]
            if not keep_regular_nodes and len(sources) == 1 and len(bands) == 1:
                continuation_source = sources[0]
            else:
                node = next_node
                next_node += 1
                R.add_node(node, value, reset_pos=False)
                for source in sources:
                    R.add_edge(source, node, reset_pos=False)
                continuation_source = node
            for band in bands:
                new_pending.append((continuation_source, band))
        if keep_regular_nodes and new_pending:
            upper_value = distinct_values[level + 1]
            # Halving first avoids overflow from value + upper_value.
            midpoint = value / 2 + upper_value / 2
            if not value < midpoint < upper_value:
                midpoint = float(np.nextafter(value, upper_value))
            if not value < midpoint < upper_value:
                raise ValueError(
                    "keep_regular_nodes=True requires a representable floating-point "
                    f"height strictly between {value!r} and {upper_value!r}; "
                    "use keep_regular_nodes=False for reduced output"
                )
            subdivided_pending = []
            for source, band in new_pending:
                sentinel = f"e_{next_half_edge}"
                next_half_edge += 1
                R.add_node(sentinel, midpoint, reset_pos=False)
                R.add_edge(source, sentinel, reset_pos=False)
                subdivided_pending.append((sentinel, band))
            new_pending = subdivided_pending
        pending = new_pending

        if verbose and (
            level == 0 or (level + 1) % 1000 == 0 or level + 1 == level_count
        ):
            print(
                f"processed {level + 1}/{level_count} levels; "
                f"Reeb graph has {R.number_of_nodes()} nodes and "
                f"{R.number_of_edges()} edges"
            )

    if pending:
        raise RuntimeError(
            "Internal error: regular components remain above the maximum"
        )
    if set_pos:
        R.set_pos_from_f()
    return R
