"""
Gromov-Wasserstein distance between Reeb graphs.

This module implements the comparison of Reeb graphs from Chambers and Meng,
*Reeb Graph Comparison via the Gromov-Wasserstein Distance*. Each Reeb graph
:math:`R_f` is turned into a metric-measure space
:math:`\\Phi(R_f) = (V_{R_f}, d_{R_f}, \\nu_{R_f})` where

- :math:`d_{R_f}` is the Symmetric Reeb Radius, the average of the two
  directed Reeb radii of Curry et al. (see :func:`symmetric_reeb_radius`), and
- :math:`\\nu_{R_f}` is a probability measure on the vertices built from the
  Persistence Image of the extended persistence diagram of :math:`R_f`
  (see :func:`pi_measure`).

Two Reeb graphs are then compared with the Gromov-Wasserstein distance between
these metric-measure spaces, computed with the
`POT <https://pythonot.github.io/>`_ library
(see :class:`GromovWasserstein`).
"""

import heapq
import itertools

import gudhi
import numpy as np
import pandas as pd

__all__ = [
    "reeb_radius",
    "reeb_radius_matrix",
    "symmetric_reeb_radius",
    "extended_persistence_pairs",
    "persistence_image",
    "pi_measure",
    "GromovWasserstein",
    "gromov_wasserstein_distance",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _function_values(R):
    """
    Get the function values of the vertices of a Reeb graph.

    For a ``MapperGraph`` the stored integer values are scaled by ``delta`` so
    that the distances are in the units of the original function.

    Parameters:
        R (ReebGraph): The Reeb graph.

    Returns:
        dict: A dictionary mapping each vertex to its function value.
    """
    # Import here to avoid a circular import.
    from ..reeb.mapper import MapperGraph

    if isinstance(R, MapperGraph):
        return {v: R.delta * R.f[v] for v in R.nodes}
    return {v: R.f[v] for v in R.nodes}


def _neighbors(R, v):
    """
    Get the neighbors of a vertex, ignoring the direction of the edges.

    Parameters:
        R (ReebGraph): The Reeb graph.
        v: A vertex of ``R``.

    Returns:
        set: The vertices that share an edge with ``v``.
    """
    if R.is_directed():
        return set(R.successors(v)) | set(R.predecessors(v))
    return set(R.neighbors(v))


def _reeb_radius_from_root(R, f, root):
    """
    Modified Dijkstra algorithm (Algorithm 1 of Curry et al.) for the Reeb
    radius from a fixed root vertex.

    Parameters:
        R (ReebGraph): The Reeb graph.
        f (dict): The function values of the vertices.
        root: The root vertex.

    Returns:
        dict: A dictionary mapping each vertex ``v`` to the Reeb radius from
        ``root`` to ``v``.
    """
    radius = {v: np.inf for v in R.nodes}
    radius[root] = 0.0
    f_root = f[root]

    # The counter breaks ties so that vertex names are never compared.
    counter = itertools.count()
    queue = [(0.0, next(counter), root)]
    while queue:
        r, _, v = heapq.heappop(queue)
        if r > radius[v]:
            continue
        for w in _neighbors(R, v):
            # f is monotone along an edge of a Reeb graph, so the largest
            # deviation on the edge v-w is attained at one of its endpoints.
            r_w = max(r, abs(f_root - f[w]))
            if r_w < radius[w]:
                radius[w] = r_w
                heapq.heappush(queue, (r_w, next(counter), w))
    return radius


def _gaussian(centers, mean, sigma):
    """
    Evaluate the 2D isotropic Gaussian kernel used in the Persistence Image.

    Parameters:
        centers (numpy.ndarray): An ``(N, 2)`` array of points.
        mean (array-like): The center of the kernel.
        sigma (float): The bandwidth of the kernel.

    Returns:
        numpy.ndarray: The kernel evaluated at each of the ``N`` points.
    """
    sq_dist = np.sum((centers - np.asarray(mean, dtype=float)) ** 2, axis=1)
    return np.exp(-sq_dist / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def _birth_persistence(pairs):
    """
    Convert extended persistence pairs to (birth, persistence) coordinates.

    Parameters:
        pairs (list): The output of :func:`extended_persistence_pairs`.

    Returns:
        numpy.ndarray: An ``(n_pairs, 2)`` array.
    """
    return np.array([(b, abs(d - b)) for _, _, b, d, _ in pairs], dtype=float).reshape(
        -1, 2
    )


def _default_sigma(point_sets):
    """
    The default bandwidth: 10% of the largest persistence value, which is the
    starting configuration recommended in the paper.

    Parameters:
        point_sets (list): A list of ``(n, 2)`` arrays of (birth, persistence)
            points.

    Returns:
        float: The bandwidth.
    """
    points = np.vstack([P for P in point_sets if len(P)] or [np.zeros((0, 2))])
    if len(points) == 0 or np.max(points[:, 1]) <= 0:
        return 1.0
    return 0.1 * float(np.max(points[:, 1]))


def _default_grid(point_sets, sigma):
    """
    A pixel grid that covers all the given points with a margin of
    :math:`3\\sigma` plus 5% of the range in each direction.

    Parameters:
        point_sets (list): A list of ``(n, 2)`` arrays of (birth, persistence)
            points.
        sigma (float): The bandwidth of the Gaussian kernel.

    Returns:
        tuple: ``(birth_range, pers_range)``, each a ``(min, max)`` pair.
    """
    points = np.vstack([P for P in point_sets if len(P)] or [np.zeros((1, 2))])
    ranges = []
    for col in (0, 1):
        lo, hi = float(np.min(points[:, col])), float(np.max(points[:, col]))
        margin = 3 * sigma + 0.05 * (hi - lo)
        ranges.append((lo - margin, hi + margin))
    # Persistence is non-negative.
    ranges[1] = (max(0.0, ranges[1][0]), ranges[1][1])
    return ranges[0], ranges[1]


# ---------------------------------------------------------------------------
# Metric: (Symmetric) Reeb radius
# ---------------------------------------------------------------------------


def reeb_radius(R, root):
    """
    Compute the Reeb radius from a root vertex :math:`x` to every vertex
    :math:`v` of the Reeb graph,

    .. math::

        \\rho_f(x, v) = \\inf_{\\gamma: x \\to v} \\sup_{t \\in [0,1]}
        |f(x) - f(\\gamma(t))|,

    that is, the smallest possible largest deviation from :math:`f(x)` along a
    path from :math:`x` to :math:`v`. This uses the modified Dijkstra algorithm
    of Curry et al. The Reeb radius is not symmetric; see
    :func:`symmetric_reeb_radius`.

    Parameters:
        R (ReebGraph): The Reeb graph. A ``MapperGraph`` is also accepted, in
            which case the function values are scaled by ``delta``.
        root: A vertex of ``R``.

    Returns:
        dict: A dictionary mapping each vertex ``v`` to
        :math:`\\rho_f(\\text{root}, v)`. Vertices in a different connected
        component get ``numpy.inf``.
    """
    if root not in R.nodes:
        raise ValueError(f"The vertex {root} is not in the Reeb graph.")
    return _reeb_radius_from_root(R, _function_values(R), root)


def reeb_radius_matrix(R):
    """
    Compute the (asymmetric) Reeb radius between all pairs of vertices.

    Parameters:
        R (ReebGraph): The Reeb graph.

    Returns:
        pandas.DataFrame: A square matrix indexed by the vertices of ``R``
        whose entry in row :math:`x` and column :math:`v` is
        :math:`\\rho_f(x, v)`.
    """
    f = _function_values(R)
    vertices = list(R.nodes)
    M = np.zeros((len(vertices), len(vertices)))
    for i, x in enumerate(vertices):
        radius = _reeb_radius_from_root(R, f, x)
        M[i] = [radius[v] for v in vertices]
    return pd.DataFrame(M, index=vertices, columns=vertices)


def symmetric_reeb_radius(R, method="mean"):
    """
    Compute the Symmetric Reeb Radius between all pairs of vertices,

    .. math::

        d_{R_f}(u, v) = \\frac{1}{2}\\left(\\rho_f(u, v) + \\rho_f(v, u)\\right).

    For generic Reeb graphs (distinct function values on distinct vertices)
    this is a metric on the vertices.

    Parameters:
        R (ReebGraph): The Reeb graph.
        method (str): Optional. ``'mean'`` (default) averages the two directed
            Reeb radii as above. ``'max'`` uses
            :math:`\\max(\\rho_f(u, v), \\rho_f(v, u))` instead, as in Curry et
            al.

    Returns:
        pandas.DataFrame: A symmetric matrix indexed by the vertices of ``R``.
    """
    rho = reeb_radius_matrix(R)
    M = rho.to_numpy()
    if method == "mean":
        D = 0.5 * (M + M.T)
    elif method == "max":
        D = np.maximum(M, M.T)
    else:
        raise ValueError(f"method must be 'mean' or 'max', not {method!r}.")
    return pd.DataFrame(D, index=rho.index, columns=rho.columns)


# ---------------------------------------------------------------------------
# Measure: Persistence Image based probability
# ---------------------------------------------------------------------------


def extended_persistence_pairs(R):
    """
    Compute the extended persistence diagram of the Reeb graph together with
    the vertices that realize each pair.

    The diagram is computed with ``gudhi`` from the lower-star filtration of
    the graph. Each pair is matched to its vertices through the simplices
    that create and destroy it, so the matching is exact even when several
    vertices have the same function value (as in Mapper graphs). Parallel
    edges are subdivided internally so that the loops they form are kept.
    Pairs with zero persistence are dropped.

    Parameters:
        R (ReebGraph): The Reeb graph.

    Returns:
        list: A list of tuples
        ``(birth_vertex, death_vertex, birth, death, type)``, where ``type``
        is one of ``'Ord0'``, ``'Rel1'``, ``'Ext0'`` and ``'Ext1'``.
    """
    f = _function_values(R)
    vertices = list(R.nodes)
    if len(vertices) == 0:
        return []

    # owner[i] is the vertex of R for simplex vertex i, or None for a vertex
    # added to subdivide a parallel edge.
    owner = list(vertices)
    values = [float(f[v]) for v in vertices]
    index = {v: i for i, v in enumerate(vertices)}

    st = gudhi.SimplexTree()
    for i, val in enumerate(values):
        st.insert([i], filtration=val)

    seen = set()
    for u, v in R.edges():
        if u == v:
            continue
        i, j = index[u], index[v]
        if frozenset((i, j)) not in seen:
            seen.add(frozenset((i, j)))
            st.insert([i, j], filtration=max(values[i], values[j]))
        else:
            m = len(values)
            owner.append(None)
            values.append(0.5 * (values[i] + values[j]))
            st.insert([m], filtration=values[m])
            st.insert([i, m], filtration=max(values[i], values[m]))
            st.insert([m, j], filtration=max(values[m], values[j]))

    if len(values) < 2 or min(values) == max(values):
        return []

    st.extend_filtration()
    st.extended_persistence()
    # extend_filtration adds a cone vertex with a label above all others.
    cone = len(values)

    def critical_vertex(simplex):
        # Simplices without the cone vertex are in the sublevel set part, so
        # they enter at their highest vertex. Coned simplices are in the
        # superlevel set part, so they enter at their lowest vertex. Ties are
        # broken in favor of vertices of R.
        rest = [i for i in simplex if i != cone]
        if not rest:
            return None, True
        if len(rest) < len(simplex):
            i = min(rest, key=lambda k: (values[k], owner[k] is None))
            return i, True
        i = max(rest, key=lambda k: (values[k], owner[k] is not None))
        return i, False

    pairs = []
    for birth_simplex, death_simplex in st.persistence_pairs():
        if len(death_simplex) == 0:
            continue
        bi, birth_relative = critical_vertex(birth_simplex)
        di, death_relative = critical_vertex(death_simplex)
        if bi is None or di is None:
            continue
        birth, death = values[bi], values[di]
        if birth == death:
            continue
        if not birth_relative and not death_relative:
            pair_type = "Ord0"
        elif not birth_relative:
            pair_type = "Ext0" if len(birth_simplex) == 1 else "Ext1"
        else:
            pair_type = "Rel1"
        pairs.append((owner[bi], owner[di], birth, death, pair_type))
    return pairs


def persistence_image(
    points,
    sigma,
    weight_power=1.0,
    birth_range=None,
    pers_range=None,
    resolution=10,
):
    """
    Compute the Persistence Image of a set of (birth, persistence) points,

    .. math::

        I[k] = \\sum_{(b, p)} p^{w_p} \\, \\phi_{(b, p)}(c_k),

    where :math:`\\phi_{(b,p)}` is the Gaussian kernel with bandwidth
    :math:`\\sigma` centered at :math:`(b, p)` and :math:`c_k` are the pixel
    centers.

    Parameters:
        points (array-like): An ``(n, 2)`` array of (birth, persistence)
            points.
        sigma (float): The bandwidth :math:`\\sigma > 0` of the Gaussian
            kernel.
        weight_power (float): Optional. The power :math:`w_p \\geq 0` in the
            weighting function :math:`w(b, p) = p^{w_p}`. Default is 1.
        birth_range (tuple): Optional. ``(min, max)`` of the pixel centers in
            the birth direction. Computed from ``points`` if not given.
        pers_range (tuple): Optional. ``(min, max)`` of the pixel centers in
            the persistence direction. Computed from ``points`` if not given.
        resolution (int or tuple): Optional. Number of pixels per axis, or a
            pair ``(n_birth, n_pers)``. Default is 10.

    Returns:
        tuple: ``(image, centers)`` where ``image`` is an array of shape
        ``(n_pers, n_birth)`` and ``centers`` is an ``(n_pers * n_birth, 2)``
        array of the pixel centers, in the same order as ``image.ravel()``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if np.isscalar(resolution):
        n_birth = n_pers = int(resolution)
    else:
        n_birth, n_pers = (int(r) for r in resolution)
    if birth_range is None or pers_range is None:
        default_birth, default_pers = _default_grid([points], sigma)
        birth_range = default_birth if birth_range is None else birth_range
        pers_range = default_pers if pers_range is None else pers_range

    X, Y = np.meshgrid(
        np.linspace(birth_range[0], birth_range[1], n_birth),
        np.linspace(pers_range[0], pers_range[1], n_pers),
    )
    centers = np.column_stack([X.ravel(), Y.ravel()])

    image = np.zeros(len(centers))
    for b, p in points:
        image += p**weight_power * _gaussian(centers, (b, p), sigma)
    return image.reshape(n_pers, n_birth), centers


def _measure_from_pairs(R, pairs, image, centers, sigma):
    """
    The PI-based probability measure given precomputed pairs and image.

    Parameters:
        R (ReebGraph): The Reeb graph.
        pairs (list): The output of :func:`extended_persistence_pairs`.
        image (numpy.ndarray): The Persistence Image of ``pairs``.
        centers (numpy.ndarray): The pixel centers of ``image``.
        sigma (float): The bandwidth of the Gaussian kernel.

    Returns:
        dict: A dictionary mapping each vertex of ``R`` to its probability.
    """
    vertices = list(R.nodes)
    if len(vertices) == 0:
        return {}
    contribution = {v: 0.0 for v in vertices}
    flat = image.ravel()
    for (bv, dv, b, d, _), point in zip(pairs, _birth_persistence(pairs)):
        c = float(flat @ _gaussian(centers, point, sigma))
        # Both endpoints of a pair get the same contribution.
        contribution[bv] += c
        contribution[dv] += c

    positive = [c for c in contribution.values() if c > 0]
    if not positive:
        return {v: 1.0 / len(vertices) for v in vertices}
    # Vertices that are not the endpoint of any pair (for example the
    # intermediate vertices of a Mapper graph) get the smallest nonzero
    # contribution, so that the measure has full support.
    floor = min(positive)
    for v in vertices:
        if contribution[v] <= 0:
            contribution[v] = floor
    total = sum(contribution.values())
    return {v: contribution[v] / total for v in vertices}


def pi_measure(
    R,
    sigma=None,
    weight_power=1.0,
    birth_range=None,
    pers_range=None,
    resolution=10,
):
    """
    Compute the Persistence Image based probability measure on the vertices
    of a Reeb graph.

    Each vertex :math:`v` is an endpoint of an extended persistence pair with
    birth-persistence coordinate :math:`(b_v, p_v)`. Its contribution is the
    inner product of the Persistence Image :math:`I_f` with the kernel
    centered at that point,

    .. math::

        \\mathrm{contrib}(v) = \\sum_k I_f[k] \\, \\phi_{(b_v, p_v)}(c_k),
        \\qquad
        \\nu_{R_f}(v) = \\frac{\\mathrm{contrib}(v)}{\\sum_u \\mathrm{contrib}(u)}.

    Both endpoints of a pair get the same mass. A vertex that is an endpoint
    of several pairs gets the sum of their contributions. Vertices that are
    not the endpoint of any pair, such as the intermediate vertices of a
    Mapper graph, get the smallest nonzero contribution so that the measure
    has full support. If the diagram is empty the measure is uniform.

    When comparing several graphs, pass the same ``birth_range``,
    ``pers_range`` and ``sigma`` for all of them so that they share one pixel
    grid. :class:`GromovWasserstein` does this automatically.

    Parameters:
        R (ReebGraph): The Reeb graph.
        sigma (float): Optional. The bandwidth of the Gaussian kernel. Default
            is 10% of the largest persistence in the diagram.
        weight_power (float): Optional. The power :math:`w_p` in the weighting
            function :math:`w(b, p) = p^{w_p}`. Default is 1.
        birth_range (tuple): Optional. ``(min, max)`` of the pixel grid in the
            birth direction.
        pers_range (tuple): Optional. ``(min, max)`` of the pixel grid in the
            persistence direction.
        resolution (int or tuple): Optional. Number of pixels per axis.
            Default is 10.

    Returns:
        dict: A dictionary mapping each vertex of ``R`` to its probability.
    """
    pairs = extended_persistence_pairs(R)
    points = _birth_persistence(pairs)
    if sigma is None:
        sigma = _default_sigma([points])
    image, centers = persistence_image(
        points, sigma, weight_power, birth_range, pers_range, resolution
    )
    return _measure_from_pairs(R, pairs, image, centers, sigma)


# ---------------------------------------------------------------------------
# Gromov-Wasserstein distance
# ---------------------------------------------------------------------------


class GromovWasserstein:
    """
    Class for computing the Gromov-Wasserstein distance between two Reeb
    graphs, following Chambers and Meng, *Reeb Graph Comparison via the
    Gromov-Wasserstein Distance*.

    Each graph is turned into a metric-measure space using the Symmetric Reeb
    Radius (:func:`symmetric_reeb_radius`) and the Persistence Image based
    measure (:func:`pi_measure`). The two Persistence Images are computed on
    one shared pixel grid. The distance is then computed with the conditional
    gradient solver of the POT library.

    Usage::

        from cereeberus import GromovWasserstein
        myGW = GromovWasserstein(R1, R2, sigma=0.5)
        myGW.fit()

    After calling :meth:`fit`, the following attributes are set:

    - ``dist``: the distance.
    - ``plan``: the optimal transport plan, as a ``pandas.DataFrame`` with
      rows indexed by the vertices of ``F`` and columns by those of ``G``.
    - ``D_F``, ``D_G``: the Symmetric Reeb Radius matrices.
    - ``mu_F``, ``mu_G``: the probability measures, as dictionaries.
    - ``pairs_F``, ``pairs_G``: the extended persistence pairs.
    - ``sigma``, ``birth_range``, ``pers_range``: the Persistence Image
      parameters that were used. The ranges stay ``None`` when
      ``shared_grid=False`` and no ranges were given.
    """

    def __init__(
        self,
        F,
        G,
        sigma=None,
        weight_power=1.0,
        resolution=10,
        birth_range=None,
        pers_range=None,
        symmetrization="mean",
        shared_grid=True,
    ):
        """
        Initialize the GromovWasserstein object.

        Parameters:
            F, G (ReebGraph): The two Reeb graphs to compare. They must be
                connected. ``MapperGraph`` objects are also accepted.
            sigma (float): Optional. The bandwidth of the Gaussian kernel in
                the Persistence Image. Default is 10% of the largest
                persistence in the two diagrams.
            weight_power (float): Optional. The power :math:`w_p` in the
                weighting function :math:`w(b, p) = p^{w_p}`. Default is 1.
            resolution (int or tuple): Optional. Number of pixels per axis of
                the Persistence Image. Default is 10.
            birth_range (tuple): Optional. ``(min, max)`` of the shared pixel
                grid in the birth direction. Default covers both diagrams.
            pers_range (tuple): Optional. ``(min, max)`` of the shared pixel
                grid in the persistence direction. Default covers both
                diagrams.
            symmetrization (str): Optional. ``'mean'`` (default) or ``'max'``.
                See :func:`symmetric_reeb_radius`.
            shared_grid (bool): Optional. If True (default), both Persistence
                Images are computed on one shared pixel grid, as in the
                definition of the measure. If False, and ``birth_range`` and
                ``pers_range`` are not given, each image is computed on a grid
                fitted to its own diagram; this is how the worked example in
                the paper was computed.
        """
        self.F = F
        self.G = G
        self.sigma = sigma
        self.weight_power = weight_power
        self.resolution = resolution
        self.birth_range = birth_range
        self.pers_range = pers_range
        self.symmetrization = symmetrization
        self.shared_grid = shared_grid

        self.dist = None
        self.plan = None

    def fit(self):
        """
        Compute the Gromov-Wasserstein distance between the two Reeb graphs.

        The returned value is the Gromov-Wasserstein objective with the square
        loss at the coupling :math:`\\pi` found by the solver,

        .. math::

            \\sum_{x, x', y, y'} |d_{R_f}(x, x') - d_{R_g}(y, y')|^2
            \\, \\pi(x, y) \\, \\pi(x', y'),

        which is the value reported in the worked example of the paper. In the
        normalization :math:`GW_2 = \\frac{1}{2}(\\cdots)^{1/2}` of Mémoli,
        this value equals :math:`(2 \\, GW_2)^2`. The problem is non-convex, so
        the solver may return a local minimum.

        Returns:
            float: The distance.
        """
        try:
            import ot
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "GromovWasserstein requires the POT package. "
                "Install it with `pip install pot`."
            ) from e

        for name, R in (("F", self.F), ("G", self.G)):
            if len(R.nodes) == 0:
                raise ValueError(f"The Reeb graph {name} has no vertices.")

        self.D_F = symmetric_reeb_radius(self.F, self.symmetrization)
        self.D_G = symmetric_reeb_radius(self.G, self.symmetrization)
        for name, D in (("F", self.D_F), ("G", self.D_G)):
            if not np.all(np.isfinite(D.to_numpy())):
                raise ValueError(
                    f"The Reeb graph {name} is not connected, so its Reeb "
                    "radius is infinite for some pairs of vertices."
                )

        self.pairs_F = extended_persistence_pairs(self.F)
        self.pairs_G = extended_persistence_pairs(self.G)
        points = [
            _birth_persistence(self.pairs_F),
            _birth_persistence(self.pairs_G),
        ]
        if self.sigma is None:
            self.sigma = _default_sigma(points)
        if self.shared_grid and (self.birth_range is None or self.pers_range is None):
            birth_range, pers_range = _default_grid(points, self.sigma)
            if self.birth_range is None:
                self.birth_range = birth_range
            if self.pers_range is None:
                self.pers_range = pers_range

        measures = []
        for R, pairs, P in zip((self.F, self.G), (self.pairs_F, self.pairs_G), points):
            image, centers = persistence_image(
                P,
                self.sigma,
                self.weight_power,
                self.birth_range,
                self.pers_range,
                self.resolution,
            )
            measures.append(_measure_from_pairs(R, pairs, image, centers, self.sigma))
        self.mu_F, self.mu_G = measures

        p = np.array([self.mu_F[v] for v in self.D_F.index], dtype=float)
        q = np.array([self.mu_G[v] for v in self.D_G.index], dtype=float)
        T, log = ot.gromov.gromov_wasserstein(
            self.D_F.to_numpy(dtype=float),
            self.D_G.to_numpy(dtype=float),
            p,
            q,
            loss_fun="square_loss",
            log=True,
        )
        self.dist = float(log["gw_dist"])
        self.plan = pd.DataFrame(T, index=self.D_F.index, columns=self.D_G.index)
        return self.dist


def gromov_wasserstein_distance(F, G, return_plan=False, **kwargs):
    """
    Compute the Gromov-Wasserstein distance between two Reeb graphs. This is a
    shortcut for ``GromovWasserstein(F, G, **kwargs).fit()``.

    Parameters:
        F, G (ReebGraph): The two Reeb graphs to compare.
        return_plan (bool): Optional. If True, also return the optimal
            transport plan.
        **kwargs: Passed to :class:`GromovWasserstein`, for example ``sigma``,
            ``weight_power``, ``resolution`` or ``symmetrization``.

    Returns:
        float: The distance, or a tuple ``(distance, plan)`` if
        ``return_plan`` is True.
    """
    gw = GromovWasserstein(F, G, **kwargs)
    dist = gw.fit()
    if return_plan:
        return dist, gw.plan
    return dist
