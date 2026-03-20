"""
Constrained layout for Reeb graphs.

Nodes are placed with y fixed at their function value f(v).  Only the
x-coordinates are free.  The optimizer minimises a spring energy:

    E(x) = sum_{(u,v) in edges}  (x_u - x_v)^2
          + repulsion * sum_{i != j, f(i)==f(j)}  1 / (x_i - x_j)^2

The first term pulls connected nodes towards the same x-position (producing
straight edges when possible).  The second term pushes apart nodes that share
the same height, preventing them from collapsing on top of each other.

The initial x-positions are computed with the Sugiyama barycenter heuristic,
which determines a crossing-minimising left-to-right ordering for each height
level before the optimizer runs.  This ensures the optimizer starts from the
correct side of the landscape and can converge to a straight-line solution.
"""

import numpy as np
from scipy.optimize import minimize


def _barycenter_init(n, f_vals, edges_list, level_nodes):
    """Sugiyama barycenter heuristic for initial x-placement.

    Iteratively assigns each node's x-position to the mean x of its
    neighbours in the adjacent level, then re-ranks nodes within each level.
    After convergence the node ordering at each level minimises edge crossings.

    Parameters:
        n (int): Number of nodes.
        f_vals (np.ndarray): Function value for each node index.
        edges_list (list of (int, int)): Directed edges (lower-f → higher-f).
        level_nodes (dict): Maps each unique f-value to a list of node indices
            at that height.

    Returns:
        np.ndarray: Initial x-coordinates of length *n*.
    """
    unique_levels = sorted(level_nodes.keys())

    predecessors = [[] for _ in range(n)]
    successors = [[] for _ in range(n)]
    for i, j in edges_list:
        successors[i].append(j)
        predecessors[j].append(i)

    # Start: evenly space nodes within each level
    x = np.zeros(n)
    for lv in unique_levels:
        group = level_nodes[lv]
        m = len(group)
        for rank, i in enumerate(group):
            x[i] = float(rank) - (m - 1) / 2.0

    # Alternating upward / downward barycenter passes
    for _ in range(10):
        # Upward pass: each node → mean x of its predecessors
        for lv in unique_levels:
            group = level_nodes[lv]
            bc = {
                i: (
                    float(np.mean([x[p] for p in predecessors[i]]))
                    if predecessors[i]
                    else x[i]
                )
                for i in group
            }
            sorted_group = sorted(group, key=lambda i: bc[i])
            m = len(sorted_group)
            for rank, i in enumerate(sorted_group):
                x[i] = float(rank) - (m - 1) / 2.0

        # Downward pass: each node → mean x of its successors
        for lv in reversed(unique_levels):
            group = level_nodes[lv]
            bc = {
                i: (
                    float(np.mean([x[s] for s in successors[i]]))
                    if successors[i]
                    else x[i]
                )
                for i in group
            }
            sorted_group = sorted(group, key=lambda i: bc[i])
            m = len(sorted_group)
            for rank, i in enumerate(sorted_group):
                x[i] = float(rank) - (m - 1) / 2.0

    return x


def reeb_x_layout(G, f, seed=None, repulsion=0.5):
    """Compute x-positions for a Reeb graph with y fixed to function values.

    Parameters:
        G: NetworkX graph whose nodes are keys in *f*.
        f (dict): Mapping node -> function value.
        seed (int or None): Random seed for a small jitter added on top of the
            barycenter initialisation (used to break exact ties).
        repulsion (float): Weight of the same-height repulsion term.  Larger
            values spread out nodes at the same height more aggressively.

    Returns:
        dict: Mapping node -> x-position.
    """
    nodes = list(G.nodes)
    n = len(nodes)
    if n == 0:
        return {}

    idx = {v: i for i, v in enumerate(nodes)}
    f_vals = np.array([f[v] for v in nodes])

    # Orient all edges lower-f → higher-f
    edges = []
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        if f_vals[i] <= f_vals[j]:
            edges.append((i, j))
        else:
            edges.append((j, i))

    # Group nodes by height level
    unique_f = sorted(set(f_vals))
    level_nodes = {lv: [i for i in range(n) if f_vals[i] == lv] for lv in unique_f}

    # Pre-compute same-height pairs for the repulsion term
    same_height_pairs = []
    for group in level_nodes.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                same_height_pairs.append((group[a], group[b]))

    # Initialise with barycenter ordering, then add tiny jitter to break ties
    x0 = _barycenter_init(n, f_vals, edges, level_nodes)
    rng = np.random.default_rng(seed)
    x0 = x0 + rng.standard_normal(n) * 1e-3

    def energy(x):
        e = 0.0
        for i, j in edges:
            e += (x[i] - x[j]) ** 2
        if repulsion > 0:
            for i, j in same_height_pairs:
                diff = x[i] - x[j]
                e += repulsion / (diff**2 + 1e-6)
        return e

    def gradient(x):
        g = np.zeros(n)
        for i, j in edges:
            d = x[i] - x[j]
            g[i] += 2 * d
            g[j] -= 2 * d
        if repulsion > 0:
            for i, j in same_height_pairs:
                diff = x[i] - x[j]
                denom = (diff**2 + 1e-6) ** 2
                grad_val = -2 * repulsion * diff / denom
                g[i] += grad_val
                g[j] -= grad_val
        return g

    result = minimize(energy, x0, jac=gradient, method="L-BFGS-B")
    x_opt = result.x

    # Normalise to [-1, 1]
    x_range = x_opt.max() - x_opt.min()
    if x_range > 1e-9:
        x_opt = 2 * (x_opt - x_opt.min()) / x_range - 1

    return {v: float(x_opt[idx[v]]) for v in nodes}
