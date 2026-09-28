"""Tests for computeReeb (exact slice sweep) on inputs of any dimension.

Two independent references are used:

* ``fiber_oracle`` below, which computes fiber components directly from all
  simplices (no 2-skeleton reduction), and
* ``computeReeb_legacy``, the previous implementation, frozen in
  ``tests/legacy_computereeb.py``.
"""

from fractions import Fraction
from itertools import combinations, product
import random
import unittest
from unittest.mock import patch

from gudhi import SimplexTree
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from cereeberus.compute.computereeb import computeReeb
from cereeberus.reeb.lowerstar import LowerStar
from cereeberus.reeb.reebgraph import ReebGraph
from cereeberus.data import Torus
from tests.legacy_computereeb import computeReeb_legacy


def make_complex(facets, values, labels=None, cls=LowerStar):
    labels = list(range(len(values))) if labels is None else labels
    k = cls()
    for v in labels:
        k.insert([v])
    for simplex in facets:
        k.insert([labels[v] for v in simplex])
    for v, value in zip(labels, values):
        k.assign_filtration(v, float(value))
    return k


def reduced_graph(r):
    g = nx.MultiDiGraph()
    g.add_nodes_from((v, {"f": r.f[v]}) for v in r)
    g.add_edges_from((a, b) for a, b, _ in r.edges(keys=True))
    return suppress(g)


def suppress(g):
    g = g.copy()
    while True:
        vertices = [v for v in g if g.in_degree(v) == 1 and g.out_degree(v) == 1]
        if not vertices:
            return g
        v = vertices[0]
        a, b = next(g.predecessors(v)), next(g.successors(v))
        g.remove_node(v)
        g.add_edge(a, b)


def fiber_oracle(k):
    """Independent full-simplex fiber intersection graph, without reduction to a skeleton."""
    simplices = [frozenset(s) for s, _ in k.get_simplices()]
    values = {
        next(iter(s)): Fraction(k.filtration(list(s))) for s in simplices if len(s) == 1
    }
    levels = sorted(set(values.values()))

    def meets(s, t):
        return bool(s) and min(values[v] for v in s) <= t <= max(values[v] for v in s)

    def components(t):
        active = [s for s in simplices if meets(s, t)]
        h = nx.Graph()
        h.add_nodes_from(active)
        h.add_edges_from((a, b) for a, b in combinations(active, 2) if meets(a & b, t))
        return list(nx.connected_components(h))

    g = nx.MultiDiGraph()
    maps = []
    for i, t in enumerate(levels):
        mapping = {}
        for j, component in enumerate(components(t)):
            node = (i, j)
            g.add_node(node, f=float(t))
            mapping.update((s, node) for s in component)
        maps.append(mapping)
    for i, (a, b) in enumerate(zip(levels, levels[1:])):
        for component in components((a + b) / 2):
            lower = {maps[i][s] for s in component}
            upper = {maps[i + 1][s] for s in component}
            assert len(lower) == len(upper) == 1
            g.add_edge(lower.pop(), upper.pop())
    return suppress(g)


class TestAgainstFiberOracle(unittest.TestCase):
    def assert_oracle(self, k):
        before = list(k.get_simplices())
        r = computeReeb(k)
        self.assertTrue(
            nx.is_isomorphic(
                reduced_graph(r),
                fiber_oracle(k),
                node_match=lambda a, b: a["f"] == b["f"],
            )
        )
        self.assertEqual(before, list(k.get_simplices()))
        self.assertIsInstance(r, ReebGraph)
        self.assertTrue(all(r.f[a] < r.f[b] for a, b in r.edges()))
        return r

    def test_empty_and_zero_dimensional(self):
        self.assertEqual(len(computeReeb(LowerStar())), 0)
        r = self.assert_oracle(make_complex([], [1, 1, 2]))
        self.assertEqual((len(r), r.number_of_edges()), (3, 0))

    def test_simplex_dimensions_one_through_eight(self):
        for dimension in range(1, 9):
            with self.subTest(dimension=dimension):
                k = make_complex([list(range(dimension + 1))], range(dimension + 1))
                r = computeReeb(k)
                self.assertEqual((len(r), r.number_of_edges()), (2, 1))
                self.assertEqual(sorted(r.f.values()), [0, dimension])
                self.assertEqual(r.graph["input_dimension"], dimension)
                self.assertEqual(r.graph["sweep_dimension"], min(dimension, 2))
                if dimension <= 5:
                    self.assert_oracle(k)

    def test_constant_simplex_dimensions_three_through_eight(self):
        for dimension in range(3, 9):
            with self.subTest(dimension=dimension):
                k = make_complex([list(range(dimension + 1))], [3] * (dimension + 1))
                r = computeReeb(k)
                self.assertEqual((len(r), r.number_of_edges()), (1, 0))
                self.assertEqual(list(r.f.values()), [3])

    def test_every_tetrahedron_height_order_and_tie_pattern(self):
        count = 0
        for values in product(range(4), repeat=4):
            if set(values) != set(range(max(values) + 1)):
                continue
            with self.subTest(values=values):
                self.assert_oracle(make_complex([[0, 1, 2, 3]], values))
            count += 1
        self.assertEqual(count, 75)

    def test_four_simplex_boundary_and_plateaus(self):
        facets = list(combinations(range(5), 4))
        for values in [[0, 1, 2, 3, 4], [0, 0, 1, 1, 2], [2] * 5]:
            self.assert_oracle(make_complex(facets, values))

    def test_tetrahedra_form_a_cycle(self):
        k = make_complex(
            [[0, 1, 3, 4], [1, 2, 5, 6], [0, 2, 7, 8]], [0, 2, 4, 1, 1, 3, 3, 2, 2]
        )
        r = self.assert_oracle(k)
        self.assertEqual((len(r), r.number_of_edges()), (2, 2))
        self.assertEqual(sorted(r.f.values()), [0, 4])

    def test_nonmanifold_shared_face(self):
        self.assert_oracle(
            make_complex([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]], [0, 1, 2, 3, 3, 3])
        )

    def test_disconnected_mixed_dimensions_and_sparse_labels(self):
        self.assert_oracle(
            make_complex(
                [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10]],
                [0, 1, 2, 2, 3, 0, 0, 2, 2, 1, 4, 5],
                labels=[17, 30, 44, 59, 71, 92, 110, 301, 507, 801, 2000, 3011],
            )
        )

    def test_random_higher_dimensional_complexes(self):
        rng = random.Random(20260923)
        for trial in range(120):
            n = rng.randint(5, 9)
            # Ensure at least one simplex above dimension two, then mix facets.
            facets = [rng.sample(range(n), rng.randint(4, min(n, 6)))]
            facets += [
                rng.sample(range(n), rng.randint(2, min(n, 6)))
                for _ in range(rng.randint(0, 3))
            ]
            values = (
                rng.sample(range(-20, 21), n)
                if trial % 2
                else [rng.randrange(3) for _ in range(n)]
            )
            with self.subTest(trial=trial):
                self.assert_oracle(make_complex(facets, values))

    def test_high_dimensional_filtration_is_still_validated(self):
        k = make_complex([[0, 1, 2, 3]], [0, 1, 2, 3])
        SimplexTree.assign_filtration(k, [0, 1, 2, 3], 100)
        with self.assertRaisesRegex(ValueError, "lower-star invariant"):
            computeReeb(k)
        r = computeReeb(k, validate=False)
        self.assertEqual((len(r), r.number_of_edges()), (2, 1))
        self.assertEqual(sorted(r.f.values()), [0, 3])

    def test_no_full_simplex_traversal_when_validation_disabled(self):
        class SkeletonOnlyLowerStar(LowerStar):
            def get_simplices(self):
                raise AssertionError("Full simplex traversal must be skipped")

        k = make_complex([list(range(7))], range(7), cls=SkeletonOnlyLowerStar)
        r = computeReeb(k, validate=False)
        self.assertEqual((len(r), r.number_of_edges()), (2, 1))
        self.assertEqual(r.graph["input_dimension"], 6)

    def test_nonfinite_vertices_and_wrong_type_are_rejected(self):
        with self.assertRaises(TypeError):
            computeReeb(SimplexTree())
        for value in [np.inf, -np.inf, np.nan]:
            with self.subTest(value=value):
                k = make_complex([[0, 1, 2, 3]], [0, 1, 2, 3])
                SimplexTree.assign_filtration(k, [0], value)
                with self.assertRaisesRegex(ValueError, "finite"):
                    computeReeb(k, validate=False)

    def test_extreme_float_heights(self):
        for values in [
            [1.0, 1.0, np.nextafter(1.0, 2.0), np.nextafter(1.0, 2.0)],
            [1e308, 1e308, 1.5e308, 1.5e308],
        ]:
            r = self.assert_oracle(make_complex([[0, 1, 2, 3]], values))
            self.assertEqual((len(r), r.number_of_edges()), (2, 1))

    def test_native_draw_and_single_final_layout(self):
        k = make_complex([[0, 1, 2, 3]], [0, 1, 2, 3])
        sizes = []
        layout = ReebGraph.set_pos_from_f

        def tracked(r, *args, **kwargs):
            sizes.append(len(r))
            return layout(r, *args, **kwargs)

        fig, ax = plt.subplots()
        try:
            with patch.object(ReebGraph, "set_pos_from_f", tracked):
                r = computeReeb(k)
                self.assertEqual(set(r.pos), set(r.nodes))
                self.assertEqual(set(r.pos_f), set(r.nodes))
                self.assertTrue(all(r.pos_f[v][1] == r.f[v] for v in r))
                r.draw(ax=ax)
                fig.canvas.draw()
            # The empty constructor layout is harmless; exactly one call
            # handles the complete graph, and drawing does not recompute it.
            self.assertEqual([n for n in sizes if n], [len(r)])
        finally:
            plt.close(fig)

    def test_default_positions_support_low_level_plot(self):
        from cereeberus.draw.draw import reeb_plot

        k = make_complex([[0, 1], [2, 3]], [0, 2, 0, 2])
        r = computeReeb(k)
        self.assertEqual(set(r.pos_f), set(r.nodes))
        fig, ax = plt.subplots()
        try:
            reeb_plot(r, ax=ax)
            fig.canvas.draw()
        finally:
            plt.close(fig)

    def test_layout_can_be_explicitly_deferred(self):
        k = make_complex([[0, 1]], [0, 2])
        sizes = []
        layout = ReebGraph.set_pos_from_f

        def tracked(r, *args, **kwargs):
            sizes.append(len(r))
            return layout(r, *args, **kwargs)

        with patch.object(ReebGraph, "set_pos_from_f", tracked):
            r = computeReeb(k, set_pos=False)
        self.assertFalse(any(sizes))
        self.assertEqual(r.pos_f, {})
        self.assertEqual((len(r), r.number_of_edges()), (2, 1))

    def test_keep_regular_nodes_single_edge_names_and_heights(self):
        k = make_complex([[0, 1]], [0, 2])
        r = computeReeb(k, keep_regular_nodes=True, set_pos=False)
        self.assertEqual(list(r), [0, "e_0", 1])
        self.assertEqual(r.f, {0: 0.0, "e_0": 1.0, 1: 2.0})
        self.assertEqual(set(r.edges()), {(0, "e_0"), ("e_0", 1)})
        self.assertTrue(r.graph["keep_regular_nodes"])
        compact = computeReeb(k, set_pos=False)
        self.assertEqual(list(compact), [0, 1])
        self.assertFalse(compact.graph["keep_regular_nodes"])

    def test_keep_regular_nodes_matches_original_unreduced_graph(self):
        examples = [
            ([[0, 1], [1, 2]], [0, 1, 2]),
            ([[0, 1], [1, 2], [0, 2]], [0, 1, 2]),
            ([[0, 1, 2]], [0, 1, 2]),
            ([[0, 1], [1, 2], [1, 3]], [0, 1, 2, 2]),
            ([[0, 1], [2, 3]], [0, 2, 0, 2]),
            ([[0, 1, 2, 3]], [0, 1, 2, 3]),
            ([[0, 1, 2, 3]], [0, 0, 2, 2]),
            ([[0, 1, 2, 3]], [1, 1, 1, 1]),
            ([[0, 1, 3, 4], [1, 2, 5, 6], [0, 2, 7, 8]], [0, 2, 4, 1, 1, 3, 3, 2, 2]),
            ([], [0, 0, 1]),
        ]

        def annotated(r):
            g = nx.MultiDiGraph()
            g.add_nodes_from(
                (v, {"f": r.f[v], "sentinel": isinstance(v, str)}) for v in r
            )
            g.add_edges_from((a, b) for a, b, _ in r.edges(keys=True))
            return g

        for facets, values in examples:
            with self.subTest(facets=facets, values=values):
                k = make_complex(facets, values)
                old = computeReeb_legacy(k)
                new = computeReeb(k, keep_regular_nodes=True, set_pos=False)
                self.assertTrue(
                    nx.is_isomorphic(
                        annotated(old), annotated(new), node_match=lambda a, b: a == b
                    )
                )
                compact = computeReeb(k, set_pos=False)
                self.assertTrue(
                    nx.is_isomorphic(
                        reduced_graph(new),
                        reduced_graph(compact),
                        node_match=lambda a, b: a["f"] == b["f"],
                    )
                )
                integers = sorted(v for v in new if type(v) is int)
                sentinels = sorted(int(v[2:]) for v in new if isinstance(v, str))
                self.assertEqual(integers, list(range(len(integers))))
                self.assertEqual(sentinels, list(range(len(sentinels))))

    def test_keep_regular_nodes_names_use_counters_and_layout_runs_once(self):
        k = make_complex([[0, 1], [1, 2], [0, 2]], [0, 1, 2])
        sizes = []
        layout = ReebGraph.set_pos_from_f

        def tracked(r, *args, **kwargs):
            sizes.append(len(r))
            return layout(r, *args, **kwargs)

        fig, ax = plt.subplots()
        try:
            with (
                patch.object(
                    ReebGraph,
                    "get_next_vert_name",
                    side_effect=AssertionError("No name scans"),
                ),
                patch.object(ReebGraph, "set_pos_from_f", tracked),
            ):
                r = computeReeb(k, keep_regular_nodes=True)
                r.draw(ax=ax)
                fig.canvas.draw()
            self.assertEqual([n for n in sizes if n], [len(r)])
        finally:
            plt.close(fig)

    def test_keep_regular_nodes_midpoints_avoid_overflow(self):
        for values in [
            [1e308, 1.5e308],
            [-1e308, 1e308],
            [0.0, float(np.nextafter(0.0, 1.0)) * 2],
        ]:
            with self.subTest(values=values):
                r = computeReeb(
                    make_complex([[0, 1]], values),
                    keep_regular_nodes=True,
                    set_pos=False,
                )
                self.assertEqual((len(r), r.number_of_edges()), (3, 2))
                self.assertTrue(np.isfinite(r.f["e_0"]))
                self.assertTrue(values[0] < r.f["e_0"] < values[1])

    def test_keep_regular_nodes_adjacent_heights_raise_clear_error(self):
        k = make_complex([[0, 1]], [1.0, np.nextafter(1.0, 2.0)])
        with self.assertRaisesRegex(ValueError, "use keep_regular_nodes=False"):
            computeReeb(k, keep_regular_nodes=True, set_pos=False)
        compact = computeReeb(k, set_pos=False)
        self.assertEqual((len(compact), compact.number_of_edges()), (2, 1))

    def test_keep_regular_nodes_empty_input(self):
        r = computeReeb(LowerStar(), keep_regular_nodes=True)
        self.assertEqual(len(r), 0)
        self.assertEqual(r.pos_f, {})


# ---------------------------------------------------------------- helpers
def _plain(R):
    """Copy a ReebGraph into a plain MultiDiGraph with f as a node attribute."""
    G = nx.MultiDiGraph()
    for v in R.nodes:
        G.add_node(v, f=float(R.f[v]))
    G.add_edges_from((u, v) for u, v, _ in R.edges(keys=True))
    return G


def _contract_regular(G):
    """Remove one-in/one-out nodes without any layout calls."""
    G = G.copy()
    for v in list(G.nodes):
        if G.in_degree(v) == 1 and G.out_degree(v) == 1:
            (u,) = G.predecessors(v)
            (w,) = G.successors(v)
            G.remove_node(v)
            G.add_edge(u, w)
    return G


def _same_reeb(G, H):
    return nx.is_isomorphic(G, H, node_match=lambda a, b: a["f"] == b["f"])


def _betti1(R):
    return (
        R.number_of_edges()
        - R.number_of_nodes()
        + nx.number_weakly_connected_components(R)
    )


def _random_complex(rng, n_vertices, n_triangles, n_extra_edges, n_levels):
    K = LowerStar()
    verts = list(range(n_vertices))
    for v in verts:
        K.insert([v])
    for _ in range(n_triangles):
        K.insert(sorted(rng.sample(verts, 3)))
    for _ in range(n_extra_edges):
        K.insert(sorted(rng.sample(verts, 2)))
    for v in verts:
        # Few levels -> many exact ties, horizontal edges and flat triangles.
        K.assign_filtration(v, float(rng.randrange(n_levels)))
    return K


def _tetra_ball(n):
    """Triangulated solid cube [0,n]^3 (Kuhn triangulation), f = z."""
    K = LowerStar()
    idx = lambda x, y, z: (x * (n + 1) + y) * (n + 1) + z
    import itertools

    for x, y, z in itertools.product(range(n), repeat=3):
        for perm in itertools.permutations(range(3)):
            p = [x, y, z]
            simplex = [idx(*p)]
            for axis in perm:
                p[axis] += 1
                simplex.append(idx(*p))
            K.insert(simplex)
    for x, y, z in itertools.product(range(n + 1), repeat=3):
        K.assign_filtration(idx(x, y, z), float(z))
    return K


# ------------------------------------------------------------------ tests
class TestAgainstLegacy(unittest.TestCase):
    def test_random_complexes_match_legacy(self):
        rng = random.Random(20260923)
        for trial in range(300):
            K = _random_complex(
                rng,
                n_vertices=rng.randint(3, 12),
                n_triangles=rng.randint(0, 10),
                n_extra_edges=rng.randint(0, 6),
                n_levels=rng.randint(1, 6),
            )
            legacy = _plain(computeReeb_legacy(K))
            full = _plain(computeReeb(K, keep_regular_nodes=True, set_pos=False))
            reduced = _plain(computeReeb(K, set_pos=False))
            with self.subTest(trial=trial):
                self.assertTrue(
                    _same_reeb(legacy, full),
                    "keep_regular_nodes=True differs from legacy",
                )
                self.assertTrue(
                    _same_reeb(_contract_regular(legacy), reduced),
                    "reduced output differs from contracted legacy",
                )

    def test_torus_class_matches_legacy(self):
        for grid, seed in [(4, 1986), (6, 1986), (8, 7)]:
            T = Torus()
            T.generate_grid(grid_size=grid)
            T.assign_random_values(0, 100, seed=seed)
            legacy = _plain(computeReeb_legacy(T))
            full = _plain(computeReeb(T, keep_regular_nodes=True, set_pos=False))
            reduced = _plain(computeReeb(T, set_pos=False))
            with self.subTest(grid=grid, seed=seed):
                self.assertTrue(_same_reeb(legacy, full))
                self.assertTrue(_same_reeb(_contract_regular(legacy), reduced))

    def test_3d_complex_matches_legacy(self):
        K = _tetra_ball(2)
        rng = random.Random(3)
        for v in range(K.num_vertices()):
            K.assign_filtration(v, float(rng.randrange(5)))
        legacy = _plain(computeReeb_legacy(K))
        full = _plain(computeReeb(K, keep_regular_nodes=True, set_pos=False))
        self.assertTrue(_same_reeb(legacy, full))


class TestKnownAnswers(unittest.TestCase):
    def test_single_triangle_is_one_edge(self):
        K = LowerStar()
        K.insert([0, 1, 2])
        for v, f in enumerate([0.0, 1.0, 2.0]):
            K.assign_filtration(v, f)
        R = computeReeb(K, set_pos=False)
        self.assertEqual((R.number_of_nodes(), R.number_of_edges()), (2, 1))
        self.assertEqual(sorted(R.f.values()), [0.0, 2.0])

    def test_hollow_triangle_is_one_loop(self):
        K = LowerStar()
        for e in ([0, 1], [1, 2], [0, 2]):
            K.insert(e)
        for v, f in enumerate([0.0, 1.0, 2.0]):
            K.assign_filtration(v, f)
        R = computeReeb(K, set_pos=False)
        self.assertEqual((R.number_of_nodes(), R.number_of_edges()), (2, 2))
        self.assertEqual(_betti1(R), 1)  # parallel edges must be preserved

    def test_flat_complex_is_single_node(self):
        K = LowerStar()
        K.insert([0, 1, 2])
        K.insert([2, 3])
        for v in range(4):
            K.assign_filtration(v, 5.0)
        R = computeReeb(K, set_pos=False)
        self.assertEqual((R.number_of_nodes(), R.number_of_edges()), (1, 0))

    def test_disconnected_pieces(self):
        K = LowerStar()
        K.insert([0, 1, 2])
        K.insert([3, 4, 5])
        K.insert([6])
        for v, f in enumerate([0, 1, 2, 0, 1, 2, 1]):
            K.assign_filtration(v, float(f))
        R = computeReeb(K, set_pos=False)
        self.assertEqual(nx.number_weakly_connected_components(R), 3)


class TestSimplification(unittest.TestCase):
    def test_keep_regular_nodes_is_legacy_format(self):
        T = Torus()
        T.generate_grid(grid_size=4)
        T.assign_random_values(0, 100, seed=1986)
        R = computeReeb(T, keep_regular_nodes=True)
        self.assertTrue(any(isinstance(v, str) and v.startswith("e_") for v in R.nodes))
        self.assertTrue(_same_reeb(_plain(R), _plain(computeReeb_legacy(T))))

    def test_remove_all_regular_vertices_matches_reduced(self):
        # remove_all_regular_vertices on the subdivided output must give the
        # reduced output, with layout recomputed once (fast, positions in sync).
        import time

        T = Torus()
        T.generate_grid(grid_size=5)
        T.assign_random_values(0, 100, seed=1986)
        R = computeReeb(T, keep_regular_nodes=True)
        start = time.perf_counter()
        R.remove_all_regular_vertices()
        elapsed = time.perf_counter() - start
        self.assertTrue(_same_reeb(_plain(R), _plain(computeReeb(T, set_pos=False))))
        self.assertEqual(set(R.nodes), set(R.pos_f))
        self.assertLess(elapsed, 2.0)  # previously ~4 s: two layouts per vertex


if __name__ == "__main__":
    unittest.main()
