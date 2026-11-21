import unittest

import numpy as np

from cereeberus import GromovWasserstein, ReebGraph, gromov_wasserstein_distance
from cereeberus.data import ex_mappergraphs, ex_reebgraphs
from cereeberus.distance.gromov_wasserstein import (
    extended_persistence_pairs,
    persistence_image,
    pi_measure,
    reeb_radius,
    reeb_radius_matrix,
    symmetric_reeb_radius,
)


def build_reeb_graph(f, edges):
    R = ReebGraph()
    for v, val in f.items():
        R.add_node(v, val, reset_pos=False)
    for u, v in edges:
        R.add_edge(u, v, reset_pos=False)
    return R


# The two Reeb graphs of the worked example (Section 3.5) in Chambers and
# Meng, "Reeb Graph Comparison via the Gromov-Wasserstein Distance".
def example_F():
    f = {"f1": 24, "f2": 18, "f3": 20, "f4": 12, "f5": 15, "f6": 6, "f7": 8, "f8": 2}
    edges = [
        ("f1", "f2"),
        ("f2", "f4"),
        ("f2", "f5"),
        ("f3", "f5"),
        ("f5", "f7"),
        ("f4", "f6"),
        ("f4", "f7"),
        ("f7", "f8"),
    ]
    return build_reeb_graph(f, edges)


def example_G():
    f = {"g1": 24, "g2": 20, "g3": 18, "g4": 16, "g5": 12, "g6": 8, "g7": 6, "g8": 4}
    edges = [
        ("g1", "g2"),
        ("g2", "g4"),
        ("g4", "g3"),
        ("g2", "g5"),
        ("g4", "g5"),
        ("g5", "g6"),
        ("g6", "g7"),
        ("g6", "g8"),
    ]
    return build_reeb_graph(f, edges)


PAPER_D_F = np.array(
    [
        [0, 6, 7, 12, 9, 18, 16, 22],
        [6, 0, 4, 6, 3, 12, 10, 16],
        [7, 4, 0, 8, 5, 14, 12, 18],
        [12, 6, 8, 0, 3.5, 6, 4, 10],
        [9, 3, 5, 3.5, 0, 9, 7, 13],
        [18, 12, 14, 6, 9, 0, 5, 8],
        [16, 10, 12, 4, 7, 5, 0, 6],
        [22, 16, 18, 10, 13, 8, 6, 0],
    ]
)

PAPER_D_G = np.array(
    [
        [0, 4, 7, 8, 12, 16, 18, 20],
        [4, 0, 3, 4, 8, 12, 14, 16],
        [7, 3, 0, 2, 6, 10, 12, 14],
        [8, 4, 2, 0, 4, 8, 10, 12],
        [12, 8, 6, 4, 0, 4, 6, 8],
        [16, 12, 10, 8, 4, 0, 2, 4],
        [18, 14, 12, 10, 6, 2, 0, 3],
        [20, 16, 14, 12, 8, 4, 3, 0],
    ]
)

PAPER_NU_F = [0.2420, 0.1190, 0.0730, 0.0660, 0.0730, 0.0660, 0.1190, 0.2420]
PAPER_NU_G = [0.3075, 0.1255, 0.0385, 0.0385, 0.1255, 0.0284, 0.0284, 0.3075]


class TestReebRadius(unittest.TestCase):
    def test_paper_example(self):
        F, G = example_F(), example_G()
        D_F = symmetric_reeb_radius(F)
        D_G = symmetric_reeb_radius(G)
        self.assertEqual(list(D_F.index), [f"f{i}" for i in range(1, 9)])
        np.testing.assert_allclose(D_F.to_numpy(), PAPER_D_F)
        np.testing.assert_allclose(D_G.to_numpy(), PAPER_D_G)

    def test_directed_radius(self):
        F = example_F()
        # From the global maximum every path goes down, so the radius is the
        # drop in function value.
        rho = reeb_radius(F, "f1")
        self.assertEqual(rho["f2"], 6)
        self.assertEqual(rho["f8"], 22)
        # f4 to f5 must pass the saddle f2 (18) or go down through f7 (8);
        # the best path has largest deviation |12 - 8| = 4.
        self.assertEqual(reeb_radius(F, "f4")["f5"], 4)
        # The Reeb radius is not symmetric.
        rho = reeb_radius_matrix(F)
        self.assertNotEqual(rho.loc["f4", "f5"], rho.loc["f5", "f4"])

    def test_metric_properties(self):
        for R in [
            example_F(),
            ex_reebgraphs.torus(),
            ex_reebgraphs.dancing_man(),
            ex_reebgraphs.juggling_man(),
        ]:
            for method in ["mean", "max"]:
                D = symmetric_reeb_radius(R, method=method).to_numpy()
                np.testing.assert_allclose(D, D.T)
                np.testing.assert_allclose(np.diag(D), 0)
                self.assertTrue(np.all(D >= 0))
                # Triangle inequality D[i, k] <= D[i, j] + D[j, k].
                lhs = D[:, None, :]
                rhs = D[:, :, None] + D[None, :, :]
                self.assertTrue(np.all(lhs <= rhs + 1e-9))

    def test_max_symmetrization(self):
        F = example_F()
        rho = reeb_radius_matrix(F).to_numpy()
        D = symmetric_reeb_radius(F, method="max").to_numpy()
        np.testing.assert_allclose(D, np.maximum(rho, rho.T))
        with self.assertRaises(ValueError):
            symmetric_reeb_radius(F, method="median")

    def test_bad_root(self):
        with self.assertRaises(ValueError):
            reeb_radius(example_F(), "not a vertex")


class TestExtendedPersistence(unittest.TestCase):
    def test_paper_example(self):
        pairs = extended_persistence_pairs(example_F())
        found = {(bv, dv, t) for bv, dv, _, _, t in pairs}
        self.assertEqual(
            found,
            {
                ("f6", "f4", "Ord0"),
                ("f3", "f5", "Rel1"),
                ("f8", "f1", "Ext0"),
                ("f2", "f7", "Ext1"),
            },
        )
        pairs = extended_persistence_pairs(example_G())
        found = {(bv, dv, t) for bv, dv, _, _, t in pairs}
        self.assertEqual(
            found,
            {
                ("g7", "g6", "Ord0"),
                ("g3", "g4", "Rel1"),
                ("g8", "g1", "Ext0"),
                ("g2", "g5", "Ext1"),
            },
        )

    def test_multiedge_loop(self):
        # The torus has a double edge between b and c, which is a loop.
        for multigraph in [True, False]:
            T = ex_reebgraphs.torus(0, 1, 4, 5, multigraph=multigraph)
            pairs = extended_persistence_pairs(T)
            found = {(bv, dv, t) for bv, dv, _, _, t in pairs}
            self.assertEqual(found, {("a", "d", "Ext0"), ("c", "b", "Ext1")})

    def test_mapper_graph_ties(self):
        # Vertices of a Mapper graph share function values, but every pair
        # must still be matched to vertices that realize it.
        M = ex_mappergraphs.torus(0, 2, 10, 12, delta=0.5)
        pairs = extended_persistence_pairs(M)
        self.assertEqual(sorted(t for *_, t in pairs), ["Ext0", "Ext1"])
        for bv, dv, b, d, _ in pairs:
            self.assertEqual(b, 0.5 * M.f[bv])
            self.assertEqual(d, 0.5 * M.f[dv])

    def test_line(self):
        pairs = extended_persistence_pairs(ex_reebgraphs.line(0, 3))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][2:], (0.0, 3.0, "Ext0"))


class TestPIMeasure(unittest.TestCase):
    def test_persistence_image(self):
        image, centers = persistence_image(
            [[1.0, 2.0]],
            sigma=1.0,
            birth_range=(0, 2),
            pers_range=(1, 3),
            resolution=(3, 5),
        )
        self.assertEqual(image.shape, (5, 3))
        self.assertEqual(centers.shape, (15, 2))
        # The largest pixel is the one centered at the point, with value
        # weight / (2 pi sigma^2) = 2 / (2 pi).
        k = np.argmax(image.ravel())
        np.testing.assert_allclose(centers[k], [1.0, 2.0])
        self.assertAlmostEqual(image.max(), 1 / np.pi)
        with self.assertRaises(ValueError):
            persistence_image([[1.0, 2.0]], sigma=0)

    def test_paper_example(self):
        mu_F = pi_measure(example_F(), sigma=2, resolution=500)
        mu_G = pi_measure(example_G(), sigma=2, resolution=500)
        np.testing.assert_allclose(
            [mu_F[f"f{i}"] for i in range(1, 9)], PAPER_NU_F, atol=1e-3
        )
        np.testing.assert_allclose(
            [mu_G[f"g{i}"] for i in range(1, 9)], PAPER_NU_G, atol=1e-3
        )

    def test_probability(self):
        for R in [
            example_F(),
            ex_reebgraphs.dancing_man(),
            ex_mappergraphs.dancing_man(),
            ex_mappergraphs.torus(),
        ]:
            mu = pi_measure(R)
            self.assertEqual(set(mu), set(R.nodes))
            self.assertAlmostEqual(sum(mu.values()), 1)
            self.assertTrue(all(m > 0 for m in mu.values()))

    def test_endpoints_share_mass(self):
        mu = pi_measure(example_F(), sigma=2, resolution=100)
        for bv, dv, *_ in extended_persistence_pairs(example_F()):
            self.assertAlmostEqual(mu[bv], mu[dv])

    def test_regular_vertices_get_smallest_mass(self):
        # In a Mapper graph the vertices that are not endpoints of a pair get
        # the smallest nonzero contribution.
        M = ex_mappergraphs.torus(0, 2, 10, 12, delta=1)
        mu = pi_measure(M, sigma=1)
        endpoints = {
            v for bv, dv, *_ in extended_persistence_pairs(M) for v in (bv, dv)
        }
        regular = [v for v in M.nodes if v not in endpoints]
        self.assertTrue(len(regular) > 0)
        smallest = min(mu.values())
        for v in regular:
            self.assertAlmostEqual(mu[v], smallest)

    def test_no_pairs_is_uniform(self):
        R = ReebGraph()
        R.add_node(0, 1.0)
        self.assertEqual(pi_measure(R), {0: 1.0})


class TestGromovWasserstein(unittest.TestCase):
    def test_paper_example(self):
        F, G = example_F(), example_G()
        # The paper's worked example computes each Persistence Image on its
        # own grid and reports GW = 7.815.
        gw = GromovWasserstein(F, G, sigma=2, resolution=500, shared_grid=False)
        self.assertAlmostEqual(gw.fit(), 7.815, places=3)
        # With one shared grid (the default) the measures move by less than
        # 1e-5 and the distance by less than 1e-3.
        gw = GromovWasserstein(F, G, sigma=2, resolution=500)
        self.assertAlmostEqual(gw.fit(), 7.815, delta=1e-3)
        np.testing.assert_allclose(gw.D_F.to_numpy(), PAPER_D_F)
        np.testing.assert_allclose(
            [gw.mu_G[f"g{i}"] for i in range(1, 9)], PAPER_NU_G, atol=1e-3
        )

    def test_plan(self):
        F, G = example_F(), example_G()
        dist, plan = gromov_wasserstein_distance(F, G, return_plan=True, sigma=2)
        self.assertEqual(list(plan.index), list(F.nodes))
        self.assertEqual(list(plan.columns), list(G.nodes))
        gw = GromovWasserstein(F, G, sigma=2)
        gw.fit()
        np.testing.assert_allclose(plan.sum(axis=1), [gw.mu_F[v] for v in F.nodes])
        np.testing.assert_allclose(plan.sum(axis=0), [gw.mu_G[v] for v in G.nodes])
        self.assertAlmostEqual(dist, gw.dist)

    def test_self_distance_and_symmetry(self):
        F, G = example_F(), example_G()
        self.assertAlmostEqual(gromov_wasserstein_distance(F, F), 0)
        self.assertAlmostEqual(
            gromov_wasserstein_distance(F, G, sigma=2),
            gromov_wasserstein_distance(G, F, sigma=2),
            places=6,
        )

    def test_mapper_graphs(self):
        M1 = ex_mappergraphs.torus(0, 2, 10, 12, delta=0.5)
        M2 = ex_mappergraphs.dancing_man(delta=0.5)
        d = gromov_wasserstein_distance(M1, M2)
        self.assertTrue(np.isfinite(d) and d > 0)

    def test_disconnected(self):
        R = example_F()
        R.add_node("x", 100, reset_pos=False)
        with self.assertRaises(ValueError):
            gromov_wasserstein_distance(R, example_G())


if __name__ == "__main__":
    unittest.main()
