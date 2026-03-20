import unittest

import networkx as nx
import numpy as np
from cereeberus.data import ex_reebgraphs as ex_rg
from cereeberus.draw.layout import reeb_x_layout


class TestDrawLayout(unittest.TestCase):
    def test_reeb_x_layout_returns_finite_x_for_all_nodes(self):
        R = ex_rg.torus(multigraph=False, seed=0)

        x_positions = reeb_x_layout(R, R.f, seed=17, repulsion=0.8)

        self.assertEqual(set(x_positions.keys()), set(R.nodes))
        for v in R.nodes:
            self.assertTrue(np.isfinite(x_positions[v]))
            self.assertLessEqual(x_positions[v], 1.0 + 1e-9)
            self.assertGreaterEqual(x_positions[v], -1.0 - 1e-9)

    def test_reeb_x_layout_is_reproducible_with_seed(self):
        R = ex_rg.torus(multigraph=False, seed=0)

        x_a = reeb_x_layout(R, R.f, seed=123, repulsion=0.8)
        x_b = reeb_x_layout(R, R.f, seed=123, repulsion=0.8)

        for v in R.nodes:
            self.assertAlmostEqual(x_a[v], x_b[v])

    def test_reeb_x_layout_empty_graph(self):
        G = nx.Graph()
        x_positions = reeb_x_layout(G, {}, seed=1, repulsion=0.5)
        self.assertEqual(x_positions, {})

    def test_reeb_x_layout_same_height_isolated_nodes(self):
        # Regression test: no edges + same-height nodes should not trigger
        # unbounded optimisation drift.
        G = nx.Graph()
        nodes = ["u", "v", "w", "z"]
        G.add_nodes_from(nodes)
        f = {node: 0.0 for node in nodes}

        x_positions = reeb_x_layout(G, f, seed=9, repulsion=1.0)

        self.assertEqual(set(x_positions.keys()), set(nodes))
        for node in nodes:
            self.assertTrue(np.isfinite(x_positions[node]))
            self.assertLessEqual(x_positions[node], 1.0 + 1e-9)
            self.assertGreaterEqual(x_positions[node], -1.0 - 1e-9)


if __name__ == "__main__":
    unittest.main()
