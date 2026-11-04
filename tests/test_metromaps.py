"""Tests for the fitted-interleaving metro view."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cereeberus import Interleave
from cereeberus.data.ex_mappergraphs import line, torus
from cereeberus.draw.metromaps import _branches, draw_metromaps


class TestMetroMaps(unittest.TestCase):
    def test_fitted_interleaving_and_color_modes(self):
        result = Interleave(line(0, 2, seed=1), line(0, 2, seed=2))
        result.fit(max_n_for_error=3)
        for color_by in ("branch", "edge"):
            figure = draw_metromaps(result, color_by=color_by, label_stations=False)
            self.assertEqual(len(figure.axes), 4)
            self.assertTrue(figure.axes[1].lines)
            self.assertTrue(figure.axes[3].lines)
            plt.close(figure)
        figure = draw_metromaps(result, direction="phi", key="n")
        self.assertEqual(len(figure.axes), 2)
        plt.close(figure)

    def test_branch_partition(self):
        graph = torus(0, 2, 10, 12, delta=1, seed=17)
        branches = _branches(graph)
        self.assertEqual(len(branches), 4)
        self.assertEqual({edge for branch in branches for edge in branch},
                         set(graph.edges(keys=True)))
        self.assertEqual(len(_branches(line(0, 12, seed=5))), 1)

    def test_requires_fitted_interleave(self):
        with self.assertRaisesRegex(TypeError, "Interleave"):
            draw_metromaps(object())
        result = Interleave(line(0, 2, seed=1), line(0, 2, seed=2))
        with self.assertRaisesRegex(ValueError, "fit"):
            draw_metromaps(result)


if __name__ == "__main__":
    unittest.main()
