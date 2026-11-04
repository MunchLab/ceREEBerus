"""Tests for the fitted-interleaving metro view."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx

from cereeberus import Interleave
from cereeberus.data.ex_mappergraphs import line, torus
from cereeberus.draw.metromaps import (
    _branches, _group_colors, _resolve_edge, color_by_edge, draw_metromaps,
)


class TestMetroMaps(unittest.TestCase):
    def test_fitted_interleaving_and_color_modes(self):
        result = Interleave(line(0, 2, seed=1), line(0, 2, seed=2))
        result.fit(max_n_for_error=3)
        for color_by in ("branch", "edge"):
            figure = draw_metromaps(result, color_by=color_by,
                                    color_scheme="tab10", label_stations=False)
            self.assertEqual(len(figure.axes), 4)
            self.assertTrue(figure.axes[1].lines)
            self.assertTrue(figure.axes[3].lines)
            plt.close(figure)
        figure = draw_metromaps(result, direction="phi", key="n")
        self.assertEqual(len(figure.axes), 2)
        plt.close(figure)

        selected = next(iter(result.F.edges(keys=True)))
        figure = color_by_edge(result, selected, color="magenta")
        self.assertEqual(len(figure.axes), 2)
        for ax in figure.axes:
            self.assertEqual(sum(to_rgba(line.get_color()) == to_rgba("magenta")
                                 for line in ax.lines), 1)
        plt.close(figure)
        reverse_edge = next(iter(result.G.edges(keys=True)))
        figure = color_by_edge(result, reverse_edge, direction="psi")
        self.assertEqual(len(figure.axes), 2)
        plt.close(figure)
        with self.assertRaisesRegex(ValueError, "absent"):
            color_by_edge(result, ("missing", "edge"))

    def test_palette_reuse_and_parallel_edge_selection(self):
        styles = [(i, str(i)) for i in range(21)]
        with self.assertWarnsRegex(UserWarning, "colors will repeat"):
            colors = _group_colors(styles, "tab20", True)
        self.assertEqual(colors[0], colors[20])
        self.assertNotEqual(_group_colors(styles, "hsv", True)[0],
                            _group_colors(styles, "hsv", True)[20])
        graph = nx.MultiDiGraph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "b")
        with self.assertRaisesRegex(ValueError, "Parallel edges"):
            _resolve_edge(graph, ("a", "b"))
        self.assertEqual(_resolve_edge(graph, ("a", "b", 1)), ("a", "b", 1))

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
