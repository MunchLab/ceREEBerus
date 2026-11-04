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
    _branches, _group_colors, _resolve_edge, color_by_branch, color_by_edge,
    draw_metromaps,
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
            self.assertTrue(any(line.get_linewidth() == 1.5
                                for line in figure.axes[1].lines))
            self.assertTrue(any(line.get_linewidth() == 2.6
                                for line in figure.axes[1].lines))
            self.assertTrue(all(size == 360 for collection in figure.axes[1].collections
                                for size in collection.get_sizes()))
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
        branch_edges = _branches(result.F)[0]
        focused = color_by_branch(result, branch_edges[0], color="magenta")
        self.assertEqual(len(focused.axes), 2)
        for ax in focused.axes:
            self.assertEqual(sum(to_rgba(line.get_color()) == to_rgba("magenta")
                                 for line in ax.lines), len(branch_edges))
        plt.close(focused)
        focused = color_by_branch(result, 0)
        self.assertEqual(focused.axes[1].get_legend().get_title().get_text(),
                         "Selected branch B0")
        self.assertIsNone(focused._suptitle)
        plt.close(focused)
        with self.assertRaisesRegex(ValueError, "out of range"):
            color_by_branch(result, 99)
        reverse_edge = next(iter(result.G.edges(keys=True)))
        figure = color_by_edge(result, reverse_edge, direction="psi")
        self.assertEqual(len(figure.axes), 2)
        plt.close(figure)
        with self.assertRaisesRegex(ValueError, "absent"):
            color_by_edge(result, ("missing", "edge"))

    def test_graph_titles_use_fitted_n_and_input_names(self):
        F = line(0, 2, seed=1)
        G = line(0, 3, seed=2)
        F.name, G.name = "Input A", "Input B"
        result = Interleave(F, G)
        self.assertEqual(result.fit(max_n_for_error=3), 1)
        figure = draw_metromaps(result)
        self.assertEqual([ax.get_title() for ax in figure.axes],
                         ["Input A", "Input B^1", "Input B", "Input A^1"])
        self.assertIsNone(figure._suptitle)
        plt.close(figure)
        auxiliary = draw_metromaps(result, direction="phi", key="n")
        self.assertEqual([ax.get_title() for ax in auxiliary.axes],
                         ["Input A^1", "Input B^2"])
        plt.close(auxiliary)

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
