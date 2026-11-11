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
    _arcs, _group_colors, _pairs, _resolve_edge, color_by_arc, color_by_edge,
    draw_metromaps,
)


class TestMetroMaps(unittest.TestCase):
    def test_fitted_interleaving_and_color_modes(self):
        result = Interleave(line(0, 2, seed=1), line(0, 2, seed=2))
        result.fit(max_n_for_error=3)
        for color_by in ("arc", "edge"):
            figure = draw_metromaps(result, color_by=color_by,
                                    show_vert_maps=False)
            self.assertEqual(len(figure.axes), 4)
            figure.canvas.draw()
            for source_ax, target_ax in zip(figure.axes[::2], figure.axes[1::2]):
                self.assertTrue(source_ax.get_shared_y_axes().joined(source_ax, target_ax))
                self.assertEqual(source_ax.get_ylim(), target_ax.get_ylim())
                self.assertAlmostEqual(source_ax.transData.transform((0, 1))[1],
                                       target_ax.transData.transform((0, 1))[1])
            for ax in figure.axes:
                for collection in ax.collections:
                    for facecolor in collection.get_facecolors():
                        self.assertEqual(tuple(facecolor), to_rgba("white"))
            source_colors = [to_rgba(line.get_color()) for line in figure.axes[0].lines
                             if line.get_linewidth() == 3]
            self.assertEqual(source_colors[0], to_rgba(plt.get_cmap("tab10")(0)))
            self.assertTrue(figure.axes[1].lines)
            self.assertTrue(figure.axes[3].lines)
            self.assertTrue(any(line.get_linewidth() == 1.5
                                for line in figure.axes[1].lines))
            self.assertTrue(any(line.get_linewidth() == 2.6
                                for line in figure.axes[1].lines))
            self.assertTrue(all(size == 240 for collection in figure.axes[1].collections
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
        arc_edges = _arcs(result.F)[0]
        focused = color_by_arc(result, arc_edges[0], color="magenta")
        self.assertEqual(len(focused.axes), 2)
        for ax in focused.axes:
            self.assertEqual(sum(to_rgba(line.get_color()) == to_rgba("magenta")
                                 for line in ax.lines), len(arc_edges))
        plt.close(focused)
        focused = color_by_arc(result, 0)
        self.assertEqual(focused.axes[1].get_legend().get_title().get_text(),
                         "Selected arc A0")
        self.assertIsNone(focused._suptitle)
        plt.close(focused)
        with self.assertRaisesRegex(ValueError, "out of range"):
            color_by_arc(result, 99)
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

    def test_arc_partition(self):
        graph = torus(0, 2, 10, 12, delta=1, seed=17)
        arcs = _arcs(graph)
        self.assertEqual(len(arcs), 4)
        self.assertEqual({edge for arc in arcs for edge in arc},
                         set(graph.edges(keys=True)))
        self.assertEqual(len(_arcs(line(0, 12, seed=5))), 1)

    def test_colored_tracks_join_target_vertices(self):
        result = Interleave(torus(0, 2, 4, 6, delta=1, seed=17),
                            line(0, 6, seed=5))
        result.fit(max_n_for_error=3)
        figure = draw_metromaps(result, direction="phi", show_vert_maps=False)
        target = result.assignment.G("n")
        positions = {tuple(xy) for xy in target.pos_f.values()}
        for track in figure.axes[1].lines:
            if track.get_linewidth() != 2.6:
                continue
            points = track.get_xydata()
            self.assertIn(tuple(points[0]), positions)
            self.assertIn(tuple(points[-1]), positions)
            self.assertTrue(all(a <= b for a, b in zip(points[:-1, 1], points[1:, 1])))
        plt.close(figure)

    def test_level_set_colors_and_shared_heights(self):
        result = Interleave(line(0, 2, seed=1), line(0, 3, seed=2))
        result.fit(max_n_for_error=3)
        edge = next(iter(result.F.edges(keys=True)))
        for draw, args in ((draw_metromaps, ()), (color_by_arc, (0,)),
                           (color_by_edge, (edge,))):
            for enabled in (False, True):
                with self.subTest(draw=draw.__name__, enabled=enabled):
                    figure = draw(result, *args, direction="phi",
                                  level_set_colors=enabled, sharey=enabled)
                    figure.canvas.draw()
                    source_ax, target_ax = figure.axes
                    self.assertEqual(source_ax.get_shared_y_axes().joined(
                        source_ax, target_ax), enabled)
                    if not enabled:
                        self.assertNotEqual(source_ax.get_ylim(), target_ax.get_ylim())
                    colors = [tuple(color) for ax in figure.axes
                              for collection in ax.collections
                              for color in collection.get_facecolors()]
                    self.assertEqual(len(set(colors)) > 1, enabled)
                    if enabled:
                        # Equal heights have equal colors in both panels.
                        by_height = {}
                        for ax in figure.axes:
                            for collection in ax.collections:
                                for (_, height), color in zip(collection.get_offsets(),
                                                              collection.get_facecolors()):
                                    color = tuple(color)
                                    self.assertEqual(by_height.setdefault(height, color), color)
                    plt.close(figure)

    def test_vertex_map_annotations_and_highlight_layering(self):
        result = Interleave(line(0, 2, seed=1), line(0, 3, seed=2))
        result.fit(max_n_for_error=3)
        edge = next(iter(result.F.edges(keys=True)))
        for draw, args in ((draw_metromaps, ()), (color_by_arc, (0,)),
                           (color_by_edge, (edge,))):
            for enabled in (False, True):
                with self.subTest(draw=draw.__name__, show_vert_maps=enabled):
                    figure = draw(result, *args, direction="phi", show_vert_maps=enabled)
                    source_ax, target_ax = figure.axes
                    annotations = [text for text in target_ax.texts
                                   if text.get_text().startswith("← ")]
                    self.assertEqual(bool(annotations), enabled)
                    top_line = max(line.get_zorder() for line in source_ax.lines)
                    self.assertTrue(all(node.get_zorder() > top_line
                                        for node in source_ax.collections))
                    self.assertTrue(all(text.get_zorder() > top_line
                                        for text in source_ax.texts))
                    plt.close(figure)

    def test_rejects_missing_target_edge_key(self):
        result = Interleave(line(0, 2, seed=1), line(0, 2, seed=2))
        result.fit(max_n_for_error=3)
        target = result.assignment.G("n")
        _, edge = _pairs(result.assignment.phi("0", "E"))[0]
        target.add_edge(edge[0], edge[1])
        target.remove_edge(*edge)
        # Another parallel edge must not stand in for the selected edge key.
        self.assertTrue(target.has_edge(edge[0], edge[1]))
        try:
            with self.assertRaisesRegex(ValueError, "absent from the target"):
                draw_metromaps(result, direction="phi")
        finally:
            plt.close("all")

    def test_requires_fitted_interleave(self):
        with self.assertRaisesRegex(TypeError, "Interleave"):
            draw_metromaps(object())
        result = Interleave(line(0, 2, seed=1), line(0, 2, seed=2))
        with self.assertRaisesRegex(ValueError, "fit"):
            draw_metromaps(result)


if __name__ == "__main__":
    unittest.main()
