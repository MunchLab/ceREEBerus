import unittest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless test runs
import matplotlib.pyplot as plt

from cereeberus import MapperGraph, computeMapper, cover
from cereeberus.draw.draw import node_label_counts, pie_plot


class TestPiePlot(unittest.TestCase):
    def setUp(self):
        # Small mapper graph with a known point-to-node assignment.
        self.pointcloud = [(0.6, 0), (-0.1, 0.5)]
        self.graph = computeMapper(
            self.pointcloud, (lambda a: a[0]), [(-1, 0), (-0.5, 0.5), (0, 1)], "trivial"
        )
        # One category label per point in self.pointcloud.
        self.labels = ["A", "B"]

    def test_node_label_counts(self):
        counts = node_label_counts(self.graph, self.labels)
        # Node 0 and node 1 both only contain point 1 ("B").
        self.assertEqual(counts[0], {"A": 0, "B": 1})
        self.assertEqual(counts[1], {"A": 0, "B": 1})
        # Node 2 only contains point 0 ("A").
        self.assertEqual(counts[2], {"A": 1, "B": 0})

    def test_node_label_counts_requires_node_points(self):
        # A manually-built MapperGraph has no per-node point membership.
        manual_graph = MapperGraph()
        manual_graph.add_node(0, 0)
        with self.assertRaises(ValueError):
            node_label_counts(manual_graph, self.labels)

    def test_pie_plot_smoke(self):
        # Just confirm this runs without error and returns an Axes.
        ax = pie_plot(self.graph, self.labels)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_draw_pie_method(self):
        ax = self.graph.draw_pie(self.labels)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_pie_plot_size_by_points(self):
        # Use a point cloud where one node gets 3 points and another gets 1,
        # so size_by_points has genuinely different counts to scale between.
        pointcloud = [(0.6, 0), (0.65, 0.05), (0.7, -0.05), (-0.1, 0.5)]
        labels = ["A", "A", "A", "B"]
        graph = computeMapper(
            pointcloud, (lambda a: a[0]), [(-1, 0), (-0.5, 0.5), (0, 1)], "trivial"
        )

        from matplotlib.offsetbox import AnnotationBbox

        counts = node_label_counts(graph, labels)
        totals = {v: sum(counts[v].values()) for v in graph.nodes}

        ax = pie_plot(graph, labels, size_by_points=True, min_zoom=0.08, max_zoom=0.3)
        boxes = ax.findobj(AnnotationBbox)
        self.assertTrue(len(boxes) > 0)

        # Map each rendered pie back to its node via position, so we can
        # check zoom against that *specific* node's point count.
        pos_to_node = {tuple(graph.pos_f[v]): v for v in graph.nodes}
        zoom_by_node = {}
        for b in boxes:
            node = pos_to_node[tuple(b.xy)]
            zoom_by_node[node] = b.offsetbox.get_zoom()

        # Every node with more points should have a zoom >= every node
        # with fewer points (monotonicity), and the max-count node must
        # be strictly bigger than the min-count node.
        by_count = sorted(totals.items(), key=lambda kv: kv[1])
        min_node, min_count = by_count[0]
        max_node, max_count = by_count[-1]
        self.assertLess(min_count, max_count)  # sanity check on the setup itself
        self.assertGreater(zoom_by_node[max_node], zoom_by_node[min_node])

        # Full monotonicity check: sorting nodes by count should give the
        # same order as sorting them by zoom.
        nodes_sorted_by_count = [n for n, _ in by_count]
        nodes_sorted_by_zoom = sorted(zoom_by_node, key=zoom_by_node.get)
        self.assertEqual(nodes_sorted_by_count, nodes_sorted_by_zoom)

        plt.close("all")

    def test_pie_plot_size_by_points_false_uses_fixed_zoom(self):
        from matplotlib.offsetbox import AnnotationBbox

        ax = pie_plot(self.graph, self.labels, zoom=0.2, size_by_points=False)
        boxes = ax.findobj(AnnotationBbox)
        zooms = [b.offsetbox.get_zoom() for b in boxes]
        # Every node should use the same fixed zoom when the flag is off.
        self.assertTrue(all(z == 0.2 for z in zooms))
        plt.close("all")

    def test_pie_plot_size_by_points_equal_counts_no_div_by_zero(self):
        from matplotlib.offsetbox import AnnotationBbox

        # self.graph's nodes all have exactly 1 point each, so
        # t_max == t_min here — this must not raise ZeroDivisionError,
        # and should fall back to the plain `zoom` value for every node.
        ax = pie_plot(self.graph, self.labels, zoom=0.2, size_by_points=True)
        boxes = ax.findobj(AnnotationBbox)
        zooms = [b.offsetbox.get_zoom() for b in boxes]
        self.assertTrue(all(z == 0.2 for z in zooms))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()