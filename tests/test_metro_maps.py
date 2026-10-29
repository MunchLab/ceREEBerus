"""Smoke tests for plotting solved ILP maps in both directions."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cereeberus import Assignment
from cereeberus.data.ex_mappergraphs import line
from cereeberus.draw.metro_maps import (
    _edge_label,
    _mapped_pairs,
    _node_label,
    draw_interleaving_map_pairs,
    draw_interleaving_metro_maps,
    draw_map_pair,
    draw_metro_map,
)


class TestMetroMaps(unittest.TestCase):
    def test_solved_maps_in_both_directions(self):
        assignment = Assignment(line(0, 2, seed=1), line(0, 2, seed=2), n=1)
        self.assertTrue(assignment.optimize())
        for direction in ("phi", "psi"):
            for key in ("0", "n"):
                self.assertTrue(_mapped_pairs(getattr(assignment, direction)(key, "V"),
                                              allow_unmapped=key == "n"))
                fig, ax = draw_metro_map(assignment, direction, key)
                self.assertTrue(ax.lines)
                plt.close(fig)
        fig, axes = draw_interleaving_metro_maps(assignment)
        self.assertEqual(len(axes), 2)
        plt.close(fig)
        fig, axes = draw_map_pair(assignment, "phi", label_stations=False)
        self.assertEqual(len(axes), 2)
        self.assertTrue(axes[0].lines)
        self.assertTrue(axes[1].lines)
        plt.close(fig)
        fig, axes = draw_interleaving_map_pairs(assignment, label_stations=False)
        self.assertEqual(axes.shape, (2, 2))
        plt.close(fig)

    def test_internal_names_match_reeb_draw(self):
        self.assertEqual(_node_label(("reeb_auto", 7)), "r7")
        self.assertEqual(_node_label(("mapper_subd", 3)), "s3")
        self.assertEqual(_edge_label((("mapper_subd", 3), "b", 0)), "s3→b")


    def test_unsolved_map_is_rejected(self):
        assignment = Assignment(line(0, 2, seed=1), line(0, 2, seed=2), n=1)
        with self.assertRaisesRegex(ValueError, "solve the ILP first"):
            draw_metro_map(assignment)


if __name__ == "__main__":
    unittest.main()
