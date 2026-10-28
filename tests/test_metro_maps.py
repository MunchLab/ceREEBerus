"""Smoke tests for plotting solved ILP maps in both directions."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cereeberus import Assignment
from cereeberus.data.ex_mappergraphs import line
from cereeberus.draw.metro_maps import _mapped_pairs, draw_interleaving_metro_maps, draw_metro_map


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


    def test_unsolved_map_is_rejected(self):
        assignment = Assignment(line(0, 2, seed=1), line(0, 2, seed=2), n=1)
        with self.assertRaisesRegex(ValueError, "solve the ILP first"):
            draw_metro_map(assignment)


if __name__ == "__main__":
    unittest.main()
