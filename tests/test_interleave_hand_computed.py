"""The paper's line–torus Mapper example and its interleaving maps."""

import unittest

import networkx as nx
import numpy as np

from cereeberus import Assignment, Interleave
from cereeberus.data.ex_mappergraphs import line, torus


def line_and_torus(loop_height):
    """Use the paper's a=0, b=2, c=b+h, d=c+2 convention."""
    top = loop_height + 4
    return (line(0, top, seed=17),
            torus(0, 2, loop_height + 2, top, delta=1, seed=17))


class TestLineTorusInterleavings(unittest.TestCase):
    def assert_genuine_maps(self, assignment):
        """Check total maps, naturality, induced thickening, and both triangles."""
        self.assertEqual(assignment.loss(), 0)
        for maptype in ("phi", "psi"):
            for key in ("0", "n"):
                for obj_type in ("V", "E"):
                    matrix = assignment.get_interleaving_map(maptype, key, obj_type)
                    for height in matrix.get_all_block_indices():
                        np.testing.assert_array_equal(
                            matrix[height].get_array().sum(axis=0),
                            np.ones(matrix[height].get_array().shape[1]),
                        )
            for direction in ("up", "down"):
                self.assertEqual(assignment.parallelogram_Edge_Vert(maptype, direction), 0)
            for obj_type in ("V", "E"):
                self.assertEqual(assignment.parallelogram(maptype, obj_type), 0)
                for s, t in assignment.cross_height_pairs(maptype, obj_type):
                    source, target = (("F", "G") if maptype == "phi" else ("G", "F"))
                    map_0 = assignment.get_interleaving_map(maptype, "0", "V")[s].get_array()
                    map_n = assignment.get_interleaving_map(maptype, "n", obj_type)[t].get_array()
                    first = map_n @ assignment.I_cross(source, "0", obj_type, s, t).get_array()
                    second = assignment.I_cross(target, "n", obj_type, s, t).get_array() @ map_0
                    np.testing.assert_array_equal(first, second)

        for start, forward, return_map in (("F", "phi", "psi"),
                                           ("G", "psi", "phi")):
            for obj_type in ("V", "E"):
                source_map = assignment.get_interleaving_map(forward, "0", obj_type)
                for height in source_map.get_all_block_indices():
                    if source_map[height].get_array().shape[1] == 0:
                        continue
                    expected = (assignment.I(start, "n", obj_type)[height].get_array()
                                @ assignment.I(start, "0", obj_type)[height].get_array())
                    actual = (assignment.get_interleaving_map(return_map, "n", obj_type)[height].get_array()
                              @ source_map[height].get_array())
                    np.testing.assert_array_equal(actual, expected)

    def test_line_torus_distance_is_ceiling_of_quarter_loop_height(self):
        # Paper: d_I(L, T_h) = ceil(h / 4). Include both ceiling jumps.
        for h, expected in ((4, 1), (5, 2), (8, 2), (9, 3), (11, 3)):
            with self.subTest(loop_height=h):
                L, T = line_and_torus(h)
                result = Interleave(L, T)
                self.assertEqual(result.fit(max_n_for_error=4), expected)
                self.assert_genuine_maps(result.assignment)
                # A zero-loss interleaving at expected - 1 would refute
                # the hand-computed lower bound.
                self.assertFalse(Assignment(L, T, n=expected - 1).optimize())

    def test_line_map_stays_on_one_side_of_loop(self):
        # At h=11 and n=3, multiple middle levels of T^n retain two
        # components. The image of L must use one branch consistently.
        L, T = line_and_torus(11)
        result = Interleave(L, T)
        self.assertEqual(result.fit(max_n_for_error=4), 3)
        self.assert_genuine_maps(result.assignment)
        assignment = result.assignment

        interior = [v for v in T.nodes if 2 < T.f[v] < 13]
        branches = list(nx.connected_components(T.subgraph(interior).to_undirected()))
        self.assertEqual(len(branches), 2)

        chosen_branches = []
        phi = assignment.phi("0", "V")  # L -> T^3
        for height in phi.get_all_block_indices():
            block = phi[height]
            if len(block.rows) != 2 or len(block.cols) == 0:
                continue
            self.assertEqual(len(block.cols), 1)  # one line vertex per level
            selected_row = block.rows[int(np.argmax(block.get_array()[:, 0]))]
            component = assignment.comp_["G"][0][selected_row]
            hits = {i for i, branch in enumerate(branches) if component & branch}
            self.assertEqual(len(hits), 1, f"ambiguous image at height {height}")
            chosen_branches.append(next(iter(hits)))

        self.assertGreaterEqual(len(chosen_branches), 2)
        self.assertEqual(len(set(chosen_branches)), 1)


if __name__ == "__main__":
    unittest.main()
