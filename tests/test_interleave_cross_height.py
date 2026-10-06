"""Regression checks for cross-height interleaving maps and their ILP losses."""

import unittest

import numpy as np

from cereeberus import Assignment, MapperGraph


def graph(nodes, edges):
    result = MapperGraph()
    for name, height in nodes:
        result.add_node(name, height, reset_pos=False)
    for u, v in edges:
        result.add_edge(u, v, reset_pos=False)
    return result


def witness_pair():
    trunk_nodes = [(f"t{i}", i) for i in range(4)]
    trunk_edges = [(f"t{i}", f"t{i + 1}") for i in range(3)]
    F = graph(trunk_nodes + [("a1", 1), ("a2", 2)],
              trunk_edges + [("t0", "a1"), ("a1", "a2")])
    G = graph(trunk_nodes + [("b2", 2)], trunk_edges + [("t1", "b2")])
    return F, G


class TestCrossHeightInterleaving(unittest.TestCase):
    def test_old_wrong_map_choice_has_positive_cross_height_loss(self):
        F, G = witness_pair()
        assignment = Assignment(F, G, n=1, initialize_random_maps=True)
        self.assertTrue(assignment.optimize())
        self.assertEqual(assignment.loss(), 0)
        for maptype in ("phi", "psi"):
            for obj_type in ("V", "E"):
                self.assertEqual(assignment.cross_parallelogram(maptype, obj_type), 0)

        # Change only psi_0(b2) to the other F component while leaving psi_n fixed.
        matrix = assignment.psi("0", "V")[2]
        column = list(matrix.cols).index("b2")
        current = int(np.argmax(matrix.get_array()[:, column]))
        alternate = 1 - current
        matrix.get_array()[:, column] = 0
        matrix.get_array()[alternate, column] = 1
        self.assertGreater(assignment.cross_parallelogram_loss("psi", "V", 2, 3), 0)

    def test_maps_are_total_at_smoothed_boundary_heights(self):
        x = graph([("u0", 0), ("u1", 1)], [("u0", "u1")])
        assignment = Assignment(x, x, n=1, initialize_random_maps=True)
        self.assertTrue(assignment.optimize())
        self.assertEqual(assignment.loss(), 0)
        for maptype in ("phi", "psi"):
            for obj_type in ("V", "E"):
                matrix = getattr(assignment, maptype)("n", obj_type)
                for height in matrix.get_all_block_indices():
                    columns = matrix[height].get_array().sum(axis=0)
                    np.testing.assert_allclose(columns, 1)


if __name__ == "__main__":
    unittest.main()
