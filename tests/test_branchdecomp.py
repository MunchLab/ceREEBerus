import unittest
import numpy as np

from cereeberus.data import ex_reebgraphs as ex_rg
from cereeberus import ReebGraph
from cereeberus.reeb.branchdecomp import BranchDecomp


class TestBranchDecomp(unittest.TestCase):

    def check_decomp(self, R, bd):
        """General validity checks on a BranchDecomp given the original ReebGraph."""
        n = len(bd.branches)

        # branches and paths have the same length
        self.assertEqual(n, len(bd.paths))

        # Each branch row has 4 columns
        self.assertEqual(bd.branches.shape, (n, 4))

        for i in range(n):
            f_low, f_high, low_attach, high_attach = bd.branches[i]

            # f_low <= f_high
            self.assertLessEqual(f_low, f_high)

            # attachment IDs are valid branch indices
            self.assertGreaterEqual(int(low_attach), 0)
            self.assertLess(int(low_attach), n)
            self.assertGreaterEqual(int(high_attach), 0)
            self.assertLess(int(high_attach), n)

            # A branch can only attach to a previous branch or itself
            self.assertLessEqual(int(low_attach), i)
            self.assertLessEqual(int(high_attach), i)

            # Path vertices are all in the original graph
            for v in bd.paths[i]:
                self.assertIn(v, R.nodes)

        # Every edge in the original graph is covered by exactly one path
        edge_count = {}
        for path in bd.paths:
            for j in range(len(path) - 1):
                edge = (path[j], path[j + 1])
                edge_count[edge] = edge_count.get(edge, 0) + 1

        for u, v, _ in R.edges:
            # Edges are directed low -> high
            self.assertIn((u, v), edge_count,
                          msg=f"Edge ({u},{v}) not covered by any branch path")

        # Every vertex appears in at least one path
        all_path_verts = {v for path in bd.paths for v in path}
        for v in R.nodes:
            self.assertIn(v, all_path_verts,
                          msg=f"Vertex {v} not in any branch path")

    def test_dancing_man(self):
        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)
        self.check_decomp(R, bd)

    def test_juggling_man_isolated_vertices(self):
        """juggling_man has isolated vertices; they should become degenerate branches."""
        R = ex_rg.juggling_man()
        bd = BranchDecomp()
        bd.decompose(R)
        self.check_decomp(R, bd)

        # Find the isolated vertices in the original graph
        isolates = {v for v in R.nodes if R.up_degree(v) == 0 and R.down_degree(v) == 0}
        self.assertGreater(len(isolates), 0, "juggling_man should have isolated vertices")

        # Each isolate must appear as a degenerate branch (f_low == f_high)
        isolate_path_verts = set()
        for i, path in enumerate(bd.paths):
            if len(path) == 1:
                f_low, f_high, _, _ = bd.branches[i]
                self.assertEqual(f_low, f_high,
                                 msg=f"Single-vertex path for branch {i} should have f_low == f_high")
                isolate_path_verts.add(path[0])

        for v in isolates:
            self.assertIn(v, isolate_path_verts,
                          msg=f"Isolated vertex {v} should have a degenerate branch")

    def test_degenerate_branch_self_attaches(self):
        """Degenerate branches (isolated vertices) should attach to themselves."""
        R = ex_rg.juggling_man()
        bd = BranchDecomp()
        bd.decompose(R)

        isolates = {v for v in R.nodes if R.up_degree(v) == 0 and R.down_degree(v) == 0}

        for i, path in enumerate(bd.paths):
            if len(path) == 1 and path[0] in isolates:
                f_low, f_high, low_attach, high_attach = bd.branches[i]
                self.assertEqual(int(low_attach), i,
                                 msg=f"Degenerate branch {i} low_attach should be itself")
                self.assertEqual(int(high_attach), i,
                                 msg=f"Degenerate branch {i} high_attach should be itself")

    def test_get_branch(self):
        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)

        for i in range(len(bd.branches)):
            branch = bd.get_branch(i)
            self.assertEqual(len(branch), 4)
            np.testing.assert_array_equal(branch, bd.branches[i])

        self.assertRaises(IndexError, bd.get_branch, -1)
        self.assertRaises(IndexError, bd.get_branch, len(bd.branches))

    def test_get_branch_path(self):
        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)

        for i in range(len(bd.paths)):
            path = bd.get_branch_path(i)
            self.assertIsInstance(path, list)
            self.assertGreater(len(path), 0)
            self.assertEqual(path, bd.paths[i])

        self.assertRaises(IndexError, bd.get_branch_path, -1)
        self.assertRaises(IndexError, bd.get_branch_path, len(bd.paths))

    def test_original_graph_unchanged(self):
        """decompose should not modify the original ReebGraph."""
        R = ex_rg.dancing_man()
        nodes_before = set(R.nodes)
        edges_before = set(R.edges)
        f_before = R.f.copy()

        bd = BranchDecomp()
        bd.decompose(R)

        self.assertEqual(set(R.nodes), nodes_before)
        self.assertEqual(set(R.edges), edges_before)
        self.assertEqual(R.f, f_before)

    def test_reebgraph_branch_decomp(self):
        """ReebGraph.branch_decomp should return a populated BranchDecomp."""
        R = ex_rg.dancing_man()
        bd = R.branch_decomp()

        self.assertIsInstance(bd, BranchDecomp)
        self.check_decomp(R, bd)

    def test_reebgraph_copy(self):
        """ReebGraph.copy should preserve nodes, multiedges, f-values, and positions."""
        R = ex_rg.dancing_man()
        R.add_edge(6, 3)

        nodes_before = set(R.nodes)
        edges_before = list(R.edges(keys=True))
        f_before = R.f.copy()
        pos_before = R.pos.copy()
        pos_f_before = R.pos_f.copy()

        R_copy = R.copy()

        self.assertIsInstance(R_copy, ReebGraph)
        self.assertIsNot(R_copy, R)
        self.assertEqual(set(R_copy.nodes), nodes_before)
        self.assertEqual(list(R_copy.edges(keys=True)), edges_before)
        self.assertEqual(R_copy.f, f_before)
        self.assertEqual(R_copy.pos, pos_before)
        self.assertEqual(R_copy.pos_f, pos_f_before)

        # Mutating the copy should not affect the original.
        R_copy.add_node("copy_only", 10, reset_pos=False)
        self.assertNotIn("copy_only", R.nodes)
        self.assertIn("copy_only", R_copy.nodes)

    def test_empty_graph(self):
        """An empty Reeb graph should produce an empty decomposition."""
        R = ReebGraph()
        bd = BranchDecomp()
        bd.decompose(R)

        self.assertEqual(len(bd.branches), 0)
        self.assertEqual(len(bd.paths), 0)

    def test_path_function_values_match_branches(self):
        """The f-values at path endpoints should match what's stored in branches."""
        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)

        for i, (f_low, f_high, _, _) in enumerate(bd.branches):
            path = bd.paths[i]
            self.assertAlmostEqual(R.f[path[0]], f_low)
            self.assertAlmostEqual(R.f[path[-1]], f_high)

    def test_reconstruct(self):
        """Reconstructed graph should have the same nodes, edges, and f-values as the original."""
        for make_graph in [ex_rg.dancing_man, ex_rg.juggling_man]:
            R = make_graph()
            bd = BranchDecomp()
            bd.decompose(R)
            R2 = bd.reconstruct()

            # Same node set
            self.assertEqual(set(R.nodes), set(R2.nodes),
                             msg=f"Node sets differ for {make_graph.__name__}")

            # Same f-values
            for v in R.nodes:
                self.assertEqual(R.f[v], R2.f[v],
                                 msg=f"f-value mismatch for vertex {v} in {make_graph.__name__}")

            # Same number of edges
            self.assertEqual(len(R.edges), len(R2.edges),
                             msg=f"Edge count differs for {make_graph.__name__}")

            # Reconstructed graph passes the standard Reeb graph structural checks
            self.assertEqual(set(R2.nodes), set(R2.f.keys()))
            self.assertEqual(set(R2.nodes), set(R2.pos_f.keys()))
            for edge in R2.edges:
                v1, v2 = edge[:2]
                self.assertGreater(R2.f[v2], R2.f[v1])

    def test_reconstruct_empty(self):
        """Reconstructing from an empty decomposition returns an empty ReebGraph."""
        bd = BranchDecomp()
        bd.decompose(ReebGraph())
        R2 = bd.reconstruct()
        self.assertEqual(len(R2.nodes), 0)
        self.assertEqual(len(R2.edges), 0)

    def test_branch_smoothing_returns_new_instance(self):
        """branch_smoothing should return a new decomposition and keep original unchanged."""
        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)

        original = bd.branches.copy()
        smoothed = bd.branch_smoothing(0.5)

        self.assertIsInstance(smoothed, BranchDecomp)
        self.assertIsNot(smoothed, bd)
        np.testing.assert_array_equal(bd.branches, original)
        self.assertTrue(np.any(np.abs(smoothed.branches[:, :2] - original[:, :2]) > 0))
        np.testing.assert_array_equal(smoothed.branches[:, 2:], original[:, 2:])

    def test_branch_smoothing_negative_eps_raises(self):
        """Negative smoothing is invalid."""
        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)

        with self.assertRaises(ValueError):
            bd.branch_smoothing(-0.1)

    def test_add_branch_appends_and_returns_row(self):
        """add_branch should append a branch with the requested endpoint values/attachments."""
        bd = BranchDecomp()

        row0 = bd.add_branch(0.0, 2.0, 0, 0)
        np.testing.assert_array_equal(row0, np.array([0.0, 2.0, 0.0, 0.0]))
        self.assertEqual(len(bd.branches), 1)

        row1 = bd.add_branch(1.0, 1.5, 0, 0)
        np.testing.assert_array_equal(row1, np.array([1.0, 1.5, 0.0, 0.0]))
        self.assertEqual(len(bd.branches), 2)
        np.testing.assert_array_equal(
            bd.branches,
            np.array(
                [
                    [0.0, 2.0, 0.0, 0.0],
                    [1.0, 1.5, 0.0, 0.0],
                ]
            ),
        )

    def test_add_branch_with_store_paths_keeps_alignment(self):
        """When store_paths=True, add_branch keeps paths aligned with branches."""
        bd = BranchDecomp(store_paths=True)
        bd.add_branch(0.0, 1.0, 0, 0, path=["a", "b"])
        bd.add_branch(0.2, 0.8, 0, 0)

        self.assertEqual(len(bd.paths), len(bd.branches))
        self.assertEqual(bd.paths[0], ["a", "b"])
        self.assertEqual(bd.paths[1], [])

    def test_add_branch_invalid_inputs_raise(self):
        """add_branch should reject malformed branches and bad attachments."""
        bd = BranchDecomp()
        bd.add_branch(0.0, 2.0, 0, 0)

        with self.assertRaises(ValueError):
            bd.add_branch(2.0, 1.0, 1, 1)

        with self.assertRaises(ValueError):
            bd.add_branch(0.5, 1.5, 2, 1)

        with self.assertRaises(ValueError):
            bd.add_branch(0.5, 1.5, 1, 2)

        with self.assertRaises(ValueError):
            bd.add_branch(-1.0, 1.0, 0, 1)

        with self.assertRaises(ValueError):
            bd.add_branch(0.5, 3.0, 1, 0)

    def test_add_branch_manual_decomp_reconstructs(self):
        """A decomposition built manually via add_branch should reconstruct."""
        bd = BranchDecomp()
        bd.add_branch(0.0, 4.0, 0, 0)
        bd.add_branch(1.0, 3.0, 0, 0)
        bd.add_branch(2.0, 2.0, 1, 1)

        R = bd.reconstruct()
        self.assertGreaterEqual(len(R.nodes), 3)
        self.assertGreaterEqual(len(R.edges), 2)


if __name__ == "__main__":
    unittest.main()
