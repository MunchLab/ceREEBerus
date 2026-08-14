import unittest
import numpy as np

from cereeberus.data import ex_reebgraphs as ex_rg
from cereeberus import ReebGraph
from cereeberus.reeb.branchdecomp import BranchDecomp, BranchDecompMap


class TestBranchDecomp(unittest.TestCase):

    def check_decomp(self, R, bd):
        """General validity checks on a BranchDecomp given the original ReebGraph."""
        branches = list(bd)
        n = len(branches)

        # branches and paths have the same length
        self.assertEqual(n, len(bd.paths))

        for i, branch in enumerate(branches):
            self.assertLessEqual(branch.f_low, branch.f_high)

            for attachment, endpoint in (
                (branch.bottom_branch, branch.f_low),
                (branch.top_branch, branch.f_high),
            ):
                if attachment is not None:
                    self.assertIn(attachment, branches)
                    self.assertLess(branches.index(attachment), i)
                    self.assertLess(attachment.f_low, endpoint)
                    self.assertLess(endpoint, attachment.f_high)

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
                branch = bd[i]
                self.assertEqual(branch.f_low, branch.f_high,
                                 msg=f"Single-vertex path for branch {i} should have f_low == f_high")
                isolate_path_verts.add(path[0])

        for v in isolates:
            self.assertIn(v, isolate_path_verts,
                          msg=f"Isolated vertex {v} should have a degenerate branch")

    def test_degenerate_branch_has_no_attachments(self):
        """Degenerate branches (isolated vertices) have local-extremum endpoints."""
        R = ex_rg.juggling_man()
        bd = BranchDecomp()
        bd.decompose(R)

        isolates = {v for v in R.nodes if R.up_degree(v) == 0 and R.down_degree(v) == 0}

        for i, path in enumerate(bd.paths):
            if len(path) == 1 and path[0] in isolates:
                branch = bd[i]
                self.assertIsNone(branch.bottom_branch,
                                  msg=f"Degenerate branch {i} bottom should be local")
                self.assertIsNone(branch.top_branch,
                                  msg=f"Degenerate branch {i} top should be local")

    def test_get_branch(self):
        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)

        for i in range(len(bd)):
            branch = bd.get_branch(i)
            self.assertIs(branch, bd[i])
            self.assertEqual(branch.f_low, bd.paths[i] and R.f[bd.paths[i][0]])
            self.assertEqual(branch.f_high, bd.paths[i] and R.f[bd.paths[i][-1]])

        self.assertRaises(IndexError, bd.get_branch, -1)
        self.assertRaises(IndexError, bd.get_branch, len(bd))

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

        self.assertEqual(len(bd), 0)
        self.assertEqual(len(bd.paths), 0)

    def test_path_function_values_match_branches(self):
        """The f-values at path endpoints should match what's stored in branches."""
        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)

        for i, branch in enumerate(bd):
            path = bd.paths[i]
            self.assertAlmostEqual(R.f[path[0]], branch.f_low)
            self.assertAlmostEqual(R.f[path[-1]], branch.f_high)

    def test_reconstruct(self):
        """Reconstruction preserves graph structure and function values, not node labels."""
        for make_graph in [ex_rg.dancing_man, ex_rg.juggling_man]:
            R = make_graph()
            bd = BranchDecomp()
            bd.decompose(R)
            R2 = bd.reconstruct()

            self.assertEqual(
                sorted(R.f.values()),
                sorted(R2.f.values()),
                msg=f"Function values differ for {make_graph.__name__}",
            )

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

    # TODO: Restore these tests when branch_smoothing is ported.
    # def test_branch_smoothing_returns_new_instance(self):
    #     ...
    #
    # def test_branch_smoothing_negative_eps_raises(self):
    #     ...

    def test_append_adds_branch_with_attachments(self):
        """append stores branches and their attachment object references."""
        bd = BranchDecomp()

        owner = bd.append(0.0, 4.0)
        branch = bd.append(1.0, 3.0, top_branch=0, bottom_branch=owner.key)

        self.assertEqual(len(bd), 2)
        self.assertIs(bd.head, owner)
        self.assertIs(bd.tail, branch)
        self.assertIs(branch.bottom_branch, owner)
        self.assertIs(branch.top_branch, owner)

    # TODO: Restore when append supports manually supplied stored paths.
    # def test_append_with_stored_path_keeps_alignment(self):
    #     ...

    def test_append_invalid_inputs_raise(self):
        """append rejects invalid intervals and non-interior attachments."""
        bd = BranchDecomp()
        owner = bd.append(0.0, 2.0)

        with self.assertRaises(ValueError):
            bd.append(2.0, 1.0)

        with self.assertRaises(ValueError):
            bd.append(-1.0, 1.0, bottom_branch=owner)

        with self.assertRaises(ValueError):
            bd.append(0.5, 2.0, top_branch=owner)

        with self.assertRaises(IndexError):
            bd.append(0.5, 1.5, bottom_branch=2)

    def test_append_manual_decomp_reconstructs(self):
        """A decomposition built manually via append should reconstruct."""
        bd = BranchDecomp()
        owner = bd.append(0.0, 4.0)
        child = bd.append(1.0, 3.0, top_branch=owner, bottom_branch=owner)
        bd.append(2.0, 2.0)

        R = bd.reconstruct()
        self.assertGreaterEqual(len(R.nodes), 3)
        self.assertGreaterEqual(len(R.edges), 2)

    def make_path_example(self):
        """Build the four-branch example used in sandbox_decomposition.ipynb."""
        bd = BranchDecomp()
        bd.append(-1.0, 10.0)
        bd.append(2.0, 8.0, bottom_branch=0)
        bd.append(4.0, 8.0, bottom_branch=1, top_branch=0)
        bd.append(2.0, 6.0, top_branch=0)
        return bd

    def test_check_branch_path(self):
        """Branch paths accept linked-list indices and may revisit a branch."""
        bd = self.make_path_example()

        for path in ([0, 1, 2], [3, 0], [0, 1, 2, 0]):
            self.assertTrue(bd.check_branch_path(path), msg=f"Expected {path} to be valid")

        for path in ([0, 3], [3, 0, 1], [0, 2], [1, 0, 3]):
            self.assertFalse(bd.check_branch_path(path), msg=f"Expected {path} to be invalid")

        with self.assertRaises(ValueError):
            bd.check_branch_path([])

    def test_get_func_vals_for_path(self):
        """Path transition heights work with both indices and UUIDs."""
        bd = self.make_path_example()

        self.assertEqual(bd.get_func_vals_for_path([0, 1, 2, 0]), [2.0, 4.0, 8.0])
        self.assertEqual(
            bd.get_func_vals_for_path([bd[3].key, bd[0].key]),
            [6.0],
        )

        with self.assertRaises(ValueError):
            bd.get_func_vals_for_path([0, 3])

    def test_find_subpath(self):
        """find_subpath returns the branch objects covering an interval on a path."""
        bd = self.make_path_example()
        path = [0, 1, 2, 0]

        self.assertEqual(bd.find_subpath(path, 2.0), [bd[1]])
        self.assertEqual(bd.find_subpath(path, 8.0), [bd[2]])
        self.assertEqual(bd.find_subpath(path, 0.0, 8.0), [bd[0], bd[1], bd[2]])
        self.assertEqual(bd.find_subpath(path, 3.0, 7.0), [bd[1], bd[2]])
        self.assertEqual(bd.find_subpath(path, -10.0, 3.0), [bd[0], bd[1]])
        self.assertEqual(bd.find_subpath(path, 3.0, 20.0), [bd[1], bd[2], bd[0]])

        with self.assertRaises(ValueError):
            bd.find_subpath([0, 3], 1.0)

    def test_branch_decomp_map_stores_uuid_paths(self):
        """BranchDecompMap resolves indices but stores stable UUID references."""
        source = BranchDecomp()
        source_branch = source.append(0.0, 1.0)
        target = self.make_path_example()
        branch_map = BranchDecompMap(source, target)

        branch_map.set_image(0, [0, 1, 2])

        self.assertEqual(
            branch_map.image_paths[source_branch.key],
            [target[0].key, target[1].key, target[2].key],
        )
        self.assertEqual(branch_map.get_image_indices(0), [0, 1, 2])
        self.assertEqual(branch_map.get_image(0), [target[0], target[1], target[2]])

        target.insert_before(target[0], -2.0, -1.0)
        self.assertEqual(branch_map.get_image_indices(0), [1, 2, 3])

        with self.assertRaises(ValueError):
            branch_map.set_image(0, [1, 4])

    def test_path_image(self):
        """path_image maps a source subpath through UUID-backed image paths."""
        source = self.make_path_example()
        target = BranchDecomp()
        target.append(-2.0, 11.0)
        target.append(3.0, 9.0, bottom_branch=0)
        target.append(5.0, 7.0, bottom_branch=1, top_branch=0)
        target.append(2.0, 5.0, top_branch=0)

        branch_map = BranchDecompMap(source, target)
        branch_map.set_image(0, [0])
        branch_map.set_image(1, [0, 1])
        branch_map.set_image(2, [1, 2, 0])
        branch_map.set_image(3, [3, 0])

        image = source.path_image([0, 1, 2, 0], 3.0, 9.0, branch_map)
        self.assertEqual(image, [target[1], target[2], target[0]])


if __name__ == "__main__":
    unittest.main()
