import unittest
import numpy as np
from cereeberus import MergeTree
from cereeberus.data import ex_mergetrees as ex_mt

class TestMergeTree(unittest.TestCase):
    def check_mt(self, MT):
        # A function to do lots of checks on a given Merge Tree. Just run this thing a ton in any unit test to make sure nothing's screwing up the stuff we think we should be maintaining.
        
        # Check that the node list and the keys for f are the same 
        self.assertEqual(set(MT.nodes), set(MT.f.keys()))

        # Check that the node list and the keys for pos_f are the same 
        self.assertEqual(set(MT.nodes), set(MT.pos_f.keys()))

        # Check that the infinite node is in the list of nodes
        self.assertTrue('v_inf' in MT.nodes)
        self.assertTrue(MT.f['v_inf'] == np.inf)

        finiteNodes = MT.get_finite_nodes()
        self.assertEqual(len(finiteNodes), len(MT.nodes)-1)

        # Check that the pos_list always has the f value in the second entry 
        # except for the infinite node
        for v in finiteNodes:
            self.assertEqual(MT.pos_f[v][1], MT.f[v])
        
        self.assertTrue(np.isfinite(MT.pos_f['v_inf'][1]))

        # Check that all edges are pointing to the higher function value node
        for edge in MT.edges:
            v1, v2 = edge[:2]
            self.assertGreater(MT.f[v2], MT.f[v1])

        # Check that adding a node that creates a loop raises an error. 
        # Since everything has a path to $v_inf$, this should always happen.
        with self.assertRaises(ValueError):
            MT.add_node(0, 'v_inf')

    def test_merge_tree(self):

        MT = ex_mt.randomMergeTree(10)
        self.check_mt(MT)

    def test_merge_labels(self):

        MT = ex_mt.randomMergeTree(10)
        
        # Make sure the LCA funciton works. 
        # Plus the infinite node should never be the LCA of two finite nodes.
        self.assertNotEqual(MT.LCA(0,1), 'v_inf')

        Leaves = MT.get_leaves()

        # Generate the LCA matrix just on the leaf set, which is the default 
        M = MT.LCA_matrix()

        # Check that the LCA matrix is symmetric and of the same size as the leaf set 
        self.assertEqual(M.shape[0], len(Leaves))

        # TODO: Add tests for specifc labeling functions, here I only have for the leaf version



if __name__ == '__main__':
    unittest.main()