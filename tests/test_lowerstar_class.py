import unittest
from cereeberus import ReebGraph, LowerStar, computeReeb
from cereeberus.data import Torus


class TestLowerStarClass(unittest.TestCase):
    
    def test_lower_star_assign_filtration(self):
        # Test that assigning filtration values to vertices correctly updates adjacent simplices.
        K = LowerStar()
        K.insert([0, 1, 2])
        K.insert([1, 3])
        K.insert([2, 3])
        
        K.assign_filtration(0, 0.0)
        self.assertEqual(K.filtration([0]), 0.0)
        self.assertEqual(K.filtration([0, 1]), 0.0)
        self.assertEqual(K.filtration([0, 2]), 0.0)
        self.assertEqual(K.filtration([0, 1, 2]), 0.0)
        
        K.assign_filtration(1, 3.0)
        self.assertEqual(K.filtration([1]), 3.0)
        self.assertEqual(K.filtration([1, 3]), 3.0)
        self.assertEqual(K.filtration([0, 1]), 3.0)  # Updated due to vertex 1
        self.assertEqual(K.filtration([0, 1, 2]), 3.0)  # Updated due to vertex 1
        
    def test_lower_star_sc_max_min_filtration(self):
        # Test that max and min filtration values are computed correctly.
        K = LowerStar()
        K.insert([0, 1])
        K.insert([1, 2])
        K.insert([2, 3])
        
        K.assign_filtration(0, 1.0)
        K.assign_filtration(1, 2.0)
        K.assign_filtration(2, 3.0)
        K.assign_filtration(3, 4.0)
        
        self.assertEqual(K.min_filtration(), 1.0)
        self.assertEqual(K.max_filtration(), 4.0)
        
    def test_computeReeb(self):
        # Test the computation of the Reeb graph from a Lower Star Simplicial Complex.
        K = LowerStar()
        K.insert([0, 1, 2])
        K.insert([1, 3])
        K.insert([2, 3])
        
        K.assign_filtration(0, 0.0)
        K.assign_filtration(1, 3.0)
        K.assign_filtration(2, 5.0)
        K.assign_filtration(3, 7.0)
        
        R = computeReeb(K)
        
        self.assertIsInstance(R, ReebGraph)
        self.assertGreater(len(R.nodes), 0)
        self.assertGreater(len(R.edges), 0)
        
    def test_computeReeb_horizontal_edge(self):
        # Regression test: when two vertices share the same filtration value and
        # are connected by an edge (a "horizontal" edge), the level-set connected
        # component is that entire edge, so the Reeb graph must collapse them to
        # a single node.  Previously this raised ValueError.
        #
        # Complex: triangle with vertices 0,1,2 and edges [0,1],[0,2],[1,2].
        # Filtration (x-coordinate): f(0)=0, f(1)=1, f(2)=0.
        # Edge [0,2] is horizontal. The Reeb graph should have exactly 1 node at
        # height 0 (not 2 separate nodes) and 2 edges leading up to vertex 1.
        K = LowerStar()
        K.insert([0, 1])
        K.insert([0, 2])
        K.insert([1, 2])

        K.assign_filtration(0, 0.0)
        K.assign_filtration(1, 1.0)
        K.assign_filtration(2, 0.0)

        R = computeReeb(K)

        self.assertIsInstance(R, ReebGraph)

        # Exactly one node at height 0 (the collapsed horizontal component)
        nodes_at_zero = [v for v in R.nodes if R.f[v] == 0.0]
        self.assertEqual(len(nodes_at_zero), 1,
                         "Horizontal edge should collapse to a single Reeb node")

        # The graph should be connected
        import networkx as nx
        self.assertTrue(nx.is_weakly_connected(R))

    def test_torus_example_class(self):
        # Test the torus example from the documentation.
        T = Torus()
        T.generate_grid(grid_size = 4)
        T.assign_random_values(0,100, seed=1986)
        
        R = computeReeb(T)
        
        self.assertIsInstance(R, ReebGraph)
        self.assertGreater(len(R.nodes), 0)
        self.assertGreater(len(R.edges), 0)