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
        
    def test_torus_example_class(self):
        # Test the torus example from the documentation.
        T = Torus()
        T.generate_grid(grid_size = 4)
        T.assign_random_values(0,100, seed=1986)
        
        R = computeReeb(T)
        
        self.assertIsInstance(R, ReebGraph)
        self.assertGreater(len(R.nodes), 0)
        self.assertGreater(len(R.edges), 0)