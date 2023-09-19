import unittest
import cereeberus.data.reeb as reeb
import cereeberus.compute.degree as degree

class TestNodeOps(unittest.TestCase):
    def test_minimal(self):
        R = reeb.torus()
        R_min = degree.minimal_reeb(R)
        nodes = [0, 1, 2, 3]
        edges = [(0, 1, 0), (1, 2, 0), (1, 2, 1), (2, 3, 0)]
        self.assertEqual(list(R_min.nodes), nodes)
        self.assertEqual(list(R_min.edges), edges)
        
    def test_add_nodes(self):
        R = reeb.torus()
        R_add = degree.add_nodes(R, fx=3.5, x = 1)
        R_add = degree.add_nodes(R_add, fx = 1.5, x = 1)
        nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        edges = [(0, 8, 0), (1, 2, 0), (1, 3, 0), (1, 8, 0), (2, 6, 0), (3, 7, 0), (4, 5, 0), (4, 6, 0), (4, 7, 0)]
        self.assertEqual(list(R_add.nodes), nodes)
        self.assertEqual(list(R_add.edges), edges)

if __name__ == '__main__':
    unittest.main()