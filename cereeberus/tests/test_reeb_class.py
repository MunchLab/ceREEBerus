import unittest
import cereeberus.reeb as Reeb
import cereeberus.data.graphs as graphs

class ReebClass(unittest.TestCase):
    def test_reeb_load(self):
        G = graphs.simple_loops()
        R = Reeb.Reeb(G)
        S = {'nodes': 4, 'edges': 5}
        self.assertEqual(R.summary, S)

if __name__ == '__main__':
    unittest.main()