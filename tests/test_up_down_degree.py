import unittest
import cereeberus.data.reeb as reeb

class TestUpDownDegree(unittest.TestCase):
    def test_up_degree(self):
        actual_up_degree = {0: 0, 1: 1, 2: 1, 3: 2, 4: 0, 5:1, 6:2, 7:1}
        dm = reeb.dancing_man()
        test_up_degree = dm.up_deg
        self.assertEqual(actual_up_degree, test_up_degree)

    def test_down_degree(self):
        actual_down_degree = {0: 1, 1: 2, 2: 2, 3: 1, 4:1, 5:0, 6:1, 7:0}
        dm = reeb.dancing_man()
        test_down_degree = dm.down_deg
        self.assertEqual(actual_down_degree, test_down_degree)

if __name__ == '__main__':
    unittest.main()