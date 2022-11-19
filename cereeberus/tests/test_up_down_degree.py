import unittest

class TestUpDownDegree(unittest.TestCase):
    def test_up_degree(self):
        actual_up_degree = ud = {0: 0, 1: 1, 2: 1, 3: 2, 4: 0, 5:1, 6:2, 7:1}

    def test_down_degree(self):
        actual_down_degree = dd = {0: 1, 1: 2, 2: 2, 3: 1, 4:1, 5:0, 6:1, 7:0}