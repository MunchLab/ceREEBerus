import unittest
from cereeberus import MapperGraph, computeMapper, cover

class TestReebClass(unittest.TestCase):
    def test_cover(self):
        #Checks to see if the cover generating function is working
        examplecover = [(-1.125, -0.375), (-0.625, 0.125), (-0.125, 0.625), (0.375, 1.125)]
        testcover = cover(min=-1, max=1, numcovers=4, percentoverlap=.5)
        self.assertEqual(examplecover, testcover)


if __name__ == '__main__':
    unittest.main()