import unittest
from cereeberus.data.ex_mappergraphs import torus, line
from cereeberus.distance.labeled_blocks import LabeledMatrix as LM 
from cereeberus.distance.labeled_blocks import LabeledBlockMatrix as LBM
import numpy as np
from cereeberus import Interleave 

class TestInterleaving(unittest.TestCase):
    def test_torus_line(self):
        T = torus(0, 2, 10, 12, delta = 1, seed = 17)
        L = line(0, 12)

        I = Interleave(T, L)