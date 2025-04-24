import unittest
from cereeberus.distance.labeled_blocks import LabeledMatrix as LM 
from cereeberus.distance.labeled_blocks import LabeledBlockMatrix as LBM
import numpy as np

class TestBlockMatrixClasses(unittest.TestCase):

    def test_check_labeled_matrix(self):
        # 2 x 3
        Mat1 = LM([[0, 1, 2], [-3, -4, -5]], ['u', 'v'], ['a', 'b', 'c'])

        # 3 x 3
        Mat2 = LM([[0, 1], [1, 0], [0,0]], ['a', 'b', 'c'], ['x', 'y', 'z'])

        # 2 x 2
        Mat4 = LM([[0, 1], [1, 0]], ['a', 'b'], ['x', 'y'])

        # 3 x 3
        Mat5 =LM([[0, 1], [1, 0], [3,-2]], ['a', 'c', 'b'], ['x', 'y', 'z']) 

        # 2 x 2 
        Mat6 = LM([[0, 1], [1, 0]], ['u', 'v'], ['x', 'y'])
        Mat7 = LM([[0, 1], [1, 0]], ['a','b'], ['u', 'v'])

        # Check that the matrix is the right size
        self.assertEqual(Mat1.shape(), (2, 3))
        self.assertEqual(Mat1.size(), 6)
        self.assertEqual(Mat2.shape(), (3, 2))
        self.assertEqual(Mat2.size(), 6)

        # Check that we can multiply the matrices and the row and column labels are propogated correctly
        Mat3 = Mat1 @ Mat2
        self.assertEqual(Mat3.shape(), (Mat1.shape()[0], Mat2.shape()[1]))
        self.assertEqual(Mat3.rows, Mat1.rows)
        self.assertEqual(Mat3.cols, Mat2.cols)

        with self.assertRaises(ValueError):
            # Make sure an error is raised if we try to multiply two matrices of incompatible size. 
            Mat1 @ Mat4

            # Mat1 and Mat 5 have compatable sizes, but the row labels are not in the same order. This should raise an error.
            Mat1 @ Mat5

        # Check that we can ask for a matrix entry by label 
        self.assertEqual(Mat1['v', 'b'], -4)
        # and we can set a matrix entry by label
        Mat1['v', 'b'] = -100

        # We can add matrices if they have the same labeles 
        Mat1 + Mat1

        # We can't add matrices if they have different labels even if they have the same size 
        with self.assertRaises(ValueError):
            Mat1 + Mat6
            Mat1 + Mat7 

        # We have cases where we want to create a matrix that ends up being empty. 
        EmptyMat = LM(rows = ['a', 'b'], cols = [])
        self.assertEqual(EmptyMat.shape(), (2, 0))
        self.assertEqual(EmptyMat.size(), 0)
        EmptyMat2 = LM()
        self.assertEqual(EmptyMat2.shape(), (0, 0))
        self.assertEqual(EmptyMat2.size(), 0)

        # Check transpose function 
        Mat7 = Mat1.T()
        self.assertEqual(Mat7.shape(), (3, 2))
        self.assertEqual(Mat7.rows, Mat1.cols)
        self.assertEqual(Mat7.cols, Mat1.rows)

        # Column sum check 
        self.assertEqual(len(Mat1.col_sum()), Mat1.shape()[1])
        self.assertTrue(Mat2.is_col_sum_1())

        # Max and abs max check 
        self.assertEqual(Mat1.max(), 2)
        self.assertEqual(Mat1.absmax(), 100)
        self.assertTrue(np.isnan(EmptyMat.max()))
        self.assertTrue(np.isnan(EmptyMat.max()))
        # print('got here')

    def test_block_labeled_matrix(self):
        # Make the example from the front of the documentation 
        cols_dict = {1: ['a','b'], 2: ['c','d','e']}
        rows_dict = {0: ['u'], 1: ['v','w'], 2: ['x']}
        map_dict = {'a': 'v', 'b':'v', 'c': 'x', 'd':'x', 'e':'x'}
        lbm = LBM(map_dict, rows_dict, cols_dict)

        # A second example with rows and columns switched
        cols_dict = {0: ['u'], 1: ['v','w'], 2: ['x']}
        rows_dict = {0: ['x','y'], 1: ['z','a','b'], 2: ['c'], 3: ['d','e']}
        map_dict = {'u': 'y', 'v':'z', 'w': 'b', 'x':'c'}
        lbm2 = LBM(map_dict, rows_dict, cols_dict)


        # Check that the block matrix is the right size
        self.assertEqual(lbm.shape(), (4,5))

        # Check that the max and min functions work correctly 
        self.assertEqual(lbm.max(), 1)
        self.assertEqual(lbm.min(), 0)

        # Make sure we can take a transpose 
        lbm2T = lbm2.T()

        # Make sure we can add 
        lbm + lbm

        # Make sure we can't add if the labels are different
        with self.assertRaises(ValueError):
            lbm + lbm2

        # Make sure we can multiply, and that nothing freaks out if we have 
        # both a block matrix and a labeled matrix in the multiplication 
        lbm @ lbm.T() # Both block matrices

        bm = lbm.to_labeled_matrix()

        bm @ lbm.T() # Block matrix @ labeled matrix

        lbm @ bm.T() # Labeled matrix @ block matrix
