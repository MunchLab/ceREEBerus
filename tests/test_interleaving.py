import os
import unittest
from unittest.mock import patch

import numpy as np
from cereeberus.data.ex_mappergraphs import line, torus
from cereeberus.distance.ilp import select_pulp_solver
from cereeberus.distance.labeled_blocks import LabeledBlockMatrix as LBM
from cereeberus.distance.labeled_blocks import LabeledMatrix as LM

from cereeberus import Assignment, Interleave, MapperGraph


class TestInterleaving(unittest.TestCase):
    def test_torus_line(self):
        T = torus(0, 2, 10, 12, delta = 1, seed = 17)
        L = line(0, 12)

        myAssgn = Assignment(T, L, n=1, initialize_random_maps=True)

        # Make sure all the matrices have the right row and column labels 

        #--- B ---# 

        for H, graph in [(myAssgn.F, 'F'), (myAssgn.G, 'G')]: 
            for key in ['0', 'n']:
                mygraph = H(key)

                rows_ = mygraph.sorted_vertices()
                cols_ = mygraph.sorted_edges()

                B_mat = myAssgn.B(graph,key)
                self.assertEqual(rows_, B_mat.rows)
                self.assertEqual(cols_, B_mat.cols)

                # Check all columns sum to 2 
                self.assertTrue(np.all(B_mat.col_sum()[[2]]))
                
        #--- I ---# 

        for H, graph in [(myAssgn.F, 'F'), (myAssgn.G, 'G')]: 
            for key in ['0', 'n']:
                mygraph = H(key)

                if key == '0':
                    nextkey = 'n'
                elif key == 'n':
                    nextkey = '2n'

                mynextgraph = H(nextkey)

                # vertex version 
                myI = myAssgn.I(graph, key, 'V')
                cols_ = mygraph.sorted_vertices()
                rows_ = mynextgraph.sorted_vertices()
                self.assertEqual(cols_, myI.get_all_cols())
                self.assertEqual(rows_, myI.get_all_rows())

                # Check all columns sum to 1
                self.assertTrue(np.all(myI.col_sum()[[1]]))

                # edge version
                myI = myAssgn.I(graph, key, 'E')
                cols_ = mygraph.sorted_edges()
                rows_ = mynextgraph.sorted_edges()
                self.assertEqual(cols_, myI.get_all_cols())
                self.assertEqual(rows_, myI.get_all_rows())

                # Check all columns sum to 1
                self.assertTrue(np.all(myI.col_sum()[[1]]))

        #--- Phi ---# 

        # Checks that the intial setting of Phi is correct

        for key, nextkey in [('0','n'), ('n', '2n')]:
            for obj_type in ['V', 'E']:
                myphi = myAssgn.phi(key, obj_type)
                cols = myAssgn.F(key).sorted_vertices() if obj_type == 'V' else myAssgn.F(key).sorted_edges()
                rows = myAssgn.G(nextkey).sorted_vertices() if obj_type == 'V' else myAssgn.G(nextkey).sorted_edges()
                # print(f'key: {key}, nextkey: {nextkey}, obj_type: {obj_type}')
                self.assertEqual(rows, myphi.get_all_rows())
                self.assertEqual(cols, myphi.get_all_cols())

                # Check all columns sum to 1
                self.assertTrue( np.all(myphi.col_sum()== 1)) 


        #--- Psi ---#

        # Checks that the intial setting of Psi is correct

        for key, nextkey in [('0','n'), ('n', '2n')]:
            for obj_type in ['V', 'E']:
                mypsi = myAssgn.psi(key, obj_type)
                cols = myAssgn.G(key).sorted_vertices() if obj_type == 'V' else myAssgn.G(key).sorted_edges()
                rows = myAssgn.F(nextkey).sorted_vertices() if obj_type == 'V' else myAssgn.F(nextkey).sorted_edges()
                # print(f'key: {key}, nextkey: {nextkey}, obj_type: {obj_type}')
                self.assertEqual(rows, mypsi.get_all_rows())
                self.assertEqual(cols, mypsi.get_all_cols())

                # Check all columns sum to 1
                self.assertTrue( np.all(mypsi.col_sum()== 1))

        
        # Check that the non-random setting of the phi_n and psi_n maps works 
        # If this is true, then those diagrams already commute. 

        myAssgn.set_random_assignment(random_n = False)

        for maptype in ['phi', 'psi']:
            for obj_type in ['V', 'E']:
                P = myAssgn.parallelogram_matrix(maptype, obj_type)
                self.assertTrue(np.all(P.col_sum() == 0))

        # check that the optimize function works and has the same output value as recomputing internally
        result = myAssgn.optimize()
        self.assertFalse(result) 

        newmyAssgn = Assignment(T, L, n=2, initialize_random_maps=True)
        new_result = newmyAssgn.optimize()
        self.assertTrue(new_result)

                
        # Check that the dist_optimize function works and has the same output value as recomputing internally 
        
        loss_out = myAssgn.dist_optimize()
        loss_in = myAssgn.loss()
        self.assertEqual(loss_out, loss_in)

    def test_edgeless_level_boundary_blocks(self):
        """Regression test: levels with vertices but no edges should still have boundary blocks."""
        F = MapperGraph()
        F.add_node("f0", 0)
        F.add_node("f1", 1)
        F.add_node("f2", 2)
        F.add_edge("f0", "f1")

        G = MapperGraph()
        G.add_node("g0", 0)
        G.add_node("g1", 1)
        G.add_node("g2", 2)
        G.add_edge("g0", "g1")

        myAssgn = Assignment(F, G, n=1, initialize_random_maps=True)

        # Level 1 has vertices but no outgoing edges, so boundary blocks should exist with zero columns.
        self.assertIn(1, myAssgn.B_up("F", "0").get_all_block_indices())
        self.assertIn(1, myAssgn.B_down("F", "0").get_all_block_indices())
        self.assertEqual(myAssgn.B_up("F", "0")[1].get_array().shape[1], 0)
        self.assertEqual(myAssgn.B_down("F", "0")[1].get_array().shape[1], 0)

        # This used to raise KeyError when solve_ilp accessed missing boundary blocks.
        myAssgn.optimize()

    @patch("cereeberus.distance.ilp.shutil.which", return_value="/custom/cbc")
    @patch("cereeberus.distance.ilp.pulp.COIN_CMD")
    def test_select_pulp_solver_prefers_custom_cbc_path(self, mock_coin_cmd, _mock_which):
        mock_coin_cmd.return_value = "coin-solver"

        with patch.dict(
            os.environ,
            {"CEREEBERUS_SOLVER": "CBC", "CEREEBERUS_CBC_PATH": "/custom/cbc"},
            clear=True,
        ):
            solver = select_pulp_solver()

        self.assertEqual(solver, "coin-solver")
        mock_coin_cmd.assert_called_once_with(msg=0, path="/custom/cbc")

    @patch("cereeberus.distance.ilp.pulp.GLPK_CMD")
    def test_select_pulp_solver_explicit_argument_overrides_env(self, mock_glpk_cmd):
        mock_glpk_cmd.return_value = "glpk-solver"

        with patch.dict(
            os.environ,
            {"CEREEBERUS_SOLVER": "CBC", "PULP_SOLVER": "CBC"},
            clear=True,
        ):
            solver = select_pulp_solver("GLPK")

        self.assertEqual(solver, "glpk-solver")
        mock_glpk_cmd.assert_called_once_with(msg=0)

    @patch("cereeberus.distance.ilp.shutil.which", return_value=None)
    @patch("cereeberus.distance.ilp.pulp.GLPK_CMD")
    @patch("cereeberus.distance.ilp.pulp.listSolvers", return_value=["GLPK_CMD"])
    def test_select_pulp_solver_falls_back_to_available_solver(
        self, mock_list_solvers, mock_glpk_cmd, _mock_which
    ):
        mock_glpk_cmd.return_value = "fallback-glpk"

        with patch.dict(os.environ, {}, clear=True):
            solver = select_pulp_solver()

        self.assertEqual(solver, "fallback-glpk")
        mock_list_solvers.assert_called_once_with(onlyAvailable=True)
        mock_glpk_cmd.assert_called_once_with(msg=0)

    @patch("cereeberus.distance.ilp.shutil.which", return_value=None)
    @patch("cereeberus.distance.ilp.pulp.listSolvers", return_value=[])
    def test_select_pulp_solver_uses_coin_fallback_when_no_solver_is_available(
        self, mock_list_solvers, _mock_which
    ):
        with patch.dict(os.environ, {}, clear=True):
            with patch("cereeberus.distance.ilp.pulp.COIN_CMD", return_value="coin-fallback") as mock_coin_cmd:
                solver = select_pulp_solver()

        self.assertEqual(solver, "coin-fallback")
        mock_list_solvers.assert_called_once_with(onlyAvailable=True)
        mock_coin_cmd.assert_called_once_with(msg=0)







