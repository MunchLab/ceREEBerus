import unittest
from cereeberus import ReebGraph, MapperGraph
from cereeberus.data import ex_graphs as ex_g
from cereeberus.data import ex_reebgraphs as ex_rg
from cereeberus.data import ex_mappergraphs as ex_mg
import numpy as np

class TestMapperClass(unittest.TestCase):

    def check_mapper(self, MG):
        # A function to do lots of checks on a given Mapper graph. Just run this thing a ton in any unit test to make sure nothing's screwing up the stuff we think we should be maintaining.
        # This is largely the same as the Reeb graph check, but with some additional checks for the Mapper graph.

        # Check that all funciton values are integers
        self.assertTrue(all([isinstance(MG.f[v], int) for v in MG.nodes]))

        # Check that all edges have adjacent function values
        for edge in MG.edges:
            v1, v2 = edge[:2]
            print (v1, v2)
            print(MG.f[v1], MG.f[v2])
            self.assertTrue(MG.f[v2] - MG.f[v1] == 1)

        # Check that the node list and the keys for f are the same 
        self.assertEqual(set(MG.nodes), set(MG.f.keys()))

        # Check that the node list and the keys for pos_f are the same 
        self.assertEqual(set(MG.nodes), set(MG.pos_f.keys()))

        # Check that the pos_list always has delta*f-value in the second entry 
        for v in MG.nodes:
            self.assertEqual(MG.pos_f[v][1], MG.delta*MG.f[v])

        # Check that all edges are pointing to the higher function value node
        for edge in MG.edges:
            v1, v2 = edge[:2]
            self.assertGreater(MG.f[v2], MG.f[v1])

    def test_mapper_load(self):
        # This test makes sure you can pass in any of the example graph types and convert it to a Reeb graph.
        for ex_graph in [ex_mg.simple_loops, ex_mg.torus, ex_mg.dancing_man, ex_mg.juggling_man]:
            MG = ex_graph()

            self.check_mapper(MG)

    
    def test_mapper_add_nodes(self):
        # This test makes sure you can add nodes to a Reeb graph.
        R = ex_mg.simple_loops()
        self.assertRaises(ValueError, R.add_node, 0, 0.5)
        R.add_node('w', 5)
        self.check_mapper(R)

        # adding a bunch of nodes 
        v_list = [i for i in range(5, 10)]
        input_f = [6,7,8, 9,10,]
        input_f_dict = {v : f for v, f in zip(v_list, input_f)}
        R.add_nodes_from(v_list, input_f_dict)

        S = {'nodes': 10, 'edges': 5}
        self.assertEqual(R.summary(), S)

        # General check of the mapper graph
        self.check_mapper(R)


    def test_add_edge(self):
        # This test makes sure you can add an edge to a Reeb graph.
        R = ex_mg.juggling_man()

        # Ok, now this should work 
        # Add an edge between vertices that don't already have an edge
        n = len(R.nodes)
        m = len(R.edges)
        R.add_edge(0, 7)

        # Add an edge between vertices that already have an edge
        n = len(R.nodes)
        m = len(R.edges)
        R.add_edge(0, 1)

        # Add an edge between vertices that are at the same function value. Should merge the vertices into one
        n = len(R.nodes)
        m = len(R.edges)
        R.add_edge(1,4)

        # Check that you can add a bunch of edges
        # These include edges already there, so should get a multi-edge 
        edge_list = [(7,5), (5,10), (5,10), (1,9), (0,9), (7,0)]
        n = len(R.nodes)
        m = len(R.edges)
        R.add_edges_from(edge_list)

        # General check of the mapper graph
        self.check_mapper(R)

    
    def test_induced_subgraph(self):
        # This test makes sure you can get an induced subgraph from a Reeb graph.
        R = ex_mg.juggling_man()
        v_list = [7,6,3,10,9,1,0]
        H = R.induced_subgraph(v_list)
        self.assertEqual(len(H.nodes), len(v_list))
        self.assertIsInstance(H, ReebGraph)
        self.check_mapper(H)
    
    def test_slice(self):
        # This test makes sure you can slice a Reeb graph.
        R = ex_mg.juggling_man()
        R.add_edge(7,9)


        # Example chosen so that we have vertices with value on the endpoints (we're assuming open interval so shouldn't be included)
        # We also have at least one edge that completely crosses the interval in question
        H = R.slice( 2,6)

        self.assertEqual(H.number_connected_components(),3 )

    def test_dist_matrix(self):
        # This test makes sure you can get the distance matrix from the mapper graph.
        R = ex_mg.juggling_man()
        M = R.thickening_distance_by_level(4)
        self.assertEqual(M[5,6], 1)
        self.assertEqual(M[5,5], 0)

        # These should have infinity entries since there are disconnected components at that level
        M = R.thickening_distance_by_level(5)
        self.assertEqual(M[2,10], np.inf)
        self.assertEqual(M[2,3], 1)

        # Check the whole put together matrix
        M = R.thickening_distance_matrix()
        self.assertEqual(M[5][3,10], np.inf)
       


if __name__ == '__main__':
    unittest.main()