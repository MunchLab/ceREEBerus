import unittest
from cereeberus import ReebGraph
from cereeberus.data import ex_graphs as ex_g
from cereeberus.data import ex_reebgraphs as ex_rg

class TestReebClass(unittest.TestCase):

    def check_reeb(self, R):
        # A function to do lots of checks on a given Reeb graph. Just run this thing a ton in any unit test to make sure nothing's screwing up the stuff we think we should be maintaining.

        # Check that the node list and the keys for f are the same 
        self.assertEqual(set(R.nodes), set(R.f.keys()))

        # Check that the node list and the keys for pos_f are the same 
        self.assertEqual(set(R.nodes), set(R.pos_f.keys()))

        # Check that the pos_list always has the f value in the second entry 
        for v in R.nodes:
            self.assertEqual(R.pos_f[v][1], R.f[v])

        # Check that all edges are pointing to the higher function value node
        for edge in R.edges:
            v1, v2 = edge[:2]
            self.assertGreater(R.f[v2], R.f[v1])


    def test_reeb_load(self):
        # This test makes sure you can pass in any of the example graph types and convert it to a Reeb graph.
        example_graphs = []
        for ex_graph in [ex_g.simple_loops, ex_g.torus_graph, ex_g.dancing_man, ex_g.juggling_man]:
            G = ex_graph()
            R = ReebGraph(G)
            example_graphs.append(R)
       
            S = {'nodes': len( G.nodes), 'edges': len(G.edges)}
            self.assertEqual(R.summary(), S)

            self.check_reeb(R)
    
    def test_reeb_add_nodes(self):
        # This test makes sure you can add nodes to a Reeb graph.
        R = ex_rg.simple_loops()
        # Test adding a node without specificying the name 
        R.add_node(None, 3)

        self.assertRaises(ValueError, R.add_node, 0, 0.5)
        R.add_node('chicken', 0.5)
        S = {'nodes': 6, 'edges': 5}
        self.assertEqual(R.summary(), S)
        self.check_reeb(R)

        # adding a bunch of nodes 
        v_list = [i for i in range(5, 10)]
        input_f = [0.5, 0.6, 0.7, 0.8, 0.9]
        input_f_dict = {v : f for v, f in zip(v_list, input_f)}
        R.add_nodes_from(v_list, input_f_dict)

        S = {'nodes': 11, 'edges': 5}
        self.assertEqual(R.summary(), S)


        # General check of the Reeb graph
        self.check_reeb(R)


    def test_add_edge(self):
        # This test makes sure you can add an edge to a Reeb graph.
        R = ex_rg.juggling_man()

        # first in, second not
        self.assertRaises(ValueError, R.add_edge, 0, 15)
        # First not in, second in
        self.assertRaises(ValueError, R.add_edge, 15, 0)
        # Neither in the graph
        self.assertRaises(ValueError, R.add_edge, 15, 'chicken')

        # Ok, now this should work 
        # Add an edge between vertices that don't already have an edge
        n = len(R.nodes)
        m = len(R.edges)
        R.add_edge(0, 7)
        S = {'nodes': n, 'edges': m+1}
        self.assertEqual(R.summary(), S)

        # Add an edge between vertices that already have an edge
        n = len(R.nodes)
        m = len(R.edges)
        R.add_edge(0, 1)
        S = {'nodes': n, 'edges': m+1}
        self.assertEqual(R.summary(), S)

        # Add an edge between vertices that are at the same function value. Should merge the vertices into one
        n = len(R.nodes)
        m = len(R.edges)
        R.add_edge(1,4)
        S = {'nodes': n-1, 'edges': m}
        self.assertEqual(R.summary(), S)


        # Check that you can add a bunch of edges
        # These include edges already there, so should get a multi-edge 
        edge_list = [(7,5), (5,10), (5,10), (1,9), (0,9), (7,0)]
        n = len(R.nodes)
        m = len(R.edges)
        R.add_edges_from(edge_list)
        S = {'nodes': n, 'edges': m+6}
        self.assertEqual(R.summary(), S)


        # General check of the Reeb graph
        self.check_reeb(R)

    def test_subdivide_edge(self):
        # This test makes sure you can subdivide an edge in a Reeb graph.
        R = ex_rg.juggling_man()

        # Pass in an existing edge but the function value isn't right 
        self.assertRaises(ValueError, R.subdivide_edge, 0, 1,'chicken', 0.5)

        # Pass in an edge where vertices exist but the edge doesn't
        self.assertRaises(ValueError, R.subdivide_edge, 1,8,'chicken', 0.5)

        # Ok, now actually do it

        n = len(R.nodes)
        m = len(R.edges)
        R.subdivide_edge(6, 7, 'new_v', 2.5)
        # Nodes and edges should be each increased by exactly one 
        S = {'nodes': n+1, 'edges': m+1}
        self.assertEqual(S, R.summary())

        # General check of the Reeb graph
        self.check_reeb(R)

    def test_remove_regular_vertx(self):
        # This test makes sure you can remove a node from a Reeb graph.
        R = ex_rg.torus(multigraph = False)

        # Error if passing in a vertex that isn't already there
        self.assertRaises(ValueError, R.remove_regular_vertex, 'chicken')

        # Error if passing in a vertex but its not regular
        self.assertRaises(ValueError, R.remove_regular_vertex, 'a')
        self.assertRaises(ValueError, R.remove_regular_vertex, 'b')

        # Ok, now actually do it
        n = len(R.nodes)
        m = len(R.edges)
        R.remove_regular_vertex('e')
        S = {'nodes': n-1, 'edges': m-1}
        self.assertEqual(S, R.summary())

        # General check of the Reeb graph
        self.check_reeb(R)

        # One more time with all the regular vertices
        R = ex_rg.torus(multigraph = False)
        n = len(R.nodes)
        m = len(R.edges)
        R.remove_all_regular_vertices()
        S = {'nodes': n-2, 'edges': n-2}
        self.assertEqual(S, R.summary())

        # General check of the Reeb graph
        self.check_reeb(R)

    def test_remove_isolates(self):
        # This test makes sure you can remove isolated nodes from a Reeb graph.
        R = ex_rg.juggling_man()
        R.remove_isolates()

        list_isolates = []
        for v in R.nodes:
            if R.up_degree(v) == 0 and R.down_degree(v) == 0:
                list_isolates.append(v)
        self.assertEqual(len(list_isolates), 0)

        # General check of the Reeb graph
        self.check_reeb(R)
    
    def test_induced_subgraph(self):
        # This test makes sure you can get an induced subgraph from a Reeb graph.
        R = ex_rg.juggling_man()
        v_list = [7,6,3,10,9,1,0]
        H = R.induced_subgraph(v_list)
        self.assertEqual(len(H.nodes), len(v_list))
        self.assertIsInstance(H, ReebGraph)
        self.check_reeb(H)
    
    def test_slice(self):
        # This test makes sure you can slice a Reeb graph.
        R = ex_rg.juggling_man()
        R.add_edge(7,9)


        # Example chosen so that we have vertices with value on the endpoints (we're assuming open interval so shouldn't be included)
        # We also have at least one edge that completely crosses the interval in question
        H = R.slice( 2,5)

        self.assertEqual(H.number_connected_components(),3 )

        # Example chosen so that we have vertices with value on the endpoints (we're using closed interval so now these should be included)
        H = R.slice( 2,5, type = 'closed')

        self.assertEqual(H.number_connected_components(),2 )

    def test_smoothing(self):
        # This test makes sure you can smooth a Reeb graph.
        R = ex_rg.juggling_man()

        R_eps = R.smoothing(0.1)
        self.assertIsInstance(R_eps, ReebGraph)
        self.check_reeb(R_eps)

        # This makes sure that the smoothing behaves with multiedges 
        R = ex_rg.torus()
        R_eps = R.smoothing(0.1)
        # The Euler characteristic should be 0 for a small smoothing
        self.assertEqual(len(R_eps.nodes) - len(R_eps.edges), 0)

    def test_matrices(self):
        # This test makes sure you can get the adjacency matrix and boundary matrix of a Reeb graph.
        R = ex_rg.juggling_man()
        A = R.adjacency_matrix()
        self.assertEqual(A.shape, (len(R.nodes), len(R.nodes)))

        B = R.boundary_matrix()
        self.assertEqual(B.shape, (len(R.nodes), len(R.edges)))




       


if __name__ == '__main__':
    unittest.main()