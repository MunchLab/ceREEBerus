import unittest
from cereeberus.reeb import embeddedgraph
from cereeberus.data.ex_embedgraphs import exampleEmbeddedGraph
import numpy as np


class TestEmbeddedGraph(unittest.TestCase):
    def test_example_graph(self):
        # Make sure we can build a grpah in the first place
        G = exampleEmbeddedGraph()
        self.assertEqual( len(G.nodes), 6)  # assuming my_function squares its input


    def test_add_node(self):
        # Make sure adding a vertex updates the coordiantes list 
        G = exampleEmbeddedGraph()
        G.add_node('G', 1, 2)
        self.assertEqual( len(G.nodes), 7)
        self.assertEqual( len(G.coordinates), 7)

    def test_add_edge(self):
        # Make sure adding an edge updates the edge list
        G = exampleEmbeddedGraph()
        G.add_edge('A', 'B')
        self.assertEqual( len(G.edges), 6)

    def test_get_coordinates(self):
        # Make sure we can get the coordinates of a vertex
        G = exampleEmbeddedGraph(mean_centered=False)
        coords = G.get_coordinates('A')
        self.assertEqual( coords, (1, 2))

    def test_coords_list(self):
        # Make sure the keys in the coordinates list are the same as the nodes
        G = exampleEmbeddedGraph(mean_centered=False)
        self.assertEqual( len(G.nodes), len(G.coordinates))
        self.assertEqual( set(G.nodes), set(G.coordinates.keys()))

    def test_mean_centered_coordinates(self):
        # Make sure the mean centered coordinates are correct
        G = exampleEmbeddedGraph(mean_centered=False)
        G.set_mean_centered_coordinates()
        x_coords = [x for x, y in G.coordinates.values()]

        self.assertAlmostEqual( np.average(x_coords), 0, places = 1)

    def test_reeb_graph(self):
        # Make sure we can build a reeb graph, and that it will collapse vertices
        # This example will collapse the vertices A and B. Even though F is at the same
        # function value, it shouldn't get collapsed. So we should go down by exactly 
        # one vertex and exactly one edge.
        G = exampleEmbeddedGraph()
        R = R = G.reeb_graph_from_direction(3*np.pi/4)
        self.assertEqual( len(R.nodes), len(G.nodes)-1)
        self.assertEqual( len(R.edges), len(G.edges)-1)

    

if __name__ == '__main__':
    unittest.main()