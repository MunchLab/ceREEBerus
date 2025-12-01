import unittest
from cereeberus import MapperGraph, computeMapper, cover
import networkx as nx
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN

class TestReebClass(unittest.TestCase):
    def test_cover(self):
        #Checks to see if the cover generating function is working
        examplecover = [(-1.125, -0.375), (-0.625, 0.125), (-0.125, 0.625), (0.375, 1.125)]
        testcover = cover(min=-1, max=1, numcovers=4, percentoverlap=.5)
        self.assertEqual(examplecover, testcover)

    def test_computeMapper_trivial(self):
        #checking the simple two point graph (mostly to check if the trivial clustering function works, it has been the problem child of this whole project)
        examplegraph1 = MapperGraph()
        examplegraph1.add_node(0,0)
        examplegraph1.add_node(1,1)
        examplegraph1.add_node(2,2)
        examplegraph1.add_edge(0,1)
        testgraph1 = computeMapper([(0.6, 0), (-0.1, 0.5)], (lambda a : a[0]), [(-1,0),(-0.5,0.5),(0,1)], "trivial")
        check = nx.utils.graphs_equal(examplegraph1, testgraph1)
        self.assertEqual(check, True)
        
    def test_computeMapper_nontrivial(self):
        #checking the default graph for the compute_mapper notebook to see if it remains the same
        data, labels = make_circles(n_samples=500, factor=0.4, noise=0.05, random_state=0)
        testgraph2 = computeMapper(data, (lambda a : a[0]), cover(min=-1, max=1, numcovers=7, percentoverlap=.5), DBSCAN(min_samples=2,eps=0.3).fit)
        examplegraph2 = MapperGraph()
        examplegraph2.add_node(0,0)
        examplegraph2.add_node(1,1)
        examplegraph2.add_node(2,1)
        examplegraph2.add_node(3,1)
        examplegraph2.add_node(4,2)
        examplegraph2.add_node(5,2)
        examplegraph2.add_node(6,2)
        examplegraph2.add_node(7,3)
        examplegraph2.add_node(8,3)
        examplegraph2.add_node(9,3)
        examplegraph2.add_node(10,3)
        examplegraph2.add_node(11,4)
        examplegraph2.add_node(12,4)
        examplegraph2.add_node(13,4)
        examplegraph2.add_node(14,5)
        examplegraph2.add_node(15,5)
        examplegraph2.add_node(16,5)
        examplegraph2.add_node(17,6)
        examplegraph2.add_edge(0,1)
        examplegraph2.add_edge(0,2)
        examplegraph2.add_edge(3,4)
        examplegraph2.add_edge(1,6)
        examplegraph2.add_edge(2,5)
        examplegraph2.add_edge(4,7)
        examplegraph2.add_edge(4,8)
        examplegraph2.add_edge(6,10)
        examplegraph2.add_edge(5,9)
        examplegraph2.add_edge(7,11)
        examplegraph2.add_edge(8,11)
        examplegraph2.add_edge(10,13)
        examplegraph2.add_edge(9,12)
        examplegraph2.add_edge(12,16)
        examplegraph2.add_edge(13,15)
        examplegraph2.add_edge(11,14)
        examplegraph2.add_edge(16,17)
        examplegraph2.add_edge(15,17)
        check = nx.utils.graphs_equal(examplegraph2, testgraph2)
        self.assertEqual(check, True)

if __name__ == '__main__':
    unittest.main()