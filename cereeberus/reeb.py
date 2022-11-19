import networkx as nx
import numpy as np
from compute import degree

class Reeb:
    """ Class for Reeb Graph

    :ivar G: Graph: G
    :ivar fx: function values associated with G
    :ivar pos: spring layout position calculated from G
    :ivar pos_fx: position values corresponding to x = fx and y = y value from pos
    """

    def __init__(self, G, fx = {}):
        #Convert to MultiGraph to allow for Parallel Edges and Self-Loops
        if type(G) != 'networkx.classes.multigraph.MultiGraph':
            self.G = nx.MultiGraph(G)
        else:
            self.G = G
        if fx == {}:
            self.fx = nx.get_node_attributes(self.G,'fx')
        else:
            self.fx = fx
        if self.fx == {}:
            raise AttributeError("No function values provided - please provide a function value for each node or update your graph to have the 'fx' attribute")
        self.pos = nx.spring_layout(self.G)
        self.pos_fx = {}
        for i in range(0,len(self.pos)):
            self.pos_fx[i] = (self.fx[i], self.pos[i][1])

        self.nodes = G.nodes
        self.edges = G.edges

        # compute upper and lower degree of reeb graph
        self.up_deg = degree.up_degree(G, self.fx)
        self.down_deg = degree.down_degree(G, self.fx)

        # adjacency matrix
        self.adjacency = nx.adjacency_matrix(G)

        # show basic properties of reeb graph
        self.summary = {'nodes': len(self.nodes), 'edges': len(self.edges)}

    def plot_reeb(self, position = {}):
        """ Plot a Reeb Graph given a graph with a position
        
        """
        if position == {}:
            pos = self.pos_fx
        else:
            pos = position
        nx.draw(self.G, pos = pos)
    


