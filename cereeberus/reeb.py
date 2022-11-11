import networkx as nx
import numpy as np
class Reeb:
    """ Class for Reeb Graph

    :ivar G: Graph: G
    :ivar fx: function values associated with G
    :ivar pos: spring layout position calculated from G
    :ivar pos_fx: position values corresponding to x = fx and y = y value from pos
    """

    def __init__(self, G, fx = {}):
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

        # compute upper and lower degree of reeb graph
        self.up_deg = self._up_deg(self.G, self.fx)
        self.down_deg = self._down_deg(self.G, self.fx)

    def _up_deg(self, G, fx = {}):
        """ Compute Upper Degree of Reeb Graph

        Args:
            G (networx graph): networkx graph to use for reeb graph computation

        Returns:
            up_deg (dict): dictionary of up degrees by node
        
        """
        n = len(G.nodes)
        up_adj = np.zeros((n,n))

        for i in range(0,n):
            for j in range(i,n):
                if fx[i] < fx[j]:
                    e = list(G.edges(i))
                    if (i,j) in e:
                        up_adj[j,i]+=1
                if fx[i] > fx[j]:
                    e = list(G.edges(i))
                    if (i,j) in e:
                        up_adj[i,j]+=1

        d = sum(up_adj)

        up_deg = {}
        for i in range(0,n):
            up_deg[i] = int(d[i])
        return up_deg

    def _down_degree(self, G, fx ={ }):

        """ Compute Down Degree of Reeb Graph

        Args:
            G (networx graph): networkx graph to use for reeb graph computation

        Returns:
            down_deg (dict): dictionary of down degrees by node
        
        """
        n = len(G.nodes)
        down_adj = np.zeros((n,n))
    
        for i in range(0,n):
            for j in range(i,n):
                if fx[i] > fx[j]:
                    e = list(G.edges(i))
                    if (i,j) in e:
                        down_adj[j,i]+=1
                if fx[i] < fx[j]:
                    e = list(G.edges(i))
                    if (i,j) in e:
                        down_adj[i,j]+=1

        d = sum(down_adj)

        down_deg = {}
        for i in range(0,n):
            down_deg[i] = int(d[i])
        return down_deg

    def plot_reeb(self, position = {}):
        """ Plot a Reeb Graph given a graph with a position
        
        """
        if position == {}:
            pos = self.pos_fx
        else:
            pos = position
        nx.draw(self.G, pos = pos)
    


