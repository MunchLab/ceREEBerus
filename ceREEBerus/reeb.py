import networkx as nx

class Reeb:
    """ Class for Reeb Graph
    
    """

    def __init__(self, G):
        self.G = G
        self.pos = nx.get_node_attributes(self.G,'pos')
    
    def plot_reeb(self):
        """ Plot a Reeb Graph given a graph with a position
        
        """
        nx.draw(self.G,self.pos)
