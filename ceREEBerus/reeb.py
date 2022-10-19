import networkx as nx

class reeb:
    """ Class for Reeb Graph
    
    """

    def node_position(self, G):
        """ Return Node Position for a Reeb Graph
        
        """
        pos = nx.get_node_attributes(G,'pos')
        return pos
    
    def plot_reeb(self, G, pos):
        """ Plot a Reeb Graph given a position
        
        """
        nx.draw(G,pos)
