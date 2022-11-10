class Reeb:
    """ Class for Reeb Graph

    :ivar G: Graph: G
    :ivar fx: function values associated with G
    :ivar pos: spring layout position calculated from G
    :ivar pos_fx: position values corresponding to x = fx and y = y value from pos
    """
    import networkx as nx
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
    
    def plot_reeb(self, position = {}):
        """ Plot a Reeb Graph given a graph with a position
        
        """
        if position == {}:
            pos = self.pos_fx
        else:
            pos = position
        nx.draw(self.G, pos = pos)
    


