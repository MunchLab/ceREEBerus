import networkx as nx
import cereeberus.compute.degree as degree
import cereeberus.compute.draw as draw

class Reeb:
    """ Class for Reeb Graph

    :ivar G: Graph: G
    :ivar fx: function values associated with G
    :ivar pos: spring layout position calculated from G
    :ivar pos_fx: position values corresponding to x = fx and y = y value from pos
    """

    def __init__(self, G, fx = {}, horizontalDrawing = False, verbose = False):
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
        
        pos = nx.get_node_attributes(G,"pos")
        if  pos == {}:
            self.pos = nx.spring_layout(self.G)
        else:
            self.pos = pos

        self.nodes = G.nodes
        self.edges = G.edges

        # add heights for merge tree function
        self.heights = degree.heights(self)

        self._horizontalDrawing = horizontalDrawing

        node_properties = {}
        

        self.set_pos_fx(verbose = verbose)
    # compute upper and lower degree of reeb graph
        self.up_deg = degree.up_degree(G, self.fx)
        self.down_deg = degree.down_degree(G, self.fx)
    


        # adjacency matrix
        #self.adjacency = nx.adjacency_matrix(G)
        

        for i in self.nodes:
            node_properties[i] = {'node': i, 'pos': self.pos[i], 'pos_fx': self.pos_fx[i], 'up_deg': self.up_deg[i],
            'down_deg': self.down_deg[i]}
        self.node_properties = node_properties

        # show basic properties of reeb graph
        self.summary = {'nodes': len(self.nodes), 'edges': len(self.edges)}
    
    def neighbors(self, node):
        return self.G.neighbors(node)

    def set_pos_fx(self, resetSpring = False, verbose = False):
        """
        Returns the position data for drawing the Reeb graph. 
        If self.horizongalDrawing = False, we are drawing the vertices
        at locations 
        (spring_layout_x, functionvalue(v)) 
        Otherwise, we are drawing the points at 
        (functionvalue(v), spring_layout_x ) 
        resetSpring will make it recalculate the spring layout to overwrite
        """    

        if resetSpring:
            self.pos = nx.spring_layout(self.G)

        if self._horizontalDrawing:
            if verbose:
                print('Saving positions to be horizontal')
            self.pos_fx = {}
            for node in self.pos:
                self.pos_fx[node] = (self.fx[node], self.pos[node][0])
        else:
            if verbose:
                print('Saving positions to be vertical')
            self.pos_fx = {}
            for node in self.pos:
                self.pos_fx[node] = (self.pos[node][0], self.fx[node])

    def plot_reeb(self, position = {}, resetSpring = False, horizontalDrawing = False, verbose = False, cpx=.1, cpy=.1):
        """ Plot a Reeb Graph given a graph with a position.
        If no position passed, the position attributes from the reeb graph will be used. 
        In this case, it will either be drawn vertically or 
        horizontally, depending on the horizontalDrawing (boolean) 
        passed in.
        
        """
        if position == {}: # the user didn't pass positions

            # if the horizontal drawing setting is not the same as 
            # what is already saved, reset the pos_fx 
            if self._horizontalDrawing != horizontalDrawing:
                self._horizontalDrawing = horizontalDrawing
                   
            self.set_pos_fx(resetSpring, verbose) 
            # then hand over the internally saved positions
            pos = self.pos_fx
            
        else:
            pos = position

        draw.reeb_plot(self, pos, cpx, cpy)