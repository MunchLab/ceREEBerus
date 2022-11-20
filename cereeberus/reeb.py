import networkx as nx
import numpy as np
import cereeberus.compute.degree as degree
import cereeberus.compute.draw as draw

class Reeb:
    """ Class for Reeb Graph

    :ivar G: Graph: G
    :ivar fx: function values associated with G
    :ivar pos: spring layout position calculated from G
    :ivar pos_fx: position values corresponding to x = fx and y = y value from pos
    :ivar horizontalDrawing: Default to False. If true, fx is drawn as a height function. 
    """
    def __init__(self, G, fx = {},
                        horizontalDrawing = False, 
                        verbose = False):
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

        self._horizontalDrawing = horizontalDrawing

        self.set_pos_fx(verbose = verbose)
        


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
            for i in range(0,len(self.pos)):
                self.pos_fx[i] = (self.fx[i], self.pos[i][0])
        else:
            if verbose:
                print('Saving positions to be vertical')
            self.pos_fx = {}
            for i in range(0,len(self.pos)):
                self.pos_fx[i] = (self.pos[i][0], self.fx[i])

        


    def plot_reeb(self, position = {}, 
                        horizontalDrawing = False,
                        verbose = False):
        """ Plot a Reeb Graph given a graph with a position.
        If no position passed, it will take the spring layout version. 
        In this case, it will either be drawn vertically or 
        horizontally, depending on the horizontalDrawing (boolean) 
        passed in.
        
        """
        if position == {}: # the user didn't pass positions

            # if the horizontal drawing setting is not the same as 
            # what is already saved, reset the pos_fx 
            if self._horizontalDrawing != horizontalDrawing:
                self._horizontalDrawing = horizontalDrawing
                self.set_pos_fx(verbose)    

            # then hand over the internally saved positions
            pos = self.pos_fx
            
        else:
            pos = position
        nx.draw(self.G, pos = pos)
        self.pos = nx.spring_layout(self.G)
        self.pos_fx = {}
        for i in range(0,len(self.pos)):
            self.pos_fx[i] = (self.pos[i][0], self.fx[i])

        self.nodes = self.G.nodes
        self.edges = self.G.edges

        # compute upper and lower degree of reeb graph
        self.up_deg = degree.up_degree(self.G, self.fx)
        self.down_deg = degree.down_degree(self.G, self.fx)

        # adjacency matrix
        #self.adjacency = nx.adjacency_matrix(G)

        # show basic properties of reeb graph
        self.summary = {'nodes': len(self.nodes), 'edges': len(self.edges)}

    def plot_reeb(self, cp=.5):
        """ Plot a Reeb Graph
        
        """
        draw.reeb_plot(self, cp)
    


