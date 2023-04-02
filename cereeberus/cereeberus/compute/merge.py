"""
Code started by Liz Nov 2022. 
The goal is to get a merge tree class with the following properties. 
- Should accept a Reeb graph class as input. 
- Needs a check to make sure the input is actually a Reeb graph. In particular, only down forks and a root node with function value `np.inf`. 
- Has drawing capabilities, in particular can handle that `np.inf` node. 
- If the Reeb graph passed in isn't a merge tree, likely I just want to generate the merge tree of the input Reeb graph. 
"""


from cereeberus.reeb.graph import Reeb
import networkx as nx
import numpy as np
from cereeberus.compute.degree import up_degree


def isMerge(T,fx):
    """
    This function takes in a networkx tree or reeb graph and function, and checks to see if it is a 
    merge tree.  This assumes that the root node(s) has/have a function value of np.inf. 
    """
    import numpy as np
    import networkx as nx
    from cereeberus.compute.degree import up_degree
    from cereeberus.reeb.graph import Reeb
    
    if type(T) is nx.classes.multigraph.MultiGraph:
        node_list = list(T.nodes)
        up_deg = up_degree(T, fx)
        for i in node_list:
            if (up_deg[i]==1 or (fx[i]==np.inf and up_deg[i]==0)) == True:
                1 == 1
            else:
                return False

    elif type(T) is Reeb:
        node_list = list(T.nodes)
        for i in node_list:
            if (T.node_properties[i]['up_deg']==1 or (fx[i]==np.inf and T.node_properties[i]['up_deg']==0)) == True:
                1 == 1
            else:
                return False
    
    else:
        raise TypeError('Graph is not a networkx graph or Reeb graph')
    
    return True

def ComputeMergeTree(R):
    """
    This function takes in a Reeb graph and returns the merge tree of that Reeb graph. 
    """
    pass 


class Merge(Reeb):
    """ Class for Merge tree
    :ivar T: Graph: T
    :ivar fx: function values associated with T
    :ivar pos: spring layout position calculated from G
    :ivar pos_fx: position values corresponding to x = fx and y = y value from pos
    :ivar horizontalDrawing: Default to False. If true, fx is drawn as a height function. 
    """

    def __init__(self, T, 
                    fx = {}, 
                    horizontalDrawing = False, 
                    verbose = False):

        # Run a check to see if the tree and 
        # function actually satisfy the merge
        # tree requirements.
        if not isMerge(T,fx):
            raise AttributeError("The tree and function you passed in do not satisfy the requirements of a merge tree. ")

        # Set the maximum finite value. Needs to happen before runnning the Reeb init
        # because of how I overwrote the set_pos_fx function.
        self.maxFiniteVal = max(np.array(fx)[np.isfinite(fx)])
        
        # Do everything from the Reeb graph setup step
        Reeb.__init__(self,T,fx)

        # Mark the root vertex. If there's more than one, we'll store an array of them.
        roots = np.where(np.isinf(fx))[0]
        self.numComponents = len(roots)


        if self.numComponents==1:
            self.rootIndex = roots[0]
        elif self.numComponents>1:
            self.rootIndex = roots 
        else:
            raise AttributeError("This has no function value at np.inf, so this is not a merge tree satisfying our requirements.")
        
        # Update position drawing 
        self.fix_pos_fx()





    def fix_pos_fx(self):
        # Update drawing locations to deal with the fact that we have np.inf around.

        # First, figure out where the inf is that we'll have to update, based on whether we want horizontal or vertical drawings 
        if self._horizontalDrawing:
            functionCoord = 0 
            otherCoord = 1
        else:
            functionCoord = 1
            otherCoord = 0

        drawingLocation = [None,None]
        drawingLocation[functionCoord] = self.maxFiniteVal + 3

        if self.numComponents >1:
            for i in self.rootIndex: #Note this is an array of roots
                
                drawingLocation[otherCoord] = self.pos_fx[i][otherCoord]
                self.pos_fx[i] = list(drawingLocation)
        else:
            drawingLocation[otherCoord] = self.pos_fx[self.rootIndex][otherCoord]
            self.pos_fx[self.rootIndex] = list(drawingLocation)


#     def set_pos_fx(self,resetSpring = False, verbose = False):
#         Reeb.set_pos_fx(self,resetSpring = False, verbose = False)

#         self.fix_pos_fx()



if __name__=="__main__":
    from cereeberus.data.randomMergeTrees import randomMerge

    R = randomMerge(10)
    M = Merge(R.G, R.fx)
    M.plot_reeb()