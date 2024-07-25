from cereeberus import ReebGraph
import networkx as nx
import numpy as np
from cereeberus.compute.merge import isMerge
from cereeberus.compute.merge import computeMergeTree

class MergeTree(ReebGraph):
    """ 
    A merge tree stored as a ``ReebGraph`` object. Like a Reeb graph, this is a directed graph with a function defined on the vertices. However, in a merge tree, the function is required to have a (single?) root with function value treated as :math:`\infty`.

    """

    def __init__(self, 
                    T = None, root = None,
                    f = {}, 
                    seed = None, 
                    verbose = False):
        """
        Initialize a merge tree object.
        
        Parameters:
            T: nx.graph, optional. If not none, it should be a tree with a specified root and function values. 
            TODO: Perhaps we want to allow to just pass a tree and root and induce the function? 
        """

        if T == None:
            super().__init__(seed = seed, verbose = verbose)
        else:
            # Check if the input is valid and add it
            T,f = self.inputToDirRootTree(T,root,f)
            super().__init__(T, f, seed, verbose)

        # Fix up the drawing locations
        

        


    def inputToDirRootTree(self, T, root, f):
        """
        Convert the input to a directed rooted tree that respects the internal structure. 

        If T is undirected, it will be converted to a directed tree with the root specified.

        If f is not specified, a function will be induced from the tree from number of edges from the root. 

        A vertex will be added with funciton value infinity. 
        """
        if T == None:
            # Returns Tree, function
            return None, None
        
        elif nx.is_tree(T) == False:
            raise ValueError("The input graph is not a tree.")
        
        elif nx.is_directed(T) == False:
            if root == None:
                raise ValueError("You must specify a root for the tree.")
            else:
                # Convert to directed graph eminating from the root
                # Because of the ReebGraph convention, edges point up to higher function values.
                T = nx.bfs_tree(T,source = root).reverse()
        else: # T is already directed
            # Make sure that this thing will be valid, 
            # I think this is the same as "arboresence" in networkx
            # which is true if maximum in-degree equal to 1. 
            # Since our graphs point upwards, this is the same as 
            # making sure the reverse of our tree is an arboresence.

            if not nx.is_arborescence(T.reverse()):
                raise ValueError("The input tree is not a valid tree.")

        # No matter what, add a dummy node, point the old root at it and 
        # eventually give that new node function value infinity. 
        T.add_node('v_inf')
        T.add_edge(root, 'v_inf')


        # Check that the function is valid 
        if f == {}:
            # Define a function with distance from the root (but reversed). Rigged so the minimum is always at 0. 
            f = nx.shortest_path_length(T, target = root)
            M = max(f.values())
            f = {i:M-f[i] for i in f.keys()}
            f['v_inf'] = np.inf
            
        else:
            for e in T.edges:
                if f[e[0]] < f[e[1]]:
                    raise ValueError("The function values should point upwards in the tree.")
        
        return T,f

        
    def fix_pos_f(self):
        """
        Update drawing locations to deal with the fact that we have np.inf around.
        
        This sets the drawing location of the infinite node to be directly above the maximum finite node of the tree.
        """
        # Get the neighbor of the infinite node. Note we're assuming a single connected component so there's only one.
        top_vertex = list(self.predecessors('v_inf'))[0]

        # Get  min and max function values
        Lmin =  min(self.f.values())
        Lmax = self.f[top_vertex]
        height = Lmax - Lmin

        # Get the drawing location of the neighbor to have same x coordinate as the neighbor, but y coordinate at .3*height above the maximum function value.
        self.pos_f['v_inf'] = (self.pos_f[top_vertex][0], Lmax + .3 * height)


        



        # Run a check to see if the tree and 
        # function actually satisfy the merge
        # tree requirements.
        # if not isMerge(T,fx):
        #     print("The tree and function you passed in do not satisfy the requirements of a merge tree. Creating Merge Tree")
        #     T = computeMergeTree(T)

        # # Set the maximum finite value. Needs to happen before runnning the Reeb init
        # # because of how I overwrote the set_pos_fx function.
        # if type(fx)==list:
        #     self.maxFiniteVal = max(np.array(fx)[np.isfinite(fx)])
        # if type(fx)==dict:
        #     values = np.array(list(fx.items()), dtype=float)[:,1]
        #     self.maxFiniteVal = max(values[np.isfinite(values)])
        
        # # Do everything from the Reeb graph setup step
        # Reeb.__init__(self,T,fx)

        # # Mark the root vertex. If there's more than one, we'll store an array of them.
        # roots = np.where(np.isinf(fx))[0]
        # self.numComponents = len(roots)


        # if self.numComponents==1:
        #     self.rootIndex = roots[0]
        # elif self.numComponents>1:
        #     self.rootIndex = roots 
        # else:
        #     raise AttributeError("This has no function value at np.inf, so this is not a merge tree satisfying our requirements.")
        
        # # Update position drawing 
        # self.fix_pos_fx()

    # def fix_pos_fx(self):
    #     # Update drawing locations to deal with the fact that we have np.inf around.

    #     # First, figure out where the inf is that we'll have to update, based on whether we want horizontal or vertical drawings 

    #     if self._horizontalDrawing:
    #         functionCoord = 0 
    #         otherCoord = 1
    #     else:
    #         functionCoord = 1
    #         otherCoord = 0

    #     drawingLocation = [None,None]
    #     drawingLocation[functionCoord] = self.maxFiniteVal + 3

    #     if self.numComponents >1:
    #         for i in self.rootIndex: #Note this is an array of roots
                
    #             drawingLocation[otherCoord] = self.pos_fx[i][otherCoord]
    #             self.pos_fx[i] = list(drawingLocation)
    #     else:
    #         drawingLocation[otherCoord] = self.pos_fx[self.rootIndex][otherCoord]
    #         self.pos_fx[self.rootIndex] = list(drawingLocation)


#     def set_pos_fx(self,resetSpring = False, verbose = False):
#         Reeb.set_pos_fx(self,resetSpring = False, verbose = False)

#         self.fix_pos_fx()



if __name__=="__main__":
    from cereeberus.data.randomMergeTrees import randomMerge

    R = randomMerge(10)
    M = Merge(R.G, R.fx)
    M.plot_reeb()