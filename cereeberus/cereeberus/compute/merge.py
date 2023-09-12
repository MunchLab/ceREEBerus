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

def ComputeMergeTree(R, filtration: tuple[tuple[float,float], tuple[float, float], int], infAdjust: int=None, precision: int=5, size: int=0, verbose: bool=False):
    """
    main function to build merge tree for a given graph and filtration
    """
    from cereeberus.compute.degree import remove_isolates
    from cereeberus.compute.uf import UnionFind
    from cereeberus.compute.uf import getSortedNodeHeights
    import networkx as nx
   
    Rmt = remove_isolates(R)
    
    # digraph for LCA calcs and it's a tree
    mt = nx.DiGraph()
    
    # to handle special indexes post projections (has nodes named >n)
    if size != 0:
        uf = UnionFind(size, verbose=verbose)
    else:
        uf = UnionFind(Rmt.number_of_nodes(), verbose=verbose)
        
    visited = set()
    numComponents = 0
    heights = getSortedNodeHeights(Rmt, filtration, precision)
    if verbose:
        print(heights)
    # this is the first node of min height since list
    topMerge = heights[0][0]
    
    for node, height in heights:
        # track visited nodes (helps deal with equal heights)
        if verbose:
            print(f"now processing node{node}, with {numComponents} already found components")
        visited.add(node)     
        
        # check to see if the node has been told it is the endpoint of a previous grouping (the endpoint of an already found edge)
        # perform find to make sure these groupings are not the same
        possibleGroups = Rmt.nodes[node].get('groups', [])

        # if this edge has never been told anything, no existing edges
        # add this node in merge tree as start of a new branch
        if possibleGroups == []:
            if verbose:
                print(f"{node} is unconnected, about to add {numComponents}, {height}")
            mt.add_node(node, pos=(numComponents, height), height=height)
            numComponents += 1

        else:
            # iterate through possible groups via unionFind to determine if this is a merge point or one connected component
            componentSet = set()
            for possibleGroup in possibleGroups:
                componentSet.add(uf.find(possibleGroup))

            componentList = list(componentSet)
            
            if verbose:
                print( f"received {componentList} membership")
            
            # if they are all the same group, this node is also part of this group
            # ignore on merge tree if its part of original graph
            # place this on if its a key label from the other graph's merge tree
            if len(componentList) == 1:
                myRoot = componentList.pop()
                uf.union(node, myRoot)
                # only add to merge tree if a special key label point
                if Rmt.nodes[node].get('projected', False):
                    if verbose:
                        print(f"although connected, key label{node}, existing connected to {myRoot}, adding still")
                    mt.add_node(node, pos=(mt.nodes[myRoot]['pos'][0], height), height=height)
                    mt.add_edge(node, myRoot)
                    
                    # change the root to represent the current head of merge point
                    if verbose:
                        print(f"rerooting component {myRoot} to {node}")
                    uf.rerootComponent(node, node)
                    
                    # this point could be a top merge point to infinity root
                    topMerge = node
                    
                else:
                    # skip if on the same graph
                    if verbose:
                        print(f"skipping node{node}, existing connected to {myRoot}")
            else:
                # else this node is the merge point, add on merge tree and perform union
                if verbose:
                    print(f"about to add {numComponents}, {height}, updating topMerge")
                
                topMerge = node
                
                mt.add_node(node, pos=(numComponents-len(componentList), height), height=height)
                
                for component in componentList:
                    componentRoot = uf.find(component)
                    if verbose:
                        print(f"unioning node{node} and componentRoot of node {componentRoot}")
                        
                    # union each component
                    uf.union(node, componentRoot)
                    # track on merge tree
                    mt.add_edge(node, componentRoot)

                    if verbose:
                        print(f"rerooting {componentRoot} to merge point {node}")
                        
                    # change the root to represent the current head of merge point
                    uf.rerootComponent(node, node)
                    
                    numComponents -= 1
                    
        # pass along the finalized group to all the edges above
        myGroup = uf.find(node)
        for neighbor in nx.all_neighbors(Rmt, node):
            # lower height neighbors seen before
            if verbose:
                print( f"neighbor{neighbor}")
            if neighbor in visited:
                if verbose:
                    print(f"visited{neighbor} already")
                continue

            # pass new info
            groups = Rmt.nodes[neighbor].get('groups', [])
            Rmt.nodes[neighbor]['groups'] = groups + [myGroup]
        
    # add final "inf" point, but visualize as max height + 10% of height range unless passed in
   
    if infAdjust is None:
        infAdjust = (heights[-1][1] - heights[0][1] ) * 0.1
    infHeight = heights[-1][1] + infAdjust
    mt.add_node('inf', pos=(0, infHeight), height=float('inf'))
    mt.add_edge('inf', topMerge)
    
    return mt
 


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