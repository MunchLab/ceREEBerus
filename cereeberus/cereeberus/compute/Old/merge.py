import networkx as nx
import numpy as np

def isMerge(T,fx):
    """
    This function takes in a networkx tree or reeb graph and function, and checks to see if it is a 
    merge tree.  This assumes that the root node(s) has/have a function value of np.inf. 

    Args:
        T (Reeb Graph): Networkx graph or reeb graph
        fx (dict): function values

    Returns:
        isMerge(bool): True if T is a merge tree, False if T is not a merge tree
    """
    import numpy as np
    import networkx as nx
    from cereeberus.compute.degree import up_degree
    from cereeberus.reeb.reebgraph import Reeb
    
    if type(T) is nx.classes.multigraph.MultiGraph or type(T) is nx.classes.digraph.DiGraph:
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

def computeMergeTree(R, filtration: "tuple[tuple[float,float], tuple[float, float], int]" = [[8,0], [8, 7], 1], infAdjust: int=None, precision: int=5, size: int=0, verbose: bool=False, filter: bool = False):
    """
    main function to build merge tree for a given graph and filtration
    
    Args:
        R (Reeb Graph): Reeb Graph
        filtration (tuple of tuples): filtration for merge tree
        infAdjust (int): parameter to adjust infinite value for root node
        precision (int): precision
        size (int): size
        verbose (bool): verbose

    Returns:
        rmt: Merge Tree as a Reeb Graph object
    """
    from cereeberus.compute.degree import remove_isolates
    from cereeberus.compute.uf import UnionFind
    from cereeberus.compute.uf import getSortedNodeHeights
    import networkx as nx
    from cereeberus.reeb.merge import mergeTree
   
    Rmt = remove_isolates(R)
    
    # digraph for LCA calcs and it's a tree
    mt = nx.DiGraph()
    
    # to handle special indexes post projections (has nodes named >n)
    if size != 0:
        uf = UnionFind(size, verbose=verbose)
    else:
        uf = UnionFind(len(Rmt.nodes), verbose=verbose)

        
    visited = set()
    numComponents = 0
    if filter==True:
        heights = getSortedNodeHeights(Rmt, filtration, precision)
    else:
        heights = R.heights
    if verbose:
        print("heights:" + str(heights))
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
            mt.add_node(node, pos=(numComponents, height), fx=height)
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
                    mt.add_node(node, pos=(mt.nodes[myRoot]['pos'][0], height), fx=height)
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
                
                mt.add_node(node, pos=(numComponents-len(componentList), height), fx=height)
                
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
        for neighbor in Rmt.neighbors(node):
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
    mt.add_node('inf', pos=(0, infHeight), fx=float('inf'))
    mt.add_edge('inf', topMerge)
    mt = nx.MultiGraph(mt)
    fx=nx.get_node_attributes(mt, 'fx')
    rmt = mergeTree(mt, fx)
    
    return rmt
