class UnionFind:
    '''
        Array index implementation of UnionFind inspired by William Fiset's java implementation (github.com/williamfiset/data-structures) with special rerooting function to handle merge tree construction
    '''
    def __init__(self, size: int, verbose: bool=False) -> None:
        '''create internal union find structure represented as array with all nodes pointing to themselves (individual components)'''
        
        assert size > 0, f"Invalid Size of {size}"
        
        self.size = size
        self.uf = [x for x in range(self.size)]
        
        # track initial 1 size of components (will track at the root connected component)
        self.sizes = [1] * self.size
        
        # component count
        self.numComponents = self.size
        self.verbose = verbose
        return
        
    def getNumComponents(self) -> int:
        '''get number of total connected components'''
        return self.numComponents
    
    def getSizeOfComponent(self, c: int) -> int:
        '''get size of c's connected component '''
        return self.sizes[self.find(c)]
    
    def getSize(self) -> int:
        '''get input max size of UF structure'''
        return self.size
    
    def _pathCompress(self, c: int, root: int) -> None:
        # internal compression function called when find is run for optimization
        while c != root:
            node = self.uf[c]
            self.uf[c] = root
            c = node
        return
    
    def rerootComponent(self, c: int, newRoot: int) -> None:
        '''given a component and any connected node, make that node the new root of the component - key for building up a mergetree'''
        oldRoot = self.find(c)
        
        newRootOld = self.find(newRoot)
        assert oldRoot == newRootOld, "must be connected to reroot"
        
        if self.verbose:
            print(f"rerooting component {oldRoot} to be called {newRoot}")
        
        copy = self.uf.copy()
        for idx, c in enumerate(self.uf):
            if self.find(c) == oldRoot:
                copy[idx] = newRoot
        self.uf = copy
        
        # copy over the size of the old c to the newRoot
        self.sizes[newRoot] = self.sizes[oldRoot]
        return
        
    def find(self, c: int) -> int:
        '''return the root of the connected component of c'''
        if c > self.size:
            raise Exception("index out of range")
        
        root = c
        while root != self.uf[root]:
            root = self.uf[root]
        
        self._pathCompress(c, root)
        return(root)
        
    def union(self, c1: int, c2: int) -> None:
        '''union c1 and c2 connected components'''
        if c1 >= self.size or c2 > self.size:
            raise Exception("index out of range")
        
        # use roots to represent each connected compoment
        root1 = self.find(c1)
        root2 = self.find(c2)
    
        # nothing to do if already same
        if root1 == root2:
            if self.verbose:
                print("nothing to merge")
            return
        else:
            if self.verbose:
                print(f"merging {root1} and {root2}")
            # perform union of smaller into larger
            if self.sizes[root2] < self.sizes[root1]:
                self.uf[root2] = root1
                self.sizes[root1] += self.sizes[root2]
            else:
                self.uf[root1] = root2
                self.sizes[root2] += self.sizes[root1]
        
        self.numComponents -= 1
        return
    
    def isFullyConnected(self) -> bool:
        '''if all "size" # of nodes are fully connected, return True'''
        return self.numComponents == 1
    
    def printAll(self) -> None:
        '''print all nodes and the connected components of each'''
        for c in self.uf:
            print( self.find(c) )
        return
    
    
import math

def signedDistToLine2Pts(pt: tuple, p0: tuple, p1: tuple) -> float:
    '''return a signed distance to a line where line is defined as two points
    
    positive sign refers to "above" the line or "left" of a vertical line
    to get the expected sign of "right" is positive, the vertical line will be inverted back under the "angle_sign" in _computeNodeHeights() of MergeTree.py
    '''
    return ((p0[0]-pt[0])*(p1[1]-p0[1]) - (p1[0]-p0[0])*(p0[1]-pt[1])) / math.dist(p0,p1)

def distToLine2Pts(pt: tuple, p0: tuple, p1: tuple) -> float:
    '''pt to line as defined by p0, p1 https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line'''
    
    return abs(signedDistToLine2Pts(pt, p0, p1))

def intersectionToLine2Pts(pt: tuple, p0: tuple, p1: tuple) -> tuple:
    '''intersection of a point to a line defined by two points'''
    a = p0[1] - p1[1]
    b = p1[0] - p0[0]
    c = p0[0]*p1[1] - p1[0]*p0[1]
    
    x = (b*(b*pt[0]-a*pt[1])-a*c) / (a**2+b**2)
    y = (a*(-b*pt[0]+a*pt[1])-b*c) / (a**2+b**2)

    return (x,y)
    
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import bisect

from numpy.lib.stride_tricks import sliding_window_view
from sortedcontainers import *

def _ptProjectToGraph(pt: tuple[int, int], graph: nx.Graph) -> nx.Graph:
    # project a point to its nearest edge on input graph
    
    minDist = float('inf')
    min_v0 = -1
    min_v1 = -1
    projected = [None, None]
    
    # for every line segment in graph
    # need to track min distance over all line segments
    for v0, v1 in graph.edges():
        
        p0 = graph.nodes[v0]['pos']
        p1 = graph.nodes[v1]['pos']
                    
        # if intersecting the line segment, then this is shortest path
        # else it will be the min of the endpoints
        intersection = intersectionToLine2Pts(pt, p0, p1)
        
        # where it intersects is on the line so only need to check domain
        maxX = max(p0[0], p1[0])
        minX = min(p0[0], p1[0])    

        # this needs to be done to handle vertical edges
        maxY = max(p0[1], p1[1])
        minY = min(p0[1], p1[1])  
        
        if intersection[0] <= maxX and intersection[0] >= minX and intersection[1] <= maxY and intersection[1] >= minY:
            dist = distToLine2Pts(pt, p0, p1)
            matchType = 2
        else:
            dist_p0 = math.dist(pt, p0)
            dist_p1 = math.dist(pt, p1)
            
            if dist_p0 < dist_p1:
                matchType = 0
                dist = dist_p0
            else:
                matchType = 1
                dist = dist_p1
        
        # update if min dist is shortest
        if dist < minDist:
            minDist = dist
            min_v0, min_v1 = v0, v1
            # match/case if python 3.10
            if matchType == 2:
                projected = intersection
            elif matchType == 0:
                projected = p0
            else:
                projected = p1
            
    return projected, min_v0, min_v1

def _projectNodesOnto(giver: nx.Graph, receiver_orig: nx.Graph) -> nx.Graph:
    # project all nodes of input giver graph onto receiver graph
    receiver = receiver_orig.copy()
    
    # NOTE this requires the nodes of the graph to be labelled from 0
    # project giver onto receiver (nodes should already be mapped)
    for pt, data in giver.nodes(data=True):
        projected, p_v0, p_v1 = _ptProjectToGraph(data['pos'], receiver)
        receiver.add_node(pt)
        receiver.nodes[pt]['pos'] = projected
        receiver.nodes[pt]['p_v0'] = p_v0
        receiver.nodes[pt]['p_v1'] = p_v1
        receiver.nodes[pt]['projected'] = True
    
    return receiver

# project graphs onto each onto the other
def prepareTwoGraphs(g1_orig: nx.Graph, g2_orig: nx.Graph) -> nx.Graph:
    '''project nodes of graph onto each other, needs integer node names and will offset graph 2 node names by the max of the node name found in graph 1'''

    #g2 = g2_orig.copy()
    # node ordering is inherited from the graph
    #g2 = nx.convert_node_labels_to_integers(g2, first_label=max(g1_orig.nodes())+1, label_attribute='origLabel')
    
    g2 = nx.relabel_nodes(g2_orig, lambda x: x + max(g1_orig.nodes())+1 )
    g2_proj = _projectNodesOnto(g1_orig, g2)
    g1_proj = _projectNodesOnto(g2, g1_orig)
    
    return g1_proj, g2_proj

def _computeNodeHeights(graph: nx.Graph, filtration: tuple[tuple[float,float], tuple[float, float], int], precision: int=5) -> dict[int, tuple[float,bool]]:
    # given a filtration line and direction, compute heights of each node and return as dict of tuple (height, projected)
    
    # defining a line as 2 points and an inversion flag
    p0 = filtration[0]
    p1 = filtration[1]
    angle_sign = filtration[2] # based on the critical angle
    
    # only need to iterate in sorted order (so no need for SortedDicts/BSTs)
    # could also save heights into the graph properties, but for now utilizing this other data structure
    heights = {}
    for node, data in graph.nodes(data=True):
        # need to calculate when the existing point is "above" or "below"
        # so it's not just a raw absolute distance to line, but tracking position using
        # y < f(x) or y > f(x)
        height = round(angle_sign * signedDistToLine2Pts(data['pos'], p0, p1), precision)
        projected = data.get('projected', False)
        heights[node] = (height, projected)
    
    return heights
    
def getSortedNodeHeights(graph: nx.Graph, filtration: tuple[tuple[float,float], tuple[float, float], int], precision: int=5) -> list[tuple[int,float]]:
    '''compute heights of each node given filtration line and return as sorted list of node height tuples, rounded to given precision'''
    
    # sorted first by height, then non-projected first
    heightTuples = _computeNodeHeights(graph, filtration, precision=precision)
    
    # only need the nodes and height so rehspaing
    return [(x[0], x[1][0]) for x in sorted(heightTuples.items(), key=lambda x:x[1])]

def buildMergeTree(g_orig: nx.Graph, filtration: tuple[tuple[float,float], tuple[float, float], int], infAdjust: int=None, precision: int=5, show: bool=True, size: int=0, verbose: bool=False ) -> nx.DiGraph:
    '''main function to build merge tree for a given graph and filtration'''
    
    # make copy of graph since we save node properties into graph
    g = g_orig.copy()
    
    # we skip isolates here which is not the most efficient, but is a key step
    # in identifying nodes that are not important (not kept in a merge tree for the second graph)
    isolates = list(nx.isolates(g))
    
    # digraph for LCA calcs and it's a tree
    mt = nx.DiGraph()
    
    # to handle special indexes post projections (has nodes named >n)
    if size != 0:
        uf = UnionFind(size, verbose=verbose)
    else:
        uf = UnionFind(g.number_of_nodes(), verbose=verbose)
        
    visited = set()
    numComponents = 0
    heights = getSortedNodeHeights(g, filtration, precision)
    if verbose:
        print(heights)
    # this is the first node of min height since list
    topMerge = heights[0][0]
    
    for node, height in heights:
        # track visited nodes (helps deal with equal heights)
        if verbose:
            print(f"now processing node{node}, with {numComponents} already found components")
        visited.add(node)

        # if graph node is floating, can ignore
        if node in isolates:
            if verbose:
                print(f"found isolated {node} at 0, {height}, skipping")
            continue        
        
        # check to see if the node has been told it is the endpoint of a previous grouping (the endpoint of an already found edge)
        # perform find to make sure these groupings are not the same
        possibleGroups = g.nodes[node].get('groups', [])

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
                if g.nodes[node].get('projected', False):
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
        for neighbor in nx.all_neighbors(g, node):
            # lower height neighbors seen before
            if verbose:
                print( f"neighbor{neighbor}")
            if neighbor in visited:
                if verbose:
                    print(f"visited{neighbor} already")
                continue

            # pass new info
            groups = g.nodes[neighbor].get('groups', [])
            g.nodes[neighbor]['groups'] = groups + [myGroup]
        
    # add final "inf" point, but visualize as max height + 10% of height range unless passed in
   
    if infAdjust is None:
        infAdjust = (heights[-1][1] - heights[0][1] ) * 0.1
    infHeight = heights[-1][1] + infAdjust
    mt.add_node('inf', pos=(0, infHeight), height=float('inf'))
    mt.add_edge('inf', topMerge)
    
    if show:
        nx.draw(mt, pos=nx.get_node_attributes(mt, 'pos'), node_color='#FFFFFF', with_labels=True)
        plt.show()
    
    return mt

def _cacheEdgeOfProjectedNodes(graph: nx.Graph) -> dict[int,list[int]]:
    # utility for sortedFlatten but not used by merge tree construction
    cache = {}
    for node in graph.nodes():
        if not 'p_v0' in graph.nodes[node]:
            continue
        
        p_v0 = graph.nodes[node]['p_v0']    
        p_v1 = graph.nodes[node]['p_v1']
        
        edgeKey = tuple(sorted((p_v0, p_v1)))
        
        if edgeKey in cache:
            cache[edgeKey].append(node)
        else:
            cache[edgeKey] = [node]
        
    return cache

def _sortedFlatten(graph_orig: nx.Graph) -> nx.Graph:
    # complete flattening, A-(C,D)-B to A-C-D-B, but not used by merge tree construction
    
    graph = graph_orig.copy()

    cache = _cacheEdgeOfProjectedNodes(graph)
    
    for edge in cache.keys():        
        # tuple is already sorted
        root = edge[0]
        rootPos = graph.nodes[root]['pos']
        
        # sort by distance
        dist = {}
        for node in cache[edge]:
            dist[node] = math.dist(graph.nodes[node]['pos'], rootPos)
        
        dist = dict(sorted(dist.items(), key=lambda x: x[1]))
        sortedNodes = dist.keys()
        
        # remove original edge
        graph.remove_edge( edge[0], edge[1])
        prevNode = edge[0]
        
        # add each line segment
        for node in sortedNodes:
            graph.add_edge( prevNode, node )
            prevNode = node
    
        # add final segment
        graph.add_edge( prevNode, edge[1])
    
    return graph

def _flattenToEndpoints(graph_orig: nx.Graph) -> nx.Graph:
    # flatten with many edges with the same endpoints, A-(C,D)-B becomes A-B, A-C-B, A-D-B, currently used in merge tree construction

    graph = graph_orig.copy()

    for node, data in graph.nodes(data=True):
        if data.get('projected', False):
            # add final segment
            graph.add_edge( node, data['p_v0'] )
            graph.add_edge( node, data['p_v1'] )
    
    return graph

def _mtToOtherGraph(mt_orig: nx.DiGraph, graph_orig: nx.Graph, verbose: bool=False) -> nx.Graph:
    # take nodes of mt and only keep projections of those nodes on the graph
    
    graph = graph_orig.copy()
    
    # remove unneeded projected points
    mt_nodes = mt_orig.nodes()
    
    # using graph_orig to remove nodes on graph (the return value copy)
    for node in graph_orig.nodes():
        if node not in mt_nodes and graph_orig.nodes[node].get('projected', False):
            if verbose:
                print(f"removing projected but unimportant node {node}")
            graph.remove_node(node)
        
    # full flattening of the remainder key nodes    
    graph = _flattenToEndpoints(graph)
        
    return graph

# get critical angles as list
def computeCriticalAngles(graph: nx.Graph) -> SortedSet[float]:
    '''get sorted set of all critical angles for a graph'''
    angles = SortedSet()
    for v0, v1 in graph.edges():
        p0 = graph.nodes[v0]['pos']
        p1 = graph.nodes[v1]['pos']
        
        y = p1[1]-p0[1]
        x = p1[0]-p0[0]
        
        # add angle + its supplement
        if y == 0:
            # horizontal line, normal is vertical
            angles.add(math.pi/2)
            angles.add(3*math.pi/2)
            
        else:
            # this is the angle formed to the normal, the domain adjusted to 0<theta<2*pi
            atan = math.atan(-x/y) % (2*math.pi)
            angles.add(atan)
            angles.add((math.pi + atan) % (2*math.pi))

    return angles

def computeAllCriticalAngles(g1: nx.Graph, g2: nx.Graph) -> list[float]:
    '''union all critical angles of two graphs in sorted set'''
    return computeCriticalAngles(g1).union(computeCriticalAngles(g2))

def computeAllAngles(g1: nx.Graph, g2: nx.Graph) -> tuple[SortedSet[float], list[float]]:
    '''get all critical angles of 2 graphs, sorted, as well as midpoints between those critical angles'''
    critical = list(computeAllCriticalAngles(g1, g2))

    # handling boundary case
    critical.append(critical[0] + 2*math.pi)

    window = sliding_window_view(critical, window_shape=2)
    midpoints = [(x[0]+x[1])/2 % (2*math.pi) for x in window]

    # clean up the boundary case
    critical = critical[:-1]

    # critical angles already sorted, this final sort is for the boundary condition
    return critical, sorted(midpoints)

def findFiltration(theta: float, origin: tuple[float,float]=(0,0)) -> tuple[tuple[float,float], tuple[float,float], int]:    
    '''get line of theta slope going through origin point, also track an inversion flag for dist'''
    # could assert this domain but can just adjust for it
    if theta >= 2*math.pi:
        theta = theta % (2*math.pi)
    
    if theta == 0:
        p1 = (origin[0], origin[1]+1)
    else:    
        # slope is normal to the angle
        slope = -1 / math.tan(theta)
        p1 = (origin[0]+1, origin[1]+slope)
    
    # flip distance function for certain angles
    # here, since "left" in our signed dist function is positive, inverting here
    sign = 1 if theta <= math.pi and theta > 0 else -1
    
    return (origin, p1, sign)

# type hinting might be slightly inaccurate here
def getHeightMatrix(mt: nx.DiGraph, verbose: bool=False) -> tuple[np.matrix, np.matrix]: 
    ''' LCA from networkx, assuming it to be better than Eulerian tour + spare table min range query, returning a matrix of pairwise LCA heights and then the matrix of LCA nodes
        
        this uses Tarjan's offline LCA optimized with inverse Ackermann function so O(Nodes+Queries) approximately
    '''
    # remove inf node
    size = mt.number_of_nodes()-1
    heightMatrix = np.zeros((size,size))
    nodeMatrix = np.zeros((size,size))
    
    # map actual node indexes to sorted matrix indexes
    allNodes = list(mt.nodes)
    allNodes.remove('inf')
    allNodes = sorted(allNodes)
    nodesDict = dict( zip(allNodes, range(len(allNodes))) )
    
    for LCA in nx.tree_all_pairs_lowest_common_ancestor(mt):
        # this is of the form ( (node1, node2), LCA )
        # handle root separately
        if LCA[0][0]=='inf' or LCA[0][1]=='inf':
            continue
        
        i = nodesDict[LCA[0][0]]
        j = nodesDict[LCA[0][1]]
        heightMatrix[i,j] = mt.nodes[LCA[1]]['height']
        heightMatrix[j,i] = mt.nodes[LCA[1]]['height']
        
        nodeMatrix[i,j] = LCA[1]
        nodeMatrix[j,i] = LCA[1]
        
    heightMatrix = np.matrix(heightMatrix)
    nodeMatrix = np.matrix(nodeMatrix)
    
    if verbose:
        print(nodesDict)
        print(heightMatrix)
        print(nodeMatrix)
        
    return heightMatrix, nodeMatrix

def calcDistanceMatrix(mt0: nx.DiGraph, mt1: nx.DiGraph, verbose: bool=False) -> tuple[np.matrix, np.matrix, np.matrix]:
    '''calculate difference matrix of LCA heights, also return LCA nodes of mt0 and mt1'''
    h0, nodes0 = getHeightMatrix(mt0, verbose=verbose)
    h1, nodes1 = getHeightMatrix(mt1, verbose=verbose)
    
    diff = np.subtract(h1, h0) 
    
    if verbose:
        print(diff)
        
    return diff, nodes0, nodes1

def calcDistanceInfNorm(mt0: nx.DiGraph, mt1: nx.DiGraph, verbose: bool=False) -> float:
    '''calculate infinity norm of two graphs, not used in the complete MT flow but useful as separate utility'''
    return np.max(np.abs(calcDistanceMatrix(mt0, mt1, verbose=verbose)[0]))

def getUnitVector(angle: float) -> tuple[float,float]:
    '''get 2D unit vector of input angle'''
    x = math.cos(angle)
    y = math.sin(angle)
    return (x,y)

def getMidpointKey(arr: list[float], target: float) -> float:
    '''get the midpoint key of the region that contains the target'''
    b = bisect.bisect_left( arr, target )
    if b>=len(arr) or b == 0:
        return ((arr[-1]+arr[0])/2 + math.pi) % (math.pi*2)
    else:
        return (arr[b]+arr[b-1])/2 
    
def _innerProduct(unitVec: tuple[float,float], pos: tuple[float,float]) -> float:
    # compute inner product of 2D vector with position
    return(unitVec[0] * pos[0] + unitVec[1] * pos[1])

def computeGraphDistanceAtAngle(angle: float, G1: nx.Graph, G2: nx.Graph, criticalDict: dict[float, np.matrix], midpointDict, verbose=False) -> float:
    '''compute distance of two embedded graphs at a given angle using cached precomputation and inferring all other angles'''
    angle = angle % (math.pi*2)
    
    criticalAngles = list(criticalDict.keys())
    if angle in criticalAngles:
        if verbose:
            print(f'using critical angle cache for: {angle}\n')
            
        diff = np.max(np.abs(criticalDict[angle]['diff']))
    else:
        # convert angle to unit vector
        unitVec = getUnitVector(angle)
        
        # find the computed midpoint matrix to work off of
        key = getMidpointKey(criticalAngles, angle)
        key = math.floor(key*1e9) / 1e9
        if verbose:
            print(f'found midpoint key {key} for angle {angle}\n')
        
        # intentionally ignoring the heights computed for the midpoint
        LCA0 = midpointDict[key]['LCA0']
        LCA1 = midpointDict[key]['LCA1']
        
        # both matrices should be same shape
        assert(LCA0.shape==LCA1.shape)
        A0 = np.matrix(np.zeros(LCA0.shape))
        A1 = np.matrix(np.zeros(LCA1.shape))
        
        # inner product of unit vector and LCA position
        for i in range(A0.shape[0]):
            for j in range(A0.shape[1]):
                pos0 = G1.nodes[LCA0[i,j]]['pos']
                A0[i,j] = _innerProduct(unitVec, pos0)
                
                pos1 = G2.nodes[LCA1[i,j]]['pos']
                A1[i,j] = _innerProduct(unitVec, pos1)
                
        diff = np.max(np.abs(np.subtract(A1, A0)))
    
    return diff

def computeDistanceAtAngleFromMT(g1_orig: nx.Graph, g2_orig: nx.Graph, angle: float, precision: int=5, show: bool=False, verbose: bool=False) -> tuple[np.matrix, np.matrix, np.matrix]:   
    '''given an angle and two graphs, compute full difference matrix and LCA node matrix'''
    g1 = g1_orig.copy()
    g2 = g2_orig.copy()
    
    filtration = findFiltration(angle)
    
    # assuming all the nodes are integers - using the conversion isn't reliable
    mt1 = buildMergeTree(g1, filtration, precision=precision, show=show, size=max(g1.nodes())+1, verbose=verbose)
    mt2 = buildMergeTree(g2, filtration, precision=precision, show=show, size=max(g2.nodes())+1, verbose=verbose)
    
    g1_proj = _mtToOtherGraph(mt2, g1, verbose=verbose)
    g2_proj = _mtToOtherGraph(mt1, g2, verbose=verbose)
    
    mt1_proj = buildMergeTree(g1_proj, filtration, precision=precision, show=show, size=max(g1_proj.nodes())+1, verbose=verbose)
    mt2_proj = buildMergeTree(g2_proj, filtration, precision=precision, show=show, size=max(g2_proj.nodes())+1, verbose=verbose)
    
    assert mt1_proj.number_of_nodes() == mt2_proj.number_of_nodes(), f"Error graph sizes don't match at {angle}, Graph 1: {mt1_proj.nodes()}, Graph 2: {mt2_proj.nodes()}"
    
    return calcDistanceMatrix(mt1_proj, mt2_proj)

def computeDistanceFull(g1: nx.Graph, g2: nx.Graph, precision: int=5, show: bool=False, verbose: bool=False) -> tuple[dict[str,np.matrix], dict[str,np.matrix]]:
    '''calculate all necessary pre-computations of height and LCA matrices for every critical angle and midpoint of two embedded graphs'''
    criticalAngles, midpoints = computeAllAngles(g1, g2)
    
    c_dict = {}
    for angle in criticalAngles:
        if show:
            print(f"using critical angle: {angle}")
        diff, LCA0, LCA1 = computeDistanceAtAngleFromMT(g1, g2, angle, precision=precision, show=show, verbose=verbose)
        c_dict[angle] = {'diff':diff, 'LCA0':LCA0, 'LCA1':LCA1}
    
    m_dict = {}
    for angle in midpoints:
        angle = math.floor(angle * 1e9) / 1e9
        if show:
            print(f"using midpoint: {angle}")
        diff, LCA0, LCA1 = computeDistanceAtAngleFromMT(g1, g2, angle, precision=precision, show=show, verbose=verbose)
        m_dict[angle] = {'diff':diff, 'LCA0':LCA0, 'LCA1':LCA1}
    
    return c_dict, m_dict

def getGraphDistance(G1_orig: nx.Graph, G2_orig: nx.Graph, precision: int=5, plot: bool=True, show: bool=False, verbose: bool=False, xMin: float=0, xMax: float=2*math.pi, n: int=10000) -> tuple[np.array, np.array]:
    '''get distances of two embedded graphs over linspace of n points from xMin to xMax, full driver code, plot to show graph, show is for internal merge trees'''
    G1, G2 = prepareTwoGraphs(G1_orig, G2_orig)

    # precache given the prepared graphs
    c_dict, m_dict = computeDistanceFull(G1, G2, precision=precision, show=show, verbose=verbose)
    
    X = np.linspace(xMin, xMax, n)
    Y = [computeGraphDistanceAtAngle(x, G1, G2, c_dict, m_dict) for x in X]

    if plot:
        plt.scatter(X,Y, marker=',', s=1)
        plt.show()
    
    return X, Y