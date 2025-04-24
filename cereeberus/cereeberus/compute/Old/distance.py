import networkx as nx
import math
import numpy as np

def edit(R1, R2):
    """Function to return the edit distance between two Reeb graphs.  Uses the graph_edit_distance function from https://networkx.org/documentation/stable/reference/algorithms/similarity.html.

    Args:
        R1 (reeb graph): Reeb graph or Merge tree
        R2 (reeb graph): Reeb graph or Merge tree

    Returns:
        edit_distance (int): graph edit distance
    """
    import networkx as nx
    edit_distance = nx.graph_edit_distance(R1.G, R2.G)
    return edit_distance

def findFiltration(theta: float, origin: "tuple[float,float]"=(0,0)) -> "tuple[tuple[float,float], tuple[float,float], int]":    
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

def getHeightMatrix(mt: nx.DiGraph, verbose: bool=False) -> "tuple[np.matrix, np.matrix]": 
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
        heightMatrix[i,j] = mt.nodes[LCA[1]]['fx']
        heightMatrix[j,i] = mt.nodes[LCA[1]]['fx']
        
        nodeMatrix[i,j] = LCA[1]
        nodeMatrix[j,i] = LCA[1]
        
    heightMatrix = np.matrix(heightMatrix)
    nodeMatrix = np.matrix(nodeMatrix)
    
    if verbose:
        print(nodesDict)
        print(heightMatrix)
        print(nodeMatrix)
        
    return heightMatrix, nodeMatrix

def calcDistanceMatrix(mt0: nx.DiGraph, mt1: nx.DiGraph, verbose: bool=False) -> "tuple[np.matrix, np.matrix, np.matrix]":
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

def getUnitVector(angle: float) -> "tuple[float,float]":
    '''get 2D unit vector of input angle'''
    x = math.cos(angle)
    y = math.sin(angle)
    return (x,y)

def getMidpointKey(arr: "list[float]", target: float) -> float:
    '''get the midpoint key of the region that contains the target'''
    import bisect
    b = bisect.bisect_left( arr, target )
    if b>=len(arr) or b == 0:
        return ((arr[-1]+arr[0])/2 + math.pi) % (math.pi*2)
    else:
        return (arr[b]+arr[b-1])/2 
    
def _innerProduct(unitVec: "tuple[float,float]", pos: "tuple[float,float]") -> float:
    # compute inner product of 2D vector with position
    return(unitVec[0] * pos[0] + unitVec[1] * pos[1])

def computeGraphDistanceAtAngle(angle: float, G1: nx.Graph, G2: nx.Graph, criticalDict: "dict[float, np.matrix]", midpointDict, verbose=False) -> float:
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


def computeDistanceAtAngleFromMT(g1_orig: nx.Graph, g2_orig: nx.Graph, angle: float, precision: int=5, show: bool=False, verbose: bool=False) -> "tuple[np.matrix, np.matrix, np.matrix]":   
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

def computeDistanceFull(g1: nx.Graph, g2: nx.Graph, precision: int=5, show: bool=False, verbose: bool=False) -> "tuple[dict[str,np.matrix], dict[str,np.matrix]]":
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

def directedMergeTreeDistance(G1_orig: nx.Graph, G2_orig: nx.Graph, precision: int=5, plot: bool=True, show: bool=False, verbose: bool=False, xMin: float=0, xMax: float=2*math.pi, n: int=10000) -> "tuple[np.array, np.array]":
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