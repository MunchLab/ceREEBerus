import networkx as nx
import math

def signedDistToLine2Pts(pt: tuple, p0: tuple, p1: tuple) -> float:
    '''return a signed distance to a line where line is defined as two points
    
    positive sign refers to "above" the line or "left" of a vertical line
    to get the expected sign of "right" is positive, the vertical line will be inverted back under the "angle_sign" in _computeNodeHeights() of MergeTree.py
    '''
    return ((p0[0]-pt[0])*(p1[1]-p0[1]) - (p1[0]-p0[0])*(p0[1]-pt[1])) / math.dist(p0,p1)

def _computeNodeHeights(graph: nx.Graph, filtration: "tuple[tuple[float,float], tuple[float, float], int]", precision: int=5) -> "dict[int, tuple[float,bool]]":
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

def getSortedNodeHeights(graph: nx.Graph, filtration: "tuple[tuple[float,float], tuple[float, float], int]", precision: int=5) -> "list[tuple[int,float]]":
    '''compute heights of each node given filtration line and return as sorted list of node height tuples, rounded to given precision'''
    
    # sorted first by height, then non-projected first
    heightTuples = _computeNodeHeights(graph, filtration, precision=precision)
    
    # only need the nodes and height so rehspaing
    return [(x[0], x[1][0]) for x in sorted(heightTuples.items(), key=lambda x:x[1])]

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