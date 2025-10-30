
from ..distance.labeled_blocks import LabeledBlockMatrix as LBM
from ..distance.labeled_blocks import LabeledMatrix as LM
from ..compute.unionfind import UnionFind
import numpy as np

from .reebgraph import ReebGraph
class MapperGraph(ReebGraph):
    r"""
    A mapper graph structure. This inherits the properties of the Reeb graph in that it is a graph with a function given on the vertices, but with some additional requirements.

    - The values are integers in some range, [n_low, n_low+1, \cdots, n_high], although we consider the funciton values to be [\delta * n_low, \delta* (n_low+1), \cdots, \delta * n_high] for a stored delta 
    - If an edge crosses a value, it has a vertex (so that the inverse image of any integer is only vertices, not interiors of edges)
    - An internal delta is stored so that this can be interpreted as function values [\delta * n_low, \delta* (n_low+1), \cdots, \delta * n_high] 
    """

    def __init__(self, 
                 G=None, f={}, delta = None, seed = None, verbose=False):

        # Check that $f$ values are only integers
        if not all([isinstance(f[v], int) for v in f]):
            raise ValueError("Function values must be integers.")
        if delta is None:
            self.delta = 1
        else: 
            self.delta = delta 

        super().__init__(G, f, seed, verbose)

        self.mapperify()

    def add_edge(self, u, v, reset_pos=True):
        """
        Add an edge to the graph. This will also update the internal structure to make sure it satisfies the mapper properties.
        """
        super().add_edge(u, v, reset_pos)
        self.mapperify()



    def mapperify(self):
        """
        Take the internal structure and make sure it satisfies the requirement that all edges have adjacent function values. 
        """

        # If we're initializing with nothing, this should pass. 
        # Note that if self.n_low is None, then self.n_high and self.delta
        # are both None as well but I am not currently checking that.
        try:
            n_low = min(self.f.values())
            n_high = max(self.f.values())

        except:
            return
        
        
        for i in range(n_low,n_high+1):
            e_list = [e for e in self.edges() if self.f[e[0]] < i and self.f[e[1]] > i]

            for e in e_list:
                w_name = self.get_next_vert_name()
                self.subdivide_edge(*e,w_name, i)

            
    
    def add_node(self, vertex, f_vertex, reset_pos=True):

        """
        Same as adding a node in Reeb, but with the additional requirement that the function value is an integer.
        """

        if not isinstance(f_vertex, int):
            raise ValueError("Function values must be integers.")
        return super().add_node(vertex, f_vertex, reset_pos)
    


    def set_pos_from_f(self, seed=None, verbose=False):

        """
        Same as the Reeb graph function, but we want to draw the vertex locations at delta*function value.
        """
        super().set_pos_from_f(seed, verbose)

        for v in self.nodes():
            self.pos_f[v] = (self.pos_f[v][0],self.delta * self.f[v])

    def induced_subgraph(self, nodes):
        """
        Returns the subgraph of the mapper graph induced by the nodes in the list nodes.

        Parameters:
            nodes (list): The list of nodes to include in the subgraph.
        
        Returns:
            MapperGraph
        """
        R = super().induced_subgraph(nodes)
        return R.to_mapper(self.delta)


    def smoothing_and_maps(self, n = 1):
        """
        Compute the smoothing of a mapper graph as given in todo: Cite the paper. Note that the input :math:`n` parameter is related to the integer function values, not the delta-scaled function values.

        Parameters:
            n (int): The amount of smoothing
        
        Returns:
            tuple: MapperGraph, vertex_map, edge_map 
        """
        if type(n) != int:
            raise ValueError("Smoothing amount must be an integer.")
        
        M_n, V_map, E_map = super().smoothing_and_maps(n)
        
        # E_map is a dictonary with output lists of edges, but we should only have one edge to one edge in the mapper graph  case
        # This just strips E_map[key] = [(u,v,0)] to instead be E_map[key] = (u,v,0)
        E_map = {key : E_map[key][0] for key in E_map}
        
        M_n = M_n.to_mapper(self.delta)
        
        return M_n, V_map, E_map 

    def smoothing(self, n=1):
        """
        Compute the smoothing of a mapper graph as given in todo: Cite the paper. Note that the input :math:`n` parameter is related to the integer function values, not the delta-scaled function values.

        Args:
            n (int, optional): Smoothing amount. Defaults to 1.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        M_n, _, _ = self.smoothing_and_maps(n)
        return M_n

    #------------------------------#
    # Functions for computing thickening distance matrix
    #------------------------------#
    def thickening_distance_by_level(self, level, obj_type = 'V'):
        """
        Get the thickening distance matrix at a given level. This distance is the amount of thickening needed before the given pair of vertices at that level map to the same connected component.

        Parameters:

            level : int. The level to get the thickening distance matrix at.
            obj_type : str. 'V' or 'E' to get the distance matrix for vertices or edges, respectively. Default is 'V'.

        Returns:
            LabeledMatrix
            
        """
        # Dict to get list of vertices at a level
        LevelVerts = self.func_to_vertex_dict()

        # Dict to get lists of edges at a level 
        LevelEdges = self.func_to_edge_dict()

        # Current level to be checking 
        L = level 

        # Objects to be checking
        if obj_type == 'V':
            # Vertices at the current level
            rowLabels = LevelVerts[L]
        elif obj_type == 'E':
            # Edges at the current level
            rowLabels = LevelEdges[L]
        else:
            raise ValueError("Type must be 'V' or 'E'.")


        # If there's only one object, the distance is 0. 
        # Return the little matrix block
        if len(rowLabels) == 1:
            D = LM(rows = rowLabels, cols = rowLabels)
            return D

        # Max difference to check 
        max_diff = max(L-self.min_f(), self.max_f()-L)

        # Distance matrix for this level
        D = np.zeros(shape = (len(rowLabels), len(rowLabels))) - 1
        D += np.identity(len(rowLabels))
        D = LM(D, rows = rowLabels, cols = rowLabels)

        # Initialize a union find object
        UF = UnionFind(list(self.nodes()))



        # Loop through the levels to add edges, and update teh distance matrix if 
        # a pair of objects not already merged.
        for k in range(1, max_diff+1):
            
            if obj_type == 'V':
                up_level = L+k-1
                down_level = L-k
            elif obj_type == 'E':
                up_level = L+k-1
                down_level = L-k+1

            U = []
            # Add edges at each up and down level
            # try/except is to ignore entries without edges
            try:
                # up verts 
                U.extend(LevelEdges[up_level])
            except:
                pass

            try:
                # down verts
                U.extend(LevelEdges[down_level])
            except:
                pass

            # Add these edges to the union find object
            for e in U:
                UF.union(e[0], e[1])

            if obj_type == 'V':
                for v in rowLabels:
                    for u in rowLabels:
                        if v != u and D.array[rowLabels.index(v)][rowLabels.index(u)] == -1:
                            if UF.find(v) == UF.find(u):
                                # print(f"found {v} and {u}")
                                D.array[rowLabels.index(v)][rowLabels.index(u)] = k
                                D.array[rowLabels.index(u)][rowLabels.index(v)] = k
            elif obj_type == 'E':
                for i, e in enumerate(rowLabels):
                    for j, f in enumerate(rowLabels):
                        if e != f and D.array[i][j] == -1:
                            if UF.find(e[0]) == UF.find(f[0]): # and UF.find(e[1]) == UF.find(f[1]):
                                # print(f"found {e} and {f}")
                                D.array[i][j] = k
                                D.array[j][i] = k
                
            
            # Check if there are still any entries of -1 
            # If there are none, no need to keep adding edges 
            if not np.any(D.array == -1):
                break
        
        # If there are still -1's, set them to np.inf
        if np.any(D.array == -1):
            D.array[D.array == -1] = np.inf

        return D
    
    def thickening_distance_matrix(self, obj_type = 'V'):
        """
        Get the thickening distance matrix for the entire mapper graph. This is a labeled block matrix with rows and columns indexed by vertices, and entries given by the thickening distance between the two vertices.

        Returns:
            LabeledBlockMatrix
        """
        if obj_type == 'V':
            V_dict = self.func_to_vertex_dict()
            DistMat = LBM(rows_dict= V_dict, cols_dict = V_dict)
        elif obj_type == 'E':
            E_dict = self.func_to_edge_dict()
            DistMat = LBM(rows_dict= E_dict, cols_dict = E_dict)

        for i in DistMat.get_all_block_indices():
            D = self.thickening_distance_by_level(i, obj_type = obj_type)
            DistMat[i] = D

        return DistMat
    
