from cereeberus import MapperGraph
import numpy as np
import networkx as nx
from scipy.linalg import block_diag
from matplotlib import pyplot as plt

class Interleave:
    """
    A class to bound the interleaving distance between two Mapper graphs.
    """
    def __init__(self, F, G, n = 1):
        """
        Initialize the Interleave object.
        
        Parameters:
            F : MapperGraph
                The first Mapper graph.
            G : MapperGraph
                The second Mapper graph.
            n : int
                The interleaving parameter. The default is 1.
        """
        
        self.n = n

        # ---
        # Containers for matrices for later 
        # self.A = {'F':{}, 'G':{}} # adjacency matrix
        self.B = {'F':{}, 'G':{}} # boundary matrix
        self.D = {'F':{}, 'G':{}} # distance matrix
        self.I = {'F':{}, 'G':{}} # induced maps

        self.val_to_verts = {'F':{}, 'G':{}} # dictionaries from function values to vertices
        self.val_to_edges = {'F':{}, 'G':{}} # dictionaries from function values to edges

        # ----
        # Make F graphs and smoothed versions
        self.F = {}
        self.F['0'] = F 
        self.F['n'], I_0 = F.smoothing(self.n, return_map = True)
        self.F['2n'], I_n = self.F['n'].smoothing(self.n, return_map = True)

        # Get the dictionaries needed for the induced maps' block structure 
        for key in ['0', 'n', '2n']:
            self.val_to_verts['F'][key] = self.F[key].func_to_vertex_dict()
            self.val_to_edges['F'][key] = self.F[key].func_to_edge_dict()

        
        
        # Make the induced map from F_0 to F_n
        self.I['F']['0'] = {}
        self.I['F']['0']['V'] = self.map_dict_to_matrix(I_0, 
                                                        self.val_to_verts['F']['n'], 
                                                        self.val_to_verts['F']['0'])
        I_0_edges = {(e[0], e[1], 0): (I_0[e[0]], I_0[e[1]],0) for e in self.F['0'].edges()}
        self.I['F']['0']['E'] = self.map_dict_to_matrix(I_0_edges, 
                                                        self.val_to_edges['F']['n'], 
                                                        self.val_to_edges['F']['0'])

        # Make the induced map from F_n to F_2n
        self.I['F']['n'] = {}
        self.I['F']['n']['V'] = self.map_dict_to_matrix(I_n, 
                                                        self.val_to_verts['F']['2n'], 
                                                        self.val_to_verts['F']['n'])
        # Note that in this setting, the induced map on edges is the same as the map sending the edge to the edge with endpoints given by the vertices since there are no double edges for any smoothing >= 1. 
        I_n_edges = {(e[0], e[1], 0): (I_n[e[0]], I_n[e[1]],0) for e in self.F['n'].edges()}
        self.I['F']['n']['E'] = self.map_dict_to_matrix(I_n_edges, 
                                                        self.val_to_edges['F']['2n'], 
                                                        self.val_to_edges['F']['n'])

        # ----
        # Now do the same for G
        self.G = {}
        self.G['0'] = G 
        self.G['n'], I_0 = G.smoothing(self.n, return_map = True)
        self.G['2n'], I_n = self.G['n'].smoothing(self.n, return_map = True)

        # Get the dictionaries needed for the induced maps' block structure 
        for key in ['0', 'n', '2n']:
            self.val_to_verts['G'][key] = self.G[key].func_to_vertex_dict()
            self.val_to_edges['G'][key] = self.G[key].func_to_edge_dict()

        # Make the induced map from G_0 to G_n
        self.I['G']['0'] = {}
        self.I['G']['0']['V'] = self.map_dict_to_matrix(I_0, 
                                self.val_to_verts['G']['n'], 
                                self.val_to_verts['G']['0'])
        I_0_edges = {(e[0], e[1], 0): (I_0[e[0]], I_0[e[1]],0) for e in self.G['0'].edges()}
        self.I['G']['0']['E'] = self.map_dict_to_matrix(I_0_edges, 
                                self.val_to_edges['G']['n'], 
                                self.val_to_edges['G']['0'])

        # Make the induced map from G_n to G_2n
        self.I['G']['n'] = {}
        self.I['G']['n']['V'] = self.map_dict_to_matrix(I_n, 
                                self.val_to_verts['G']['2n'], 
                                self.val_to_verts['G']['n'])
        I_n_edges = {(e[0], e[1], 0): (I_n[e[0]], I_n[e[1]],0) for e in self.G['n'].edges()}
        self.I['G']['n']['E'] = self.map_dict_to_matrix(I_n_edges, 
                                self.val_to_edges['G']['2n'], 
                                self.val_to_edges['G']['n'])
        
        # End making smoothings and induced maps
        # ----
        # ---
        # Build boundary matrices 

        # Boundary matrix for F
        for key in ['0', 'n', '2n']:
            
            self.B['F'][key] = self.F[key].boundary_matrix(astype = 'dict')

        # Boundary matrix for G
        for key in ['0', 'n', '2n']:
            self.B['G'][key] = self.G[key].boundary_matrix(astype = 'dict')

        # End boundary matrices
        # ---

        # ---
        # Build the distance matrices 
        for (metagraph, name) in [ (self.F,'F'), (self.G,'G')]:
            for key in ['0', 'n', '2n']:
                M = metagraph[key]

                length = dict(nx.all_pairs_shortest_path_length(M.to_undirected()))

                val_to_verts = self.val_to_verts[name][key]

                block_dict = {}
                for f_i in val_to_verts:
                    vert_set = val_to_verts[f_i]
                    D_i = np.zeros((len(vert_set), len(vert_set)))
                    for i, v1 in enumerate(vert_set):
                        for j, v2 in enumerate(vert_set):
                            if i!=j:
                                D_i[i, j] = length[v1][v2]/2
                    block_dict[f_i] = {'rows': vert_set, 'cols': vert_set, 'array': D_i}
                
                self.D[name][key] = block_dict



        # ----
        # phi: F -> G^n

        # Initialize the phi matrices. These will have all 0 entries.
        self.phi_V = self.map_dict_to_matrix(None, self.val_to_verts['G']['n'], self.val_to_verts['F']['0'])


        self.phi_E = {}
        # TODO: Edge version!


        # End phi
        # ---



        # ----
        # psi: G -> F^n
        
        # Initialize the psi matrices. These will have all 0 entries.
        self.psi_V = self.map_dict_to_matrix(None, self.val_to_verts['F']['n'], self.val_to_verts['G']['0'])

        self.psi_E = {}
        # TODO: Edge version!

        # End psi
        # ---

    def map_dict_to_matrix(self, map_dict, 
                           row_val_to_verts, 
                           col_val_to_verts):
        """
        Convert a dictionary of maps to a matrix. 
        We have row objects and column objects (for example vertices from $F$ for columns and vertices from $G_n$ for rows although this all should work for edge versions as well).
        Input 'map_dict' is a dictionary from column objects to row objects. However, 'map_dict' can also be passed in as None, in which case this will set up the structure of the block dictionary but not fill in the array.

        Also provided are dictionaries from function values to a list of objects for the row and column objects.

        We will break this up into a dictionary of matrices, one for each function value. Out[i] has keys 'rows', 'cols', and 'array' for each function value i. The array should be a numpy array with the rows corresponding to the rows and the columns corresponding to the columns.

        Parameters:
            map_dict : dict
                A dictionary from column objects to row objects.
            row_val_to_verts : dict
                A dictionary from function values to a list of row objects.
            col_val_to_verts : dict
                A dictionary from function values to a list of column objects.
        
        Returns:
            dict
                A dictionary of matrices, one for each function value.

        """
        Out = {}
        for i in col_val_to_verts:
            # i is the function value of the vertices 
            matrix_dict = {}
            matrix_dict['rows'] = row_val_to_verts[i]
            matrix_dict['cols'] = col_val_to_verts[i]
            matrix_dict['array'] = np.zeros((len(row_val_to_verts[i]), len(col_val_to_verts[i])))

            if map_dict is not None:
                # only fill in the array if map data is provided
                for col_i, vert in enumerate(matrix_dict['cols']):
                        row_j = matrix_dict['rows'].index(map_dict[vert])
                        matrix_dict['array'][row_j, col_i] = 1


            Out[i] = matrix_dict

        return Out

    def block_dict_to_matrix(self, block_dict):
        """
        Convert a dictionary of blocks to a block diagonal matrix.
        The dictionary should have keys that are integers and values that are dictionaries with keys 'rows', 'cols', and 'array' that are lists of rows, columns, and a numpy array respectively.

        parameters:
            block_dict : dict
                A dictionary of blocks.
        
        returns:
            dict
                A dictionary with keys 'rows', 'cols', and 'array' that are lists of rows, columns, and a numpy array respectively.
        """
        Blocks = block_dict

        a = np.min(list(Blocks.keys()) )
        b = np.max(list(Blocks.keys()) )

        rows = [Blocks[i]['rows'] for i in range(a, b+1)]
        rows = sum(rows, []) # flatten the list
        cols = [Blocks[i]['cols'] for i in range(a, b+1)]
        cols = sum(cols, []) # flatten the list

        arrays = [ Blocks[i]['array'] for i in range(a, b+1)]

        BigMatrix = block_diag(*arrays)

        return {'rows': rows, 'cols': cols, 'array': BigMatrix}
    
    def draw_matrix(self, matrix_dict, **kwargs):
        """
        Draw a matrix with row and column labels.

        The dictionary can either be passed in as a block dictionary or a matrix dictionary.

        A block dictionary has keys that are integers and values that are dictionaries with keys 'rows', 'cols', and 'array' that are lists of rows, columns, and a numpy array respectively.

        A matrix dictionary has keys 'rows', 'cols', and 'array' that are lists of rows, columns, and a numpy array respectively.
        
        Parameters:
            matrix_dict : dict
                A dictionary (or dictionary of dictionaries) with keys 'rows', 'cols', and 'array' that are lists of rows, columns, and a numpy array respectively.
        """

        if 'array' not in matrix_dict:
            matrix_dict = self.block_dict_to_matrix(matrix_dict)

        plt.matshow(matrix_dict['array'], **kwargs)
            
        # Add vertices as the row and column labels
        plt.xticks(range(len(matrix_dict['cols'])), matrix_dict['cols'], rotation = 90)
        plt.yticks(range(len(matrix_dict['rows'])), matrix_dict['rows'])

    def draw_I(self, graph = 'F', key = '0', type = 'V', **kwargs):
        """
        Draw the induced map from one Mapper graph to another.

        Parameters:
            graph : str
                The graph to draw the induced map for. Either 'F' or 'G'.
            key : str
                The key for the induced map. Either '0' or 'n'.
        """
        matrixDict = self.I[graph][key][type]
        self.draw_matrix(matrixDict, **kwargs)
        plt.xlabel(f"{graph}_{key}")
        if key == '0':
            plt.ylabel(f"{graph}_n")
        else:
            plt.ylabel(f"{graph}_2n")

    def draw_B(self, graph = 'F', key = '0', **kwargs):
        """
        Draw the boundary matrix for a Mapper graph.

        Parameters:
            graph : str
                The graph to draw the boundary matrix for. Either 'F' or 'G'.
            key : str
                The key for the boundary matrix. Either '0', 'n', or '2n'.
        """
        self.draw_matrix(self.B[graph][key], **kwargs)
        plt.xlabel(f"E({graph}_{key})")
        plt.ylabel(f"V({graph}_{key})")

    def draw_D(self, graph = 'F', key = '0', **kwargs):
        """
        Draw the distance matrix for a Mapper graph.

        Parameters:
            graph : str
                The graph to draw the distance matrix for. Either 'F' or 'G'.
            key : str
                The key for the distance matrix. Either '0', 'n', or '2n'.
        """
        self.draw_matrix(self.D[graph][key], **kwargs)
        plt.xlabel(f"V({graph}_{key})")
        plt.ylabel(f"V({graph}_{key})")
        plt.colorbar()
