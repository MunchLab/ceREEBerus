from cereeberus import MapperGraph
import numpy as np
import networkx as nx
from scipy.linalg import block_diag
from matplotlib import pyplot as plt

class Interleave:
    """
    A class to bound the interleaving distance between two Mapper graphs.
    """
    def __init__(self, F, G, 
                        n = 1, 
                        initialize_random_maps = False, seed = None):
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

                val_to_verts = self.val_to_verts[name][key]

                block_dict = {}
                for f_i in val_to_verts:
                    vert_set = val_to_verts[f_i]
                    D_i = np.zeros((len(vert_set), len(vert_set)))
                    for i in range(len(vert_set)):
                        for j in range(i+1, len(vert_set)):
                            D_i[i, j] = M.thickening_distance(vert_set[i], vert_set[j])
                            D_i[j, i] = D_i[i, j]
                    block_dict[f_i] = {'rows': vert_set, 'cols': vert_set, 'array': D_i}
                
                self.D[name][key] = block_dict



        # ----
        # phi: F -> G^n

        # Initialize the phi matrices. These will have all 0 entries.
        self.phi = {'V':{}, 'E':{}}
        self.phi['V'] = self.map_dict_to_matrix(None, 
                                    self.val_to_verts['G']['n'], 
                                    self.val_to_verts['F']['0'],
                                    random_initialize = initialize_random_maps)


        if initialize_random_maps:
            B_Gn = self.B['G']['n']['array']
            phi = self.block_dict_to_matrix(self.phi['V'])['array']
            B_F = self.B['F']['0']['array']

            A = B_Gn.T @ (phi @ B_F)
            A = np.floor(A/2)
            self.phi['E'] = self.matrix_to_block_dict(A,  self.val_to_edges['G']['n'], self.val_to_edges['F']['0'])

        else:
            self.phi['E'] = self.map_dict_to_matrix(None, self.val_to_edges['G']['n'], self.val_to_edges['F']['0'])

        # End phi
        # ---



        # ----
        # psi: G -> F^n
        
        # Initialize the psi matrices. These will have all 0 entries.
        self.psi = {'V':{}, 'E':{}}

        self.psi['V'] = self.map_dict_to_matrix(None, 
                                    self.val_to_verts['F']['n'], 
                                    self.val_to_verts['G']['0'],
                                    random_initialize = initialize_random_maps)

        if initialize_random_maps:
            B_Fn = self.B['F']['n']['array']
            psi = self.block_dict_to_matrix(self.psi['V'])['array']
            B_G = self.B['G']['0']['array']

            A = B_Fn.T @ (psi @ B_G)
            A = np.floor(A/2)
            self.psi['E'] = self.matrix_to_block_dict(A,  self.val_to_edges['F']['n'], self.val_to_edges['G']['0'])

        else:
            self.psi['E'] = self.map_dict_to_matrix(None, self.val_to_edges['F']['n'], self.val_to_edges['G']['0'])

        # End psi
        # ---
    
    def random_initialize(self, block_dict, seed = None):
        """
        Randomly initialize the block dictionary.

        Parameters:
            block_dict : dict
                A dictionary with keys 'rows', 'cols', and 'array' that are lists of rows, columns, and a numpy array respectively.
        """
        block_dict = block_dict.copy()
        for i in block_dict.keys():
            A = block_dict[i]['array']
            
            rng = np.random.default_rng(seed)
            col_1s = rng.integers(0, A.shape[0], size = A.shape[1])
            A[col_1s, list(range(A.shape[1]))] = 1
            block_dict[i]['array'] = A

        return block_dict

    def map_dict_to_matrix(self, map_dict, 
                           row_val_to_verts, 
                           col_val_to_verts,
                           random_initialize = False,
                           seed = None):
        """
        Convert a dictionary of maps to a matrix. 
        We have row objects and column objects (for example vertices from $F$ for columns and vertices from $G_n$ for rows although this all should work for edge versions as well).
        Input 'map_dict' is a dictionary from column objects to row objects. However, 'map_dict' can also be passed in as None, in which case this will set up the structure of the block dictionary but not fill in the array. If `map_dict` is None and `random_initialize` is True, then the array will be filled with a random 1 in each column.

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
            else:
                if random_initialize:
                    A = matrix_dict['array']             
                    rng = np.random.default_rng(seed = seed)
                    col_1s = rng.integers(0, A.shape[0], size = A.shape[1])
                    A[col_1s, list(range(A.shape[1]))] = 1
                    matrix_dict['array'] = A


            Out[i] = matrix_dict

        for i in row_val_to_verts.keys() ^ col_val_to_verts.keys():
            if i in row_val_to_verts:
                Out[i] = {'rows': row_val_to_verts[i], 'cols': [], 'array': []}
            else:
                raise ValueError('There is a bug, there are column function values not in the row values and this should not happen')

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

    def matrix_to_block_dict(self, matrix, row_val_to_verts, col_val_to_verts):
        """
        Turn a matrix back into a block dictionary.

        Parameters:
            matrix : np.array
                A matrix.
            row_val_to_verts : dict
                A dictionary from function values to a list of row objects.
            col_val_to_verts : dict
                A dictionary from function values to a list of column objects.
        """
        min_i = min(row_val_to_verts.keys() | col_val_to_verts.keys())
        max_i = max(row_val_to_verts.keys() | col_val_to_verts.keys())

        block_dict = {}

        curr_row = 0
        curr_col = 0

        for i in range(min_i, max_i + 1):
            try:
                rows = row_val_to_verts[i]
                next_row = curr_row + len(rows)
            except KeyError:
                rows = []
                next_row = curr_row
            
            try:
                cols = col_val_to_verts[i]
                next_col = curr_col + len(col_val_to_verts[i])
            except KeyError:
                cols = []
                next_col = curr_col 

            A = matrix[curr_row:next_row, curr_col:next_col]
            block_dict[i] = {'rows': rows, 'cols': cols, 'array':A}

            curr_row = next_row
            curr_col = next_col

        return block_dict

    def check_column_sum(self, matrix_dict, verbose = False):
        """
        Check that the sum of each column is 1.

        Parameters:
            matrix_dict : dict
                Either a dictionary with keys 'rows', 'cols', and 'array' that are lists of rows, columns, and a numpy array respectively), or a block dictionary where keys are function values and output is a dictionary of the above form. 
            verbose : bool
                Prints information on which matrices have columns that do not sum to 1 if True. The default is False.

        Returns:
            bool
                True if the columns sum to 1, False otherwise.
        """
        
        if 'array' in matrix_dict:
            # This will be false if any of the columns does not sum to 1
            check = np.all(matrix_dict['array'].sum(axis = 0) == 1)

            if not check and verbose : 
                print('The columns of the distance matrix do not sum to 1')

        else:
            for i in matrix_dict.keys():
                D_small = matrix_dict[i]
                check = np.all(D_small['array'].sum(axis = 0) == 1) 

                if not check and verbose: 
                    print(f'The columns of the distance matrix for function value {i} do not sum to 1')

        return check
    
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
