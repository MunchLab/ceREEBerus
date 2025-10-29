
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
from ..compute.unionfind import UnionFind
from .labeled_blocks import LabeledBlockMatrix as LBM
from .labeled_blocks import LabeledMatrix as LM
from .ilp import solve_ilp
from ..compute.utils import HiddenPrints
import sys, os

class Interleave:
    # Import MapperGraph inside methods as needed to avoid circular import
    """
    A class to compute the interleaving distance between two Mapper graphs, denoted :math:`F` and :math:`G`. The interleaving distance is a measure of how similar two Mapper graphs are, based on the induced maps between them.
    
    Once the ``Interleave.fit()`` command has been run, the optimal bound for the distance for the input mapper graphs is stored as ``Interleave.n``. The resulting maps can be found using the ``Interleave.phi()`` and ``Interleave.psi()`` functions. For more detailed structure, the full interleaving is stored using an `Assignment` class, documented below, and can be accessed using ``Interleave.assignment``.
    
    """
    
    def __init__(self, F, G):
        """
        Initialize the Interleave object.
        
        Parameters:
            F (MapperGraph):
                The first Mapper graph.
            G (MapperGraph):
                The second Mapper graph.
        """
        
        self.F = F
        self.G = G
        self.n = np.inf
        self.assignment = None
        
    def old_fit(self, pulp_solver = None, verbose = False, max_n_for_error = 100, 
            ):
        """
        Compute the interleaving distance between the two Mapper graphs.
        
        Parameters:
            pulp_solver (pulp.LpSolver): 
                The solver to use for the ILP optimization. If None, the default solver is used.
            verbose (bool, optional): 
                If True, print the progress of the optimization. Defaults to False.
            max_n_for_error (int, optional): 
                The maximum value of `n` to search for. If the interleaving distance is not found by this value, a ValueError is raised. Defaults to 100.
            printOptimizerOutput (bool, optional):
                If True, the output of the PULP optimizer is printed. Defaults to False.
        
        Returns:
            Interleave : 
                The Interleave object with the computed interleaving distance.
        """
        # -- Search through possible n values to find the smallest one that still allows for interleaving
        
        # -- Dictionary to store the search data 
        # -- This will store the Loss for each value of n so we don't 
        # -- have to recompute it each time. 
        # -- The distance bound can be determined by search_data[key] = Loss implies d_I <= key + Loss
        search_data = {} 
        
        # -- Set the initial value of the distance bound to be infinity
        distance_bound = np.inf

        # -- Do an expoential search for the smallest n that allows for interleaving
        # -- Start with n = 1 and double it until the Loss is less than or equal to 0
        N = 1 
        found_max = False
            
        while not found_max:
            # -- Compute the assignment for the current value of n
            # -- If the Loss is less than or equal to 0, we have found a valid n
            # -- Otherwise, double n and try again

            try:
                if verbose: 
                    print(f"\n-\nTrying n = {N}...")
                myAssgn = Assignment(self.F, self.G, n = N)
                
                Loss = myAssgn.optimize()
                search_data[N] = Loss
                
                if verbose: 
                    print(f"n = {N}, Loss = {Loss}, distance_bound = {N + Loss}")
                
            except ValueError as e:
                # -- If we get a ValueError, it means the current n is too small
                # -- So we double n and try again
                search_data[N] = np.inf
                N *= 2
                continue
            
            if Loss <= 0:
                # -- If the Loss is less than or equal to 0, we have found a valid n
                max_N = N
                found_max = True
                if verbose:
                    print(f"Found valid maximum n = {N} with Loss = {Loss}. Moving on to binary search.")
            else:
                # -- If the Loss is greater than 0, we need to try a larger n
                N *= 2
                
            if N > max_n_for_error:
                # -- If we have exceeded the maximum value of n, raise an error. Useful while we're checking this while loop
                raise ValueError(f"Interleaving distance not found for n <= {max_n_for_error}.")
                
            

        # now binary search in [0, max_N] to find the smallest n that gives a loss of 0
        
        # -- Set the initial values for the binary search
        low = 1
        high = max_N
        while low < high:
            mid = (low + high) // 2
            
            try:
                myAssgn = Assignment(self.F, self.G, n = mid)
                Loss = myAssgn.optimize()
                search_data[mid] = Loss
            except ValueError as e:
                # -- If we get a ValueError, it means the current n is too small
                search_data[mid] = np.inf
                low = mid + 1
                continue
            
            if Loss <= 0:
                # -- If the Loss is less than or equal to 0, we have found a valid n
                high = mid
            else:
                low = mid + 1
                
        # -- Set the final value of n to be the smallest one that gives a loss of 0
        self.n = low
        # -- Set the assignment to be the one that minimizes the interleaving distance
        self.assignment = Assignment(self.F, self.G, n = self.n)
        Loss = self.assignment.optimize(pulp_solver = pulp_solver,)
        
        
        # Raise error if the loss isn't 0 
        if Loss > 0:
            raise ValueError(f"Final fit object is not an interleaving. Loss = {Loss}. N = {self.n}.")
        
        return self.n
    
    def fit(self, pulp_solver = None, verbose= False, max_n_for_error = 100):
        """
        Compute the interleaving distance between the two Mapper graphs.
        
        Parameters:
            pulp_solver (pulp.LpSolver): 
                The solver to use for the ILP optimization. If None, the default solver is used.
            verbose (bool, optional): 
                If True, print the progress of the optimization. Defaults to False.
            max_n_for_error (int, optional): 
                The maximum value of `n` to search for. If the interleaving distance is not found by this value, a ValueError is raised. Defaults to 100. ####NOTE: this can be replaced by the bounding box.
            
        Returns:
            Int:
                The interleaving distance, that is, the smallest n for which loss is zero.
        """

        # step 0: search for n=0 
        myAssgn = Assignment(self.F, self.G, n = 0)
        Loss = myAssgn.optimize(pulp_solver = pulp_solver)

        if verbose:
            print(f"\n-\nTrying n = 0...")
            print(f"n = 0, Loss = {Loss}, distance_bound = {0 + Loss}")

        # if loss is 0, we're done
        if Loss == 0:
            self.n = 0
            self.assignment = myAssgn
            return self.n
        
        # step 1: exponential search for the upperbound
        low, high = 1, 1
        found_valid_n = False

        while high <= max_n_for_error:
            try:
                myAssgn = Assignment(self.F, self.G, n = high)
                Loss = myAssgn.optimize(pulp_solver = pulp_solver)

                if verbose:
                    print(f"\n-\nTrying n = {high}...")
                    print(f"n = {high}, Loss = {Loss}, distance_bound = {high + Loss}")

                if  Loss == 0:
                    found_valid_n = True
                    break
                low, high = high, high*2
            except ValueError: # infeasible assignment
                low, high = high, high*2

        if not found_valid_n:
            raise ValueError(f"Interleaving distance not found for n <= {max_n_for_error}.")
        
        high = min(high, max_n_for_error)  # Clamp to max allowed
        # step 2: binary search for the optimal n
        
        low = (high//2) + 1 if high > 1 else 1
        best_n = high

        while low <= high:
            mid = (low + high) // 2 
            try:
                myAssgn = Assignment(self.F, self.G, n = mid)
                Loss = myAssgn.optimize(pulp_solver = pulp_solver)
                
                if verbose:
                    print(f"\n-\nTrying n = {mid}...")
                    print(f"n = {mid}, Loss = {Loss}, distance_bound = {mid + Loss}")
                
                if Loss == 0:
                    best_n = mid
                    high = mid - 1 # decrease n to increase the loss
                else:
                    low = mid + 1
            except ValueError: # infeasible assignment
                low = mid + 1  
        
        # validate the final solution
        self.n = best_n
        self.assignment = Assignment(self.F, self.G, n = self.n)
        final_loss = self.assignment.optimize(pulp_solver = pulp_solver)

        if final_loss != 0:
            raise ValueError(f"Unexpected non-zero loss (Loss={final_loss}) for n={self.n}")
        
        return self.n
            
    

    def phi(self, key = '0', obj_type = 'V'):
        """
        Get the interleaving map :math:`\\varphi: F \\to G^n` if ``key == '0'`` or :math:`\\varphi_n: F^n \\to G^{2n}` if ``key == 'n'``.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
        
        Returns:
            LabeledBlockMatrix : 
                The interleaving map.
        """
        if self.assignment is None:
            raise ValueError("You must call fit() before getting the interleaving map.")
        return self.assignment.phi(key, obj_type)
    
    def psi(self, key = '0', obj_type = 'V'):
        """
        Get the interleaving map :math:`\\psi: G \\to F^n` if ``key == '0'`` or :math:`\\psi_n: G^n \\to F^{2n}` if ``key == 'n'``.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
        
        Returns:
            LabeledBlockMatrix : 
                The interleaving map.
        """
        if self.assignment is None:
            raise ValueError("You must call fit() before getting the interleaving map.")
        return self.assignment.psi(key, obj_type)
    
    def get_interleaving_map(self, maptype = 'phi', key = '0', obj_type = 'V'):
        """
        Get the relevant interleaving map. Helpful for iterating over options. 

        Parameters:
            maptype (str) : 
                The type of map. Either ``'phi'`` or ``'psi'``.
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.

        Returns:
            LabeledBlockMatrix : 
                The relevant interleaving map.
        """
        
        if self.assignment is None:
            raise ValueError("You must call fit() before getting the interleaving map.")
        return self.assignment.get_interleaving_map(maptype, key, obj_type)
    
    def draw_all_graphs(self, figsize = (15,10), **kwargs):
        """Draw all the graphs stored in the Interleave object.

        Args:
            figsize (tuple, optional): Sets the size of the figure. Defaults to (15,10).

        Returns:
            tuple: The figure and axes objects.
        """
        
        if self.assignment is None:
            raise ValueError("You must call fit() before drawing the graphs.")
        return self.assignment.draw_all_graphs() 
    
    def draw_all_phi(self, figsize = (15,10), **kwargs):
        """Draw all the phi maps stored in the Interleave object.

        Args:
            figsize (tuple, optional): Sets the size of the figure. Defaults to (15,10).
            **kwargs: Additional keyword arguments to pass to the drawing function.
            
        Returns:
            tuple: The figure and axes objects.
        """
        
        if self.assignment is None:
            raise ValueError("You must call fit() before drawing the phi maps.")
        return self.assignment.draw_all_phi(figsize, **kwargs)
    
    def draw_all_psi(self, figsize = (15,10), **kwargs):
        """Draw all the psi maps stored in the Interleave object.

        Args:
            figsize (tuple, optional): Sets the size of the figure. Defaults to (15,10).
            **kwargs: Additional keyword arguments to pass to the drawing function.
            
        Returns:
            tuple: The figure and axes objects.
        """
        
        if self.assignment is None:
            raise ValueError("You must call fit() before drawing the psi maps.")
        return self.assignment.draw_all_psi(figsize, **kwargs)
    
    # Disable Printing 
    def _blockPrint():
        sys.stdout = open(os.devnull, 'w')

    # Restore Printing
    def _enablePrint():
        sys.stdout = sys.__stdout__
        

#============================
# Assignment Class
#============================

class Assignment:
    """
    A class to determine the loss for a given assignment, and thus bound the interleaving distance between two Mapper graphs, denoted :math:`F` and :math:`G` throughout.

    We use keys ``['0', 'n', '2n']`` to denote the Mapper graphs :math:`F = F_0`, :math:`F_n`, and :math:`F_{2n}` and similarly for :math:`G`.

    Note that the difference in the ranges of the two Mapper graphs must be within ``'n'``.
    """
    def __init__(self, F, G, 
                        n = 1, 
                        initialize_random_maps = False, seed = None):
        """
        Initialize the Interleave object.
        
        Parameters:
            F (MapperGraph):
                The first Mapper graph.
            G (MapperGraph):
                The second Mapper graph.
            n (int):
                The interleaving parameter. The default is 1.
        """
        
        self.n = n

        # --- Check that the ranges are compatible
        if np.abs(F.min_f()-G.min_f()) > n or np.abs(F.max_f()-G.max_f()) > n:
            raise ValueError(f"Function values for F and G are too far apart to interleave with n = {n}. Try initializing with a larger n.")

        # ---
        # Containers for matrices for later 

        self.B_down_ = {'F':{}, 'G':{}} # boundary matrix
        self.B_up_ = {'F':{}, 'G':{}} # boundary matrix
        self.D_ = {'F':{}, 'G':{}} # distance matrix
        self.I_ = {'F':{}, 'G':{}} # induced maps

        self.val_to_verts = {'F':{}, 'G':{}} # dictionaries from function values to vertices
        self.val_to_edges = {'F':{}, 'G':{}} # dictionaries from function values to edges

        # ----
        # Make F graphs and smoothed versions
        self.F_ = {}
        self.F_['0'] = F 
        self.F_['n'], I_0_V, I_0_E = F.smoothing_and_maps(self.n)
        self.F_['2n'], I_n_V, I_n_E = self.F('n').smoothing_and_maps(self.n)
        
        # Get the dictionaries needed for the induced maps' block structure 
        for key in ['0', 'n', '2n']:
            self.val_to_verts['F'][key] = self.F(key).func_to_vertex_dict()
            self.val_to_edges['F'][key] = self.F(key).func_to_edge_dict()

        
        # Make the induced map from F_0 to F_n
        self.I_['F']['0'] = {}
        self.I_['F']['0']['V'] = LBM(map_dict = I_0_V, 
                                     rows_dict = self.val_to_verts['F']['n'], 
                                     cols_dict = self.val_to_verts['F']['0'])
        self.I_['F']['0']['E'] = LBM(map_dict = I_0_E, 
                                     rows_dict = self.val_to_edges['F']['n'], 
                                     cols_dict = self.val_to_edges['F']['0'])

        # Make the induced map from F_n to F_2n
        self.I_['F']['n'] = {}
        self.I_['F']['n']['V'] = LBM(I_n_V, 
                                     self.val_to_verts['F']['2n'], 
                                     self.val_to_verts['F']['n'])
        self.I_['F']['n']['E'] = LBM(I_n_E, 
                                     self.val_to_edges['F']['2n'], 
                                     self.val_to_edges['F']['n'])

        # ----
        # Now do the same for G
        self.G_ = {}
        self.G_['0'] = G 
        self.G_['n'], I_0_V, I_0_E = G.smoothing_and_maps(self.n)
        self.G_['2n'], I_n_V, I_n_E = self.G_['n'].smoothing_and_maps(self.n)

        # Get the dictionaries needed for the induced maps' block structure 
        for key in ['0', 'n', '2n']:
            self.val_to_verts['G'][key] = self.G_[key].func_to_vertex_dict()
            self.val_to_edges['G'][key] = self.G_[key].func_to_edge_dict()

        # Make the induced map from G_0 to G_n
        self.I_['G']['0'] = {}
        self.I_['G']['0']['V'] = LBM(rows_dict = self.val_to_verts['G']['n'],
                                    cols_dict = self.val_to_verts['G']['0'], 
                                    map_dict = I_0_V)
        self.I_['G']['0']['E'] = LBM(map_dict = I_0_E, 
                                     rows_dict = self.val_to_edges['G']['n'], 
                                     cols_dict = self.val_to_edges['G']['0'])
        
        self.I_['G']['0']['E'] = LBM(I_0_E, 
                                self.val_to_edges['G']['n'], 
                                self.val_to_edges['G']['0'])

        # Make the induced map from G_n to G_2n
        self.I_['G']['n'] = {}
        self.I_['G']['n']['V'] = LBM(I_n_V, 
                                self.val_to_verts['G']['2n'], 
                                self.val_to_verts['G']['n'])
        self.I_['G']['n']['E'] = LBM(I_n_E, 
                                self.val_to_edges['G']['2n'], 
                                self.val_to_edges['G']['n'])
        
        # End making smoothings and induced maps
        # ----
        # ---
        # Build boundary matrices 

        for Graph, graph_name in [(self.F, 'F'), (self.G, 'G')]:
            for key in ['0', 'n']: # Note, we don't need to do this for 2n because the matrices are never used.

                B_down = LBM()
                B_up = LBM()

                for i in self.val_to_verts[graph_name][key]:
                    if i in self.val_to_edges[graph_name][key]:
                        edges = self.val_to_edges[graph_name][key][i]
                        verts_down = self.val_to_verts[graph_name][key][i]
                        verts_up = self.val_to_verts[graph_name][key][i+1]
                        B_down[i] = LM(rows = verts_down, cols = edges)
                        B_up[i] = LM(rows = verts_up, cols = edges)

                        for e in edges:
                            B_down[i][e[0],e] = 1
                            B_up[i][e[1],e] = 1

                min_i = min(list(self.val_to_verts[graph_name][key].keys()))
                max_i = max(list(self.val_to_verts[graph_name][key].keys()))

                min_verts = self.val_to_verts[graph_name][key][min_i]
                max_verts = self.val_to_verts[graph_name][key][max_i]

                B_up[min_i-1] = LM(rows = min_verts, cols = [])
                B_down[max_i] = LM(rows = max_verts, cols = [])

                self.B_down_[graph_name][key] = B_down
                self.B_up_[graph_name][key] = B_up

        # End boundary matrices
        # ---

        # ---
        # Build the distance matrices 
        for (metagraph, name) in [ (self.F_,'F'), (self.G_,'G')]:
            for key in ['n', '2n']: # Note, we don't need to do this for 0 because the matrices are never used.
                self.D_[name][key] = {'V':{}, 'E':{}}

                self.D_[name][key]['V'] = metagraph[key].thickening_distance_matrix()
                self.D_[name][key]['E'] = metagraph[key].thickening_distance_matrix(obj_type = 'E')

        # End distance matrices

        # ----
        # phi: F -> G^n

        # Initialize the phi matrices. These will have all 0 entries.
        self.phi_ = {'0':{}, 'n':{}}
        self.phi_['0'] = {'V':{}, 'E':{}}
        self.phi_['0']['V'] = LBM(None, 
                                    self.val_to_verts['G']['n'], 
                                    self.val_to_verts['F']['0'],
                                    random_initialize = initialize_random_maps)

        self.phi_['0']['E'] = LBM(None,
                                    self.val_to_edges['G']['n'],
                                    self.val_to_edges['F']['0'],
                                    random_initialize = initialize_random_maps)
        self.phi_['n'] = {'V':{}, 'E':{}}
        self.phi_['n']['V'] = LBM(None, 
                                    self.val_to_verts['G']['2n'], 
                                    self.val_to_verts['F']['n'],
                                    random_initialize = initialize_random_maps)

        self.phi_['n']['E'] = LBM(None,
                                    self.val_to_edges['G']['2n'],
                                    self.val_to_edges['F']['n'],
                                    random_initialize = initialize_random_maps)

        # End phi
        # ---



        # ----
        # psi: G -> F^n
        
        # Initialize the psi matrices. These will have all 0 entries.
        self.psi_ = {'0':{}, 'n':{}}

        self.psi_['0'] = {'V':{}, 'E':{}}

        self.psi_['0']['V'] = LBM(None, 
                                    self.val_to_verts['F']['n'], 
                                    self.val_to_verts['G']['0'],
                                    random_initialize = initialize_random_maps)
        self.psi_['0']['E'] = LBM(None, 
                                    self.val_to_edges['F']['n'], 
                                    self.val_to_edges['G']['0'],
                                    random_initialize = initialize_random_maps)

        self.psi_['n'] = {'V':{}, 'E':{}}
        self.psi_['n']['V'] = LBM(None, 
                                    self.val_to_verts['F']['2n'], 
                                    self.val_to_verts['G']['n'],
                                    random_initialize = initialize_random_maps)
        self.psi_['n']['E'] = LBM(None, 
                                    self.val_to_edges['F']['2n'], 
                                    self.val_to_edges['G']['n'],
                                    random_initialize = initialize_random_maps)
        

        # End psi
        # ---

 
        

    ### ----------------
    # Functions for getting stuff out of all the dictionaries 
    ### ----------------

    def F(self, key = '0'):
        """
        Get the MapperGraph for :math:`F` with key.

        Parameters:
            key (str) :
                The key for the MapperGraph. Either ``'0'``, ``'n'``, or ``'2n'``. Default is ``'0'``.
        
        Returns:
            MapperGraph : 
                The MapperGraph for :math:`F` with key.
        """
        return self.F_[key]
    
    def G(self, key = '0'):
        """
        Get the MapperGraph for :math:`G` with key.

        Parameters:
            key (str) : 
                The key for the MapperGraph. Either ``'0'``, ``'n'``, or ``'2n'``. Default is ``'0'``.
                
        Returns:
            MapperGraph : 
                The MapperGraph for :math:`G` with key.
        """
        return self.G_[key]

    def B(self, graph = 'F', key = '0'):
        """
        Get the boundary matrix for a Mapper graph. This is the matrix with entry :math:`B[v,e]` equal to 1 if vertex :math:`v` is an endpoint of edge :math:`e` and 0 otherwise. Also, the boundary matrix is not computed for ``key = '2n'`` because it is not used in the optimization.

        Parameters:
            graph (str) : 
                The graph to get the boundary matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the boundary matrix. Either ``'0'`` or ``'n'``.
        
        Returns:
            LabeledBlockMatrix : 
                The boundary matrix for the Mapper graph.
        """
        return self.B_down_[graph][key].to_labeled_matrix() + self.B_up_[graph][key].to_labeled_matrix()

    def B_down(self, graph = 'F', key = '0'):
        """
        Get the downward boundary matrix for a Mapper graph. This is the matrix with entry :math:`B[v,e]` equal to 1 if vertex :math:`v` is a *lower* endpoint of edge :math:`e` and 0 otherwise. Also, the boundary matrix is not computed for ``key = '2n'`` because it is not used in the optimization.


        Parameters:
            graph (str) : 
                The graph to get the boundary matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the boundary matrix. Either ``'0'``, or ``'n'``.
        
        Returns:
            LabeledBlockMatrix : 
                The downward boundary matrix for the Mapper graph.
        """
        return self.B_down_[graph][key]

    def B_up(self, graph = 'F', key = '0'):
        """
        Get the upward boundary matrix for a Mapper graph. This is the matrix with entry :math:`B[v,e]` equal to 1 if vertex :math:`v` is an *upper* endpoint of edge :math:`e` and 0 otherwise. Also, the boundary matrix is not computed for ``key = '2n'`` because it is not used in the optimization.
        
        If ``shift_indices`` is True, the indices of the matrix will be shifted (DOWN?) by one to make matrix multiplication work later.

        Parameters:
            graph (str) : 
                The graph to get the boundary matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the boundary matrix. Either ``'0'`` or  ``'n'``.
                
        Returns:
            LabeledBlockMatrix : 
                The upward boundary matrix for the Mapper graph.
        """
        
        return self.B_up_[graph][key]

    def I(self, graph = 'F', key = '0', obj_type = 'V'):
        """
        Get the induced map from one Mapper graph to another, specifically from ``graph_key`` to ``graph_(key+n)`` sending ``obj_type`` to the same type. For example, ``I('G', 'n', 'E')`` is the map for edges from :math:`G_n` to :math:`G_{2n}`.
        
        This is the matrix with entry :math:`I[u, v] = 1` if vertex :math:`v` in the first graph maps to vertex :math:`u` in the second graph.

        Parameters:
            graph (str) : 
                The graph to get the induced map for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the induced map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
                
        Returns:
            LabeledBlockMatrix : 
                The induced map from ``graph_key`` to ``graph_(key+n)``.
        """
        return self.I_[graph][key][obj_type]
        

    def D(self, graph = 'F', key = 'n', obj_type = 'V'):
        """
        Get the distance matrix for a Mapper graph. This is the matrix with entry :math:`D[u, v]` equal to the minimum thickening needed for vertices :math:`u` and :math:`v` to map to the same connected component (similarly for edges). Note this distance is only defined for vertices or edges at the same function value. Also, the distance matrix is not computed for key = '0' because it is not used in the optimization.

        Parameters:
            graph (str) : 
                The graph to get the distance matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the distance matrix. Either ``'n'``, or ``'2n'``.

        Returns:
            LabeledBlockMatrix : 
                The distance matrix for the Mapper graph.
        """
        return self.D_[graph][key][obj_type]

    def phi(self, key = '0', obj_type = 'V'):
        """
        Get the interleaving map :math:`\\varphi: F \\to G^n` if `key == '0'` or :math:`\\varphi_n: F^n \\to G^{2n}` if `key == 'n'`.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
        
        Returns:
            LabeledBlockMatrix : 
                The interleaving map.
        """
        return self.phi_[key][obj_type]

    def psi(self, key = '0', obj_type = 'V'):
        """
        Get the interleaving map :math:`\\psi: G \\to F^n` if `key == '0'` or :math:`\\psi_n: G^n \\to F^{2n}` if `key == 'n'`.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
        
        Returns:
            LabeledBlockMatrix : 
                The interleaving map.
        """
        return self.psi_[key][obj_type]

    def get_interleaving_map(self, maptype = 'phi', key = '0', obj_type = 'V'):
        """
        Get the relevant interleaving map. Helpful for iterating over options. 

        Parameters:
            maptype (str) : 
                The type of map. Either ``'phi'`` or ``'psi'``.
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.

        Returns:
            LabeledBlockMatrix : 
                The relevant interleaving map.
        """

        if maptype == 'phi':
            return self.phi(key, obj_type)
        elif maptype == 'psi':
            return self.psi(key, obj_type)
        else:
            raise ValueError(f"Unknown maptype {maptype}. Must be 'phi' or 'psi'.")


    ### ----------------
    # Functions to set phi and psi matrices for the interleaving (instead of random)
    ### ----------------

    def set_interleaving_maps(self, phi_dict = None, psi_dict = None):
        """
        Set the phi and psi matrices to a given value. Instead of replacing the matrices, set the values block by block.

        Parameters:
            phi_dict (dict): 
                A dictionary of the form ``{'0': {'V': phi_0_V, 'E': phi_0_E}, 'n': {'V': phi_n_V, 'E': phi_n_E}}`` where each ``phi_i_j`` is a LabeledBlockMatrix.
            psi_dict (dict): 
                A dictionary of the form ``{'0': {'V': psi_0_V, 'E': psi_0_E}, 'n': {'V': psi_n_V, 'E': psi_n_E}}`` where each ``psi_i_j`` is a LabeledBlockMatrix.
        """
        
        if phi_dict is not None:
            for thickening in ['0', 'n']:
                for obj_type in ['V', 'E']:

                    # fist get the funciton values of the map that's coming in
                    keys = phi_dict[thickening][obj_type].get_all_block_indices()

                    # Now set the values
                    for key in keys:
                        self.phi_[thickening][obj_type][key] = phi_dict[thickening][obj_type][key]

        if psi_dict is not None:
            for thickening in ['0', 'n']:
                for obj_type in ['V', 'E']:

                    # fist get the funciton values of the map that's coming in
                    keys = psi_dict[thickening][obj_type].get_all_block_indices()

                    # Now set the values
                    for key in keys:
                        self.psi_[thickening][obj_type][key] = psi_dict[thickening][obj_type][key]

    def set_single_assignment(self, obj_a, obj_b, 
                              maptype = 'phi', 
                              key = '0', 
                              objtype = 'V'):
        """Set a single assignment in the interleaving map.
        
        This will be maptype_key(obj_a) = obj_b.

        Note that edges can be passed as a triple (u,v,count) where u and v are the vertices and count is the key for the edge. If passed as a pair, the count will be assumed to be 0.
        
        Args:
            obj_a (int): Object in the domain of the map.
            obj_b (int): Object in the codomain of the map.
            maptype (str, optional): Map to be set, either 'phi' or 'psi'. Defaults to 'phi'.
            objtype (str, optional): Type of object as input, either 'V' or 'E'. Defaults to 'V'.
            
        """
        if maptype == 'phi':
            start_graph = self.F(key = key)
            if key == '0':
                end_graph = self.G(key = 'n')
            elif key == 'n':
                end_graph = self.G(key = '2n')
            else:
                raise ValueError("'key' must be '0' or 'n'.")
        elif maptype == 'psi':
            start_graph = self.G(key = key)
            if key == '0':
                end_graph = self.F(key = 'n')
            elif key == 'n':
                end_graph = self.F(key = '2n')
            else:
                raise ValueError("'key' must be '0' or 'n'.")
        else:
            raise ValueError("'maptype' must be 'phi' or 'psi'.")
        
        if objtype == 'V':
            # Check that the objects have the same function value 
            if start_graph.f[obj_a] != end_graph.f[obj_b]:
                i = start_graph.f[obj_a]
                j = end_graph.f[obj_b]
                end_graph.draw()
                raise ValueError(f"The objects must have the same function value. {obj_a}: {i} != {obj_b}: {j}. Nodes: {start_graph.f.keys()}, {end_graph.f.keys()}")

            i = start_graph.f[obj_a]
        elif objtype == 'E':
            if len(obj_a) == 2:
                obj_a = obj_a + (0,)
            elif len(obj_a) != 3:
                raise ValueError("Edges must be passed as pairs or triples.")
            if len(obj_b) == 2:
                obj_b = obj_b + (0,)
            elif len(obj_b) != 3:
                raise ValueError("Edges must be passed as pairs or triples.")
            
            if start_graph.f[obj_a[0]] != end_graph.f[obj_b[0]]:
                
                i = start_graph.f[obj_a[0]]
                j = end_graph.f[obj_b[0]]
                raise ValueError(f"The objects must have the same function value. {i} != {j}")
            i = start_graph.f[obj_a[0]]
        
        else:
            raise ValueError("'objtype' must be 'V' or 'E'.")
        
        if maptype == 'phi': 
            row = self.phi_[key][objtype][i].rows.index(obj_b)
            col = self.phi_[key][objtype][i].cols.index(obj_a)
            self.phi_[key][objtype][i].array[:, col] = 0
            self.phi_[key][objtype][i].array[row, col] = 1
        else: # Maptype = 'psi'
            row = self.psi_[key][objtype][i].rows.index(obj_b)
            col = self.psi_[key][objtype][i].cols.index(obj_a)
            self.psi_[key][objtype][i].array[:, col] = 0
            self.psi_[key][objtype][i].array[row, col] = 1
            
    def set_random_assignment(self, random_n = True, seed = None):
        """
        Set the phi and psi matrices to random values. No matter what, the maps from F to Gn and G to Fn will be randomly set.  If ``'random_n'`` is True, the maps from Fn to G2n and Gn to F2n will also be randomly set. Otherwise, we will use matrix tricks to figure out the later from the former. 

        Note this functions assumes the phi and psi dictionaries were set on initialization. It will overwrite any contents that are there. 
        """
        # ----
        # phi: F -> G^n

        # Set the phi matrices. These will have all random entries.
        self.phi_['0']['V'] = LBM(None, 
                                    self.val_to_verts['G']['n'], 
                                    self.val_to_verts['F']['0'],
                                    random_initialize = True,
                                    seed = seed)

        self.phi_['0']['E'] = LBM(None,
                                    self.val_to_edges['G']['n'],
                                    self.val_to_edges['F']['0'],
                                    random_initialize = True,
                                    seed = seed)
        
        if random_n:
            self.phi_['n']['V'] = LBM(None, 
                                    self.val_to_verts['G']['2n'], 
                                    self.val_to_verts['F']['n'],
                                    random_initialize = random_n,
                                    seed = seed)

            self.phi_['n']['E'] = LBM(None,
                                    self.val_to_edges['G']['2n'],
                                    self.val_to_edges['F']['n'],
                                    random_initialize = random_n,
                                    seed = seed)
        else:
            for obj_type in ['V', 'E']:
                self.phi_['n'][obj_type] = self.I('G', 'n', obj_type) @ self.phi('0', obj_type) @ self.I('F', '0', obj_type).T()
                self.phi_['n'][obj_type] = self.phi_['n'][obj_type].to_indicator()

        # End phi
        # ---

        # ----
        # psi: G -> F^n

        # Set the psi matrices. These will have all random entries.
        self.psi_['0']['V'] = LBM(None, 
                                    self.val_to_verts['F']['n'], 
                                    self.val_to_verts['G']['0'],
                                    random_initialize = True)

        self.psi_['0']['E'] = LBM(None,
                                    self.val_to_edges['F']['n'],
                                    self.val_to_edges['G']['0'],
                                    random_initialize = True)
        
        if random_n:
            self.psi_['n']['V'] = LBM(None, 
                                    self.val_to_verts['F']['2n'], 
                                    self.val_to_verts['G']['n'],
                                    random_initialize = random_n)

            self.psi_['n']['E'] = LBM(None,
                                    self.val_to_edges['F']['2n'],
                                    self.val_to_edges['G']['n'],
                                    random_initialize = random_n)
        else:
            for obj_type in ['V', 'E']:
                self.psi_['n'][obj_type] = self.I('F', 'n', obj_type) @ self.psi('0', obj_type) @ self.I('G', '0', obj_type).T()
                self.psi_['n'][obj_type] = self.psi_['n'][obj_type].to_indicator()

        # End psi
        # ---


    ### ----------------
    # Functions for drawing stuff
    ### ----------------

    def draw_all_graphs(self, figsize = (15,10), **kwargs):
        """Draw all the graphs stored in the Interleave object.

        Args:
            figsize (tuple, optional): Sets the size of the figure. Defaults to (15,10).

        Returns:
            tuple: The figure and axes objects.
        """
        fig, axs = plt.subplots(2, 3, figsize=figsize, constrained_layout = True, sharey = True)

        self.F().draw(ax = axs[0,0], **kwargs)
        axs[0,0].set_title(r'$F_0$')

        self.F('n').draw(ax = axs[0,1], **kwargs)
        axs[0,1].set_title(r'$F_n$')

        self.F('2n').draw(ax = axs[0,2], **kwargs)
        axs[0,2].set_title(r'$F_{2n}$')

        self.G().draw(ax = axs[1,0], **kwargs)
        axs[1,0].set_title(r'$G_0$')

        self.G('n').draw(ax = axs[1,1], **kwargs)
        axs[1,1].set_title(r'$G_n$')

        self.G('2n').draw(ax = axs[1,2], **kwargs)
        axs[1,2].set_title(r'$G_{2n}$')

        return fig, axs

    def draw_I(self, graph = 'F', key = '0', obj_type = 'V', ax = None, **kwargs):
        """
        Draw the induced map from one Mapper graph to another.

        Parameters:
            graph (str) : 
                The graph to draw the induced map for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the induced map. Either ``'0'`` or ``'n'``.
        
        Returns:
            matplotlib.axes.Axes
                The axes the matrix was drawn on.
        """
        if ax is None:
            ax = plt.gca()

        self.I(graph, key, obj_type).draw(ax, **kwargs)

        ax.set_xlabel(f"{graph}_{key}")
        if key == '0':
            ax.set_ylabel(f"{graph}_n")
        else:
            ax.set_ylabel(f"{graph}_2n")
        
        return ax

    def draw_all_I(self, graph = 'F',  figsize = (10,10),  **kwargs):
        """
        Draw all the induced maps.
        
        Parameters:
            graph (str) : 
                The graph to draw the induced maps for. Either ``'F'`` or ``'G'``.
            figsize (tuple) : 
                The size of the figure. Default is (10,10).
        
        Returns:
            tuple: The figure and axes objects.
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

        self.draw_I(graph, '0', 'V', ax = axs[0, 0], **kwargs)
        title = r'Vertices: $' + graph + r'_0 \to ' + graph + r'_{n}$'
        axs[0,0].set_title(title)

        title = r'Edges: $' + graph + r'_0 \to ' + graph + r'_{n}$'
        self.draw_I(graph, '0', 'E', ax = axs[1,0], **kwargs)
        axs[1,0].set_title(title)

        self.draw_I(graph, 'n', 'V', ax = axs[0, 1], **kwargs)
        title = r'Vertices: $' + graph + r'_n \to ' + graph + r'_{2n}$'
        axs[0,1].set_title(title)


        title = r'Edges: $' + graph + r'_n \to ' + graph + r'_{2n}$'
        self.draw_I(graph, 'n', 'E', ax = axs[1,1], **kwargs)
        axs[1,1].set_title(title)
        
        return fig, axs

    def draw_B(self, graph = 'F', key = '0', ax = None, **kwargs):
        """
        Draw the boundary matrix for a Mapper graph.

        Parameters:
            graph (str) : 
                The graph to draw the boundary matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the boundary matrix. Either ``'0'``, ``'n'``, or ``'2n'``.
        
        Returns:
            matplotlib.axes.Axes
                The axes the matrix was drawn on.
        """
        if ax is None:
            ax = plt.gca()

        self.B(graph,key).draw(ax = ax, **kwargs)
        ax.set_title(f"B({graph}_{key})")
        ax.set_xlabel(f"E({graph}_{key})")
        ax.set_ylabel(f"V({graph}_{key})")

        return ax

    def draw_all_B(self, figsize = (18,18)):
        """
        Draw all the boundary matrices.
        
        Parameters:
            figsize (tuple) : 
                The size of the figure. Default is (24,18).
        
        Returns:
            tuple: The figure and axes objects.
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        self.draw_B('F', '0', ax = axs[0, 0])
        axs[0,0].set_title(r'$B(F_0)$')
        self.draw_B('F', 'n', ax = axs[0, 1])
        axs[0,1].set_title(r'$B(F_n)$')
        # self.draw_B('F', '2n', ax = axs[0, 2])
        # axs[0,2].set_title(r'$B(F_{2n})$')

        self.draw_B('G', '0', ax = axs[1, 0])
        axs[1,0].set_title(r'$B(G_0)$')
        self.draw_B('G', 'n', ax = axs[1, 1])
        axs[1,1].set_title(r'$B(G_n)$')
        # self.draw_B('G', '2n', ax = axs[1, 2])
        # axs[1,2].set_title(r'$B(G_{2n})$')
        
        return fig, axs

    def draw_D(self, graph = 'F', key = 'n', obj_type = 'V', 
                    colorbar = True, ax = None,  **kwargs):
        """
        Draw the distance matrix for a Mapper graph.

        Parameters:
            graph (str) : 
                The graph to draw the distance matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the distance matrix. Either  ``'n'``, or ``'2n'``.
            obj_type (str) : 
                The type of matrix. Either ``'V'`` or ``'E'``.
            colorbar (bool) : 
                Whether to draw a colorbar. Default is True.
            ax ( matplotlib.axes.Axes) :
                The axes to draw the matrix on. If None, the current axes will be used.
            **kwargs (dict) : 
                Additional keyword arguments to pass to the drawing function.

        Returns:
            matplotlib.axes.Axes
                The axes the matrix was drawn on.

        """
        if ax is None:
            ax = plt.gca()
        
        self.D(graph, key, obj_type).draw(ax = ax, colorbar = colorbar, **kwargs)
        ax.set_xlabel(f"V({graph}_{key})")
        ax.set_ylabel(f"V({graph}_{key})")

        return ax

    def draw_phi(self, key = '0', obj_type = 'V', ax = None, **kwargs):
        """
        Draw the map :math:`\\psi: F \\to G^n`.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
                
        Returns:
            matplotlib.axes.Axes
                The axes the matrix was drawn on.
        """
        if ax is None:
            ax = plt.gca()

        
        self.phi(key = key, obj_type = obj_type).draw(ax = ax, **kwargs)

        if key == '0':
            G_key = ''
            F_key = '_n'
        elif key == 'n':
            G_key = '_n'
            F_key = '_2n'

        ax.set_ylabel(f"{obj_type}(G{F_key})")
        ax.set_xlabel(f"{obj_type}(F{G_key})")

        return ax
    
    def draw_psi(self, key = '0', obj_type = 'V', ax = None, **kwargs):
        """
        Draw the map :math:`\\psi: G \\to F^n`.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
                
        Returns:
            matplotlib.axes.Axes
                The axes the matrix was drawn on.
        """
        if ax is None:
            ax = plt.gca()

        self.psi(key = key, obj_type = obj_type).draw(ax = ax, **kwargs)
        if key == '0':
            F_key = ''
            G_key = '_n'
        elif key == 'n':
            F_key = '_n'
            G_key = '_2n'
            
        ax.set_ylabel(f"{obj_type}(G{G_key})")
        ax.set_xlabel(f"{obj_type}(F{F_key})")

        return ax

    def draw_all_phi(self, figsize = (10,10), **kwargs):
        """
        Draw all the ``phi`` maps.
        
        Parameters:
            figsize (tuple) : 
                The size of the figure. Default is (10,10).
        
        Returns:
            tuple: The figure and axes objects.

        """
        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        self.draw_phi('0', 'V', ax = axs[0, 0], **kwargs)
        axs[0, 0].set_title(r'$\varphi_0^V$')
        self.draw_phi('n', 'V', ax = axs[0, 1], **kwargs)
        axs[0, 1].set_title(r'$\varphi_n^V$')

        self.draw_phi('0', 'E', ax = axs[1, 0], **kwargs)
        axs[1, 0].set_title(r'$\varphi_0^E$')
        self.draw_phi('n', 'E', ax = axs[1, 1], **kwargs)
        axs[1, 1].set_title(r'$\varphi_n^E$')

        return fig, axs

    def draw_all_psi(self, figsize = (10,10), **kwargs):
        """
        Draw all the ``psi`` maps.
        
        Parameters:
            figsize (tuple) : 
                The size of the figure. Default is (10,10).
        
        Returns:
            tuple: The figure and axes objects.
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        self.draw_psi('0', 'V', ax = axs[0,0],  **kwargs)
        axs[0,0].set_title(r'$\psi_0^V$')
        self.draw_psi('n', 'V', ax = axs[0,1],  **kwargs)
        axs[0,1].set_title(r'$\psi_n^V$')

        self.draw_psi('0', 'E', ax = axs[1,0],  **kwargs)
        axs[1,0].set_title(r'$\psi_0^E$')
        self.draw_psi('n', 'E', ax = axs[1,1],  **kwargs)
        axs[1,1].set_title(r'$\psi_n^E$')

        return fig, axs

    # =======================
    # Functions for checking commutative diagrams 
    # =======================

    def _draw_matrix_mult(self, A, B, C, D, 
                           titles = ['', '', '', ''], 
                           figsize = (12,3)):
        '''
        A drawing function to check the matrix mutliplications for each of the three diagram types. 
        '''

        fig, axs = plt.subplots(1, 4, figsize = figsize, constrained_layout = True)
        if type(A) == LBM:
            A.draw(ax = axs[0], vmin = -1, vmax = 1, filltype = 'nan')
        else:
            A.draw(ax = axs[0], vmin = -1, vmax = 1)
        axs[0].set_title(titles[0])
        
        if type(B) == LBM:
            B.draw(ax = axs[1], vmin = -1, vmax = 1, filltype = 'nan')
        else:
            B.draw(ax = axs[1], vmin = -1, vmax = 1)
        axs[1].set_title(titles[1])
        
        if type(C) == LBM:
            C.draw(ax = axs[2], vmin = -1, vmax = 1,colorbar = True, filltype = 'nan')
        else:
            C.draw(ax = axs[2], vmin = -1, vmax = 1, colorbar = True)
        axs[2].set_title(titles[2])
        D.draw(ax = axs[3], colorbar = True,  cmap = 'PuOr')
        axs[3].set_title(titles[3])

        return fig, axs

    #---- Parallelogram: Edge-Vertex ----

    def parallelogram_Edge_Vert_matrix(self, 
                                maptype = 'phi', 
                                up_or_down = 'down',
                                func_val = None, 
                                draw = False,  
                                ):
        """
        Check that the parallelogram for the pair :math:`(S_{\\tau_i}\\subset S_{\\sigma_i})` commutes.
        This is the one that relates the edge maps to the vertex maps. Because a function value has both an up and down version, we need to specify which one we want to check with the ``up_or_down`` parameter.

        If ``func_val`` is not None, we will only check the parallelogram for that function value. 

        Parameters:
            maptype (str) : 
                The type of map for the relevant diagram. Either ``'phi'`` or ``'psi'``.
            up_or_down (str) :
                Whether to check the up or down version of the parallelogram. Either ``'up'``, ``'down'``, or ``'both'``. Default is ``'down'``.
            func_val (int) :
                The function value to check the parallelogram for. If None, we will check all function values for the full matrix.
            draw (bool) : 
                Whether to draw the maps. Default is ``False``.
        
        Returns:
            LabeledMatrix : 
                The matrix that gives the thickening required to make the diagram commute. 
        """

        if maptype == 'phi':
            start_graph = 'F'
            end_graph = 'G'
            maptype_latex = r'\varphi'
        elif maptype == 'psi':
            start_graph = 'G'
            end_graph = 'F'
            maptype_latex = r'\psi'
        
        if up_or_down == 'down':
            B = self.B_down
            arrow = '\\downarrow'
            interleaving_map_top = self.get_interleaving_map(maptype, '0', 'V')
            interleaving_map_bottom = self.get_interleaving_map(maptype, '0', 'E')
        elif up_or_down == 'up':
            B = self.B_up
            arrow = '\\uparrow'

            # Shift of the top caused by indexing problem: 
            interleaving_map_top = self.get_interleaving_map(maptype, '0', 'V').to_shifted_blocks(-1)
            
            interleaving_map_bottom = self.get_interleaving_map(maptype, '0', 'E')
            
        elif up_or_down == 'both':
            interleaving_map_top = self.get_interleaving_map(maptype, '0', 'V')
            interleaving_map_bottom = self.get_interleaving_map(maptype, '0', 'E')
            B = self.B
            arrow = '{ }'
        else:
            raise ValueError(f"Unknown up_or_down {up_or_down}. Must be 'up', 'down', or 'both'.")

        if func_val is None:
            Top = interleaving_map_top @ B(start_graph, '0')
            Bottom = B(end_graph, 'n') @ interleaving_map_bottom

            Result = Top - Bottom

            if up_or_down == 'up':
                # Need to undo the shift fix applied earlier
                Result = Result.to_shifted_blocks(1)

            Result_Dist = self.D(end_graph, 'n', 'V') @ Result
        else: 
            if up_or_down == 'down': # tau_i \to \sigma_i
                Top = self.get_interleaving_map(maptype, '0', 'V')[func_val] @ self.B_down(start_graph, '0')[func_val]
                Bottom = self.B_down(end_graph, 'n')[func_val] @ self.get_interleaving_map(maptype, '0', 'E')[func_val]
                Result = Top - Bottom
                Result_Dist = self.D(end_graph, 'n', 'V')[func_val] @ Result
            elif up_or_down == 'up': # \tau_i \to \sigma_{i+1}
                Top = self.get_interleaving_map(maptype, '0', 'V')[func_val+1] @ self.B_up(start_graph, '0')[func_val]
                Bottom = self.B_up(end_graph, 'n')[func_val] @ self.get_interleaving_map(maptype, '0', 'E')[func_val]
                Result = Top - Bottom
                Result_Dist = self.D(end_graph, 'n', 'V')[func_val+1] @ Result
            else:
                raise ValueError(f"Unknown up_or_down {up_or_down}. Must be 'up' or 'down'.")


        if draw: 
            titles = ['', '', '', '']
            titles[0] = f"$M_{maptype_latex}^V \\cdot B_{start_graph}^{arrow}$"
            titles[1] = f"$B_{end_graph}^{arrow} \\cdot M_{maptype_latex}^E$"
            titles[2] = titles[0][:-1] + ' - ' + titles[1][1:]
            titles[3] = f"$D_{{{end_graph}^{{n}}}}^{{V}} \\cdot  ({titles[2][1:-1]})$"

            fig, axs = self._draw_matrix_mult(Top, Bottom, Result, Result_Dist, titles = titles)

        return Result_Dist

    def parallelogram_Edge_Vert(self,maptype = 'phi', 
                                up_or_down = 'down',
                                func_val = None):
        """
        Check that the parallelogram for the pair :math:`(S_{\\tau_i}\\subset S_{\\sigma_i})` commutes, and return the maximum value in the matrix. 

        """

        Result = self.parallelogram_Edge_Vert_matrix(maptype, up_or_down, func_val)
        return Result.absmax()

    #---

    #---- Parallelogram: Thickening ----

    def parallelogram_matrix(self, 
                            maptype = 'phi', 
                            obj_type = 'V', 
                            func_val = None, 
                            draw = False, 
                            ):
                            
        """
        Get the paralellograms for checking that it's a nat trans.
        These are types 3-6 from Liz's Big List. 

        If ``func_val`` is not None, we will only check the parallelogram for that function value.

        Parameters:
            maptype (str) : 
                The type of map. Either 'phi' or 'psi'.
            obj_type (str) : 
                The type of object. Either ``'V'`` or ``'E'``.
            func_val (int) :
                The function value to check the parallelogram for. If None, we will check all function values for the full matrix.
            draw (bool) : 
                Whether to draw the maps. Default is False.
        
        Returns:
            LabeledMatrix : 
                The matrix that gives the thickening required to make the diagram commute. 
        """

        if maptype == 'phi':
            start_graph = 'F'
            end_graph = 'G'
            maptype_latex = r'\varphi'
        elif maptype == 'psi':
            start_graph = 'G'
            end_graph = 'F'
            maptype_latex = r'\psi'

        if func_val is None:
            Top = self.get_interleaving_map(maptype, 'n', obj_type) @ self.I(start_graph, '0', obj_type)
            Bottom = self.I(end_graph, 'n', obj_type) @ self.get_interleaving_map(maptype, '0', obj_type)
            Result = Top - Bottom
            # Result= Result.to_labeled_matrix() # To make the matrix labeledmatrix
            
            Result_Dist = self.D(end_graph, '2n', obj_type) @ Result
        else:
            # Do this for a single input function value
            Top = self.get_interleaving_map(maptype, 'n', obj_type)[func_val] @ self.I(start_graph, '0', obj_type)[func_val]
            Bottom = self.I(end_graph, 'n', obj_type)[func_val] @ self.get_interleaving_map(maptype, '0', obj_type)[func_val]
            Result = Top - Bottom
            Result_Dist = self.D(end_graph, '2n', obj_type)[func_val] @ Result

        # --- Drawing--- #
        if draw:
            titles = ['', '', '', '']
            titles[0] = f"$M_{{{maptype_latex}_n}}^{obj_type} \\cdot I_{start_graph}^{obj_type}$"
            titles[1] = f"$I_{{{end_graph}^n}} \\cdot M_{maptype_latex}^{obj_type}$"
            titles[2] = titles[0][:-1] + ' - ' + titles[1][1:]
            titles[3] = f"$D_{{{end_graph}^{{2n}}}}^{obj_type} \\cdot ({titles[2][1:-1]})$"

            fig, axs = self._draw_matrix_mult(Top, Bottom, Result, Result_Dist, titles = titles)

            
        return Result_Dist

    def parallelogram(self, maptype = 'phi',
                        obj_type = 'V',
                        func_val = None):
        """
        Get the loss value for the thickening paralellograms
        """
        Result = self.parallelogram_matrix(maptype, obj_type, func_val).absmax()
        return Result

    # --- Triangles ----

    def triangle_matrix(self, 
                        start_graph = 'F', 
                        obj_type = 'V', 
                        func_val = None, 
                        draw = False, ):
        """
        Get the triangle for checking that it's an interleaving. 

        If ``func_val`` is not None, we will only check the parallelogram for that function value.

        Parameters:
            start_graph (str) : 
                The starting graph. Either 'F' or 'G'.
            obj_type (str) : 
                The type of object. Either ``'V'`` or ``'E'``.
            func_val (int) :
                The function value to check the parallelogram for. If None, we will check all function values for the full matrix.
            draw (bool) : 
                Whether to draw the maps. Default is False.
        
        Returns:
            LabeledMatrix : 
                The matrix that gives the thickenin grequired to make the diagram commute 
        """

        if start_graph == 'F':
            end_graph = 'G'
            map1 = 'phi'
            map2 = 'psi'
            map1_latex = r'\varphi'
            map2_latex = r'\psi'
        elif start_graph == 'G':
            end_graph = 'F'
            map1 = 'psi'
            map2 = 'phi'
            map1_latex = r'\psi'
            map2_latex = r'\varphi'
        else:
            raise ValueError(f"Unknown start_graph {start_graph}. Must be 'F' or 'G'.")

        if func_val is None:
            Top = self.I(start_graph, 'n', obj_type) @ self.I(start_graph, '0', obj_type)
            Bottom = self.get_interleaving_map(maptype = map2, key = 'n', obj_type = obj_type) @ self.get_interleaving_map(maptype = map1, key = '0', obj_type = obj_type)
            Result = Top - Bottom
            # Result = Result.to_labeled_matrix() # To make the matrix labeledmatrix

            Result_Dist = self.D(start_graph, '2n', obj_type) @ Result
        
        else:
            Top = self.I(start_graph, 'n', obj_type)[func_val] @ self.I(start_graph, '0', obj_type)[func_val]
            Bottom = self.get_interleaving_map(maptype = map2, key = 'n', obj_type = obj_type)[func_val] @ self.get_interleaving_map(maptype = map1, key = '0', obj_type = obj_type)[func_val]
            Result = Top - Bottom
            
            Result_Dist = self.D(start_graph, '2n', obj_type)[func_val] @ Result

        if draw:
            titles = ['', '', '', '']
            titles[0] = f"$I_{{{start_graph}^n}}^{obj_type} \\cdot I_{start_graph}^{obj_type}$"
            titles[1] = f"$M_{{{map2_latex}^n}}^{obj_type} \\cdot M_{{{map1_latex}}}^{obj_type}$"
            titles[2] = titles[0][:-1] + ' - ' + titles[1][1:]
            titles[3] = f"$D_{{{start_graph}^{{2n}}}}^{obj_type} \\cdot ({titles[2][1:-1]})$"
            fig, axs = self._draw_matrix_mult(Top, Bottom, Result, Result_Dist, titles = titles)

        return Result_Dist
    
    def triangle(self, start_graph = 'F', obj_type = 'V', func_val = None):
        """
        Get the loss value for the triangle
        """
        Result = self.triangle_matrix(start_graph, obj_type, func_val).absmax()
        Result = np.ceil(Result/2)
        return Result

    # --- Loss functions ----

    def loss_table(self):
        """
        Returns a table with the loss for each term in the bound. The actual loss is the maximum of these values, and can be found with the ``loss`` method.

        Returns:
            pd.DataFrame : 
                A table with the loss value for each term. The 'Loss' column has the loss value.
        """

        loss_list = []

        # All the edge-vertex parallelogram maps 
        for maptype in ['phi', 'psi']:
            for up_or_down in ['up', 'down']:
                result = self.parallelogram_Edge_Vert(maptype = maptype, up_or_down = up_or_down)
                loss = result
                loss_list.append([f'Edge-Vertex', maptype, up_or_down, loss])

        # All the parallelogram maps 
        for maptype in ['phi', 'psi']:
            for obj_type in ['V', 'E']:
                result = self.parallelogram(maptype = maptype, obj_type = obj_type)
                loss = result
                loss_list.append( [f'Thickening', maptype, obj_type, loss])

        # ALl the triangle maps
        for obj_type in ['V', 'E']:
            for start_graph in ['F', 'G']:
                result = self.triangle(start_graph = start_graph, obj_type = obj_type)
                loss = result
                loss_list.append([f'Triangle', obj_type, start_graph, loss])
        loss_table = pd.DataFrame(loss_list, columns = ['Dgm Type', 'Req A', 'Req B', 'Loss'])
        return loss_table

    def loss(self, verbose = False):
        """
        Computes the loss for the interleaving distance. Specifically, if the loss is :math:`L` and the interleaving class was initiated with :math:`n`, then the interleaving distance is at most :math:`n + L`.

        Returns:
            float : 
                The loss value.
        """
        loss_table = self.loss_table()
        
        loss = loss_table['Loss'].max()
        
        if verbose: 
            print(loss_table)
            print(f"\nLoss: {loss}")
            print(f"Interleaving distance bound: {self.n} + {loss} = {self.n + loss}")
        return loss

    def loss_by_block(self):
        """
        Computes the loss for each block of the interleaving distance. 

        Returns:
            dict : 
                A dictionary with the loss for each block.
        """

        # This is not at all done yet, consider this a placeholder!
        all_func_vals = set(self.F().get_function_values()) | set(self.G().get_function_values()) 
        all_func_vals = list(all_func_vals)
        all_func_vals.sort()

        loss_dict = {}

        for i in all_func_vals:
            #====
            # Check the matrices with F(\sigma_i) or G(\sigma_i) in the top left 
            #====
            loss_list = []

            # -- Type 3, 5. Vertex type parallelogram 
            for (Graph, graph_name, maptype) in [(self.F, 'F', 'phi'), (self.G, 'G', 'psi')]:
                obj_type = 'V'
                if i in Graph().get_function_values():
                    loss = self.parallelogram(maptype = maptype, obj_type = obj_type, func_val = i)
                    # loss = result.absmax() 
                    # print(f"parallelogram loss for {graph_name} {obj_type} {i} is {loss}")
                    loss_list.append(loss)
                else:
                    # This catch is because the two graphs might have different ranges of function values
                    pass 

            # -- Type 7, 9. Vertex type triangle
            for (Graph, graph_name) in [(self.F, 'F'), (self.G, 'G')]:
                obj_type = 'V'
                if i in Graph().get_function_values():
                    loss = self.triangle(start_graph = graph_name, obj_type = obj_type, func_val = i)
                    # loss = result.absmax()
                    loss_list.append(loss)
                else:
                    # This catch is because the two graphs might have different ranges of function values
                    pass
            
            #====
            # Check the matrices with F(\tau_i) or G(\tau_i) in the top left 
            #====

            # -- Type 1, 2. Mixed type parallelogram 
            for (Graph, graph_name, maptype) in [(self.F, 'F', 'phi'), (self.G, 'G', 'psi')]:
                obj_type = 'E'
                edge_vals = Graph().get_function_values()
                edge_vals.pop(edge_vals.index(max(edge_vals)))
                if i in edge_vals:
                    for up_or_down in ['up', 'down']:
                        loss = self.parallelogram_Edge_Vert(maptype = maptype, func_val = i, up_or_down=up_or_down)
                        # loss = result.absmax()
                        # print(f"mixed parallelogram loss for {graph_name} {obj_type} {i} is {loss}")
                        loss_list.append(loss)
                else:
                    # This catch is because the two graphs might have different ranges of function values
                    pass

            # -- Type 4, 6. Edge type parallelogram
            for (Graph, graph_name, maptype) in [(self.F, 'F', "phi"), (self.G, 'G', "psi")]:
                obj_type = 'E'
                edge_vals = Graph().get_function_values()
                edge_vals.pop(edge_vals.index(max(edge_vals)))
                if i in edge_vals:
                    loss = self.parallelogram(maptype = maptype, obj_type = obj_type, func_val = i)
                    # loss = result.absmax()
                    # print(f"parallelogram loss for {graph_name} {obj_type} {i} is {loss}")
                    loss_list.append(loss)
                else:
                    # This catch is because the two graphs might have different ranges of function values 
                    pass

            # -- Type 8, 10. Edge type triangle 
            for (Graph, graph_name) in [(self.F, 'F'), (self.G, 'G')]:
                obj_type = 'E'
                edge_vals = Graph().get_function_values()
                edge_vals.pop(edge_vals.index(max(edge_vals)))
                if i in edge_vals:
                    loss = self.triangle(start_graph = graph_name, obj_type = obj_type, func_val = i)
                    # loss = result.absmax()
                    # print(f"triangle loss for {graph_name} {obj_type} {i} is {loss}")
                    loss_list.append(loss)
                else:
                    # This catch is because the two graphs might have different ranges of function values
                    pass

            # Store the max loss for this function value 
            loss_dict[i] = max(loss_list)


        # Get max loss over all function values
        # flatten the dictionary
        loss_list = list(loss_dict.values())
        return max(loss_list)


    def all_func_vals(self):
        """
        Get all the function values that are in the graphs. 

        Returns:
            list : 
                A list of all the function values.
        """

        all_func_vals = set(self.F().get_function_values()) | set(self.G().get_function_values()) 
        all_func_vals = list(all_func_vals)
        all_func_vals.sort()

        return all_func_vals
    
    def optimize(self, pulp_solver = None):
        """Uses the ILP to find the best interleaving distance bound, returns the loss value found. Further, it stores the optimal phi and psi maps which can be returned using the ``self.phi`` and ``self.psi`` attributes respectively.
        This function requires the `pulp` package to be installed.
        
        Parameters:
            pulp_solver (pulp.LpSolver): the solver to use for the ILP optimization. If None, the default solver is used.
        Returns:
            float : 
                The loss value found by the ILP solver.
        """
        
        map_dict, loss_val = solve_ilp(self, pulp_solver = pulp_solver)
            
        self.phi_['0'] = {'V': map_dict['phi_0_V'], 'E': map_dict['phi_0_E']}
        self.phi_['n'] = {'V': map_dict['phi_n_V'], 'E': map_dict['phi_n_E']}
        self.psi_['0'] = {'V': map_dict['psi_0_V'], 'E': map_dict['psi_0_E']}
        self.psi_['n'] = {'V': map_dict['psi_n_V'], 'E': map_dict['psi_n_E']}
        
        
        return loss_val
    

    
