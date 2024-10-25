from cereeberus import MapperGraph
import numpy as np
import networkx as nx
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
from .labeled_blocks import LabeledBlockMatrix as LBM
from .labeled_blocks import LabeledMatrix as LM

class Interleave:
    """
    A class to bound the interleaving distance between two Mapper graphs, denoted :math:`F` and :math:`G` throughout.

    We use keys ``['0', 'n', '2n']`` to denote the Mapper graphs :math:`F = F_0`, :math:`F_n`, and :math:`F_{2n}` and similarly for :math:`G`.
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

        # ---
        # Containers for matrices for later 


        # self.A = {'F':{}, 'G':{}} # adjacency matrix
        self.B_ = {'F':{}, 'G':{}} # boundary matrix
        self.D_ = {'F':{}, 'G':{}} # distance matrix
        self.I_ = {'F':{}, 'G':{}} # induced maps

        self.val_to_verts = {'F':{}, 'G':{}} # dictionaries from function values to vertices
        self.val_to_edges = {'F':{}, 'G':{}} # dictionaries from function values to edges

        # ----
        # Make F graphs and smoothed versions
        self.F_ = {}
        self.F_['0'] = F 
        self.F_['n'], I_0 = F.smoothing(self.n, return_map = True)
        self.F_['2n'], I_n = self.F('n').smoothing(self.n, return_map = True)

        # Get the dictionaries needed for the induced maps' block structure 
        for key in ['0', 'n', '2n']:
            self.val_to_verts['F'][key] = self.F(key).func_to_vertex_dict()
            self.val_to_edges['F'][key] = self.F(key).func_to_edge_dict()

        
        
        # Make the induced map from F_0 to F_n
        self.I_['F']['0'] = {}
        self.I_['F']['0']['V'] = LBM(map_dict = I_0, 
                                     rows_dict = self.val_to_verts['F']['n'], 
                                     cols_dict = self.val_to_verts['F']['0'])
        I_0_edges = {(e[0], e[1], 0): (I_0[e[0]], I_0[e[1]],0) for e in self.F_['0'].edges()}
        self.I_['F']['0']['E'] = LBM(map_dict = I_0_edges, 
                                     rows_dict = self.val_to_edges['F']['n'], 
                                     cols_dict = self.val_to_edges['F']['0'])

        # Make the induced map from F_n to F_2n
        self.I_['F']['n'] = {}
        self.I_['F']['n']['V'] = LBM(I_n, 
                                     self.val_to_verts['F']['2n'], 
                                     self.val_to_verts['F']['n'])
        # Note that in this setting, the induced map on edges is the same as the map sending the edge to the edge with endpoints given by the vertices since there are no double edges for any smoothing >= 1. 
        I_n_edges = {(e[0], e[1], 0): (I_n[e[0]], I_n[e[1]],0) for e in self.F('n').edges()}
        self.I_['F']['n']['E'] = LBM(I_n_edges, 
                                     self.val_to_edges['F']['2n'], 
                                     self.val_to_edges['F']['n'])

        # ----
        # Now do the same for G
        self.G_ = {}
        self.G_['0'] = G 
        self.G_['n'], I_0 = G.smoothing(self.n, return_map = True)
        self.G_['2n'], I_n = self.G_['n'].smoothing(self.n, return_map = True)

        # Get the dictionaries needed for the induced maps' block structure 
        for key in ['0', 'n', '2n']:
            self.val_to_verts['G'][key] = self.G_[key].func_to_vertex_dict()
            self.val_to_edges['G'][key] = self.G_[key].func_to_edge_dict()

        # Make the induced map from G_0 to G_n
        self.I_['G']['0'] = {}
        self.I_['G']['0']['V'] = LBM( rows_dict = self.val_to_verts['G']['n'],
                                    cols_dict = self.val_to_verts['G']['0'], 
                                    map_dict = I_0)

        # self.map_dict_to_matrix(I_0, 
        #                         self.val_to_verts['G']['n'], 
        #                         self.val_to_verts['G']['0'])
        I_0_edges = {(e[0], e[1], 0): (I_0[e[0]], I_0[e[1]],0) for e in self.G_['0'].edges()}
        self.I_['G']['0']['E'] = LBM(I_0_edges, 
                                self.val_to_edges['G']['n'], 
                                self.val_to_edges['G']['0'])

        # Make the induced map from G_n to G_2n
        self.I_['G']['n'] = {}
        self.I_['G']['n']['V'] = LBM(I_n, 
                                self.val_to_verts['G']['2n'], 
                                self.val_to_verts['G']['n'])
        I_n_edges = {(e[0], e[1], 0): (I_n[e[0]], I_n[e[1]],0) for e in self.G_['n'].edges()}
        self.I_['G']['n']['E'] = LBM(I_n_edges, 
                                self.val_to_edges['G']['2n'], 
                                self.val_to_edges['G']['n'])
        
        # End making smoothings and induced maps
        # ----
        # ---
        # Build boundary matrices 

        for key in ['0', 'n', '2n']:
            B_dict = self.F(key).boundary_matrix(astype = 'map')

            vert_dict = self.val_to_verts['F'][key]
            func_vals = list(vert_dict.keys())
            func_vals.sort()
            rows = [vert_dict[f] for f in func_vals]
            rows = [item for sublist in rows for item in sublist]

            edge_dict =  self.val_to_edges['F'][key]
            func_vals = list(edge_dict.keys())
            func_vals.sort()
            cols = [edge_dict[f] for f in func_vals]
            cols = [item for sublist in cols for item in sublist]

            self.B_['F'][key] = LM(rows = rows, cols = cols)
            for e in cols:
                self.B_['F'][key][e[0], e] = 1
                self.B_['F'][key][e[1], e] = 1

        for key in ['0', 'n', '2n']:
            B_dict = self.G(key).boundary_matrix(astype = 'map')

            vert_dict = self.val_to_verts['G'][key]
            func_vals = list(vert_dict.keys())
            func_vals.sort()
            rows = [vert_dict[f] for f in func_vals]
            rows = [item for sublist in rows for item in sublist]

            edge_dict =  self.val_to_edges['G'][key]
            func_vals = list(edge_dict.keys())
            func_vals.sort()
            cols = [edge_dict[f] for f in func_vals]
            cols = [item for sublist in cols for item in sublist]

            self.B_['G'][key] = LM(rows = rows, cols = cols)
            for e in cols:
                self.B_['G'][key][e[0], e] = 1
                self.B_['G'][key][e[1], e] = 1

        # End boundary matrices
        # ---

        # ---
        # Build the distance matrices 
        # Note.... I don't think I actually need all of these, but I'm going to build them all anyway.
        for (metagraph, name) in [ (self.F_,'F'), (self.G_,'G')]:
            for key in ['0', 'n', '2n']:
                self.D_[name][key] = {'V':{}, 'E':{}}

                # Vertex version 
                M = metagraph[key]

                val_to_verts = self.val_to_verts[name][key]

                block_D = LBM()
                for f_i in val_to_verts:
                    vert_set = val_to_verts[f_i]
                    D_i = np.zeros((len(vert_set), len(vert_set)))
                    for i in range(len(vert_set)):
                        for j in range(i+1, len(vert_set)):
                            D_i[i, j] = M.thickening_distance(vert_set[i], vert_set[j])
                            D_i[j, i] = D_i[i, j]
                    block_D[f_i] = LM( rows =  vert_set, cols =  vert_set,array =  D_i)
                
                self.D_[name][key]['V'] = block_D

                # Edge version 
                # Note that in this setting, the distance between two edges is the max distance between any pair of vertices in the two edges.
                val_to_edges = self.val_to_edges[name][key]

                block_D = LBM()
                for f_i in val_to_edges:
                    edge_set = val_to_edges[f_i]
                    D_i = np.zeros((len(edge_set), len(edge_set)))
                    for i in range(len(edge_set)):
                        for j in range(i+1, len(edge_set)):
                            u_i, v_i, _ = edge_set[i]
                            u_j, v_j, _ = edge_set[j]

                            # Lower vertex checking for u_i and u_v
                            D_lower = self.D(name, key, 'V')[f_i]
                            u_i_index = D_lower.rows.index(u_i)
                            u_j_index = D_lower.rows.index(u_j)
                            lower_k = D_lower.array[u_i_index, u_j_index]

                            # Upper vertex checking 
                            D_upper = self.D(name, key, 'V')[f_i+1]
                            v_i_index = D_upper.rows.index(v_i)
                            v_j_index = D_upper.rows.index(v_j)
                            upper_k = D_upper.array[v_i_index, v_j_index]

                            # Distance for the edge is the largest k so that both endpoints have mapped to the same thing. 
                            D_i[i, j] = max(lower_k, upper_k)
                            D_i[j, i] = D_i[i, j]
                    block_D[f_i] = LM( rows =  edge_set, cols =  edge_set,array =  D_i)

                self.D_[name][key]['E'] = block_D



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
        """
        return self.F_[key]
    
    def G(self, key = '0'):
        """
        Get the MapperGraph for :math:`G` with key.

        Parameters:
            key (str) : 
                The key for the MapperGraph. Either ``'0'``, ``'n'``, or ``'2n'``. Default is ``'0'``.
        """
        return self.G_[key]

    def B(self, graph = 'F', key = '0'):
        """
        Get the boundary matrix for a Mapper graph. This is the matrix with entry :math:`B[v,e]` equal to 1 if vertex :math:`v` is an endpoint of edge :math:`e` and 0 otherwise.

        Parameters:
            graph (str) : 
                The graph to get the boundary matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the boundary matrix. Either ``'0'``, ``'n'``, or ``'2n'``.
        """
        return self.B_[graph][key]

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
        """
        return self.I_[graph][key][obj_type]
        

    def D(self, graph = 'F', key = '0', obj_type = 'V'):
        """
        Get the distance matrix for a Mapper graph. This is the matrix with entry :math:`D[u, v]` equal to the minimum thickening needed for vertices :math:`u` and :math:`v` to map to the same connected component (similarly for edges). Note this distance is only defined for vertices or edges at the same function value. 

        Parameters:
            graph (str) : 
                The graph to get the distance matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the distance matrix. Either ``'0'``, ``'n'``, or ``'2n'``.
        """
        return self.D_[graph][key][obj_type]

    def phi(self, key = '0', obj_type = 'V'):
        """
        Get the interleaving map :math:`F \\to G^n`.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
        """
        return self.phi_[key][obj_type]

    def psi(self, key = '0', obj_type = 'V'):
        """
        Get the interleaving map :math:`\psi: G  \\to F^n` on either vertices or edges.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
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
    #
    ### ----------------
    



    ### ----------------
    # Functions for drawing stuff
    ### ----------------

    def draw_all_graphs(self):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey = True)

        self.F().draw(ax = axs[0,0])
        axs[0,0].set_title(r'$F_0$')

        self.F('n').draw(ax = axs[0,1])
        axs[0,1].set_title(r'$F_n$')

        self.F('2n').draw(ax = axs[0,2])
        axs[0,2].set_title(r'$F_{2n}$')

        self.G().draw(ax = axs[1,0])
        axs[1,0].set_title(r'$G_0$')

        self.G('n').draw(ax = axs[1,1])
        axs[1,1].set_title(r'$G_n$')

        self.G('2n').draw(ax = axs[1,2])
        axs[1,2].set_title(r'$G_{2n}$')

    def draw_I(self, graph = 'F', key = '0', obj_type = 'V', ax = None, **kwargs):
        """
        Draw the induced map from one Mapper graph to another.

        Parameters:
            graph (str) : 
                The graph to draw the induced map for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the induced map. Either ``'0'`` or ``'n'``.
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

    def draw_all_I(self):
        """
        Draw all the induced maps.
        """
        fig, axs = plt.subplots(2, 2, figsize=(13, 13))
        plt.subplots_adjust(wspace=.4, hspace=.4)
        self.draw_I('G', '0', 'V', ax = axs[0, 0])
        axs[0,0].set_title(r'Vertices: $G_0 \to G_{n}$')
        self.draw_I('G', '0', 'E', ax = axs[1,0])
        axs[1,0].set_title(r'Edges: $G_0 \to G_{n}$')

        self.draw_I('G', 'n', 'V', ax = axs[0, 1])
        axs[0,1].set_title(r'Vertices: $G_n \to G_{2n}$')
        self.draw_I('G', 'n', 'E', ax = axs[1,1])
        axs[1,1].set_title(r'Edges: $G_n \to G_{2n}$')

    def draw_B(self, graph = 'F', key = '0', ax = None, **kwargs):
        """
        Draw the boundary matrix for a Mapper graph.

        Parameters:
            graph (str) : 
                The graph to draw the boundary matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the boundary matrix. Either ``'0'``, ``'n'``, or ``'2n'``.
        """
        if ax is None:
            ax = plt.gca()

        self.B(graph,key).draw(ax = ax, **kwargs)
        ax.set_title(f"B({graph}_{key})")
        ax.set_xlabel(f"E({graph}_{key})")
        ax.set_ylabel(f"V({graph}_{key})")

        return ax

    def draw_all_B(self, figsize = (24,18), spacing = (.1,.1)):
        """
        Draw all the boundary matrices.
        """
        fig, axs = plt.subplots(2, 3, figsize=figsize)
        plt.subplots_adjust(wspace=spacing[0], hspace=spacing[1])
        self.draw_B('F', '0', ax = axs[0, 0])
        axs[0,0].set_title(r'$B(F_0)$')
        self.draw_B('F', 'n', ax = axs[0, 1])
        axs[0,1].set_title(r'$B(F_n)$')
        self.draw_B('F', '2n', ax = axs[0, 2])
        axs[0,2].set_title(r'$B(F_{2n})$')

        self.draw_B('G', '0', ax = axs[1, 0])
        axs[1,0].set_title(r'$B(G_0)$')
        self.draw_B('G', 'n', ax = axs[1, 1])
        axs[1,1].set_title(r'$B(G_n)$')
        self.draw_B('G', '2n', ax = axs[1, 2])
        axs[1,2].set_title(r'$B(G_{2n})$')

    def draw_D(self, graph = 'F', key = '0', obj_type = 'V', 
                    colorbar = True, ax = None,  **kwargs):
        """
        Draw the distance matrix for a Mapper graph.

        Parameters:
            graph (str) : 
                The graph to draw the distance matrix for. Either ``'F'`` or ``'G'``.
            key (str) : 
                The key for the distance matrix. Either ``'0'``, ``'n'``, or ``'2n'``.
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
        Draw the map :math:`\psi: F \\to G^n`.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
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

        ax.set_ylabel(f"{type}(G{F_key})")
        ax.set_xlabel(f"{type}(F{G_key})")

        return ax
    
    def draw_psi(self, key = '0', obj_type = 'V', ax = None, **kwargs):
        """
        Draw the map :math:`\psi: G \\to F^n`.

        Parameters:
            key (str) : 
                The key for the map. Either ``'0'`` or ``'n'``.
            obj_type (str) : 
                The type of map. Either ``'V'`` or ``'E'``.
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
            
        ax.set_ylabel(f"{type}(G{G_key})")
        ax.set_xlabel(f"{type}(F{F_key})")

        return ax

    def draw_all_phi(self, figsize = (10,10), spacing = (0,.5), **kwargs):
        """
        Draw all the ``phi`` maps.

        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        plt.subplots_adjust(wspace=spacing[0], hspace=spacing[1])
        self.draw_phi('0', 'V', ax = axs[0, 0], **kwargs)
        axs[0, 0].set_title(r'$\varphi_0^V$')
        self.draw_phi('n', 'V', ax = axs[0, 1], **kwargs)
        axs[0, 1].set_title(r'$\varphi_n^V$')

        self.draw_phi('0', 'E', ax = axs[1, 0], **kwargs)
        axs[1, 0].set_title(r'$\varphi_0^E$')
        self.draw_phi('n', 'E', ax = axs[1, 1], **kwargs)
        axs[1, 1].set_title(r'$\varphi_n^E$')

    def draw_all_psi(self, figsize = (10,10), spacing = (0,.5),     **kwargs):
        """
        Draw all the ``psi`` maps.
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        plt.subplots_adjust(wspace=spacing[0], hspace=spacing[1])
        self.draw_psi('0', 'V', ax = axs[0,0],  **kwargs)
        axs[0,0].set_title(r'$\psi_0^V$')
        self.draw_psi('n', 'V', ax = axs[0,1],  **kwargs)
        axs[0,1].set_title(r'$\psi_n^V$')

        self.draw_psi('0', 'E', ax = axs[1,0],  **kwargs)
        axs[1,0].set_title(r'$\psi_0^E$')
        self.draw_psi('n', 'E', ax = axs[1,1],  **kwargs)
        axs[1,1].set_title(r'$\psi_n^E$')

    # ==========
    # Functions for checking commutative diagrams 

    def parallelogram_Edge_Vert(self, maptype = 'phi', returntype = 'dist', 
                                    draw = False, drawtype = 'all', 
                                    **kwargs):
        """
        Check that the parallelogram for the pair :math:`(U_{\\tau_I}\subset U_{\sigma_i})` commutes.
        This is the one that relates the edge maps to the vertex maps.
        (These are types 1 (when maptype = 'phi') and 2 (when maptype = 'psi') from Liz's Big List )

        Parameters:
            maptype (str) : 
                The type of map. Either ``'phi'`` or ``'psi'``.
            returntype (str) : 
                The type of return. Either ``'dist'`` if you want the matrix that gives the thickening required to make the diagram commute; or ``'commute'`` to just give the map mismatch.
            draw (bool) : 
                Whether to draw the maps. Default is ``False``.
            drawtype (str) : 
                The type of drawing. Either ``'all'`` or ``'result'``. Default is ``'all'``.
            **kwargs : 
                Additional keyword arguments to pass to the drawing function.
        
        Returns:
            LabeledMatrix : 
                The matrix that gives the thickenin grequired to make the diagram commute if returntype is ``'dist'``; or the map mismatch if returntype is ``'commute'``
        """

        if maptype == 'phi':
            start_graph = 'F'
            end_graph = 'G'
            maptype_latex = r'\varphi'
        elif maptype == 'psi':
            start_graph = 'G'
            end_graph = 'F'
            maptype_latex = r'\psi'

        Top = self.get_interleaving_map(maptype, '0', 'V') @ self.B(start_graph, '0')

        Bottom = self.B(end_graph, 'n') @ self.get_interleaving_map(maptype, '0', 'E') 
        Result = Top - Bottom

        Result_Dist = self.D(end_graph, 'n', 'V') @ Result

        if draw and drawtype == 'all':
            fig, axs = plt.subplots(1, 4, figsize = (15, 5))
            Top.draw(ax = axs[0], vmin = -1, vmax = 1)
            Top_title = f"${maptype_latex}_{{0,V}} \cdot B_{start_graph}$"
            axs[0].set_title(Top_title)

            Bottom.draw(axs[1], vmin = -1, vmax = 1)
            Bottom_title = f"$B_{end_graph} \cdot {maptype_latex}_{{0,E}}$"
            axs[1].set_title(Bottom_title)

            Result.draw(axs[2], vmin = -1, vmax = 1, colorbar = True)
            Result_title = Top_title[:-1] + ' - ' + Bottom_title[1:]
            axs[2].set_title(Result_title)

            Result_Dist.draw(axs[3], colorbar = True, cmap = 'PuOr')
            Result_Dist_title = f"$D_{{{end_graph},n,V}} \cdot ({Result_title[1:-1]})$"
            axs[3].set_title(Result_Dist_title)

        elif draw and drawtype == 'result':
            fig, ax = plt.subplots(1, 1, figsize = (5, 5))
            Result_Dist.draw(ax = ax, colorbar = True, cmap = 'PuOr')
            ax.set_title(f"Parallelogram for ${maptype_latex}$ (Edge-Vertex)")

        if returntype == 'dist':
            return Result_Dist
        elif returntype == 'commute':
            return Result
        else:
            raise ValueError(f"Unknown returntype {returntype}. Must be 'dist' or 'commute'.")

    #---

    def parallelogram(self, maptype = 'phi', obj_type = 'V', 
                            returntype = 'dist', 
                            draw = False, 
                            drawtype = 'all'):
                            
        """
        Get the paralellograms for checking that it's a nat trans.
        These are types 3-6 from Liz's Big List. 

        Parameters:
            maptype (str) : 
                The type of map. Either 'phi' or 'psi'.
            obj_type (str) : 
                The type of object. Either ``'V'`` or ``'E'``.
            returntype (str) : 
                The type of return. Either ``'dist'`` if you want the matrix that gives the thickenin grequired to make the diagram commute; or ``'commute'`` to just give the map mismatch.
            draw (bool) : 
                Whether to draw the maps. Default is False.
            drawtype (str) : 
                The type of drawing. Either 'all' or 'result'. Default is 'all'.
        
        Returns:
            LabeledMatrix : 
                The matrix that gives the thickening required to make the diagram commute if returntype is ``'dist'``; or the map mismatch if returntype is ``'commute'``.
        """

        if maptype == 'phi':
            start_graph = 'F'
            end_graph = 'G'
            maptype_latex = r'\varphi'
        elif maptype == 'psi':
            start_graph = 'G'
            end_graph = 'F'
            maptype_latex = r'\psi'

        Top = self.get_interleaving_map(maptype, 'n', obj_type) @ self.I(start_graph, '0', obj_type)
        Bottom = self.I(end_graph, 'n', obj_type) @ self.get_interleaving_map(maptype, '0', obj_type)
        Result = Top - Bottom
        Result_Dist = self.D(end_graph, '2n', obj_type) @ Result


        if draw and drawtype == 'all':
            fig, axs = plt.subplots(1, 4, figsize = (15, 5))
            Top.draw(ax = axs[0], vmin = -1, vmax = 1)
            Top_title = f"${maptype_latex}_{{n,{obj_type}}} \cdot I_{{0,{obj_type}}}$"
            axs[0].set_title(Top_title)

            Bottom.draw(axs[1], vmin = -1, vmax = 1)
            Bottom_title = f"$I_{{n,{obj_type}}} \cdot {maptype_latex}_{{0,{obj_type}}}$"
            axs[1].set_title(Bottom_title)

            Result.draw(axs[2], vmin = -1, vmax = 1, colorbar = True)
            Result_title = Top_title[:-1] + ' - ' + Bottom_title[1:]
            axs[2].set_title(Result_title)

            Result_Dist.draw(axs[3], colorbar = True, cmap = 'PuOr')
            Result_dist_title = f"$D_{{{end_graph},2n,{obj_type}}} \cdot ({Result_title[1:-1]})$"
            axs[3].set_title(Result_dist_title)

        elif draw and drawtype == 'result':
            fig, ax = plt.subplots(1, 1, figsize = (5, 5))
            Result_Dist.draw(ax = ax, colorbar = True, cmap = 'PuOr')
            ax.set_title(f"Parallelogram for ${maptype_latex}$ (Edge-Vertex)")


        if returntype == 'dist':
            return Result_Dist
        elif returntype == 'commute':
            return Result
        else:
            raise ValueError(f"Unknown returntype {returntype}. Must be 'dist' or 'commute'.")

    def triangle(self, start_graph = 'F', obj_type = 'V', returntype = 'dist',  draw = False, drawtype = 'all'):
        """
        Get the triangle for checking that it's an interleaving 

        Parameters:
            start_graph (str) : 
                The starting graph. Either 'F' or 'G'.
            obj_type (str) : 
                The type of object. Either ``'V'`` or ``'E'``.
            returntype (str) : 
                The type of return. Either ``'dist'`` if you want the matrix that gives the thickenin grequired to make the diagram commute; or ``'commute'`` to just give the map mismatch.
            draw (bool) : 
                Whether to draw the maps. Default is False.
            drawtype (str) : 
                The type of drawing. Either 'all' or 'result'. Default is 'all'.
        
        Returns:
            LabeledMatrix : 
                The matrix that gives the thickenin grequired to make the diagram commute if returntype is ``'dist'``; or the map mismatch if returntype is ``'commute'``.
        """

        if start_graph == 'F':
            end_graph = 'G'
            map1 = 'phi'
            map2 = 'psi'
        elif start_graph == 'G':
            end_graph = 'F'
            map1 = 'psi'
            map2 = 'phi'
        else:
            raise ValueError(f"Unknown start_graph {start_graph}. Must be 'F' or 'G'.")

        Top = self.I(start_graph, 'n', obj_type) @ self.I(start_graph, '0', obj_type)
        Bottom = self.get_interleaving_map(maptype = map2, key = 'n', obj_type = obj_type) @ self.get_interleaving_map(maptype = map1, key = '0', obj_type = obj_type)
        Result = Top - Bottom

        Result_Dist = self.D(start_graph, '2n', obj_type) @ Result

        if draw and drawtype == 'all':
            fig, axs = plt.subplots(1, 4, figsize = (15, 5))
            Top.draw(ax = axs[0], vmin = -1, vmax = 1)
            Top_title = f"$I_{{0,{obj_type}}} \cdot I_{{n,{obj_type}}}$"
            axs[0].set_title(Top_title)

            Bottom.draw(axs[1], vmin = -1, vmax = 1)
            Bottom_title = f"$I_{{n,{obj_type}}} \cdot I_{{2n,{obj_type}}}$"
            axs[1].set_title(Bottom_title)

            Result.draw(axs[2], vmin = -1, vmax = 1, colorbar = True)
            Result_title = Top_title[:-1] + ' - ' + Bottom_title[1:]
            axs[2].set_title(Result_title)

            Result_Dist.draw(axs[3], colorbar = True, cmap = 'PuOr')
            Result_dist_title = f"$D_{{{start_graph},2n,{obj_type}}} \cdot ({Result_title[1:-1]})$"
            axs[3].set_title(Result_dist_title)

        elif draw and drawtype == 'result':
            fig, ax = plt.subplots(1, 1, figsize = (5, 5))
            Result.draw(ax = ax, colorbar = True)
            ax.set_title(f"Triangle for {start_graph} (Edge-Vertex)")

        if returntype == 'dist':
            return Result_Dist
        elif returntype == 'commute':
            return Result
        else:
            raise ValueError(f"Unknown returntype {returntype}. Must be 'dist' or 'commute'.")

    def loss(self):
        """
        Computes the actual loss :math:`k` for the interleaving distance. This means that there is an :math:`(n+k)`-interleaving. 

        Returns:
            float : 
                The loss value.
        """

        loss_list = []

        # All the edge-vertex parallelogram maps 
        for maptype in ['phi', 'psi']:
            result = self.parallelogram_Edge_Vert(maptype = maptype)
            loss = result.absmax()
            loss_list.append(loss)

        # All the parallelogram maps 
        for maptype in ['phi', 'psi']:
            for obj_type in ['V', 'E']:
                result = self.parallelogram(maptype = maptype, obj_type = obj_type)
                loss = result.absmax()
                loss_list.append(loss)

        # ALl the triangle maps
        for obj_type in ['V', 'E']:
            for drawtype in ['all', 'result']:
                result = self.triangle(start_graph = 'F', obj_type = obj_type)
                loss = result.absmax()
                loss_list.append(loss)

        return max(loss_list)
        
