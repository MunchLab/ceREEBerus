import networkx as nx
from ..compute import draw
import matplotlib.pyplot as plt
import numpy as np


class ReebGraph(nx.MultiDiGraph):
    """
    A Reeb graph stored as a networkx ``MultiDiGraph``. The function values are stored as a dictionary. The directedness of the edges follows the convention that the edge goes from the lower function value to the higher function value node.
    
    """

    def __init__(self, G=None, f={}, seed = None, verbose=False):
        """Initializes a Reeb graph object.
        
        Parameters:
            G : nx.graph, optional. If not None, a graph to initialize the Reeb graph.
            f : dict, optional. If not an empty dictionary, a dictionary of function values associated with the graph nodes.
            seed : int, optional. If not None, a seed to pass to the spring layout.
            verbose: bool, optional. If True, will print out additional information during initialization.
        """

        super().__init__()

        if G is None:
            self.f = {}
        elif f != {}:
            self.f = f
        else:
            # Some example graphs still have the function values stored as 'fx'
            self.f = nx.get_node_attributes(G, 'fx')
        
        if verbose:
            print(f'Function values: {self.f}')
        
        if G is not None:
            if verbose:
                print(f'Nodes: {G.nodes}')
            for v in G.nodes:
                self.add_node(v, self.f[v], reset_pos=False)

            # self.f = nx.get_node_attributes(G, 'f')

            for e in G.edges:
                e = e[:2]
                if verbose:
                    print(f'Edge: {e}')
                # Multigraph means edges are stored as a triple (u,v,n) where n is the number of the edge. 
                # This is an old note, maybe not true anymore?
                edge = (e[0], e[1])
                self.add_edge(*edge, reset_pos = False)

        self.set_pos_from_f(seed = seed, verbose=verbose)
    
    def __str__(self):
        return f'ReebGraph with {len(self.nodes)} nodes and {len(self.edges)} edges.'

    def summary(self):
        """Summary of the Reeb graph.
        
        Returns:
            dict
                A dictionary with the number of nodes and edges in the Reeb graph.
        """
        return {'nodes': len(self.nodes), 'edges': len(self.edges)}

    #-----------------#
    # Methods for getting info about the Reeb graph and its nodes
    #-----------------#

    def get_function_values(self):
        """
        Get the function values of the nodes in the Reeb graph.

        Returns:
            list
                A sorted list of function values.
        """
        L =  [self.f[v] for v in self.nodes]
        L.sort()
        return L

    def min_f(self):
        """
        Get the minimum function value in the Reeb graph.

        Returns:
            float
                The minimum function value.
        """
        return min(self.f.values())
        
    def max_f(self):
        """
        Get the maximum function value in the Reeb graph.

        Returns:
            float
                The maximum function value.
        """
        return max(self.f.values())
    

    def up_degree(self, node):
        """Get the up degree of a node.

        Parameters:
            node : int. The node to get the up degree of.
        
        Returns:
            int  
                The up degree of the node.
        """
        return self.out_degree(node)
    
    def down_degree(self, node):
        """
        Get the down degree of a node.

        Parameters:
            node : int
                The node to get the down degree of.
        
        Returns:
            int
                The down degree of the node.
        """
        return self.in_degree(node)

    def number_connected_components(self):
        """
        Get the number of connected components in the Reeb graph.
        
        Returns:
            int
                The number of connected components in the Reeb graph.
        """
        return nx.number_connected_components(self.to_undirected())
    
    def func_to_vertex_dict(self):
        """
        Get a dictionary mapping function values to all vertices at that height.

        Returns:
            dict
                A dictionary mapping function values to vertices.
        """
        f_to_v = {}
        for v in self.nodes:
            f = self.f[v]
            if f in f_to_v:
                f_to_v[f].append(v)
            else:
                f_to_v[f] = [v]
        return f_to_v

    def sorted_vertices(self):
        """
        Get a list of vertices sorted by function value. Same order as passed by the func_to_vertex_dict method.

        Returns:
            list
                A list of vertices sorted by function value.
        """
        
        func_dict = self.func_to_vertex_dict()
        keys = list(func_dict.keys())
        keys.sort()
        vertices = [ func_dict[k] for k in keys]
        vertices = [v for sublist in vertices for v in sublist]

        return vertices 


    def func_to_edge_dict(self):
        """
        Get a dictionary mapping function values to all edges with lower endpoint at that height.

        Returns:
            dict
                A dictionary mapping function values to edges.
        """
        f_to_e = {}
        for e in self.edges:
            f = self.f[e[0]]
            if f in f_to_e:
                f_to_e[f].append(e)
            else:
                f_to_e[f] = [e]
        return f_to_e
    
    def sorted_edges(self):
        """
        Get a list of edges sorted by function value. Same order as passed by the func_to_edge_dict method.

        Returns:
            list
                A list of edges sorted by function value.
        """
        func_dict = self.func_to_edge_dict()
        keys = list(func_dict.keys())
        keys.sort()
        edges = [ func_dict[k] for k in keys]
        edges = [item for row in edges for item in row]

        return edges

    def relabel_nodes(self, mapping):
        """
        Relabel the nodes of the Reeb graph using a mapping. The mapping should be a dictionary with keys as old node names and values as new node names.

        Parameters:
            mapping : dict. A dictionary mapping old node names to new node names.
        """
        
        # Store copies of the old stuff 
        nodes_old = list(self.nodes()).copy()
        edges_old = list(self.edges(keys=True)).copy()
        f_old = self.f.copy()
        
        # Delete all nodes and edges from the graph
        for n in nodes_old:
            self.remove_node(n)
            
        # Add the nodes and edges back in with the new names
        for v_old in nodes_old:
            self.add_node(mapping[v_old], f_old[v_old])
            
        for e in edges_old:
            self.add_edge(mapping[e[0]], mapping[e[1]], e[2])
            
    def inv_image(self, f_val):
        """
        Get the set of vertices and/or edges at a given function value. This will return $v$ for which $f(v) = $`f_val`, as well as edges $e = (u,v)$ for which $f(u) < $`f_val` and $f(v) > $`f_val`.

        Parameters:
            f_val : float. The function value to get the objects for.

        Returns:
            tuple : (set, set)
                The set of vertices and the set of edges at the function value `f_val`.
        """
        verts = {v for v in self.nodes if self.f[v] == f_val}
        edges = {e for e in self.edges if self.f[e[0]] < f_val and self.f[e[1]] > f_val}
        return verts, edges
        

    #-----------------------------------------------#
    # Methods for getting distances in the Reeb graph
    #-----------------------------------------------#

    def thickening_distance(self, u, v):
            """
            Get the thickening distance between two vertices in the Reeb graph. This is the amount of thickening needed before the two vertices map to the same connected component. Note that u and v need to be at the same function value, so f(u) = f(v). 

            Parameters:
                u, v : int. The vertices to get the thickening distance between.
            
            Returns:
                int
                    The thickening distance between the two vertices.
            """
            if self.f[u] != self.f[v]:
                raise ValueError(f"Vertices {u} and {v} are not at the same function value.")

            a = self.f[u]

            # Get function values 
            all_f = self.get_function_values()
            all_f = list(set(all_f))
            all_f.sort()

            # Get differences between the function value and all other function values
            diff = np.abs(np.array(all_f) - a)
            diff = diff[diff != 0]
            diff.sort()

            # For each difference, build the slice and see if the two vertices are in the same connected component
            for n in diff:
                S = self.slice(a-n, a+n, type = 'closed')
                try:
                    nx.shortest_path(S.to_undirected(), u, v)
                    return n
                except:
                    pass

   
    #-----------------------------------------------#
    # Methods for adding and removing nodes and edges 
    #-----------------------------------------------#
 



    def get_next_vert_name(self):
        """Get the next name for a vertex in the Reeb graph that isn't already included. If there are no nodes, it will return 0. Otherwise, it will name the next vertex as one more than the maximum integer labeled vertex, ignoring any strings. 

        Returns:
            str or int
                The next name in the sequence.
        """
        integer_nodes = [v for v in self.nodes.keys() if type(v) == int]
        if len(integer_nodes) == 0:
            return 0
        else:
            return max(integer_nodes) + 1


    def add_node(self, vertex, f_vertex, reset_pos=True):
        """Add a vertex to the Reeb graph. 
        If the vertex name is given as None, it will be assigned via the get_next_vert_name method.

        Parameters:
            vertex (hashable like int or str, or None) : The name of the vertex to add.
            f_vertex (float) : The function value of the vertex being added.
            reset_pos (bool, optional) 
                If True, will reset the positions of the nodes based on the function values.
        """
        if vertex in self.nodes:
            raise ValueError(f'The vertex {vertex} is already in the Reeb graph.')

        if vertex is None:
            vertex = self.get_next_vert_name()
            
        super().add_node(vertex)

        self.f[vertex] = f_vertex

        if reset_pos:
            self.set_pos_from_f()

    def add_nodes_from(self, nodes, f_dict, reset_pos=True):
        """Add a list of nodes to the Reeb graph.

        Parameters:
            nodes (list) 
                The list of node names to add.
            f_dict (dict): A dictionary of function values associated with the nodes. Should be ``f_dict[node] = f_value``.
            reset_pos (bool, optional): If True, will reset the positions of the nodes based on the function values.
        """
        for node in nodes:
            self.add_node(node, f_dict[node], reset_pos=False)

        for node in nodes:
            self.f[node] = f_dict[node]
        
        if reset_pos:
            self.set_pos_from_f()

    def remove_node(self, vertex, reset_pos=True):
        """Remove a vertex from the Reeb graph.

        Parameters:
            vertex (hashable like int or str): The name of the vertex to remove.
            reset_pos (bool, optional): If True, will reset the positions of the nodes based on the function values.
        """
        if vertex not in self.nodes:
            raise ValueError(f'The vertex {vertex} is not in the Reeb graph.')
        super().remove_node(vertex)
        del self.f[vertex]

        if reset_pos and hasattr(self, 'pos_f'):
            del self.pos_f[vertex]
            self.set_pos_from_f()

    def remove_nodes_from(self, nodes, reset_pos=True):
        """Remove a list of nodes from the Reeb graph.

        Parameters:
            nodes (list): The list of node names to remove.
            reset_pos (bool, optional): If True, will reset the positions of the nodes based on the function values.
        """
        for node in nodes:
            self.remove_node(node, reset_pos=False)

        if reset_pos:
            self.set_pos_from_f()

    def add_edge(self, u,v, reset_pos = True):
        """Add an edge to the Reeb graph. Make sure that the edge points to the vertex with higher function value. 

        Note that if the edge added is between two vertices with the same function value, it will collapse the two vertices into one.

        Parameters:
            u,v: The edge to add.
            reset_pos (bool): Optional. If True, will reset the positions of the nodes based on the function values.
        """
        for vertex in [u,v]:
            try:
                self.f[vertex]
            except KeyError:
                raise ValueError(f'The vertex {vertex} must be in the Reeb graph to add an edge between them.')
                
        if self.f[u] < self.f[v]:
            edge = (u,v)
            super().add_edge(*edge)
        elif self.f[u] > self.f[v]:
            edge = (v,u)
            super().add_edge(*edge)
        else: 
            # the function values are the same, so the edge collapses the two vertices
            # wlog we're going to get rid of v, and add all its edges to u

            # get the edges of v
            edges_in = self.in_edges(v)
            edges_out = self.out_edges(v)

            # add the edges to u
            for e in edges_in:
                self.add_edge(e[0], u)
            for e in edges_out:
                self.add_edge(u, e[1])

            # Remove v
            self.remove_node(v)

        if reset_pos:
            self.set_pos_from_f()
    
    def add_edges_from(self, edges, reset_pos = True):
        """Add a list of edges to the Reeb graph.

        Parameters:
            edges (list): The list of edges to add.
            reset_pos (bool): Optional. If True, will reset the positions of the nodes based on the function values.
        """
        for edge in edges:
            self.add_edge(*edge, reset_pos=False)

        if reset_pos:
            self.set_pos_from_f()

    
    def subdivide_edge(self, u, v, w, f_w):
        """Subdivide an edge with a new vertex.

        Parameters:
            u,v: The edge to subdivide.
            w: The new vertex to add.
            f_w: The function value of the new vertex.
        """
        f_lower = min(self.f[u], self.f[v])
        f_upper = max(self.f[u], self.f[v])
        if f_w < f_lower or f_w > f_upper:
            raise ValueError('The function value of the new vertex must be between the function values of the edge vertices.')
        
        edge = sorted([u,v], key = lambda x: self.f[x])
        if edge not in self.edges:
            raise ValueError('The edge must be in the Reeb graph to subdivide it.')

        self.f[w] = f_w

        self.remove_edge(*edge)
        self.add_node(w, f_w)
        self.add_edge(u, w)
        self.add_edge(w, v)

        self.set_pos_from_f()

    def remove_regular_vertex(self, v):
        """Remove a regular vertex from the Reeb graph. A regular vertex is one for which down degree = up degree = 1, so it can be removed and replaed with a single edge.

        Parameters:
            v (int): The vertex to remove.
        """
        if v not in self.nodes:
            raise ValueError('The vertex must be in the Reeb graph to be removed.')
        
        if self.up_degree(v) != 1 or self.down_degree(v) != 1:
            raise ValueError('The vertex must have up degree 1 and down degree 1 to be removed as a regular vertex.')
        
        u = list(self.predecessors(v))[0]
        w = list(self.successors(v))[0]

        self.add_edge(u,w)
        self.remove_node(v)

    def remove_all_regular_vertices(self):
        """
        Remove all regular vertices from the Reeb graph.
        """
        regular_vertices = [v for v in self.nodes if self.up_degree(v) == 1 and self.down_degree(v) == 1]

        for v in regular_vertices:
            self.remove_regular_vertex(v)

    def remove_isolates(self):
        """
        Remove all isolated vertices from the Reeb graph.
        """
        self.remove_nodes_from(list(nx.isolates(self)))

    #----------------------------------#
    # Methods for drawing the Reeb graph
    #----------------------------------#
    def set_pos_from_f(self, seed = None, verbose=False):
        """Set the position of the nodes based on the function values. The result will be the (spring layout x, function value y). Note that this will overwrite the previous positions.

        Parameters:
            verbose (bool): Optional. If True, will print out the function values and the positions.
        """
        if len(self.nodes) == 0:
            if verbose:
                print('Nothing to be done, no nodes here')
            self.pos = {}
            self.pos_f = {}
        else:
            pos = nx.spring_layout(self, seed = seed)
            self.pos = pos

            self.pos_f = {}

            for v in self.nodes:
                self.pos_f[v] = (self.pos[v][0], self.f[v])

            if verbose:
                print('Function values:', self.f)
                print('Positions:', self.pos_f)

    def draw(self, with_labels = True, with_colorbar = False, cpx = .1, ax = None, **kwargs):
        """
        A drawing of the Reeb graph. Uses the fancy version from cereeberus.compute.draw.

        Parameters:
            with_labels (bool): Optional. If True, will include the labels of the nodes.
            cpx (float): Optional. A parameter that controls "loopiness" of multiedges in the drawing.
        
        Returns:
            None
        """

        if ax is None:
           ax = plt.gca()

        # This really shouldn't be called ever since this is supposed to be maintained by the class
        if not set(self.pos_f.keys()) == set(self.nodes()):
            print('The positions are not set correctly. Setting them now.')
            self.set_pos_from_f()
        


    
        draw.reeb_plot(self, with_labels = with_labels, with_colorbar = with_colorbar, cpx=cpx, cpy=0, ax = ax, **kwargs)



    

    #-----------------------------------------------#
    # Methods for getting portions of the Reeb graph
    #-----------------------------------------------#
    
    def induced_subgraph(self, nodes):
        """
        Returns the subgraph of the Reeb graph induced by the nodes in the list nodes.

        Parameters:
            nodes (list): The list of nodes to include in the subgraph.
        
        Returns:
            ReebGraph: The subgraph of the Reeb graph induced by the nodes in the list nodes.
        """
        H = ReebGraph(self.subgraph(nodes), f={v: self.f[v] for v in nodes})

        return H
    

    def slice(self,a,b, type = 'open', verbose = False):
        """
        Returns the subgraph of the Reeb graph with image in the open interval (a,b) if `type = 'open'` or in the closed interval [a,b] of `type = 'closed'`.
        This will convert any edges that cross the slice into vertices.

        Parameters:
            a (float): The lower bound of the slice.
            b (float): The upper bound of the slice.
            type (str): Optional. The type of interval used to take the slice. Can be 'open' or 'closed'. Default is 'open'.
        
        Returns:
            ReebGraph: The subgraph of the Reeb graph with image in (a,b).
        """
        if type == 'open':
            v_list = [v for v in self.nodes() if self.f[v] > a and self.f[v] < b]
        elif type == 'closed':
            v_list = [v for v in self.nodes() if self.f[v] >= a and self.f[v] <= b]

        # Keep the edges where either endpoint (or both) is in (a,b)
        e_list = [e for e in self.edges() if e[0] in v_list or e[1] in v_list ]
        #Include the edges that cover the entire slice. 
        # Note this assumes that all edges are ordered twoards teh upper function value
        e_list.extend([e for e in self.edges() if self.f[e[0]]<a and self.f[e[1]]>b])

        # Make a dictionary of counts to deal with multiedges
        e_dict = {e:e_list.count(e) for e in e_list}



        if verbose:
            print('Vertices (v,f(v)):', [(v, self.f[v]) for v in v_list])
            print('Edges:', e_list)
            print('Edge dictionary:', e_dict)


        # Build the subgraph 
        H = ReebGraph()

        for v in v_list:
            H.add_node(v, self.f[v])

        for e in e_dict:
            if e[0] in v_list and e[1] in v_list:
                # The edge is entirely in the set, so both vertices are already there
                if verbose:
                    print(f'Adding {e_dict[e]} of edge {e} entirely inside slice:')

                for i in range(e_dict[e]): # Add an edge for each copy in the list
                    H.add_edge(e[0], e[1])


            elif e[0] not in v_list and e[1] not in v_list:
                # The edge is entirely crossing the slice, so we add two vertices and an edge
                if verbose:
                    print(f'Adding {e_dict[e]} of edge {e} with both endpoints outside slice')
                
                for i in range(e_dict[e]):
                    v1 = '-'.join([str(v) for v in e])+'_'+str(i)+'_lower'
                    v2 = '-'.join([str(v) for v in e])+'_'+str(i)+'_upper'
                    H.add_node(v1, a)
                    H.add_node(v2, b)
                    H.add_edge(v1,v2)
            else:
                # One vertex is in the set and one is out. 
                # Need to check (for the closed case) that this isn't an edge going up from the top bound or down from the bottom bound
                if e[0] in v_list and self.f[e[0]] == b or e[1] in v_list and self.f[e[1]] == a:
                    if verbose:
                        print(f'Edge {e} is not in the slice')
                    continue

                if verbose:
                    print(f'adding {e_dict[e]} of edge {e} with one endpoint outside slice')
                
                # Make a name for the new vertex to create
                for i in range(e_dict[e]):
                    edge_name = '-'.join([str(v) for v in e])+'_'+str(i)

                    if e[0] in v_list:
                        if verbose:
                            print(f'Adding part of edge {e[0],e[1]} with function values {self.f[e[0]],[e[1]]} and {self.f[e[1]]}')
                            print(f'This should add edge {e[0]}-{edge_name} with function values {self.f[e[0]]} and {b}')

                        # The lower edge is in the set, so the other vertex must have 
                        # value above the max 
                        assert self.f[e[1]] >= b
                        vert_in = e[0]
                        func_val = b

                        # Add a new vertex called edge_name with value b
                        H.add_node(edge_name, func_val)
                        H.add_edge(e[0], edge_name)
                    else:
                        # The higher edge is in the set, so the other vertex must have 
                        # value below the min 
                        if verbose:
                            print(f'Adding part of edge {e[0],e[1]} with function values {self.f[e[0]]} and {self.f[e[1]]}')
                            print(f'This should add edge ({edge_name}, {e[1]}) with function values {a} and {self.f[e[1]]}')

                        assert self.f[e[0]] <= a
                        vert_in = e[1]
                        func_val = a

                        # Add a new vertex called edge_name with value a
                        H.add_node(edge_name, func_val)
                        H.add_edge(edge_name, e[1])
        return H
    
    def connected_components(self):
        """
        Returns the connected components of the Reeb graph by providing a set of vertices for each connected component. If you want to actually get the subgraphs of ReebGraph `R` returned as Reeb graphs, you can use::
            
            [R.induced_subgraph(component) for component in R.connected_components()]
        
        Returns:
            A generator of sets of nodes, one for each component of the Reeb graph.

        """

        return nx.connected_components(self.to_undirected())
    
    
    def get_largest_component(self):
        """
        Returns the largest connected component of the Reeb graph as a set of nodes. Note: this is not well defined unless the Reeb graph is minimal.

        Returns:
            ReebGraph: The largest connected component of the Reeb graph based on number of vertices. 
        """
        
        v_list = max(self.connected_components(), key=len)
        return self.induced_subgraph(v_list)
    
    def get_multi_edges(self):
        '''
        Returns a set of the edges that have count higher than 1. Useful for checking for multiedges in the Reeb graph.
        
        Returns:
            set: A set of edges that have count higher than 1.
        '''
        
        mult_edges = [(e[0],e[1]) for e in self.edges if self.number_of_edges(e[0], e[1]) > 1]
        return set(mult_edges)
        

    #-----------------------------------------------#
    # Methods for getting matrices for the graph 
    #-----------------------------------------------#

    def adjacency_matrix(self, symmetrize = True):
        """
        Returns the adjacency matrix of the Reeb graph. Symmetrizes the matrix by default; if symmetrize = False, it will return the directed adjacency matrix.

        Parameters:
            symmetrize (bool): Optional. If True, will return the symmetrized adjacency matrix. If False, will return the directed adjacency matrix.

        Returns:
            numpy.ndarray: The adjacency matrix of the Reeb graph.
        """
        A = nx.to_numpy_array(self)
        if symmetrize:
            A = A + A.T

        return A

    def plot_adjacency_matrix(self, symmetrize = True):
        plt.matshow(self.adjacency_matrix(symmetrize = symmetrize))
        
        # Add vertices as the row and column labels
        plt.xticks(range(len(self.nodes)), self.nodes, rotation = 90)
        plt.yticks(range(len(self.nodes)), self.nodes)
    
    def boundary_matrix(self, astype = 'numpy'):
        """
        Creates an boundary matrix for the graph, where :math:`B[v,e] = 1` if vertex :math:`v` is an endpoint of edge :math:`e` and :math:`B[v,e] = 0` otherwise. If astype is `map`, it will return a dictionary with keys as edges and values as lists of vertices that are endpoints of that edge.

        Args:
            astype (str): Optional. The type of output to return. Can be 'numpy', or 'map'. Default is 'numpy'. 

        Returns:
            numpy.ndarray: The boundary matrix.
        """

        V = list(self.nodes())
        V.sort(key = lambda x: self.f[x])
        E = list(self.edges())
        E.sort(key = lambda x: self.f[x[0]])
        if astype == 'numpy':
            B = np.zeros((len(V), len(E)))

            for j, e in enumerate(E):
                i = V.index(e[0])
                B[i, j] = 1

                i = V.index(e[1])
                B[i, j] = 1

            return B

        elif astype == 'map':
            B = {}
            for e in E:
                if not len(e)== 3:
                    e = (e[0], e[1], 0) 
                B[e] = [e[0], e[1]]
            return B

        else:
            raise ValueError('astype must be either "numpy" or "map".')

    
    def plot_boundary_matrix(self):
        plt.matshow(self.boundary_matrix())
        
        # Add vertices as the row labels
        plt.yticks(range(len(self.nodes)), self.nodes)
        plt.xticks(range(len(self.edges)), self.edges, rotation = 90)


    #-----------------------------------------------#
    # Operations on Reeb graph to get new Reeb graph
    #-----------------------------------------------#

    def smoothing_and_maps(self, eps = 1, 
                            verbose = False):
        """
        Builds the ``eps``-smoothed Reeb graph from the input. One way to define this is for a given Reeb graph :math:`(X,f)`, the smoothed graph is the Reeb graph of the product space :math:`(X \\times [-\\varepsilon, \\varepsilon], f(x) +  t)`. This function also returns the maps (i) from the vertices of the original Reeb graph to the vertices of the smoothed Reeb graph and (ii) from the edges of the original to the edges of the smoothed graph.
        These maps are given as a dictionary with keys as vertices (resp. edges) in the original Reeb graph and values as vertices (resp. edges) in the smoothed Reeb graph.

        Parameters:
            eps (float): The amount of smoothing to apply.
            verbose (bool): Optional. If True, will print out additional information during the smoothing process.
        
        Returns:
            tuple: ReebGraph, vertex_map, edge_map.
        """

        # If eps is 0, return the original graph
        if eps ==0:
            return self, {v:v for v in self.nodes}, {e:[e] for e in self.edges(keys = True)}

        # Get the list of critical values to place new nodes 
        crit_vals = list(set(self.f.values()))
        new_crit_vals = [cv + eps for cv in crit_vals]
        new_crit_vals.extend([cv - eps for cv in crit_vals])
        
        # Add vertices at the locations of the function values of the inputs
        new_crit_vals.extend(list(self.f.values()))
        new_crit_vals = list(set(new_crit_vals))
        new_crit_vals.sort()

        # get smallest diff between adjacent critical values
        # This delta is used for the shift to figure out edges below
        # the vertices at level cv
        min_diff = min([new_crit_vals[i+1] - new_crit_vals[i] for i in range(len(new_crit_vals)-1)])
        delta = min(min_diff/2, eps/2)

        # Create the new Reeb graph
        R_eps = ReebGraph()

        cv_0 = new_crit_vals[0]-eps # Starting value is arbitrary below the first critical value
        H_0 = self.slice(cv_0-eps, cv_0+eps, type = 'closed') # This should initialize empty graph 
        C_0 = list(H_0.connected_components()) # This should give empty list
        comp_to_new_vert_0 = {}

        # This will be a dictionary with keys as vertices in R and values as vertices in R_eps
        map_V = {}
        map_E = {e: [] for e in self.edges(keys = True)}

        for i in range(len(new_crit_vals)):
            cv = new_crit_vals[i]
            H = self.slice(cv-eps, cv+eps, type = 'closed')
            C = list(H.connected_components())
            
            # ==== Add vertices at level cv =====

            comp_to_new_vert = {}
            # comp_to_new_vert is a dictionary with keys as indices of connected components in the list C and values as the name of the vertex in R_eps
            for i,c  in enumerate(C):
                vert_name = R_eps.get_next_vert_name()
                R_eps.add_node(vert_name, cv, reset_pos = False)
                comp_to_new_vert[i] = vert_name

                # Track the vertex map
                for v in c:
                    if v in self.nodes() and self.f[v] == cv:
                        map_V[v] = vert_name

            # ===== Add edges below the vertices at level cv =====

            # Get the slice slightly below the vertex
            H_edge = self.slice(cv-eps-delta, cv + eps-delta, type = 'closed')
            C_edge = list(H_edge.connected_components())

            # Strip the upper and lower from the strings to be able to check overlaps 
            for conn_comp_list in [C, C_0, C_edge]:
                for j, conn_comp in enumerate(conn_comp_list):
                    conn_comp_new = []
                    for i, v in enumerate(conn_comp):
                        if type(v) == str and (v[-6:] == '_upper' or v[-6:] == '_lower'):
                            conn_comp_new.append(v[:-6])
                        else:
                            conn_comp_new.append(v)
                    conn_comp_list[j] = set(conn_comp_new)

            # Add edges 
            # Track the stuff that needs to be included in the map
            V_inv, E_inv = self.inv_image(cv-delta)
            
            for i,c in enumerate(C_edge):
                overlap_down = np.array([len(c.intersection(c_0)) for c_0 in C_0])
                lower_vert = [comp_to_new_vert_0[u] for u in  np.where(overlap_down > 0)[0]]
                if len(lower_vert) > 1:
                    print(f'{i,c} has multiple lower vertices')
                else:
                    lower_vert = lower_vert[0]
                
                overlap_up = np.array([len(c.intersection(c_0)) for c_0 in C])
                upper_vert = [comp_to_new_vert[u] for u in  np.where(overlap_up > 0)[0]]
                if len(upper_vert) > 1:
                    print(f'{i,c} has multiple upper vertices')
                else:
                    upper_vert = upper_vert[0]
                
                R_eps.add_edge(lower_vert, upper_vert)
                
                new_edge_name = (lower_vert, upper_vert, R_eps.number_of_edges(lower_vert, upper_vert)-1) # Because we just added the new edge, it should be the last one
                
                # Right now this assumes an endpoint vertex is in the sliced graph, is it possible for it to be edges only? 
                for e in E_inv: 
                    if e[0] in c or e[1] in c:
                        map_E[e].append(new_edge_name)
                


            # Set up for the next round
            cv_0 = cv
            H_0 = H
            C_0 = C 
            comp_to_new_vert_0 = comp_to_new_vert

        R_eps.set_pos_from_f()
        
        return R_eps, map_V, map_E

    def smoothing(self, eps = 1):
        """
        Builds the ``eps``-smoothed Reeb graph from the input. One way to define this is for a given Reeb graph :math:`(X,f)`, the smoothed graph is the Reeb graph of the product space :math:`(X \\times [-\\varepsilon, \\varepsilon], f(x) +  t)`. 

        Parameters:
            eps (float): The amount of smoothing to apply.
        
        Returns:
            ReebGraph: The smoothed Reeb graph.
        """
        R_eps, _, _ = self.smoothing_and_maps(eps = eps)
        return R_eps

    #-----------------------------------------------#
    # Methods for converting to other graph types
    #-----------------------------------------------#
    

    def to_mapper(self, delta = None):
        """
        Convert the Reeb graph to a Mapper graph as long as all function values are integers. Note this is NOT the same as computing the mapper graph of a given Reeb graph as the input topological space.  This will create a new Mapper graph object with the same nodes and edges as the Reeb graph.

        Parameters:
            delta (float): Optional. The delta value to use for the Mapper graph. If None, will use 1.

        Returns:
            MapperGraph: The Mapper graph representation of the Reeb graph.
        """

        # Check that all function values are integers before proceeding
        if not all([isinstance(self.f[v], int) for v in self.f]):
            raise ValueError("Function values must be integers to convert to a Mapper graph.")

        from cereeberus.reeb.mapper import MapperGraph
        return MapperGraph(self, f=self.f, delta = delta)