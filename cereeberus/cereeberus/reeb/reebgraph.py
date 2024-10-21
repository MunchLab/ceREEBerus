import networkx as nx
import cereeberus.compute.draw as draw
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
                # Can't figure out why the graphs are outputting a 
                # 3rd part of the tuple from our examples
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
    
    #-----------------------------------------------#
    # Methods for adding and removing nodes and edges 
    #-----------------------------------------------#
    def next_vert_name(self, s):
        """ 
        Making a simple name generator for vertices. 
        If you're using integers, it will just up the count by one. 
        Letters will be incremented in the alphabet. If you reach 'Z', it will return 'AA'. If you reach 'ZZ', it will return 'AAA', etc.

        Parameters:
            s (str or int): The name of the vertex to increment.

        Returns:
            str or int
                The next name in the sequence.
        """

        if type(s) == int:
            return s+1
        elif type(s) == str and len(s) == 1:
            if not s == 'Z':
                return chr(ord(s)+1)
            else:
                return 'AA'
        elif type(s) == str and len(s) > 1:
            if s[-1] == 'Z':
                return (len(s)+1)* 'A'
            else:
                return len(s)* chr(ord(s[-1])+1)
        else:
            ValueError('Input must be a string or an integer')

    def get_next_vert_name(self):
        """Get the next name for a vertex in the Reeb graph. If there are no nodes, it will return 0.

        Returns:
            str or int
                The next name in the sequence.
        """
        if len(self.nodes) == 0:
            return 0
        else:
            return self.next_vert_name(max(self.nodes))


    def add_node(self, vertex, f_vertex, reset_pos=True):
        """Add a vertex to the Reeb graph. 
        If the vertex name is given as None, it will be assigned via the next_vert_name method.

        Parameters:
            vertex (hashable like int or str, or None) : The name of the vertex to add.
            f_vertex (float) : The function value of the vertex being added.
            reset_pos (bool, optional) 
                If True, will reset the positions of the nodes based on the function values.
        """
        if vertex in self.nodes:
            raise ValueError(f'The vertex {vertex} is already in the Reeb graph.')

        if vertex is None:
            vertex = self.next_vert_name(max(self.nodes))
            
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
        del self.pos_f[vertex]

        if reset_pos:
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
        else: # the function values are the same, so the edge collapses the two vertices
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

    def draw(self, with_labels = True, with_colorbar = False, cpx = .1):
        """
        A drawing of the Reeb graph. Uses the fancy version from cereeberus.compute.draw.

        Parameters:
            with_labels (bool): Optional. If True, will include the labels of the nodes.
            cpx (float): Optional. A parameter that controls "loopiness" of multiedges in the drawing.
        
        Returns:
            None
        """

        # This really shouldn't be called ever since this is supposed to be maintained by the class
        if not set(self.pos_f.keys()) == set(self.nodes()):
            print('The positions are not set correctly. Setting them now.')
            self.set_pos_from_f()
        


    
        draw.reeb_plot(self, with_labels = with_labels, with_colorbar = with_colorbar, cpx=cpx, cpy=0)



    

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
    
    def boundary_matrix(self):
        """
        Returns the boundary matrix of the Reeb graph.

        Returns:
            numpy.ndarray: The boundary matrix of the Reeb graph.
        """
        return nx.linalg.graphmatrix.incidence_matrix(self, oriented = True).toarray()
    
    def boundary_matrix(self):
        """
        Creates an boundary matrix for the graph, where :math:`B[v,e] = 1` if vertex :math:`v` is an endpoint of edge :math:`e` and :math:`B[v,e] = 0` otherwise.

        Args:
            G (networkx.Graph): The graph.
            node_subset (list): A list of nodes representing the subset.

        Returns:
            numpy.ndarray: The boundary matrix.
        """

        V = list(self.nodes())
        E = list(self.edges())

        B = np.zeros((len(V), len(E)))

        for j, e in enumerate(E):
            i = V.index(e[0])
            B[i, j] = 1

            i = V.index(e[1])
            B[i, j] = 1

        return B
    
    def plot_boundary_matrix(self):
        plt.matshow(self.boundary_matrix())
        
        # Add vertices as the row labels
        plt.yticks(range(len(self.nodes)), self.nodes)
        plt.xticks(range(len(self.edges)), self.edges, rotation = 90)


    #-----------------------------------------------#
    # Operations on Reeb graph to get new Reeb graph
    #-----------------------------------------------#

    def smoothing(self, eps = 1, 
                  return_map = False, 
                  map_type = 'dict',
                  verbose = False):
        """
        Builds the `eps`-smoothed Reeb graph from the input. One way to define this is for a given Reeb graph :math:`(X,f)`, the smoothed graph is the Reeb graph of the product space :math:`(X \\times [-\\varepsilon, \\varepsilon], f(x) +  t)`. 

        Parameters:
            eps (float): The amount of smoothing to apply.
            return_map (bool): Optional. If True, will return a map from the vertices of the original Reeb graph to the vertices of the smoothed Reeb graph.
            map_type (str): Optional. The type of map to return. Can be 'matrix' or 'dictionary'. Default is 'matrix'.
            verbose (bool): Optional. If True, will print out additional information during the smoothing process.
        
        Returns:
            ReebGraph: The smoothed Reeb graph.
            If ``return_map`` is True, it returns a tuple with the second object giving a dictionary with keys as vertices in the original Reeb graph and values as vertices in the smoothed Reeb graph.
        """

        # Get the list of critical values to place new nodes 
        crit_vals = list(set(self.f.values()))
        new_crit_vals = [cv + eps for cv in crit_vals]
        new_crit_vals.extend([cv - eps for cv in crit_vals])
        if return_map == True:
            # Only need to create vertices at this level if we want to return the map from
            # the original vertices to the new vertices
            new_crit_vals.extend(list(self.f.values()))
        new_crit_vals = list(set(new_crit_vals))
        new_crit_vals.sort()

        # Create the new Reeb graph
        R_eps = ReebGraph()

        cv_0 = new_crit_vals[0]-eps # is arbitrary
        H_0 = self.slice(cv_0-eps, cv_0+eps, type = 'closed') # This should give empty graph 
        C_0 = list(H_0.connected_components()) # This should give empty list
        comp_to_new_vert_0 = {}

        if return_map == True:
            # This will be a dictionary with keys as vertices in R and values as vertices in R_eps
            map = {}


        for i in range(len(new_crit_vals)):
            cv = new_crit_vals[i]
            # print(f"Current critical value: {cv}")
            H = self.slice(cv-eps, cv+eps, type = 'closed')
            C = list(H.connected_components())

            # Add vertices 
            comp_to_new_vert = {}
            for i,c  in enumerate(C):
                vert_name = R_eps.get_next_vert_name()
                R_eps.add_node(vert_name, cv, reset_pos = False)
                comp_to_new_vert[i] = vert_name

                if return_map == True:
                    for v in c:
                        if v in self.nodes() and self.f[v] == cv:
                            map[v] = vert_name

            # Add edges 
            for i,c in enumerate(C):
                for i_0, c_0 in enumerate(C_0):
                    if len(c.intersection(c_0)) > 0:
                        R_eps.add_edge(comp_to_new_vert[i], comp_to_new_vert_0[i_0])

            # Set up for the next round
            cv_0 = cv
            H_0 = H
            C_0 = C 
            comp_to_new_vert_0 = comp_to_new_vert

        R_eps.set_pos_from_f()

        if return_map == True:
            if map_type == 'matrix':
                # Create the matrix version of the map
                map_matrix = np.zeros((len(self.nodes), len(R_eps.nodes)))
                for v in map:
                    map_matrix[v, R_eps.nodes().index(map[v])] = 1
                return R_eps, map_matrix
            else:
                return R_eps, map
        else:
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