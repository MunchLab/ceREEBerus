import networkx as nx
import cereeberus.compute.draw as draw
import matplotlib.pyplot as plt

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
    
    #-----------------------------------------------#
    # Methods for adding and removing nodes and edges 
    #-----------------------------------------------#

    def add_node(self, vertex, f_vertex, reset_pos=True):
        """Add a vertex to the Reeb graph. 

        Parameters:
            vertex (hashable like int or str) : The name of the vertex to add.
            f_vertex (float) : The function value of the vertex being added.
            reset_pos (bool, optional) 
                If True, will reset the positions of the nodes based on the function values.
        """
        if vertex in self.nodes:
            raise ValueError(f'The vertex {vertex} is already in the Reeb graph.')
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

        self.add_node(w, f_w)
        self.add_edge(u, w)
        self.add_edge(w, v)
        self.remove_edge(*edge)

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

    def draw(self, cpx = .1, cpy = .1):
        """
        A drawing of the Reeb graph. Uses the fancy version from cereeberus.compute.draw.

        Parameters:
            cpx (float): Optional. A parameter that controls "loopiness" of multiedges in the drawing.
            cpy (float): Optional. A parameter that controls "loopiness" of multiedges in the drawing.
        
        Returns:
            None
        """

        if not set(self.pos_f.keys()) == set(self.nodes()):
            print('The positions are not set correctly. Setting them now.')
            self.set_pos_from_f()

    
        draw.reeb_plot(self, pos=self.pos_f, cpx=cpx, cpy=cpy)


    def draw_straight_line(self, ax=None, **kwargs):
        """
        A straight line drawing of the Reeb graph. Very simple, but if you have multi-edges, they won't show up.

        Parameters:
            ax (matplotlib axis): Optional. If not passed, will create a new figure and axis.
            **kwargs: Additional keyword arguments to pass to the draw.
        
        Returns:
            ax (matplotlib axis): The matplotlib axis object used for the drawing.
        """
        if ax is None:
            fig, ax = plt.subplots()

        nx.draw(self, pos=self.pos_f, **kwargs)
        plt.axis('on')
        ax.tick_params(left=True, labelleft=True)

        return ax
    

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
    

    def slice(self,a,b, verbose = False):
        """
        Returns the subgraph of the Reeb graph with image in the open interval (a,b).
        This will convert any edges that cross the slice into vertices.

        Parameters:
            a (float): The lower bound of the slice.
            b (float): The upper bound of the slice.
        
        Returns:
            ReebGraph: The subgraph of the Reeb graph with image in (a,b).
        """
        v_list = [v for v in self.nodes() if self.f[v] > a and self.f[v] < b]

        # Keep the edges where either endpoint (or both) is in (a,b)
        e_list = [e for e in self.edges() if e[0] in v_list or e[1] in v_list ]
        #Include the edges that cover the entire slice. 
        # Note this assumes that all edges are ordered twoards teh upper function value
        e_list.extend([e for e in self.edges() if self.f[e[0]]<a and self.f[e[1]]>b])
        # e_list = ['-'.join([str(v) for v in edge]) for edge in e_list]

        if verbose:
            print('Vertices (v,f(v)):', [(v, self.f[v]) for v in v_list])
            print('Edges:', e_list)


        # Build the subgraph 
        H = ReebGraph()

        for v in v_list:
            H.add_node(v, self.f[v])

        for e in e_list:
            if e[0] in v_list and e[1] in v_list:
                # The edge is entirely in the set, so both vertices are already there
                if verbose:
                    print('Adding edge entirely inside slice:', e)
                H.add_edge(e[0], e[1])
            elif e[0] not in v_list and e[1] not in v_list:
                # The edge is entirely crossing the slice, so we add two vertices and an edge
                if verbose:
                    print('Adding edge with both endpoints outside slice:', e)
                H.add_node('-'.join([str(v) for v in e])+'_lower', a)
                H.add_node('-'.join([str(v) for v in e])+'_upper', b)
                H.add_edge('-'.join([str(v) for v in e])+'_lower', '-'.join([str(v) for v in e])+'_upper')
            else:
                # Half the edge is included 

                # Make a name for the new vertex to create
                edge_name = '-'.join([str(v) for v in e])
                if verbose:
                    print('adding edge with one endpoint outside slice:', e)

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
    

