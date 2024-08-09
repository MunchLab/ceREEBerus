from cereeberus import ReebGraph

class MapperGraph(ReebGraph):
    """
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
        
        last_vert_name = max(self.nodes())
        
        for i in range(n_low,n_high+1):
            e_list = [e for e in self.edges() if self.f[e[0]] < i and self.f[e[1]] > i]

            for e in e_list:
                w_name = self.next_vert_name(last_vert_name)
                self.subdivide_edge(*e,w_name, i)

                last_vert_name = w_name
            
    
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
    
