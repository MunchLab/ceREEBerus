import numpy as np
import matplotlib.pyplot as plt


class BranchDecomp:
    """
    A branch decomposition of a Reeb graph.

    A branch decomposition breaks a Reeb graph into a collection of non-overlapping
    upward paths called *branches*. Each branch is a path that travels strictly upward
    in function value, starting at either a local minimum or a saddle point (where a
    previous branch split off) and ending at either a local maximum or another saddle
    point. The decomposition is built greedily and is not unique.

    **Algorithm:** Branches are extracted one at a time by repeatedly finding the
    unprocessed vertex with the lowest function value that still has outgoing edges,
    then greedily following successors upward until a local maximum is reached. The
    edges along that path are removed from a working copy of the graph, and the
    process repeats until no edges remain. Isolated vertices (no edges) are added last
    as degenerate branches.

    **Branch storage (``self.branches``):** An ``n x 4`` numpy array where each row
    represents one branch in the order they were extracted:

    - Column 0 (``f_low``): function value of the lower endpoint.
    - Column 1 (``f_high``): function value of the upper endpoint. Equal to ``f_low``
      for degenerate (isolated vertex) branches.
    - Column 2 (``low_attach``): index of the branch that owns the lower endpoint
      vertex. If the lower endpoint is a local minimum, this is the branch's own index.
    - Column 3 (``high_attach``): index of the branch that owns the upper endpoint
      vertex. If the upper endpoint is a local maximum, this is the branch's own index.
      Otherwise it points to the earlier branch whose path passes through that vertex.

    Because branches are extracted in order, attachment indices always satisfy
    ``low_attach <= branch_id`` and ``high_attach <= branch_id``.

    **Path storage (``self.paths``):** A list of lists, in the same order as
    ``self.branches``. Each entry is the ordered list of vertex names (from the
    original ReebGraph) that make up the branch path.

    **Reconstruction:** The original Reeb graph can be recovered exactly from the
    stored paths and function values via :meth:`reconstruct`.

    Example::

        import cereeberus.data.ex_reebgraphs as ex_rg
        from cereeberus.reeb.branchdecomp import BranchDecomp

        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)
        bd.draw()
        R2 = bd.reconstruct()
    """
    def __init__(self):
        self.branches = np.empty((0, 4))
        self.paths = []
        

    @staticmethod
    def _lowest_available_vertex(graph):
        """
        Find the vertex with the lowest function value that still has outgoing edges.
        
        Parameters:
            graph: ReebGraph object
            
        Returns:
            The vertex with lowest function value and out_degree > 0, or None if no such vertex exists.
        """
        available_vertices = [v for v in graph.nodes if graph.up_degree(v) > 0]
        
        if not available_vertices:
            return None
        
        return min(available_vertices, key=lambda v: graph.f[v])
    
    @staticmethod
    def _largest_upward_path(graph, start_vertex):
        """
        Greedily follow upward edges from a starting vertex until reaching a local maximum.
        
        Parameters:
            graph: ReebGraph object
            start_vertex: vertex to start from
            
        Returns:
            A list of vertices representing the upward path from start to a local max.
        """
        path = [start_vertex]
        current = start_vertex
        
        while graph.up_degree(current) > 0:
            # Pick any successor arbitrarily
            next_vertex = next(graph.successors(current))
            path.append(next_vertex)
            current = next_vertex
        
        return path
    
    @staticmethod
    def _remove_path_edges(graph, path):
        """Remove one edge along each step of the chosen path."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
    
    def decompose(self, reebgraph):
        '''
        Decomposes the Reeb graph into branches. 
        '''
        working = reebgraph.copy()

        self._f = reebgraph.f.copy()
        self.paths = []
        branch_rows = []

        # Track endpoint attachments to previously created branches by shared vertex.
        endpoint_owner = {}

        while len(working.edges) > 0:
            start = self._lowest_available_vertex(working)
            if start is None:
                break

            path = self._largest_upward_path(working, start)
            branch_id = len(self.paths)
            start_v = path[0]
            end_v = path[-1]

            low_attach = endpoint_owner.get(start_v, branch_id)
            high_attach = endpoint_owner.get(end_v, branch_id)

            branch_rows.append(
                (working.f[start_v], working.f[end_v], low_attach, high_attach)
            )
            self.paths.append(path)

            for v in path:
                endpoint_owner.setdefault(v, branch_id)

            self._remove_path_edges(working, path)

        # Handle any remaining isolated vertices (never part of any path) as degenerate branches
        for v in working.nodes:
            if v not in endpoint_owner:
                branch_id = len(self.paths)
                f_v = working.f[v]
                branch_rows.append((f_v, f_v, branch_id, branch_id))
                self.paths.append([v])

        if len(branch_rows) == 0:
            self.branches = np.empty((0, 4))
        else:
            self.branches = np.array(branch_rows, dtype=float)

        return self.branches
    
    def get_branches(self):
        '''
        Returns the branches of the Reeb graph. 
        '''
        return self.branches
    
    def get_branch(self, branch_id):
        '''
        Returns the branch with the given ID. 
        '''
        if branch_id < 0 or branch_id >= len(self.branches):
            raise IndexError("branch_id out of range")

        return self.branches[branch_id]
    
    def get_branch_path(self, branch_id):
        '''
        Returns the list of vertices constituting the path for the given branch ID.
        
        Parameters:
            branch_id (int): The ID of the branch.
            
        Returns:
            list: The list of vertices in the path for the given branch.
        '''
        if branch_id < 0 or branch_id >= len(self.paths):
            raise IndexError("branch_id out of range")
        
        return self.paths[branch_id]
    
    def reconstruct(self):
        '''
        Reconstructs the Reeb graph from the branch decomposition stored in self.paths and self._f.
        
        Returns:
            ReebGraph: The reconstructed Reeb graph.
        '''
        from .reebgraph import ReebGraph

        if not self.paths:
            return ReebGraph()

        R = ReebGraph()

        # Add all unique vertices across all paths
        seen = set()
        for path in self.paths:
            for v in path:
                if v not in seen:
                    R.add_node(v, self._f[v], reset_pos=False)
                    seen.add(v)

        # Add one edge per step in each path
        for path in self.paths:
            for i in range(len(path) - 1):
                R.add_edge(path[i], path[i + 1], reset_pos=False)

        R.set_pos_from_f()
        return R
    
    def draw(self, ax=None, figsize=(12, 8)):
        '''
        Draw the branch decomposition with branches ordered left to right.
        
        Each branch is drawn as a vertical line at its proper function values.
        Endpoint labels show the branch attachment information:
        - Lower endpoint labeled with low_attach branch ID
        - Upper endpoint labeled with high_attach branch ID
        
        Parameters:
            ax: matplotlib axis object. If None, creates a new figure.
            figsize: tuple of figure size (width, height)
            
        Returns:
            ax: the matplotlib axis object
        '''
        if len(self.branches) == 0:
            print("No branches to draw.")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        n_branches = len(self.branches)
        
        # Evenly space branches horizontally
        x_positions = np.linspace(0, n_branches - 1, n_branches)
        
        # Darker colors
        purple = '#6B4C7A'  # Darker purple
        green = '#5A8C5A'   # Darker green
        
        # Draw each branch as a vertical line
        for i, (f_low, f_high, low_attach, high_attach) in enumerate(self.branches):
            x = x_positions[i]
            
            # Draw the branch line (black)
            ax.plot([x, x], [f_low, f_high], 'k-', linewidth=2.5)
            
            # Draw points at endpoints
            ax.plot(x, f_low, 'o', color=purple, markersize=10)      # Lower endpoint
            ax.plot(x, f_high, 'o', color=green, markersize=10)      # Upper endpoint
            
            # Label lower endpoint with attachment info
            ax.text(x - 0.12, f_low, f'B{int(low_attach)}', fontsize=12, 
                   ha='right', va='center', color=purple, fontweight='bold')
            
            # Label upper endpoint with attachment info
            ax.text(x - 0.12, f_high, f'B{int(high_attach)}', fontsize=12, 
                   ha='right', va='center', color=green, fontweight='bold')
        
        ax.set_xlabel('Branch (ordered left to right)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Function Value', fontsize=14, fontweight='bold')
        ax.set_title('Branch Decomposition of Reeb Graph', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'B{i}' for i in range(n_branches)], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # Add padding on left side to accommodate labels
        ax.set_xlim(-0.5, n_branches - 0.5)
        
        return ax