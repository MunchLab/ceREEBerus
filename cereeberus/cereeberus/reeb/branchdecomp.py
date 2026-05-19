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

    **Branch storage:** Internally, branches are split into two arrays:

    - ``self._branch_values`` (``n x 2``, float): ``[f_low, f_high]``
    - ``self._branch_attach`` (``n x 2``, int): ``[low_attach, high_attach]``

    For backward compatibility, ``self.branches`` exposes a combined ``n x 4``
    array view where each row represents one branch in extraction order:

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

    **Path storage:** By default, decomposition paths are not persisted after
    ``decompose`` completes. Set ``store_paths=True`` to keep them.

    **Reconstruction:** :meth:`reconstruct` builds a canonical Reeb graph from
    branch endpoint values and attachment structure, so it does not require a
    stored copy of the original vertex labels or function dictionary.

    Example::

        import cereeberus.data.ex_reebgraphs as ex_rg
        from cereeberus.reeb.branchdecomp import BranchDecomp

        R = ex_rg.dancing_man()
        bd = BranchDecomp()
        bd.decompose(R)
        bd.draw()
        R2 = bd.reconstruct()
    """
    def __init__(self, store_paths=False):
        self._branch_values = np.empty((0, 2), dtype=float)
        self._branch_attach = np.empty((0, 2), dtype=int)
        self.store_paths = store_paths
        self.paths = []

    @property
    def branches(self):
        """Combined branch matrix [f_low, f_high, low_attach, high_attach]."""
        if len(self._branch_values) == 0:
            return np.empty((0, 4), dtype=float)
        return np.column_stack((self._branch_values, self._branch_attach.astype(float)))

    @property 
    def _num_branches(self):
        return len(self._branch_values)

    @branches.setter
    def branches(self, value):
        arr = np.asarray(value)
        if arr.size == 0:
            self._branch_values = np.empty((0, 2), dtype=float)
            self._branch_attach = np.empty((0, 2), dtype=int)
            return

        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("branches must be a 2D array with shape (n, 4)")

        self._branch_values = arr[:, :2].astype(float)
        self._branch_attach = arr[:, 2:].astype(int)
        

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

        self.paths = []
        branch_rows = []

        # Track endpoint attachments to previously created branches by shared vertex.
        endpoint_owner = {}

        while len(working.edges) > 0:
            start = self._lowest_available_vertex(working)
            if start is None:
                break

            path = self._largest_upward_path(working, start)
            branch_id = len(branch_rows)
            start_v = path[0]
            end_v = path[-1]

            low_attach = endpoint_owner.get(start_v, branch_id)
            high_attach = endpoint_owner.get(end_v, branch_id)

            branch_rows.append(
                (working.f[start_v], working.f[end_v], low_attach, high_attach)
            )
            if self.store_paths:
                self.paths.append(path)

            for v in path:
                endpoint_owner.setdefault(v, branch_id)

            self._remove_path_edges(working, path)

        # Handle any remaining isolated vertices (never part of any path) as degenerate branches
        for v in working.nodes:
            if v not in endpoint_owner:
                branch_id = len(branch_rows)
                f_v = working.f[v]
                branch_rows.append((f_v, f_v, branch_id, branch_id))
                if self.store_paths:
                    self.paths.append([v])

        if len(branch_rows) == 0:
            self._branch_values = np.empty((0, 2), dtype=float)
            self._branch_attach = np.empty((0, 2), dtype=int)
        else:
            rows = np.array(branch_rows, dtype=float)
            self._branch_values = rows[:, :2]
            self._branch_attach = rows[:, 2:].astype(int)

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

    def add_branch(self, f_low, f_high, low_attach, high_attach, path=None):
        """Append a branch to the existing decomposition.

        Parameters:
            f_low (float): Function value at the lower endpoint.
            f_high (float): Function value at the upper endpoint.
            low_attach (int): Branch label for lower endpoint attachment.
            high_attach (int): Branch label for upper endpoint attachment.
            path (list, optional): Optional stored path for this branch. Only used
                when ``store_paths=True``.

        Returns:
            numpy.ndarray: The appended branch row
                ``[f_low, f_high, low_attach, high_attach]``.

        Notes:
            The new branch index is ``n`` where ``n`` is the current number of
            branches. Attachment labels must be integers in ``[0, n]``. Using ``n``
            indicates that endpoint is a local extremum on the newly added branch.
        """
        f_low = float(f_low)
        f_high = float(f_high)

        if f_low > f_high:
            raise ValueError("f_low must be less than or equal to f_high")

        branch_id = len(self._branch_values)
        low_attach = int(low_attach)
        high_attach = int(high_attach)

        if not (0 <= low_attach <= branch_id):
            raise ValueError(
                f"low_attach must be an integer in [0, {branch_id}]"
            )
        if not (0 <= high_attach <= branch_id):
            raise ValueError(
                f"high_attach must be an integer in [0, {branch_id}]"
            )

        # Non-local endpoint attachments must lie on the owner branch by value.
        if low_attach != branch_id:
            owner_low, owner_high = self._branch_values[low_attach]
            if not (owner_low <= f_low <= owner_high):
                raise ValueError(
                    "Lower endpoint value must lie in the interval of low_attach branch"
                )
        if high_attach != branch_id:
            owner_low, owner_high = self._branch_values[high_attach]
            if not (owner_low <= f_high <= owner_high):
                raise ValueError(
                    "Upper endpoint value must lie in the interval of high_attach branch"
                )

        self._branch_values = np.vstack(
            (self._branch_values, np.array([[f_low, f_high]], dtype=float))
        )
        self._branch_attach = np.vstack(
            (
                self._branch_attach,
                np.array([[low_attach, high_attach]], dtype=int),
            )
        )

        if self.store_paths:
            self.paths.append([] if path is None else list(path))

        return self.branches[branch_id]
    
    def get_branch_path(self, branch_id):
        '''
        Returns the list of vertices constituting the path for the given branch ID.
        
        Parameters:
            branch_id (int): The ID of the branch.
            
        Returns:
            list: The list of vertices in the path for the given branch.
        '''
        if not self.store_paths:
            raise RuntimeError("paths were not stored; initialize with store_paths=True to access branch paths")

        if branch_id < 0 or branch_id >= len(self.paths):
            raise IndexError("branch_id out of range")
        
        return self.paths[branch_id]
    
    def reconstruct(self):
        '''Reconstruct a Reeb graph from branch endpoint values and attachments.

        Reconstruction proceeds branch-by-branch. When an endpoint attaches to a
        previous branch at a height where no vertex exists yet, the owner branch
        edge is subdivided and the new attachment vertex is reused.

        Returns:
            ReebGraph: A reconstructed Reeb graph.
        '''
        from .reebgraph import ReebGraph

        if len(self._branch_values) == 0:
            return ReebGraph()

        R = ReebGraph()
        branch_paths = {}
        tol = 1e-12
        counter = 0

        def new_vertex_name():
            nonlocal counter
            name = counter
            counter += 1
            return name

        def add_vertex_at_height(f_val):
            v = new_vertex_name()
            R.add_node(v, float(f_val), reset_pos=False)
            return v

        def ensure_vertex_on_branch(branch_id, f_target):
            f_target = float(f_target)
            path = branch_paths[int(branch_id)]

            # Existing vertex at this exact height.
            for v in path:
                if abs(R.f[v] - f_target) <= tol:
                    return v

            if len(path) == 1:
                raise ValueError("Cannot attach at a new height on a degenerate branch")

            # Subdivide the unique segment containing f_target.
            for idx in range(len(path) - 1):
                u, v = path[idx], path[idx + 1]
                f_u, f_v = float(R.f[u]), float(R.f[v])
                lo, hi = min(f_u, f_v), max(f_u, f_v)

                if lo - tol <= f_target <= hi + tol:
                    if abs(f_target - f_u) <= tol:
                        return u
                    if abs(f_target - f_v) <= tol:
                        return v

                    w = new_vertex_name()
                    R.subdivide_edge(u, v, w, f_target)
                    path.insert(idx + 1, w)
                    return w

            raise ValueError("Attachment height is not on owner branch")

        for i in range(len(self._branch_values)):
            low_f, high_f = map(float, self._branch_values[i])
            low_attach, high_attach = map(int, self._branch_attach[i])

            low_v = (
                add_vertex_at_height(low_f)
                if low_attach == i
                else ensure_vertex_on_branch(low_attach, low_f)
            )
            high_v = (
                add_vertex_at_height(high_f)
                if high_attach == i
                else ensure_vertex_on_branch(high_attach, high_f)
            )

            if low_v != high_v:
                R.add_edge(low_v, high_v, reset_pos=False)
                branch_paths[i] = [low_v, high_v]
            else:
                # Degenerate branch represented by a single isolated vertex.
                branch_paths[i] = [low_v]

        R.set_pos_from_f()
        return R

    def branch_smoothing(self, eps):
        """Return a new BranchDecomp equal to the eps-smoothed decomposition.

        This method is functional: it does not mutate the current instance.

        Before shifting endpoints, branches with both endpoints non-local and
        span < 2*eps are pruned when no later branch attaches to them.

        Args:
            eps (float/int): The amount to smooth the Reeb graph by.

        Returns:
            BranchDecomp: A new smoothed decomposition.
        """
        if eps < 0:
            raise ValueError("eps must be non-negative")

        smoothed = BranchDecomp(store_paths=self.store_paths)

        
        # For each branch 
        for i in range(len(self.branches)):
            f_low = self._branch_values[i, 0]
            f_high = self._branch_values[i, 1]
            low_attach = self._branch_attach[i, 0]
            high_attach = self._branch_attach[i, 1]
            
            # If both endpoints are local extrema 
            if low_attach == i and high_attach == i:
                # add a new branch to the smoothed decoomposition with shifted endpoints
                pass 
                
            
            # If top is local max and bottom is not local min
            elif low_attach != i and high_attach == i:
                pass 
            
            # if botom is local min and top is not local max
            elif low_attach == i and high_attach != i:
                pass
            
            # if both are not min/max 
            else:
                pass

        return smoothed
    
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
        if len(self._branch_values) == 0:
            print("No branches to draw.")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        n_branches = len(self._branch_values)
        
        # Evenly space branches horizontally
        x_positions = np.linspace(0, n_branches - 1, n_branches)
        
        # Darker colors
        purple = '#6B4C7A'  # Darker purple
        green = '#5A8C5A'   # Darker green
        
        # Draw each branch as a vertical line
        for i in range(n_branches):
            f_low, f_high = self._branch_values[i]
            low_attach, high_attach = self._branch_attach[i]
            x = x_positions[i]
            
            # Draw the branch line (black)
            ax.plot([x, x], [f_low, f_high], 'k-', linewidth=2.5)
            
            # Draw points at endpoints
            ax.plot(x, f_low, 'o', color=purple, markersize=10)      # Lower endpoint
            ax.plot(x, f_high, 'o', color=green, markersize=10)      # Upper endpoint
            
            # Label lower endpoint with attachment info
            lower_label = f'(B{int(low_attach)})' if low_attach == i else f'B{int(low_attach)}'
            ax.text(x - 0.12, f_low, lower_label, fontsize=12, 
                   ha='right', va='center', color=purple, fontweight='bold')
            
            # Label upper endpoint with attachment info
            upper_label = f'(B{int(high_attach)})' if high_attach == i else f'B{int(high_attach)}'
            ax.text(x - 0.12, f_high, upper_label, fontsize=12, 
                   ha='right', va='center', color=green, fontweight='bold')
        
        ax.set_xlabel('Branch', fontsize=14 )
        ax.set_ylabel('Function Value', fontsize=14 )
        ax.set_title('Branch Decomposition of Reeb Graph', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'B{i}' for i in range(n_branches)], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # Add padding on left side to accommodate labels
        ax.set_xlim(-0.5, n_branches - 0.5)
        
        return ax