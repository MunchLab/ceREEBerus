import matplotlib.pyplot as plt
import numpy as np
import uuid

class Branch():
    """
    A single branch in a branch decomposition of a Reeb graph. Stored as a doubly linked list data structure. 

    This will store: 
    - prev_branch: a pointer to the previous branch in the decomposition. If this is the first branch, this is None.
    - next_branch: a pointer to the next branch in the decomposition. If this is the last branch, this is None.
    - f_low: function value of the lower endpoint.
    - f_high: function value of the upper endpoint. This should always be strictly higher than f_low, although at intermediate steps of the decomposition, it may be equal to f_low for degenerate branches (isolated vertices).
    - low_attach: pointer to the branch that the lower endpoint is attached to. This needs to be a UUID key.If the lower endpoint is a local minimum, this is None (TODO: do I want to do this?). 
    - high_attach: pointer to the branch that the upper endpoint is attached to. If the upper endpoint is a local maximum, this is None (TODO: do I want to do this?).
    - incoming_low_attach: a list of pointers to previous branches that have their lower endpoint attached to this branch. 
    - incoming_high_attach: a list of pointers to previous branches that have their upper endpoint attached to this branch.
    - key: a unique identifier for the branch (UUID)
    - label: A nice printable label for the branch (string)
    """
    def __init__(self, f_low, f_high, low_attach, high_attach, key,  label=None,):

        if not isinstance(low_attach, (uuid.UUID, type(None))):
            raise ValueError("low_attach must be a UUID key or None")
        if not isinstance(high_attach, (uuid.UUID, type(None))):
            raise ValueError("high_attach must be a UUID key or None")

        if f_low == f_high:
            raise Warning("f_low is equal to f_high. This is a degenerate branch and may indicate a problem with the decomposition.")
        elif f_low > f_high:
            raise ValueError("f_low must be less than f_high")

        self.f_low = f_low
        self.f_high = f_high
        self.low_attach = low_attach
        self.high_attach = high_attach
        self.prev_branch = None
        self.next_branch = None
        self.incoming_low_attach = []
        self.incoming_high_attach = []
        self.label = label
        self.key = key

    def __print__(self):
        return f"Branch(key={self.key}, f_low={self.f_low}, f_high={self.f_high}, low_attach={self.low_attach}, high_attach={self.high_attach}, label={self.label})"
    
class BranchDecomp:
    """
    A branch decomposition of a Reeb graph.

    A branch decomposition breaks a Reeb graph into a collection of non-overlapping upward paths called *branches*. Each branch is a path that travels strictly upward in function value, starting at either a local minimum or a saddle point (where a previous branch split off) and ending at either a local maximum or another saddle point. The decomposition is built greedily and is not unique.

    **Algorithm:** Branches are extracted one at a time by repeatedly finding the
    unprocessed vertex with the lowest function value that still has outgoing edges,
    then greedily following successors upward until a local maximum is reached. The
    edges along that path are removed from a working copy of the graph, and the
    process repeats until no edges remain. Isolated vertices (no edges) are added last
    as degenerate branches.

    **Branch storage:** Internally, branches are stored in a doubly linked list, with each entry given by the Branch class. 

    TODO: Does this need updating? 
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
        self.branches_by_id = {}
        self.first_branch = None
        self.last_branch = None
        self.store_paths = store_paths
        self.paths = []

    @property
    def _num_branches(self):
        """Number of branches in the decomposition."""
        return len(self.branches_by_id)


    def __len__(self):
        return self._num_branches

    def __iter__(self):
        """
        Iterate through the branches in the branch decomposition
        """
        current = self.first_branch
        while current is not None:
            yield self.branches_by_id[current]
            current = self.branches_by_id[current].next_branch

    def __getitem__(self, index):
        '''
        Get the branch at the given index in the list of branches. Note this is NOT the branch key (UUID) but rather the index in the list of branches.
        '''

        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds since there are only {len(self)} branches in the decomposition.")

        for i, branch in enumerate(self):
            print(i,branch)
            if i == index:
                return branch

        raise IndexError("Index out of bounds")

    def add_branch(self, f_low, f_high, 
                        low_attach, high_attach, 
                        prev_branch_in_list=None, next_branch_in_list=None,
                        path=None, label = None):
        """Append a branch to the existing decomposition.

        Parameters:
            f_low (float): Function value at the lower endpoint.
            f_high (float): Function value at the upper endpoint.
            low_attach (integer/UUID): 
                Either an integer, in which case it represents the index of the branch in the list, or a UUID, in which case it is the key of the branch to which the lower endpoint is attached. If the lower endpoint is a local minimum, this is None.
            high_attach (integer/UUID): 
                Either an integer, in which case it represents the index of the branch in the list, or a UUID, in which case it is the key of the branch to which the upper endpoint is attached. If the upper endpoint is a local maximum, this is None.
            prev_branch_in_list (UUID, optional): Branch key (UUID) for the previous branch in the doubly linked list. If None, this branch is added to the end of the list.
            next_branch_in_list (UUID, optional): Branch key (UUID) for the next branch in the doubly linked list. If None, this branch is added to the end of the list. Note that there will be an error if prev_branch_in_list is not None and next_branch_in_list is not None, since this would create a cycle in the list.
            path (list, optional): Optional stored path for this branch. Only used
                when ``store_paths=True``.
            label (str, optional): Optional printable label for the branch. Note this is different than the key used to identify the branch in the decomposition. 

        """

        # Check that low_attach and high_attach are either integers or UUIDs
        # if type(low_attach) == int:
        #     # Find the ith branch in the list and get its UUID key
        #     pass 


        new_branch_key = uuid.uuid4()
        f_low = float(f_low)
        f_high = float(f_high)
        if f_low > f_high:
            raise ValueError("f_low must be less than or equal to f_high")

        self.branches_by_id[new_branch_key] = Branch(f_low, f_high, low_attach, high_attach, new_branch_key, label)


        # low_attach = int(low_attach)
        # high_attach = int(high_attach)

        # if not (0 <= low_attach <= branch_id):
        #     raise ValueError(
        #         f"low_attach must be an integer in [0, {branch_id}]"
        #     )
        # if not (0 <= high_attach <= branch_id):
        #     raise ValueError(
        #         f"high_attach must be an integer in [0, {branch_id}]"
        #     )

        # # Non-local endpoint attachments must lie on the owner branch by value.
        # if low_attach != branch_id:
        #     owner_low, owner_high = self._branch_values[low_attach]
        #     if not (owner_low <= f_low <= owner_high):
        #         raise ValueError(
        #             "Lower endpoint value must lie in the interval of low_attach branch"
        #         )
        # if high_attach != branch_id:
        #     owner_low, owner_high = self._branch_values[high_attach]
        #     if not (owner_low <= f_high <= owner_high):
        #         raise ValueError(
        #             "Upper endpoint value must lie in the interval of high_attach branch"
        #         )

        # self._branch_values = np.vstack(
        #     (self._branch_values, np.array([[f_low, f_high]], dtype=float))
        # )
        # self._branch_attach = np.vstack(
        #     (
        #         self._branch_attach,
        #         np.array([[low_attach, high_attach]], dtype=int),
        #     )
        # )

        # if self.store_paths:
        #     self.paths.append([] if path is None else list(path))

        # return self.branches[branch_id]



    

    # EDITED DOWN TO HERE!

    # @branches.setter
    # def branches(self, value):
    #     arr = np.asarray(value)
    #     if arr.size == 0:
    #         self._branch_values = np.empty((0, 2), dtype=float)
    #         self._branch_attach = np.empty((0, 2), dtype=int)
    #         return

    #     arr = np.asarray(arr, dtype=float)
    #     if arr.ndim != 2 or arr.shape[1] != 4:
    #         raise ValueError("branches must be a 2D array with shape (n, 4)")

    #     self._branch_values = arr[:, :2].astype(float)
    #     self._branch_attach = arr[:, 2:].astype(int)
        

    # @staticmethod
    # def _lowest_available_vertex(graph):
    #     """
    #     Find the vertex with the lowest function value that still has outgoing edges.
        
    #     Parameters:
    #         graph: ReebGraph object
            
    #     Returns:
    #         The vertex with lowest function value and out_degree > 0, or None if no such vertex exists.
    #     """
    #     available_vertices = [v for v in graph.nodes if graph.up_degree(v) > 0]
        
    #     if not available_vertices:
    #         return None
        
    #     return min(available_vertices, key=lambda v: graph.f[v])
    
    # @staticmethod
    # def _largest_upward_path(graph, start_vertex):
    #     """
    #     Greedily follow upward edges from a starting vertex until reaching a local maximum.
        
    #     Parameters:
    #         graph: ReebGraph object
    #         start_vertex: vertex to start from
            
    #     Returns:
    #         A list of vertices representing the upward path from start to a local max.
    #     """
    #     path = [start_vertex]
    #     current = start_vertex
        
    #     while graph.up_degree(current) > 0:
    #         # Pick any successor arbitrarily
    #         next_vertex = next(graph.successors(current))
    #         path.append(next_vertex)
    #         current = next_vertex
        
    #     return path
    
    # @staticmethod
    # def _remove_path_edges(graph, path):
    #     """Remove one edge along each step of the chosen path."""
    #     for i in range(len(path) - 1):
    #         u, v = path[i], path[i + 1]
    #         if graph.has_edge(u, v):
    #             graph.remove_edge(u, v)
    
    # def decompose(self, reebgraph):
    #     '''
    #     Decomposes the Reeb graph into branches. 
    #     '''
    #     working = reebgraph.copy()

    #     self.paths = []
    #     branch_rows = []

    #     # Track endpoint attachments to previously created branches by shared vertex.
    #     endpoint_owner = {}

    #     while len(working.edges) > 0:
    #         start = self._lowest_available_vertex(working)
    #         if start is None:
    #             break

    #         path = self._largest_upward_path(working, start)
    #         branch_id = len(branch_rows)
    #         start_v = path[0]
    #         end_v = path[-1]

    #         low_attach = endpoint_owner.get(start_v, branch_id)
    #         high_attach = endpoint_owner.get(end_v, branch_id)

    #         branch_rows.append(
    #             (working.f[start_v], working.f[end_v], low_attach, high_attach)
    #         )
    #         if self.store_paths:
    #             self.paths.append(path)

    #         for v in path:
    #             endpoint_owner.setdefault(v, branch_id)

    #         self._remove_path_edges(working, path)

    #     # Handle any remaining isolated vertices (never part of any path) as degenerate branches
    #     for v in working.nodes:
    #         if v not in endpoint_owner:
    #             branch_id = len(branch_rows)
    #             f_v = working.f[v]
    #             branch_rows.append((f_v, f_v, branch_id, branch_id))
    #             if self.store_paths:
    #                 self.paths.append([v])

    #     if len(branch_rows) == 0:
    #         self._branch_values = np.empty((0, 2), dtype=float)
    #         self._branch_attach = np.empty((0, 2), dtype=int)
    #     else:
    #         rows = np.array(branch_rows, dtype=float)
    #         self._branch_values = rows[:, :2]
    #         self._branch_attach = rows[:, 2:].astype(int)

    #     return self.branches
    
    # def get_branches(self):
    #     '''
    #     Returns the branches of the Reeb graph. 
    #     '''
    #     return self.branches
    
    # def get_branch(self, branch_id):
    #     '''
    #     Returns the branch with the given ID. 
    #     '''
    #     if branch_id < 0 or branch_id >= len(self.branches):
    #         raise IndexError("branch_id out of range")

    #     return self.branches[branch_id]


    
    # def get_branch_path(self, branch_id):
    #     '''
    #     Returns the list of vertices constituting the path for the given branch ID.
        
    #     Parameters:
    #         branch_id (int): The ID of the branch.
            
    #     Returns:
    #         list: The list of vertices in the path for the given branch.
    #     '''
    #     if not self.store_paths:
    #         raise RuntimeError("paths were not stored; initialize with store_paths=True to access branch paths")

    #     if branch_id < 0 or branch_id >= len(self.paths):
    #         raise IndexError("branch_id out of range")
        
    #     return self.paths[branch_id]
    
    # def reconstruct(self):
    #     '''Reconstruct a Reeb graph from branch endpoint values and attachments.

    #     Reconstruction proceeds branch-by-branch. When an endpoint attaches to a
    #     previous branch at a height where no vertex exists yet, the owner branch
    #     edge is subdivided and the new attachment vertex is reused.

    #     Returns:
    #         ReebGraph: A reconstructed Reeb graph.
    #     '''
    #     from .reebgraph import ReebGraph

    #     if len(self._branch_values) == 0:
    #         return ReebGraph()

    #     R = ReebGraph()
    #     branch_paths = {}
    #     tol = 1e-12
    #     counter = 0

    #     def new_vertex_name():
    #         nonlocal counter
    #         name = counter
    #         counter += 1
    #         return name

    #     def add_vertex_at_height(f_val):
    #         v = new_vertex_name()
    #         R.add_node(v, float(f_val), reset_pos=False)
    #         return v

    #     def ensure_vertex_on_branch(branch_id, f_target):
    #         f_target = float(f_target)
    #         path = branch_paths[int(branch_id)]

    #         # Existing vertex at this exact height.
    #         for v in path:
    #             if abs(R.f[v] - f_target) <= tol:
    #                 return v

    #         if len(path) == 1:
    #             raise ValueError("Cannot attach at a new height on a degenerate branch")

    #         # Subdivide the unique segment containing f_target.
    #         for idx in range(len(path) - 1):
    #             u, v = path[idx], path[idx + 1]
    #             f_u, f_v = float(R.f[u]), float(R.f[v])
    #             lo, hi = min(f_u, f_v), max(f_u, f_v)

    #             if lo - tol <= f_target <= hi + tol:
    #                 if abs(f_target - f_u) <= tol:
    #                     return u
    #                 if abs(f_target - f_v) <= tol:
    #                     return v

    #                 w = new_vertex_name()
    #                 R.subdivide_edge(u, v, w, f_target)
    #                 path.insert(idx + 1, w)
    #                 return w

    #         raise ValueError("Attachment height is not on owner branch")

    #     for i in range(len(self._branch_values)):
    #         low_f, high_f = map(float, self._branch_values[i])
    #         low_attach, high_attach = map(int, self._branch_attach[i])

    #         low_v = (
    #             add_vertex_at_height(low_f)
    #             if low_attach == i
    #             else ensure_vertex_on_branch(low_attach, low_f)
    #         )
    #         high_v = (
    #             add_vertex_at_height(high_f)
    #             if high_attach == i
    #             else ensure_vertex_on_branch(high_attach, high_f)
    #         )

    #         if low_v != high_v:
    #             R.add_edge(low_v, high_v, reset_pos=False)
    #             branch_paths[i] = [low_v, high_v]
    #         else:
    #             # Degenerate branch represented by a single isolated vertex.
    #             branch_paths[i] = [low_v]

    #     R.set_pos_from_f()
    #     return R

        
    # def check_branch_path(self, path):
    #     """Given a list of branches in a branch decomposition, check that they satisfy the following properties meaning they constitute a valid upward path in the branch decomposition: 
    #     - Denote the path entires as [B_1,B_2,...,B_k]. 
    #     - Then for each i, either:
    #         - The top of B_i is attached to B_{i+1}, or 
    #         - The bottom of B_{i+1} is attached to B_i.
    #     - Between each consecutive pair of braches, the property above gives a function value where the attaching happens, call this a_i. Then we need a_i < a_{i+1} for all i.

    #     Args:
    #         path (list): integers giving the branch numbers of the path we want to check. 
    #     """
    #     prev_attach_val = self._branch_values[path[0],0] # initialize this to the bottom of the first branch.
    #     for i in range(len(path)-1):
    #         b1 = path[i]
    #         b2 = path[i+1]
    #         attach1 = self._branch_attach[b1, 1] # branch that the top of b1 is attached to
    #         attach2 = self._branch_attach[b2, 0] # branch that the bottom of b2 is attached to

    #         if attach1 == b2:
    #             # print(f"Branch {b1} is attached to {b2} at the top of {b1}.")
    #             attach_val = self._branch_values[b1, 1]
    #         elif attach2 == b1:
    #             # print(f"Branch {b1} is attached to {b2} at the bottom of {b2}.")
    #             attach_val = self._branch_values[b2, 0]
    #         else:
    #             print(f"Branches {b1} and {b2} are not properly attached (step {i} in the path).")
    #             return False
            
    #         if attach_val < prev_attach_val:
    #             print(f"Function values along the path at B{b1} and B{b2} (step {i} in the path) are not strictly increasing:")
    #             print(f"Prev val = {prev_attach_val} > {attach_val} = attach val at current step.")
    #             return False
            
    #         prev_attach_val = attach_val
        
    #     # print("The path is valid.")
    #     return True
    
    # def get_func_vals_for_path(self, path):
    #     """Given a list of branches in a branch decomposition, return the function values between entries along the path. This is only well defined if the path is valid (i.e. satisfies the properties checked by check_branch_path). Specifically, `func_vals[i]` is the function value at which branch `path[i]` is attached to branch `path[i+1]`. This means the list `func_vals` has length one less than the length of `path`. 

    #     Args:
    #         branch_decomp (BranchDecomp): the branch decomposition object we are working with.
    #         path (list): integers giving the branch numbers of the path we want to check. 
    #     """
    #     func_vals = []
    #     for i in range(len(path)-1):
    #         b1 = path[i]
    #         b2 = path[i+1]
    #         attach1 = self._branch_attach[b1, 1] # branch that the top of b1 is attached to
    #         attach2 = self._branch_attach[b2, 0] # branch that the bottom of b2 is attached to

    #         if attach1 == b2:
    #             attach_val = self._branch_values[b1, 1]
    #         elif attach2 == b1:
    #             attach_val = self._branch_values[b2, 0]
    #         else:
    #             print(f"Branches {b1} and {b2} are not properly attached (step {i} in the path).")
    #             return None
            
    #         func_vals.append(attach_val)
        
    #     return func_vals

    # def find_subpath(self, path, a, b=None):
    #     """Given a branch decomposition and a valid path in the branch decomposition, find the branches for the subset of the path with function values between [a,b] with the assumption that the path is valid. If b= None, we assume b=a, and we return a single element [i] which is the branch containing the path at height a. Otherwise, we return the list of branches that are along the path between a and b.

    #     Args:
    #         branch_decomp (BranchDecomp): the branch decomposition we are working with 
    #         path (list): a valid path in the branch decomposition, given as a list of branch numbers.
    #         a (float): the function value we want to find the branch for.   
    #         b (float, optional): the upper bound of the function value range. If None, we assume b=a. Defaults to None.
    #     Returns:
    #         list: the branch numbers of the branches that contain function values in the range [a,b]. If no branches contain values in this range, return an empty list.
    #     """
        
    #     path_list = []

    #     if b is None:
    #         b = a

    #     # check that b is above the bottom of the first branch in the path. If not, return None.
    #     first_branch = path[0]
    #     if b < self._branch_values[first_branch, 0]:
    #         return []
        
    #     # If it's above that one, then we work our way up the path checking the attaching values until we find a spot where a is between the last attaching value and the next attaching value. 
    #     for i in range(len(path)-1):
    #         b1 = path[i]
    #         b2 = path[i+1]
            
    #         # If b1 has top attached to b2, 
    #         if self._branch_attach[b1, 1] == b2:
    #             attach_val = self._branch_values[b1, 1]
    #             if a <= attach_val:
    #                 path_list.append(b1)
    #                 if b <= attach_val:
    #                     return path_list
    #         elif self._branch_attach[b2, 0] == b1:
    #             attach_val = self._branch_values[b2, 0]
    #             if a < attach_val:
    #                 path_list.append(b1)
    #                 if b <= attach_val:
    #                     return path_list
        
        
    #     # If we get through the whole path and haven't returned, either a is still above everything, so we check if the top of the last branch is above a (add it to the path and return), or if then we check if the last branch has a top below b.  
    #     last_branch = path[-1]
    #     if a <= self._branch_values[last_branch, 1]:
    #         path_list.append(last_branch)
    #     return path_list

    # def path_image(self, path,a,b, branch_decomp_im, branch_map):
    #     """Given a branch decomposition `myB` and a valid path (restricted to (a,b)) in the branch decomposition, along with a map from myB to myB2 using the branch_map, find the image of the subpath with interval (a,b) in myB2 as a valid branch path.  

    #     Args:
    #         path (list): a valid path in the branch decomposition, given as a list of branch numbers.
    #         a (float): the function value we want to find the branch for.   
    #         b (float, optional): the upper bound of the function value range. If None, we assume b=a. Defaults to None.
    #         branch_decomp_im (BranchDecomp): the branch decomposition image we are mapping to
    #         branch_map (dict): a map from branches in `branch_decomp` to branches in `branch_decomp_im`
    #     Returns:
    #         list: the branch numbers of the branches in `branch_decomp_im` that contain function values in the range (a,b). If no branches contain values in this range, return an empty list. (TODO: but this should not be possible???)
    #     """ 
        
    #     subpath = self.find_subpath(path, a, b)
    #     func_vals = self.get_func_vals_for_path(subpath)
        
    #     # get the interval for each branch in the subpath. 
    #     relevant_intervals = []
    #     for i in range(len(subpath)):
    #         if i == 0: 
    #             low = a 
    #         else:
    #             low = func_vals[i-1]
    #         if i == len(subpath)-1:
    #             high = b
    #         else:
    #             high = func_vals[i]
    #         relevant_intervals.append((low, high))
            
    #     if len(subpath) == 0:
    #         print("No branches in the path contain values in the given range in `branch_decomp`.")
    #         return []
        
    #     im_subpath = []
    #     for i in range(len(subpath)):
    #         interval = relevant_intervals[i]
    #         im_subpath_branch =  branch_map[subpath[i]]
    #         im_subpath_branch_restricted_to_interval = branch_decomp_im.find_subpath( im_subpath_branch, interval[0], interval[1])
    #         im_subpath.extend(im_subpath_branch_restricted_to_interval)
            
    #     # Get rid of adjacent duplicates in im_subpath (this can happen when we have a long interval that covers multiple branches in the image decomposition that are attached to each other).
    #     im_subpath_no_dups = []
    #     for i in range(len(im_subpath)):
    #         if i == 0 or im_subpath[i] != im_subpath[i-1]:
    #             im_subpath_no_dups.append(im_subpath[i])
    #     im_subpath = im_subpath_no_dups
        
    #     return im_subpath

    # def branch_smoothing(self, eps):

    #     """Given a branch decomposition of a Reeb graph, we want to return the branch decomposition of the smoothed Reeb graph, with parameter epsilon. We will also return the branch_map which gives the path in the smoothed Reeb graph which is the image of a branch from the input Reeb graph.

    #     Args:
    #         branch_decomp (branch.BranchDecomposition): The branch decomposition of the input Reeb graph.
    #         eps (float): The smoothing parameter.
        
    #     Returns:
    #         tuple: A tuple containing the smoothed branch decomposition and the branch map.
    #     """
        
    #     # Check that Epsilon is positive. If not, return an error.
    #     if eps <= 0:
    #         raise ValueError("Epsilon must be positive.")
            
    #     # Create a new branch decomposition object for the smoothed Reeb graph.
    #     smoothed_branch_decomp = branch.BranchDecomp()
        
    #     # Create the branch map dictionary to keep track of the mapping from branches in the input decomposition to paths in the smoothed decomposition.
    #     branch_map = {}
        
    #     for i in range(branch_decomp._num_branches):
    #         # Get the interval for the current branch.
    #         low = branch_decomp._branch_values[i, 0]
    #         high = branch_decomp._branch_values[i, 1]
            
    #         if branch_decomp._branch_attach[i, 0] == i: # Bottom is a local min
    #             if branch_decomp._branch_attach[i, 1] == i: # Top is a local max
    #                 # print(f"\nWorking on B{i}, (local min, local max) case.")
    #                 # This means that both the top and bottom are local extrema, so we add a new branch with extended interval and no attachments. 
    #                 new_branch_num = smoothed_branch_decomp._num_branches
    #                 smoothed_branch_decomp.add_branch(low-eps, high+eps, new_branch_num, new_branch_num)
    #                 branch_map[i] = [new_branch_num]
                
    #             else: # (Local min, down fork) case
    #                 # print(f"\nWorking on B{i}, (local min, downfork) case.")
    #                 # The new branch will have interval [low-eps,high-eps]. 
    #                 new_branch_num = smoothed_branch_decomp._num_branches
                    
    #                 # Starting with the down fork attachment, path_in_old will give the path of branches we get to by following the attaching information down from the top attachment of the current branch until we get to a branch that is below high-eps.
    #                 check_attach_top = branch_decomp._branch_attach[i, 1]
    #                 path_in_old = [check_attach_top] 
                    
    #                 # While the check_attach_top branch doesn't have a local min at the bottom but the function value at the bottom is still above high-eps, we keep following the attaching information down.
    #                 while branch_decomp._branch_attach[check_attach_top, 0] != check_attach_top and branch_decomp._branch_values[check_attach_top, 0] >= high - eps:
    #                     check_attach_top = branch_decomp._branch_attach[check_attach_top, 0]
    #                     path_in_old.append(check_attach_top)
                    
    #                 print(f"Path in old branch decomposition to attach new branch for B{i}: {path_in_old}")
                    
    #                 # Attachment for top of new branch is the image (in the smoothed graph) of the last branch in path_in_old at the height value high-eps.
    #                 # We get this from the function find_subpath 
    #                 new_attach_top = find_subpath(smoothed_branch_decomp, branch_map[path_in_old[-1]], high-eps)[0]
                    
    #                 # Attaching maps are bottom is attached to itself, top is attached to new_attach_top
    #                 smoothed_branch_decomp.add_branch(low-eps,high-eps, new_branch_num, new_attach_top)

    #                 # Reverse the branch map because branch paths go up in the Reeb graph, but this one went down
    #                 path_in_old = list(reversed(path_in_old))
                    
    #                 print(f"Reversed path: {path_in_old}")
                    
    #                 # Now get the image of this path in the new branch decomposition using the branch map.
    #                 # We need to restrict the image to the interval 
    #                 # (max(high-eps,low) , high)
    #                 # since that's the portion that would actually get mapped there. Then after the fact, we add the new branch as a tail at the end.
    #                 branch_map[i] = path_image(path_in_old, max(high-eps,low), high, branch_decomp, smoothed_branch_decomp, branch_map)
                    
    #                 print(f"Image of old-path {path_in_old} for range ({max(high-eps,low)},{high}) in new branch decomposition: {branch_map[i]}")

    #                 # If new high (high-eps) is above low, then the bottom of the branch gets mapped to the new branch, so we need to add that to the beginning of the image as well.
    #                 if high-eps > low :
    #                     branch_map[i].insert(0, new_branch_num)

    #                 # Add a check for valid path 
    #                 if not check_branch_path(smoothed_branch_decomp, branch_map[i]):
    #                     print(f" local min, down fork case made a bad path for branch {i}: {branch_map[i]}")
                    
    #         else: # Bottom is an up fork 
    #             if branch_decomp._branch_attach[i, 1] == i: # (upfork, local max) case
    #                 # print(f"\nWorking on B{i}, (upfork, local max) case.")
    #                 # The new branch will have interval [low+eps, high+eps]. 
    #                 new_branch_num = smoothed_branch_decomp._num_branches
                    
    #                 # Starting with the up fork attachment, path_in_old will give the path of branches we get to by following the attaching information up from the bottom attachment of the current branch until we get to a branch that contains low+eps.
    #                 check_attach_bottom = branch_decomp._branch_attach[i, 0]
    #                 path_in_old = [check_attach_bottom] 

    #                 # While 
    #                 #   - There is still a next branch to follow up at the top (which means branch_decomp._branch_attach[check_attach_bottom, 1] != check_attach_bottom )
    #                 #   - The value a low+eps is not in the check_attach_bottom branch's interval (which means branch_decomp._branch_values[check_attach_bottom, 1] < low + eps. If the top of the branch checking is above low+eps, then we know low+eps is in that branch's interval since the bottom is attached to the previous branch in the path which is below low+eps).
    #                 while branch_decomp._branch_attach[check_attach_bottom, 1] != check_attach_bottom and branch_decomp._branch_values[check_attach_bottom, 1] < low + eps:
    #                     check_attach_bottom = branch_decomp._branch_attach[check_attach_bottom, 1]
    #                     path_in_old.append(check_attach_bottom)
                    
    #                 # Attachment for bottom of new branch is the image of the last branch in path_in_old at the height value low+eps. We get this from the function find_subpath 
    #                 new_attach_low = find_subpath(smoothed_branch_decomp, branch_map[path_in_old[-1]], low+eps)[0]
                    
    #                 # Attaching maps are top is attached to itself, bottom is attached to new_attach_low
    #                 smoothed_branch_decomp.add_branch(low+eps, high+eps, new_attach_low, new_branch_num)

    #                 # Now get the image of this path in the new branch decomposition using the branch map.
    #                 # Like before, we need to restrict the image to the interval (low, min(low+eps,high)) since that's the portion that would actually get mapped there. Then after the fact, we add the new branch as a head at the beginning.
    #                 branch_map[i] = path_image(path_in_old, low, min(low+eps,high), branch_decomp, smoothed_branch_decomp, branch_map)
                    
    #                 # If new low (low+eps) is below high, then the top of the branch gets mapped to the new branch, so we need to add that to the image as well.
    #                 if high > low + eps: 
    #                     branch_map[i].append(new_branch_num)
    
    #                 # Add a check for valid path 
    #                 if not check_branch_path(smoothed_branch_decomp, branch_map[i]):
    #                     print(f" upfork, local max case made a bad path for branch {i}: {branch_map[i]}")                   
                    
    #             else: # Top and bottom are both forks. This is the hard one.
    #                 print(f"\nWarning: B{i} is an (upfork, downfork) case. - NOT IMPLEMENTED YET")
    #                 if high-low <= 2*eps:
    #                     # This is a short edge. It will go away, but how it does needs to be determined. 
                        
    #                     # TODO Update mapping 
    #                     branch_map[i] = []
    #                     pass 
                    
    #                 else:
    #                     # The top will go down and the bottom will go up, so the new branch will have interval [low+eps, high-eps]. 
    #                     new_branch_low = low+eps
    #                     new_branch_high = high-eps
    #                     new_branch_num = smoothed_branch_decomp._num_branches
    #                     # TODO Fix attaching maps
    #                     smoothed_branch_decomp.add_branch(new_branch_low, new_branch_high, new_branch_num, new_branch_num)
    #                     # TODO Update mapping
    #                     branch_map[i] = []
                        
    #     return smoothed_branch_decomp, branch_map
  

    
    # def draw(self, ax=None, figsize=(12, 8)):
    #     '''
    #     Draw the branch decomposition with branches ordered left to right.
        
    #     Each branch is drawn as a vertical line at its proper function values.
    #     Endpoint labels show the branch attachment information:
    #     - Lower endpoint labeled with low_attach branch ID
    #     - Upper endpoint labeled with high_attach branch ID
        
    #     Parameters:
    #         ax: matplotlib axis object. If None, creates a new figure.
    #         figsize: tuple of figure size (width, height)
            
    #     Returns:
    #         ax: the matplotlib axis object
    #     '''
    #     if len(self._branch_values) == 0:
    #         print("No branches to draw.")
    #         return
        
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=figsize)
        
    #     n_branches = len(self._branch_values)
        
    #     # Evenly space branches horizontally
    #     x_positions = np.linspace(0, n_branches - 1, n_branches)
        
    #     # Darker colors
    #     purple = '#6B4C7A'  # Darker purple
    #     green = '#5A8C5A'   # Darker green
        
    #     # Draw each branch as a vertical line
    #     for i in range(n_branches):
    #         f_low, f_high = self._branch_values[i]
    #         low_attach, high_attach = self._branch_attach[i]
    #         x = x_positions[i]
            
    #         # Draw the branch line (black)
    #         ax.plot([x, x], [f_low, f_high], 'k-', linewidth=2.5)
            
    #         # Draw points at endpoints
    #         ax.plot(x, f_low, 'o', color=purple, markersize=10)      # Lower endpoint
    #         ax.plot(x, f_high, 'o', color=green, markersize=10)      # Upper endpoint
            
    #         # Label lower endpoint with attachment info
    #         lower_label = f'(B{int(low_attach)})' if low_attach == i else f'B{int(low_attach)}'
    #         ax.text(x - 0.12, f_low, lower_label, fontsize=12, 
    #                ha='right', va='center', color=purple, fontweight='bold')
            
    #         # Label upper endpoint with attachment info
    #         upper_label = f'(B{int(high_attach)})' if high_attach == i else f'B{int(high_attach)}'
    #         ax.text(x - 0.12, f_high, upper_label, fontsize=12, 
    #                ha='right', va='center', color=green, fontweight='bold')
        
    #     ax.set_xlabel('Branch', fontsize=14 )
    #     ax.set_ylabel('Function Value', fontsize=14 )
    #     ax.set_title('Branch Decomposition of Reeb Graph', fontsize=16)
    #     ax.grid(True, alpha=0.3)
    #     ax.set_xticks(x_positions)
    #     ax.set_xticklabels([f'B{i}' for i in range(n_branches)], fontsize=12)
    #     ax.tick_params(axis='y', labelsize=12)
        
    #     # Add padding on left side to accommodate labels
    #     ax.set_xlim(-0.5, n_branches - 0.5)
        
    #     return ax