import numpy as np


class BranchDecomp:
    """
    This class encodes a branch decmposition of a given Reeb graph. This is a sorted list of intervals (branches), where the endpoints of the intervals have attaching information, either to a previous branch, or to itself (meaning that end of the interval is a local max/min). The Reeb graph can be reconstructed from the data as well. Note that the decomposition is not unique. 
    
    The branches are stored as an n x 4 array, where n is the number of branches. The first two columns are the real-valued endpoints of the branch, and the last two columns are integers giving the attaching information. The attaching information is a tuple of the form (branch_id_low, branch_id_high), where branch_id_low is the ID of the branch that the lower interval endpoint is attached to (can be its own row if it is a local min), and branch_id_high is the ID of the branch that the upper interval endpoint is attached to (can be its own row if it is a local max). 
   
    
    """
    def __init__(self):
        self.branches = np.empty((0, 4))
        self.paths = []
        

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
            branch_id = len(self.paths)
            start_v = path[0]
            end_v = path[-1]

            low_attach = endpoint_owner.get(start_v, branch_id)
            high_attach = endpoint_owner.get(end_v, branch_id)

            branch_rows.append(
                (working.f[start_v], working.f[end_v], low_attach, high_attach)
            )
            self.paths.append(path)

            endpoint_owner.setdefault(start_v, branch_id)
            endpoint_owner.setdefault(end_v, branch_id)

            self._remove_path_edges(working, path)

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
    
    def reconstruct(self):
        '''
        Reconstructs the Reeb graph from the branches. 
        '''
        raise NotImplementedError("reconstruct is not implemented yet")