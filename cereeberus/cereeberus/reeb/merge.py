
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from warnings import warn

# from cereeberus.compute.merge import isMerge
# from cereeberus.compute.merge import computeMergeTree

from .reebgraph import ReebGraph
class MergeTree(ReebGraph):
    r"""
    A merge tree stored as a ``ReebGraph`` object. Like a Reeb graph, this is a directed graph with a function defined on the vertices. However, in a merge tree, the function is required to iave a single root with function value treated as :math:`\infty`. 

    We also store label information to construct a labeled merge tree. Here, this is a dictionary from some set (usually [1,...,n]) to a subset of vertices of the graph. 

    """

    def __init__(self, 
                    T = None, root = None,
                    f = {}, 
                    labels = {},
                    seed = None, 
                    verbose = False):
        """
        Initialize a merge tree object.
        
        Parameters:
            T: nx.graph, optional. If not none, it should be a tree with a specified root and function values. 
            labels: dict, optional. A dictionary from labels to subsets of vertices.
            
        """

        if T == None:
            super().__init__(seed = seed, verbose = verbose)

            # Always add the infinite node
            self.add_node('v_inf', np.inf)
        else:
            # Check if the input is valid and add it
            T,f = self.inputToDirRootTree(T,root,f)
            super().__init__(T, f, seed, verbose)

            # Fix up the drawing locations
            self.fix_pos_f()

        self.labels = labels
        
    def __str__(self):
        return f'MergeTree with {len(self.nodes)} nodes and {len(self.edges)} edges.'
        
    #------------------------------------------------------#
    # Methods for listing, adding and removing nodes and edges 
    #---------------------------------------------------------#

    def get_finite_nodes(self):
        """
        Returns a list of the finite nodes of the tree. That is, everything except for `v_inf`.
        """
        return [v for v in self.nodes if v != 'v_inf']
    

    def get_leaves(self):
        """
        Returns a list of the leaves of the tree.
        """

        return [v for v in self.nodes if self.down_degree(v) == 0]

    def add_node(self,  vertex, f_vertex, reset_pos=True):
        """
        Adds a node to the tree. Note that this will break the single connected component property of the tree so we assume you will do this in the process of adding more connecting edges.
        """

        super().add_node(vertex, f_vertex, reset_pos)
        warn("Merge trees assume a single connected component. Adding a node may break this assumption.")
    
    def remove_node(self, vertex, reset_pos=True):
        """
        Removes a node from the tree. Note that this will break the single connected component property of the tree if it is not a leaf so we assume you will do this in the process of adding more connecting edges.
        """
        if vertex == 'v_inf':
            raise ValueError("You cannot remove the infinite node")
        
        if self.down_degree(vertex) > 0:
            warn("You removing a vertex with multiple children will create multiple connected components, so ensure that this property is eventually maintained.")

        super().remove_node(vertex, reset_pos) 
    
    
    def add_edge(self, u, v, reset_pos=True):
        """
        Adds the edge (u,v) to the tree. This command will not allow you to add an edge if it will create a loop. 
        """
        
        if self.f[u] < self.f[v]:
            lowerVertex = u
        else:
            lowerVertex = v

        
        if self.up_degree(lowerVertex) > 0:
            raise ValueError(f"Edge ({u,v}) cannot be added. Adding the edge will create a loop in the graph.")
        
        super().add_edge(u,v,reset_pos)
    
    

    def inputToDirRootTree(self, T, root, f):
        """
        Convert the input to a directed rooted tree that respects the internal structure. 

        If T is undirected, it will be converted to a directed tree with the root specified.

        If f is not specified, a function will be induced from the tree from number of edges from the root. 

        A vertex will be added with funciton value infinity. 
        """
        if T == None:
            # Returns Tree, function
            return None, None
        
        elif nx.is_tree(T) == False:
            raise ValueError("The input graph is not a tree.")
        
        elif nx.is_directed(T) == False:
            if root == None:
                raise ValueError("You must specify a root for the tree.")
            else:
                # Convert to directed graph eminating from the root
                # Because of the ReebGraph convention, edges point up to higher function values.
                T = nx.bfs_tree(T,source = root).reverse()
        else: # T is already directed
            # Make sure that this thing will be valid, 
            # I think this is the same as "arboresence" in networkx
            # which is true if maximum in-degree equal to 1. 
            # Since our graphs point upwards, this is the same as 
            # making sure the reverse of our tree is an arboresence.

            if not nx.is_arborescence(T.reverse()):
                raise ValueError("The input tree is not a valid tree.")

        # No matter what, add a dummy node, point the old root at it and 
        # eventually give that new node function value infinity. 
        T.add_node('v_inf')
        T.add_edge(root, 'v_inf')


        # Check that the function is valid 
        if f == {}:
            # Define a function with distance from the root (but reversed). Rigged so the minimum is always at 0. 
            f = nx.shortest_path_length(T, target = root)
            M = max(f.values())
            f = {i:M-f[i] for i in f.keys()}
            f['v_inf'] = np.inf
            
        else:
            for e in T.edges:
                if f[e[0]] < f[e[1]]:
                    raise ValueError("The function values should point upwards in the tree.")
        
        return T,f

    #-----------------------------------------------#
    # Methods for drawing 
    #-----------------------------------------------#

        
    def fix_pos_f(self):
        """
        Update drawing locations to deal with the fact that we have np.inf around.
        
        This sets the drawing location of the infinite node to be directly above the maximum finite node of the tree.
        """
        # Get the neighbor of the infinite node. Note we're assuming a single connected component so there's only one.
        top_vertices = list(self.predecessors('v_inf'))

        if len(top_vertices) == 0:
            warn("The infinite node has no neighbors so there is nothing to be done.")
        else:
            top_vertex = top_vertices[0]

            # Get  min and max function values
            Lmin =  min(self.f.values())
            Lmax = self.f[top_vertex]
            height = Lmax - Lmin

            # Get the drawing location of the neighbor to have same x coordinate as the neighbor, but y coordinate at .3*height above the maximum function value.
            self.pos_f['v_inf'] = (self.pos_f[top_vertex][0], Lmax + .3 * height)
        
    def set_pos_from_f(self, seed=None, verbose=False):
        """
        Fix the drawing locations for the function values. 
        """
        super().set_pos_from_f(seed, verbose)
        if 'v_inf' in self.pos_f.keys():
            self.fix_pos_f()


    def draw(self, with_labels = True, label_type = 'names', with_colorbar = False):
        """
        Draw the merge tree. 

        If `with_labels` is True, the labels will be drawn. This is either the vertex names if `label_type` is 'names' or the merge tree labels if `label_type` is 'labels'.
        """
        # viridis = mpl.colormaps['viridis'].resampled(16)
        fig, ax = plt.subplots()


        color_map = [self.pos_f[v][1] for v in self.nodes]

        nx.draw_networkx_nodes(self,self.pos_f,node_color = color_map)
        nx.draw_networkx_edges(self,self.pos_f)

        if with_labels:
            if label_type == 'names':
                nx.draw_networkx_labels(self,self.pos_f)
            elif label_type == 'labels':
                label_map  = {}
                
                for k, v in self.labels.items():
                    label_map[v] = label_map.get(v, []) + [k]

                nx.draw_networkx_labels(self,self.pos_f, labels = label_map)
            else:
                raise ValueError("The label type must be either 'names' or 'labels'.")

        ax.tick_params(left = True, bottom = False, labelleft = True, labelbottom = False)



    #-----------------------------------------------#
    # Methods for labeled merge trees
    # 
    # -----------------------------------------------# 
    
    def label_all_leaves(self):
        """
        Label all the leaves of the tree by adding them to the labels dictionary if they're not already there. 
        """
        leaves = self.get_leaves()

        if len(self.labels) == 0:
            key = 0
        else:
            key = max(self.labels.keys())
        for i,v in enumerate(leaves):
            if v not in self.labels.values():
                self.labels[key] = v
                key +=1
        
    def add_label(self, vertex, label = None):
        """
        Add a label to a vertex. If not provided, the label will be the next available integer.
        """
        if label == None:
            key = max(self.labels.keys()) + 1
        elif label in self.labels.keys():
            warn(f"The label {label} is already in use but will be reassigned.")
            key = label
        else:
            key = label

        self.labels[key] = vertex

    def add_label_edge(self, u, v, w, f_w, label = None):
        """
        Add a new vertex named `w` at function value `f_w` by subdividing the edge (u,v) and label it. 

        """

        # Add the vertex 
        self.subdivide_edge(u, v, w, f_w)

        # Add the label
        self.add_label(w, label)

    def LCA(self, u, v):
        """
        Compute the least common ancestor of two vertices in the tree. 
        """
        # Get the path to the root from each 
        anc_u = set(nx.shortest_path(self, u, 'v_inf'))
        anc_v = set(nx.shortest_path(self, v, 'v_inf'))

        # Get the intersection of the ancestors
        common_anc = anc_u.intersection(anc_v)

        # Get the vertex with the minimum function value
        return min(common_anc, key = lambda x: self.f[x])

    def LCA_matrix(self, type = 'leaves', return_as_df = False):
        """
        Compute the matrix of least common ancestors. 

        If type is `leaves`, then the rows and columns of the matrix are determined by the leaf sets. 

        If type is `labels`, then the rows and columns of the matrix are determined by the labels internal to the MergeTree.
        """
        if type == 'leaves':
            # print('leaf version')
            nodes = self.get_leaves()
            # keys = list(range(len(nodes)))
            # print(nodes,keys)
            col_labels = nodes

        elif type == 'labels':
            col_labels = list(self.labels.keys())
            nodes = [self.labels[i] for i in col_labels]

        else:
            raise ValueError("The input `type` of LCA matrix must be either 'leaves' or 'labels'.")


        # Initialize the matrix
        n = len(nodes)
        M = np.zeros((n,n))

        # Compute the LCA matrix
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    M[i,j] = self.f[node_i]
                else:
                    LCA_vertex = self.LCA(node_i, node_j)
                    M[i,j] = self.f[LCA_vertex]
        if return_as_df:
            return pd.DataFrame(M,col_labels, col_labels)
        else:
            return M










if __name__=="__main__":
    from cereeberus.data.randomMergeTrees import randomMerge

    R = randomMerge(10)
    M = Merge(R.G, R.fx)
    M.plot_reeb()