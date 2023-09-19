
import networkx as nx
import numpy as np
from cereeberus.reeb.graph import Reeb

def randomMerge(n = 10):
    """
    Generates a random tree with n nodes + 1 root.
    Function defined is inverted distance from root node. 
    """
    # Generate a random tree
    T = nx.random_tree(n)

    # Chose a random vertex to be the topmost vertex. This will
    # eventually be connected to the infinite function value root. 
    topNodeIndex = np.random.randint(n)

    # Generate a function by getting distance to the top node
    # then reversing so that bottom leaves have height 0.
    pathlengths = nx.shortest_path_length(T, source = topNodeIndex)
    fx = [pathlengths[i] for i in range(n)]
    fx = [max(fx)-k for k in fx]

    # Add a root node 
    T.add_node(n)
    T.add_edge(topNodeIndex,n)
    fx.append(np.inf)

    # Generate the Reeb graph class for this input 
    T_Merge = Reeb(T,fx)

    # Overwrite the position drawing to make drawing not freak out
    T_Merge.pos_fx[n] = ( T_Merge.pos[n][0], max(fx[:-1]) + 3)

    return T_Merge