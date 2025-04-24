
import networkx as nx
import numpy as np
import warnings

from networkx.generators import random_labeled_tree

from cereeberus import MergeTree




def randomMergeTree(n = 10, func_randomizer = None, range = [0,10],  seed = None):
    """
    Generates a random tree with n nodes + 1 root.
    If `func_randomizer` is None, the function defined is inverted edge count distance from root node. 
    Otherwise, func_randomizer should be an input to type the randomizeMergeFunction function. For example, this could be 'uniform' or 'exp'.
    """
    warnings.filterwarnings('ignore') 

    # Generate a random tree
    T = random_labeled_tree(n, seed = seed)

    np.random.seed(seed)
    root = np.random.randint(n)

    MT = MergeTree(T, root, seed = seed)

    if func_randomizer is not None:
        MT = randomizeMergeFunction(MT, type = func_randomizer, range = range,  seed = seed)

    return MT

def randomizeMergeFunction(MT, range = [0,10], type = 'exp', seed = None):
    """
    Returns a merge tree with the same underlying tree, but randomized function values in the given range.
    """
    np.random.seed(seed)
    f = {'v_inf':np.inf}

    for e in nx.edge_bfs(MT.to_undirected(), 'v_inf'):
        u = e[0]
        v = e[1]

        bot_f = range[0]
        if u == 'v_inf':
            top_f = range[1]
        else:
            top_f = f[u]

        # Here's the main thing. 
        # We pull a random number that's lower than the upper 
        # neighbor's funciton value. 
        # Doing this with uniform repeatedly makes for a squashed merge tree. 
        # The expoenential function is a bit better.
        if type == 'uniform':
            f[v] = np.random.uniform(bot_f, top_f)
        elif type == 'exp':
            f[v] = top_f-np.random.exponential(.1*(top_f-bot_f))
            if f[v] <bot_f:
                f[v] = bot_f

    MT.f = f
    MT.set_pos_from_f()

    return MT