import numpy as np
import networkx as nx

def up_degree(R, fx = {}):
    """ Compute Upper Degree of Reeb Graph
    degree.up_degree is deprecated. Instead use R.up_degree() to compute the up degree of a node in a Reeb graph.

    Args:
        R (reeb graph): networkx or reeb graph to use for reeb graph computation

    Returns:
        up_deg (dict): dictionary of up degrees by node
    """

    n = len(R.nodes)
    up_adj = np.zeros((n,n))
    i = 0
    
    RCopy = list(R.nodes)

    for inode in R.nodes:
        RCopy.remove(inode)
        j = i+1
        for jnode in RCopy:
            if fx[inode] < fx[jnode]:
                e = list(R.edges(inode))
                if (inode,jnode) in e:
                    up_adj[j,i]+=1
            if fx[inode] > fx[jnode]:
                e = list(R.edges(inode))
                if (inode,jnode) in e:
                    up_adj[i,j]+=1
            j+=1
        i+=1

    d = sum(up_adj)

    up_deg = {}
    i = 0
    for node in R.nodes:
        up_deg[node] = int(d[i])
        i+=1
    return up_deg

def down_degree(R, fx ={ }):

    """ Compute Down Degree of Reeb Graph
    degree.down_degree is deprecated. Instead use R.down_degree() to compute the down degree of a node in a Reeb graph.

    Args:
        R (reeb graph): networkx or reeb graph to use for reeb graph computation

    Returns:
        down_deg (dict): dictionary of down degrees by node
    
    
    """

    n = len(R.nodes)
    down_adj = np.zeros((n,n))
    i = 0
    
    RCopy = list(R.nodes)

    for inode in R.nodes:
        RCopy.remove(inode)
        j = i+1
        for jnode in RCopy:
            if fx[inode] > fx[jnode]:
                e = list(R.edges(inode))
                if (inode,jnode) in e:
                    down_adj[j,i]+=1
            if fx[inode] < fx[jnode]:
                e = list(R.edges(inode))
                if (inode,jnode) in e:
                    down_adj[i,j]+=1
            j+=1
        i+=1

    d = sum(down_adj)

    down_deg = {}
    i = 0
    for node in R.nodes:
        down_deg[node] = int(d[i])
        i+=1
    return down_deg

def add_nodes(R, fx, x=0):
    """ Function to add nodes to a Reeb Graph
    degree.add_nodes is deprecated. You can now use R.add_node() to add nodes to a Reeb graph. 
    """
    print('degree.add_nodes is deprecated. You can now use R.add_node() to add nodes to a Reeb graph.')
    from cereeberus.reeb.reebgraph import ReebGraph
    r = len(R.edges)
    e = list(R.edges)
    c = 0
    for i in range(0,r):
        pt0 = e[i][0]
        pt1 = e[i][1]
        f0 = R.fx[pt0]
        f1 = R.fx[pt1]
        if f0 < fx < f1 or f1 < fx < f0:
            R.fx[r+c] = fx
            R.G.add_edge(pt0, r+c)
            R.G.add_edge(pt1, r+c)
            R.G.add_node(r+c, fx = fx, pos = (x, fx))
            R.G.remove_edge(pt0, pt1)
            c+=1
    return ReebGraph(R.G)

def minimal_reeb(R):
    """ Function to create minimal Reeb Graph
    """
    print("degree.minimal_reeb is deprecated. You can now use R.remove_regular_vertices() to remove regular vertices from a Reeb graph.")

    R.remove_all_regular_vertices()
    return R
    # from cereeberus.reeb.reebgraph import ReebGraph
    # H = R.G.copy()
    # for i in H.nodes:
    #     if R.up_deg[i] == R.down_deg[i] == 1:
    #         e = list(H.edges(i))
    #         pt0 = e[0][1]
    #         pt1 = e[1][1]
    #         H.add_edge(pt0, pt1)
    #         H.remove_edge(i, pt0)
    #         H.remove_edge(i, pt1)
    # for i in R.nodes:
    #     if R.up_deg[i] == R.down_deg[i] == 1:
    #         H.remove_node(i)
    # H = nx.convert_node_labels_to_integers(H)
    # return ReebGraph(H)

def remove_isolates(R):
    """ 
    Function to remove isolates from Reeb Graph.  Important for computation of Merge Tree
    """
    from cereeberus.reeb.reebgraph import ReebGraph
    H = R.G.copy()
    for i in R.nodes:
        if R.up_deg[i] == R.down_deg[i] == 0:
            H.remove_node(i)
    H = nx.convert_node_labels_to_integers(H)
    return ReebGraph(H)

def heights(graph):
    h = []
    for i in graph.nodes:
        pt = (i, graph.fx[i])
        h.append(pt)
    h.sort(key = lambda x: x[1])
    return h