def up_degree(R, fx = {}):
    """ Compute Upper Degree of Reeb Graph

    Args:
        R (reeb graph): networkx or reeb graph to use for reeb graph computation

    Returns:
        up_deg (dict): dictionary of up degrees by node
    
    """

    import numpy as np
    n = len(R.nodes)
    up_adj = np.zeros((n,n))

    for i in range(0,n):
        for j in range(i,n):
            if fx[i] < fx[j]:
                e = list(R.edges(i))
                if (i,j) in e:
                    up_adj[j,i]+=1
            if fx[i] > fx[j]:
                e = list(R.edges(i))
                if (i,j) in e:
                    up_adj[i,j]+=1

    d = sum(up_adj)

    up_deg = {}
    for i in range(0,n):
        up_deg[i] = int(d[i])
    return up_deg

def down_degree(R, fx ={ }):

    """ Compute Down Degree of Reeb Graph

    Args:
        R (reeb graph): networkx or reeb graph to use for reeb graph computation

    Returns:
        down_deg (dict): dictionary of down degrees by node
    
    """

    import numpy as np
    n = len(R.nodes)
    down_adj = np.zeros((n,n))

    for i in range(0,n):
        for j in range(i,n):
            if fx[i] > fx[j]:
                e = list(R.edges(i))
                if (i,j) in e:
                    down_adj[j,i]+=1
            if fx[i] < fx[j]:
                e = list(R.edges(i))
                if (i,j) in e:
                    down_adj[i,j]+=1

    d = sum(down_adj)

    down_deg = {}
    for i in range(0,n):
        down_deg[i] = int(d[i])
    return down_deg

def add_nodes(R, fx, x=0):
    from reeb.reeb import Reeb
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
    return Reeb(R.G)

def minimal_reeb(R):
    import networkx as nx
    from reeb.reeb import Reeb
    H = R.G.copy()
    for i in H.nodes:
        if R.up_deg[i] == R.down_deg[i] == 1:
            e = list(H.edges(i))
            pt0 = e[0][1]
            pt1 = e[1][1]
            H.add_edge(pt0, pt1)
            H.remove_edge(i, pt0)
            H.remove_edge(i, pt1)
    for i in R.nodes:
        if R.up_deg[i] == R.down_deg[i] == 1:
            H.remove_node(i)
    H = nx.convert_node_labels_to_integers(H)
    return Reeb(H)