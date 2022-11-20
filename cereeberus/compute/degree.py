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