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

def line_loop_index(R):
    edge_list = list(R.edges)
    n = len(R.edges)
    loop_index=[]
    line_index=[]
    for i in range(0,n):
        if edge_list[i][2]==1:
            loop_index.append(edge_list.index(edge_list[i][0:2] + (0,)))
            loop_index.append(i)
            line_index.remove(edge_list.index(edge_list[i][0:2] + (0,)))
        else:
            line_index.append(i)

    return(line_index, loop_index)

def slope_intercept(pt0, pt1):
    m = (pt0[1] - pt1[1]) / (pt0[0] - pt1[0])
    b = pt0[1] - m * pt0[0]
    return (m, b)

def bezier_curve(pt0, midpt, pt1):
    (x1, y1, x2, y2) = (pt0[0], pt0[1], midpt[0], midpt[1])
    (a1, b1) = slope_intercept(pt0, midpt)
    (a2, b2) = slope_intercept(midpt, pt1)
    points = []

    for i in range(0, 100):
        if x1 == x2:
            continue
        else:
            (a, b) = slope_intercept((x1,y1), (x2,y2))
        x = i*(x2 - x1)/100 + x1
        y = a*x + b
        points.append((x,y))
        x1 += (midpt[0] - pt0[0])/100
        y1 = a1*x1 + b1
        x2 += (pt1[0] - midpt[0])/100
        y2 = a2*x2 + b2
    return points

def reeb_plot(R):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    viridis = mpl.colormaps['viridis'].resampled(16)
    fig, ax = plt.subplots()

    n = len(R.nodes)
    fx_sum = 0
    fx_max = 0
    fx_min = 0
    for i in range(0, n):
        fx_sum += R.fx[i]
        fx_max = max(fx_max, R.fx[i])
        fx_min = min(fx_min, R.fx[i])

    fx_mean = fx_sum/n
    colormap = []
    for i in range (0,n):
        colormap.append((R.fx[i]-fx_min)/fx_max)


    edge_list = list(R.edges)
    line_index, loop_index = line_loop_index(R)
    for i in line_index:
        node0 = edge_list[i][0]
        node1 = edge_list[i][1]
        x_pos = (R.pos_fx[node0][0], R.pos_fx[node1][0])
        y_pos = (R.pos_fx[node0][1], R.pos_fx[node1][1])
        ax.plot(x_pos, y_pos, color='grey', zorder = 0)
    
    for i in loop_index:
        node0 = edge_list[i][0]
        node1 = edge_list[i][1]
        xmid = (R.pos_fx[node0][0]+R.pos_fx[node1][0])/2
        xmid0 = xmid - .5*xmid
        xmid1 = xmid + .5*xmid
        ymid = (R.pos_fx[node0][1]+R.pos_fx[node1][1])/2
        curve = bezier_curve(R.pos_fx[node0], (xmid0, ymid), R.pos_fx[node1])
        c = np.array(curve)
        plt.plot(c[:,0], c[:,1], color='grey', zorder = 0)
        curve = bezier_curve(R.pos_fx[node0], (xmid1, ymid), R.pos_fx[node1])
        c = np.array(curve)
        plt.plot(c[:,0], c[:,1], color='grey', zorder = 0)

    for i in range(0, len(R.nodes)):
        ax.scatter(R.pos_fx[i][0], R.pos_fx[i][1], s = 250, color = viridis(colormap[i]))
    
    plt.xlabel('X')
    plt.ylabel('Y')