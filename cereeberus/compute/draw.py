def line_loop_index(R):
    """Determine if edges between two nodes should be lines or loops
    
    Args:
        R (Reeb Graph): Reeb Graph

    Returns:
        2-element tuple containing

        - **line_index (list)** : list of indices for edges to be drawn as lines
        - **loop_index (list)** : list of indices for edges to be drawn as loops
    """
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
    """Compute slope and intercept to be used in the bezier curve function
    
    Args:
        pt0 (ordered pair): first point
        pt1 (ordered pair): second point

    Returns:
        2-element tuple containing

        - **m (float)** : slope
        - **b (float)** : intercept
    """
    m = (pt0[1] - pt1[1]) / (pt0[0] - pt1[0])
    b = pt0[1] - m * pt0[0]
    return (m, b)

def bezier_curve(pt0, midpt, pt1):
    """Compute bezier curves for plotting two edges between a single set of nodes
    
    Args:
        pt0 (ordered pair): first point
        midpt (ordered pair): midpoint for bezier curve to pass through
        pt1 (ordered pair): second point

    Returns:
        points (np array): array of points to be used in plotting
    """

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

def reeb_plot(R, cp=.5):
    """Compute bezier curves for plotting two edges between a single set of nodes
    
    Args:
        R (Reeb Graph): object of Reeb Graph class
        cp (float): parameter to control curvature of loops in the plotting function

    Returns:
        plot (Reeb Graph): custom visualization of Reeb Graph
    """

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
        xmid0 = xmid - cp*xmid
        xmid1 = xmid + cp*xmid
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