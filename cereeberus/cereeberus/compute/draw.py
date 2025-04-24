import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx

def dict_to_list(d):
    return list(d.values())


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

def reeb_plot(R, with_labels = True, with_colorbar = False, cpx=.1, cpy=.1, ax = None, **kwargs):
    """Main plotting function for the Reeb Graph Class

    Parameters: 
        R (Reeb Graph): object of Reeb Graph class
        with_labels (bool): parameter to control whether or not to plot labels
        with_colorbar (bool): parameter to control whether or not to plot colorbar
        cp (float): parameter to control curvature of loops in the plotting function. For vertical Reeb graph, only mess with cpx.

    """
    if ax is None:
        fig, ax = plt.subplots()

    viridis = mpl.colormaps['viridis'].resampled(16)

    n = len(R.nodes)


    edge_list = list(R.edges)
    line_index, loop_index = line_loop_index(R)

    # Some weird plotting to make the colored and labeled nodes work.
    # Taking the list of function values from the pos_f dicationary since the infinite node should already have a position set.
    color_map = [R.pos_f[v][1] for v in R.nodes]
    pathcollection = nx.draw_networkx_nodes(R, R.pos_f, node_color=color_map, ax = ax, **kwargs)
    if with_labels:
        nx.draw_networkx_labels(R, pos=R.pos_f, font_color='black', ax = ax)
    if with_colorbar:
        plt.colorbar(pathcollection)

    for i in line_index:
        node0 = edge_list[i][0]
        node1 = edge_list[i][1]
        x_pos = (R.pos_f[node0][0], R.pos_f[node1][0])
        y_pos = (R.pos_f[node0][1], R.pos_f[node1][1])
        ax.plot(x_pos, y_pos, color='grey', zorder = 0)
    
    for i in loop_index:
        node0 = edge_list[i][0]
        node1 = edge_list[i][1]
        xmid = (R.pos_f[node0][0]+R.pos_f[node1][0])/2
        xmid0 = xmid - cpx*xmid
        xmid1 = xmid + cpx*xmid
        ymid = (R.pos_f[node0][1]+R.pos_f[node1][1])/2
        ymid0 = ymid - cpy*ymid
        ymid1 = ymid + cpy*ymid
        curve = bezier_curve(R.pos_f[node0], (xmid0, ymid0), R.pos_f[node1])
        c = np.array(curve)
        ax.plot(c[:,0], c[:,1], color='grey', zorder = 0)
        curve = bezier_curve(R.pos_f[node0], (xmid1, ymid1), R.pos_f[node1])
        c = np.array(curve)
        ax.plot(c[:,0], c[:,1], color='grey', zorder = 0)


    ax.tick_params(left = True, bottom = False, labelleft = True, labelbottom = False)