from cereeberus import ReebGraph, MapperGraph, Interleave
import cereeberus.data.ex_mappergraphs as ex_mg
import cereeberus.distance.ilp as ilp
import matplotlib.pyplot as plt
import random

from cereeberus import ReebGraph, MapperGraph, Interleave
import cereeberus.data.ex_mappergraphs as ex_mg
import cereeberus.distance.ilp as ilp
import matplotlib.pyplot as plt
import random

def run_optimization_torus_line(a,b,c,d,n, num_iter):
    """
    Run the optimization algorithm on a line and torus mapper graph for a given number of iterations.

    Parameters
    a (int): The height of the bottom most vertex at the very bottom of both the line and torus mapper graph.
    b (int): The height where the loop of the torus mapper graph starts.
    c (int): The height where the loop of the torus mapper graph ends.
    d (int): The height of the top most vertex at the very top of both the line and torus mapper graph.
    n (int): Thickening parameter.
    num_iter (list): A list of the number of iterations to run the optimization algorithm for.

    Returns
    None
    """
    
    losses_before = []
    losses_after = []

    for i in range(num_iter):
        # generate a random seed    
        loop_seed = random.seed(i)

        # generate a line and torus mapper graph
        M1 = ex_mg.line(a,d, seed=loop_seed)
        M2 = ex_mg.torus(a,b,c,d, seed=loop_seed)

        # generate an interleaving of the two mapper graphs
        myInt = Interleave(M1, M2, n = n, initialize_random_maps=True, seed = loop_seed)

        # calculate the loss before optimization
        loss_before = myInt.loss()
        losses_before.append(loss_before) # store the loss

        # optimize the interleaving
        loss_after = ilp.solve_ilp(myInt)[1]
        losses_after.append(loss_after) # store the loss


    return losses_before, losses_after

def plot_losses_and_upper_bounds(losses,n, filename=None):
    """
    Plot the losses and upper bounds for the optimization algorithm.
    
    Parameters
    losses (list): A list of the losses to plot. It contains two sublists, the first containing the losses before optimization and the second containing the losses after optimization.
    n (int): Thickening parameter.
    filename (str): The filename to read loss from.

    Returns
    None
    """

    # compute the upper bounds
    upper_bounds_before = [x+n for x in losses[0]]
    upper_bounds_after = [x+n for x in losses[1]]

    # get num_iter
    num_iter = len(losses[0])
    num_range = range(num_iter)

    # scatter the losses and upper bounds
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(num_range,losses[0], label='before', alpha=0.7, marker='x')
    plt.scatter(num_range, losses[1], label='after', alpha=0.5, marker='o')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('before optimization')
    plt.title(f'Losses for n = {n}')
    plt.subplot(1,2,2)
    plt.scatter(num_range, upper_bounds_before,label='before', alpha=0.5, marker='x')
    plt.scatter(num_range,upper_bounds_after, label='after', alpha=0.5, marker='o')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('after optimization')
    plt.title(f'Upper Bounds for n = {n}')
    plt.show()


